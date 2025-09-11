# 分布式Energy训练问题解决方案

## 🚨 **问题诊断**

**症状**: 8卡 bs_local=3 时模型几乎不学习，单卡 bs=24 时学习良好

**根本原因**: `energy_inbatch_swap_infonce` 损失函数严重依赖于批内样本的多样性
- 单卡bs=24: 每个正样本有23个负样本，信号强
- 8卡bs_local=3: 每个GPU上每个正样本只有2个负样本，信号极弱

## 🛠️ **解决方案**

### 方案1: Global In-Batch InfoNCE (推荐⭐)

**原理**: 使用`all_gather`收集所有GPU的样本，进行全局InfoNCE计算

```python
def distributed_energy_inbatch_swap_infonce(
    energy_model, h, a_pos, pad_mask, tau=0.5, world_size=None
):
    """
    分布式友好的In-batch InfoNCE损失
    """
    import torch.distributed as dist
    
    if world_size is None or world_size == 1:
        # 单卡情况，直接调用原函数
        return energy_inbatch_swap_infonce(energy_model, h, a_pos, pad_mask, tau)
    
    # Step 1: 收集所有GPU的样本
    h_list = [torch.zeros_like(h) for _ in range(world_size)]
    a_list = [torch.zeros_like(a_pos) for _ in range(world_size)]
    pm_list = [torch.zeros_like(pad_mask) for _ in range(world_size)] if pad_mask is not None else None
    
    dist.all_gather(h_list, h)
    dist.all_gather(a_list, a_pos)
    if pad_mask is not None:
        dist.all_gather(pm_list, pad_mask)
    
    # Step 2: 拼接成全局batch
    h_global = torch.cat(h_list, dim=0)  # [B*world_size, S, D]
    a_global = torch.cat(a_list, dim=0)  # [B*world_size, H, Da]
    pm_global = torch.cat(pm_list, dim=0) if pad_mask is not None else None
    
    # Step 3: 计算全局InfoNCE
    B_local = h.size(0)
    global_loss, E_pos_global, E_neg_global = energy_inbatch_swap_infonce(
        energy_model, h_global, a_global, pm_global, tau
    )
    
    # Step 4: 只取当前GPU对应的正样本能量
    rank = dist.get_rank()
    start_idx = rank * B_local
    end_idx = start_idx + B_local
    
    E_pos_local = torch.diag(compute_local_energy_matrix(energy_model, h, a_pos, pad_mask))
    E_pos_mean = E_pos_local.mean()
    
    return global_loss, E_pos_mean, E_neg_global

def compute_local_energy_matrix(energy_model, h, a_pos, pad_mask):
    """计算本地能量用于监控"""
    B, S, D = h.shape
    H, Da = a_pos.shape[1], a_pos.shape[2]
    
    h_rep = h.unsqueeze(1).expand(B, B, S, D).reshape(B*B, S, D)
    a_rep = a_pos.unsqueeze(0).expand(B, B, H, Da).reshape(B*B, H, Da)
    pm = pad_mask.unsqueeze(1).expand(B, B, -1).reshape(B*B, -1) if pad_mask is not None else None
    
    E_ij = energy_model(h_rep, a_rep, pm).view(B, B)
    return E_ij
```

**使用方法**:
```python
# 在 finetune_Energy.py 中替换第536行
# 原来：
# swap_loss, E_pos_mean, E_neg_mean = energy_inbatch_swap_infonce(
#     energy_model, context_hidden, ground_truth_actions, energy_mask
# )

# 新版本：
world_size = dist.get_world_size() if dist.is_initialized() else 1
swap_loss, E_pos_mean, E_neg_mean = distributed_energy_inbatch_swap_infonce(
    energy_model, context_hidden, ground_truth_actions, energy_mask, 
    tau=0.3, world_size=world_size
)
```

### 方案2: 基于显式负样本的InfoNCE

**原理**: 不依赖batch内样本，使用layer actions生成固定数量的高质量负样本

```python
def layer_based_energy_infonce_loss(
    energy_model, context_hidden, ground_truth_actions, layer_actions, 
    energy_mask, tau=0.3, num_negatives=8
):
    """
    基于层间预测的InfoNCE损失，不依赖batch size
    """
    B, H, Da = ground_truth_actions.shape
    
    # Step 1: 生成多样化负样本
    negatives = []
    
    # 从不同层选择动作预测作为负样本
    if len(layer_actions) >= 2:
        # 选择前几层和中间层的预测
        selected_layers = [0, len(layer_actions)//3, len(layer_actions)//2, -2]
        for layer_idx in selected_layers:
            if layer_idx < len(layer_actions):
                negatives.append(layer_actions[layer_idx])
    
    # 添加噪声扰动
    for sigma in [0.1, 0.2, 0.4]:
        noise_actions = ground_truth_actions + torch.randn_like(ground_truth_actions) * sigma
        negatives.append(noise_actions)
    
    # 时序shuffle
    shuffle_idx = torch.randperm(H)
    shuffle_actions = ground_truth_actions[:, shuffle_idx, :]
    negatives.append(shuffle_actions)
    
    # 确保有足够的负样本
    while len(negatives) < num_negatives:
        # 添加更多随机噪声
        sigma = 0.3 + 0.1 * len(negatives)
        noise_actions = ground_truth_actions + torch.randn_like(ground_truth_actions) * sigma
        negatives.append(noise_actions)
    
    # 只保留指定数量的负样本
    negatives = negatives[:num_negatives]
    A_negatives = torch.stack(negatives, dim=1)  # [B, M, H, Da]
    
    # Step 2: 计算InfoNCE损失
    from energy.energy_model import energy_infonce_loss
    loss, E_pos_mean, E_neg_mean = energy_infonce_loss(
        energy_model, context_hidden, ground_truth_actions, A_negatives, 
        energy_mask, tau=tau
    )
    
    return loss, E_pos_mean, E_neg_mean
```

**使用方法**:
```python
# 在 run_forward_pass 中替换energy损失计算
energy_loss, E_pos_mean, E_neg_mean = layer_based_energy_infonce_loss(
    energy_model, context_hidden, ground_truth_actions, layer_actions,
    energy_mask, tau=0.3, num_negatives=8
)
```

### 方案3: 混合策略 (最稳健)

```python
def adaptive_energy_loss(
    energy_model, context_hidden, ground_truth_actions, layer_actions,
    energy_mask, local_batch_size, world_size=1, step=0
):
    """
    自适应选择最佳的energy损失策略
    """
    effective_batch_size = local_batch_size * world_size
    
    # 决策逻辑
    if world_size == 1 or local_batch_size >= 8:
        # 单卡或local batch足够大时，使用in-batch方法
        if world_size > 1 and local_batch_size >= 6:
            # 多卡但local batch较大，使用分布式in-batch
            loss, E_pos, E_neg = distributed_energy_inbatch_swap_infonce(
                energy_model, context_hidden, ground_truth_actions, energy_mask,
                tau=0.3, world_size=world_size
            )
        else:
            # 单卡，使用标准in-batch
            loss, E_pos, E_neg = energy_inbatch_swap_infonce(
                energy_model, context_hidden, ground_truth_actions, energy_mask, tau=0.3
            )
    else:
        # 多卡且local batch很小，使用layer-based方法
        loss, E_pos, E_neg = layer_based_energy_infonce_loss(
            energy_model, context_hidden, ground_truth_actions, layer_actions,
            energy_mask, tau=0.3
        )
    
    return loss, E_pos, E_neg
```

### 方案4: 增大Local Batch Size

**最简单但可能受内存限制**：

```python
# 调整训练配置
# 8卡 bs_local=3 → 4卡 bs_local=6 或 2卡 bs_local=12
# 或者使用gradient accumulation增大有效batch size

# 在配置中：
batch_size: int = 6              # 增加到6
grad_accumulation_steps: int = 2 # 有效batch size = 6*2 = 12 per GPU
```

## 🚀 **推荐实施策略**

### 立即测试方案 (优先级排序):

1. **方案1 (Global InfoNCE)** - 如果通信开销可接受
2. **方案2 (Layer-based InfoNCE)** - 如果方案1通信过于昂贵  
3. **方案4 (增大local batch)** - 如果GPU内存充足
4. **方案3 (混合策略)** - 作为最终的鲁棒解决方案

### 实施步骤:

1. **先试方案2** (最简单，风险最低):
   - 不需要修改分布式逻辑
   - 直接替换损失函数即可
   
2. **效果不佳再试方案1** (效果最好，但需要通信):
   - 需要添加all_gather通信
   - 可能增加训练时间15-25%

3. **最后考虑方案4** (如果前两者都不理想):
   - 调整硬件配置和batch size

## 📊 **预期效果**

- **方案1**: 完全恢复单卡bs=24的学习效果
- **方案2**: 提供稳定的学习信号，不依赖batch size
- **方案4**: 直接解决问题，但可能受内存限制

选择哪个方案主要取决于您的计算资源和通信带宽限制。
