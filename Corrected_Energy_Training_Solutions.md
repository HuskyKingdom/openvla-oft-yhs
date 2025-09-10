# Energy训练问题的修正解决方案

## 问题重新定位

**真正的根本问题**: 不是τ太小或能量范围太大，而是：
1. **能量区分度不足** - 正负样本能量差异太小
2. **Sigmoid饱和导致梯度消失**
3. **负样本质量差**

## 🎯 **修正解决方案**

### 方案1: 保守的能量范围调整 (🔴推荐)

```python
class EnergyModel(nn.Module):
    def __init__(self, ...):
        # 不要完全移除Sigmoid，而是扩大其输入范围
        self.act = nn.Sigmoid()
        self.energy_scale = 2.0  # 温和扩大范围到 (0, 2)
        self.energy_offset = 0.1  # 避免过小值
        
    def forward(self, hN, a, pad_mask=None):
        # ... 前面计算相同 ...
        raw = self.prediction_head(energy)
        
        # 温和扩大Sigmoid输入范围，避免饱和
        scaled_raw = raw * 0.5  # 将raw缩放到合理范围
        E = self.act(scaled_raw) * self.energy_scale + self.energy_offset
        # 结果: E ∈ [0.1, 2.1]，比原来的[1e-6, 1+1e-6]大，但不会太极端
        
        return E
```

**优势**:
- 能量范围适度扩大: (0.1, 2.1)
- 避免Sigmoid完全饱和
- 梯度scale增加2倍，可控
- 保持数值稳定性

### 方案2: 改进负样本生成 (🔴最重要)

```python
def improved_negative_sampling(layer_actions, ground_truth_actions, context_hidden, energy_model):
    """关键: 生成更有区分度的负样本"""
    
    # 策略1: 选择能量接近但不相同的负样本
    with torch.no_grad():
        # 计算所有层动作的能量
        layer_energies = []
        for layer_action in layer_actions[:-1]:  # 排除最终层
            energy = energy_model(context_hidden, layer_action)
            layer_energies.append((layer_action, energy))
        
        # 按能量排序，选择中等能量的作为负样本（不是最差的）
        layer_energies.sort(key=lambda x: x[1].mean().item())
        mid_idx = len(layer_energies) // 2
        selected_negative = layer_energies[mid_idx][0]
    
    # 策略2: 添加控制强度的噪声
    B, H, Da = ground_truth_actions.shape
    noise_levels = [0.1, 0.3]  # 不同强度的噪声
    noise_negatives = []
    for sigma in noise_levels:
        noise_action = ground_truth_actions + torch.randn_like(ground_truth_actions) * sigma
        noise_negatives.append(noise_action)
    
    # 组合负样本
    all_negatives = [selected_negative] + noise_negatives
    return torch.stack(all_negatives, dim=1)  # [B, M, H, Da]
```

### 方案3: 动态温度调整 (🟡可选)

```python
def adaptive_energy_inbatch_swap_infonce(energy_model, h, a_pos, pad_mask, base_tau=0.5):
    """自适应温度调整"""
    
    B, S, D = h.shape
    H, Da = a_pos.shape[1], a_pos.shape[2]
    
    # 计算能量矩阵
    h_rep = h.unsqueeze(1).expand(B, B, S, D).reshape(B*B, S, D)
    a_rep = a_pos.unsqueeze(0).expand(B, B, H, Da).reshape(B*B, H, Da)
    pm = pad_mask.unsqueeze(1).expand(B, B, -1).reshape(B*B, -1) if pad_mask is not None else None
    
    E_ij = energy_model(h_rep, a_rep, pm).view(B, B).squeeze(-1)
    
    # 动态调整温度
    E_pos = torch.diag(E_ij)  # 正样本能量
    E_neg_mask = ~torch.eye(B, dtype=bool, device=h.device)
    E_neg = E_ij[E_neg_mask]  # 负样本能量
    
    # 基于能量差异调整温度
    energy_diff = (E_neg.mean() - E_pos.mean()).abs()
    if energy_diff < 0.1:  # 差异太小
        adaptive_tau = base_tau * 0.5  # 降低温度增强对比
    elif energy_diff > 1.0:  # 差异很大
        adaptive_tau = base_tau * 2.0  # 升高温度缓解过度对比
    else:
        adaptive_tau = base_tau
    
    # 限制温度范围
    adaptive_tau = torch.clamp(torch.tensor(adaptive_tau), 0.1, 2.0).item()
    
    logits = (-E_ij) / adaptive_tau
    labels = torch.arange(B, device=h.device)
    loss = F.cross_entropy(logits, labels)
    
    return loss, E_pos.mean(), E_neg.mean(), adaptive_tau
```

### 方案4: 渐进式训练策略 (🟠重要)

```python
def progressive_energy_training(energy_model, step, total_steps):
    """渐进式训练避免突然变化"""
    
    # 阶段1: 前5k步，降低学习率，专注稳定性
    if step < 5000:
        lr_scale = 0.1
        grad_clip = 0.3
    # 阶段2: 5k-15k步，逐步提高学习率
    elif step < 15000:
        progress = (step - 5000) / 10000
        lr_scale = 0.1 + 0.4 * progress  # 从0.1增长到0.5
        grad_clip = 0.3 + 0.2 * progress  # 从0.3增长到0.5
    # 阶段3: 15k+步，正常学习率
    else:
        lr_scale = 0.5
        grad_clip = 0.5
    
    return lr_scale, grad_clip
```

## 🚀 **最终推荐的修改**

**立即实施 (优先级🔴)**:

```python
# 1. 在 EnergyModel.__init__ 中
self.energy_scale = 2.0  # 而不是5.0
self.energy_offset = 0.1

# 2. 在 EnergyModel.forward 中  
scaled_raw = raw * 0.5  # 防止Sigmoid饱和
E = self.act(scaled_raw) * self.energy_scale + self.energy_offset

# 3. 保持原始温度或稍微降低
tau = 0.3  # 从0.5降到0.3，增强对比信号

# 4. 更保守的学习率和梯度裁剪
energy_lr = cfg.energy_learning_rate * 0.1  # 降低学习率
torch.nn.utils.clip_grad_norm_(energy_model.parameters(), max_norm=0.3)

# 5. 改进负样本生成
negatives = improved_negative_sampling(layer_actions, ground_truth_actions, 
                                     context_hidden, energy_model)
```

**为什么这样修改更合理**:

1. **能量范围**: (0.1, 2.1) vs 原来的(1e-6, 1.000001)
   - 区分度提高**2000倍**但梯度只增大2倍，平衡
   
2. **温度系数**: 0.3 vs 原来的0.5
   - 您说得对，降低τ增强对比信号
   
3. **渐进策略**: 避免训练震荡
   - 从保守开始，逐步放开约束

4. **负样本质量**: 这是最关键的
   - 好的负样本比大的能量范围更重要

## 📊 **预期效果**

- **能量差异**: 从0.1增加到0.5-1.0
- **梯度稳定**: 避免爆炸，保持可学习梯度
- **学习曲线**: 应该看到smooth下降而不是直线
- **NaN避免**: 渐进策略大幅降低NaN风险

这个修正方案解决了您担心的两个问题，同时保持训练稳定性！
