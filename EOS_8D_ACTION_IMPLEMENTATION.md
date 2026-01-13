# EOS检测实现：8维Action方案

## 概述

使用方案1实现EOS检测：扩展action维度从7维到8维，第8维作为EOS标志。

**核心思想**：
- 训练时：在substep结束位置，将action的第8维设为1.0（EOS），否则为0.0
- 推理时：检查第8维是否>0.5，如果是则表示substep结束
- 环境执行：只使用前7维action（xyz, rotation, gripper）

---

## 修改的文件

### 1. `prismatic/vla/constants.py`

**修改内容**：
- 添加 `BASE_ACTION_DIM = 7`（原始action维度）
- 修改 `ACTION_DIM = 8`（包含EOS标志）
- 适配LIBERO、ALOHA、BRIDGE三个平台

**代码示例**：
```python
LIBERO_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 8,
    "BASE_ACTION_DIM": 7,  # xyz(3) + rotation(3) + gripper(1)
    "ACTION_DIM": 8,       # base(7) + eos_flag(1)
    "PROPRIO_DIM": 8,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}
```

---

### 2. `prismatic/vla/datasets/datasets_substep.py`

**修改内容**：
- 在 `__call__` 方法中，将原始7维action扩展为8维
- 在substep结束时，将最后一个action的第8维设为1.0

**代码示例**：
```python
# 创建EOS标志：所有为0，substep结束时最后一个为1
eos_flags = np.zeros((num_actions, 1), dtype=base_actions.dtype)
if self.use_substep_eos and is_substep_end:
    eos_flags[-1, 0] = 1.0  # 标记最后一个action为substep结束

# 拼接base action和EOS标志
actions = np.concatenate([base_actions, eos_flags], axis=1)  # (num_actions, 8)
```

---

### 3. `prismatic/models/action_heads.py`

**修改内容**：
- `L1RegressionActionHead` 构造函数默认使用 `ACTION_DIM`（8维）
- 输出shape：`(batch_size, NUM_ACTIONS_CHUNK, 8)`

**代码示例**：
```python
class L1RegressionActionHead(nn.Module):
    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=None,  # 如果未指定，使用ACTION_DIM(8)
    ):
        super().__init__()
        if action_dim is None:
            action_dim = ACTION_DIM  # 使用8维
        self.action_dim = action_dim
        self.model = MLPResNet(
            num_blocks=2, input_dim=input_dim*ACTION_DIM, 
            hidden_dim=hidden_dim, output_dim=action_dim
        )
```

---

### 4. `prismatic/extern/hf/modeling_prismatic.py`

**修改内容**：
- 重写EOS检测逻辑，从第8维检测而不是从token logits
- 阈值：EOS flag > 0.5 表示substep结束

**代码示例**：
```python
# EOS检测：从第8维（索引7）检测
if return_eos_info:
    from prismatic.vla.constants import BASE_ACTION_DIM
    EOS_THRESHOLD = 0.5
    
    # 提取EOS标志（最后一维）
    eos_flags = normalized_actions[:, BASE_ACTION_DIM]  # (NUM_ACTIONS_CHUNK,)
    
    # 找到第一个EOS flag > threshold的位置
    eos_detected_mask = eos_flags > EOS_THRESHOLD
    
    if eos_detected_mask.any():
        first_eos_idx = eos_detected_mask.nonzero()[0].item()
        has_eos = True
        eos_position = first_eos_idx
        logger.info(
            f"[EOS DETECT] ✓ EOS detected at action {eos_position} "
            f"(flag value={eos_flags[eos_position]:.3f})"
        )
    else:
        logger.info(
            f"[EOS DETECT] ✗ No EOS detected. "
            f"Max EOS flag={eos_flags.max():.3f}"
        )
```

---

### 5. `experiments/robot/openvla_utils.py`

**修改内容**：
- 在返回actions时，只提取前7维给环境执行
- 第8维（EOS标志）仅用于检测，不传递给环境

**代码示例**：
```python
# 提取前7维给机器人执行
from prismatic.vla.constants import BASE_ACTION_DIM

# actions shape: (NUM_ACTIONS_CHUNK, 8)
# 只返回前7维
actions_list = [action[i, :BASE_ACTION_DIM] for i in range(len(action))]

if return_eos_info:
    return actions_list, has_eos, eos_position
else:
    return actions_list
```

---

## 数据流

### 训练时

```
RLDS数据 (7维action)
    ↓
datasets_substep.py: 扩展为8维
    ↓ (substep结束时最后一个action第8维=1.0)
Batch: (B, NUM_ACTIONS_CHUNK, 8)
    ↓
action_head: 预测8维action
    ↓
L1 Loss: 计算所有8维（包括EOS维度）
    ↓
模型学习：预测第8维=1表示substep结束
```

### 推理时

```
observation + instruction
    ↓
VLA forward
    ↓
action_head: 输出8维action
    ↓ (第8维为EOS标志)
EOS检测: 检查第8维 > 0.5
    ↓ Yes: has_eos=True, eos_position=位置
    ↓ No: has_eos=False
    ↓
提取前7维 → 环境执行
```

---

## 关键优势

### 1. **完全在L1框架内**
- 不需要token logits
- EOS作为连续值，可以渐进学习
- 统一的L1 loss，无需混合loss

### 2. **训练和推理一致**
- 训练时：L1 loss包含EOS维度
- 推理时：直接从第8维读取EOS标志
- 无token化/解token化过程

### 3. **简单直观**
- EOS就是一个0/1标志
- 阈值简单（0.5）
- 易于调试和可视化

### 4. **兼容现有架构**
- 只增加1维，不改变核心架构
- 环境仍接收7维action
- 向后兼容（可以忽略第8维）

---

## 使用方法

### 训练

```bash
python vla-scripts/finetune_substep.py \
    --vla_path openvla/openvla-7b \
    --dataset_name libero_goal_no_noops \
    --substep_labels_path substep_labels_output.json \
    --use_substep_eos True \
    --use_l1_regression True \
    --lora_rank 32 \
    --batch_size 8
```

**关键**：`--use_substep_eos True` 启用EOS标志训练

### 推理

```bash
python experiments/robot/libero/run_libero_pro_eval_substep.py \
    --pretrained_checkpoint path/to/checkpoint \
    --task_suite_name libero_spatial \
    --use_substep_decomposition True \
    --use_eos_detection True \
    --use_l1_regression True
```

**关键**：`--use_eos_detection True` 启用EOS检测

---

## 验证

### 1. 检查训练数据

```python
# 在 datasets_substep.py 中查看
print(f"Action shape: {actions.shape}")  # 应该是 (8, 8)
print(f"Last action EOS flag: {actions[-1, 7]}")  # substep结束时应该是 1.0
```

### 2. 检查模型输出

```python
# 在 modeling_prismatic.py 中查看
print(f"Predicted actions shape: {normalized_actions.shape}")  # (8, 8)
print(f"EOS flags: {normalized_actions[:, 7]}")  # 应该有值接近1.0的位置
```

### 3. 检查推理结果

查看日志：
```
[EOS DETECT] ✓ EOS detected at action 3 (flag value=0.872 > 0.5)
[EOS SWITCH] ✓ Switched to step 2/3: next substep instruction
```

---

## 与原方案对比

| 特性 | Token Logits方案 | 8维Action方案（当前） |
|------|-----------------|---------------------|
| **L1兼容性** | ❌ 不兼容（logits只在离散模式） | ✅ 完全兼容 |
| **训练Loss** | 混合loss（L1 + CrossEntropy） | 纯L1 loss |
| **实现复杂度** | 复杂（需要处理两种预测） | 简单（统一action预测） |
| **EOS语义** | 离散token | 连续标志（可渐进学习） |
| **推理速度** | 需要额外token解码 | 直接读取第8维 |
| **调试难度** | 困难（涉及tokenization） | 简单（直接查看数值） |

---

## 注意事项

1. **重新训练**：需要用8维action重新训练模型
2. **数据兼容**：旧的7维checkpoint不兼容，需要重新训练
3. **归一化**：EOS维度（第8维）的归一化统计需要正确计算
4. **阈值调优**：EOS阈值（0.5）可能需要根据实际效果调整

---

## 总结

使用8维action方案实现EOS检测：
- ✅ **训练**：L1 loss自然包含EOS维度
- ✅ **推理**：直接从第8维检测EOS（> 0.5）
- ✅ **执行**：只使用前7维控制机器人
- ✅ **优势**：简单、统一、高效、易调试

这是在L1回归模式下实现EOS检测的最佳方案。

