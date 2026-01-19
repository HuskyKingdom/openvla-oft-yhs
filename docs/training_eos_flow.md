# 训练代码中 EOS 信息的获取流程

## 数据流概览

```
标签文件 (JSON) 
    ↓
datasets_substep.py (数据加载)
    ↓
8D actions (7D base + 1D EOS flag)
    ↓
finetune.py (训练前向传播)
    ↓
L1 Loss / Diffusion Loss
```

---

## 1. 标签文件：`substep_labels_output.json`

**生成工具**: `label_substeps.py`

**格式示例**:
```json
{
  "LIBERO_spatial_no_noops": {
    "put_the_black_bowl_on_top_of_the_cabinet": {
      "episode_0": {
        "original_instruction": "put the black bowl on top of the cabinet",
        "substeps": [...],
        "timestep_labels": [
          {
            "timestep": 75,
            "action": "pick",
            "APD_step": "Pick up the black bowl from the top drawer",
            "cycle": 0,
            "is_substep_end": false
          },
          {
            "timestep": 80,
            "action": "pick",
            "APD_step": "Pick up the black bowl from the top drawer",
            "cycle": 0,
            "is_substep_end": true    // ← 标记 substep 结束
          },
          {
            "timestep": 81,
            "action": "place",
            "APD_step": "Place the black bowl on top of the cabinet",
            "cycle": 0,
            "is_substep_end": false
          }
        ]
      }
    }
  }
}
```

**关键字段**: `is_substep_end` - 标记某个 timestep 是否为 substep 的结束点

---

## 2. 数据加载：`prismatic/vla/datasets/datasets_substep.py`

### 2.1 初始化

```python
class SubstepRLDSBatchTransform:
    def __init__(self, ..., use_substep_eos=False):
        self.use_substep_eos = use_substep_eos
        
        # 加载标签文件
        with open(substep_label_path, 'r') as f:
            self.substep_labels = json.load(f)
```

### 2.2 核心逻辑：`__call__` 方法

**当前时间步**: `timestep = 75`  
**Action chunk 大小**: `NUM_ACTIONS_CHUNK = 8`  
**需要标注的 actions**: timestep 75-82 (共 8 个)

```python
from prismatic.vla.constants import BASE_ACTION_DIM, ACTION_DIM

# 1. 获取原始 7D actions
base_actions = rlds_batch["action"]  # Shape: (8, 7)
num_actions = base_actions.shape[0]  # 8

# 2. 创建 EOS flags 数组
eos_flags = np.zeros((num_actions, 1), dtype=base_actions.dtype)  # Shape: (8, 1)

# 3. 遍历 action chunk，查找 substep 结束点
if self.use_substep_eos:
    for i in range(num_actions):  # i = 0, 1, 2, ..., 7
        future_timestep = timestep + i  # 75, 76, 77, ..., 82
        
        # 查询标签
        suite_name = dataset_name.replace("_no_noops", "")
        task_name = original_instruction.lower().strip().replace(" ", "_")
        episode_key = f"episode_{episode_id}"
        
        try:
            episode_data = self.substep_labels[suite_name][task_name][episode_key]
            timestep_labels = episode_data.get("timestep_labels", [])
            
            # 查找 future_timestep 对应的标签
            current_label = next(
                (label for label in timestep_labels if label["timestep"] == future_timestep), 
                None
            )
            
            # 如果这个 timestep 是 substep 结束，标记对应位置
            if current_label and current_label.get("is_substep_end", False):
                eos_flags[i, 0] = 1.0  # ← 在 action chunk 中标记
                break  # 只标记第一个结束点
                
        except (KeyError, TypeError, IndexError):
            pass

# 4. 拼接成 8D actions
actions = np.concatenate([base_actions, eos_flags], axis=1)  # Shape: (8, 8)
```

**标注结果示例** (假设 timestep=80 是 substep 结束):
```python
# timestep: 75  76  77  78  79  80  81  82
# i:         0   1   2   3   4   5   6   7
eos_flags = [
    [0.0],  # timestep 75
    [0.0],  # timestep 76
    [0.0],  # timestep 77
    [0.0],  # timestep 78
    [0.0],  # timestep 79
    [1.0],  # timestep 80 ← is_substep_end=True
    [0.0],  # timestep 81
    [0.0],  # timestep 82
]
```

### 2.3 Tokenization

**关键**: 只对前 7 维进行 tokenization（VLM 的 token embedding 对应 7 维动作）

```python
current_action = actions[0]  # 8D action

# 只 tokenize 前 7 维
future_actions = actions[1:, :BASE_ACTION_DIM]  # Shape: (7, 7)
future_actions_string = ''.join(self.action_tokenizer(future_actions))
current_action_string = self.action_tokenizer(current_action[:BASE_ACTION_DIM])
action_chunk_string = current_action_string + future_actions_string
```

### 2.4 返回数据

```python
return_dict = dict(
    # ...
    actions=actions,  # ← 完整的 8D actions (包含 EOS flag) 作为 ground truth
    # ...
)
```

---

## 3. 训练前向传播：`vla-scripts/finetune.py`

### 3.1 提取 Hidden States

```python
from prismatic.vla.constants import BASE_ACTION_DIM

# 提取 action tokens 的 hidden states
# 注意：使用 BASE_ACTION_DIM (7)，因为只有 7 个 action tokens
actions_hidden_states = (
    text_hidden_states[current_action_mask | next_actions_mask]
    .reshape(batch_size, NUM_ACTIONS_CHUNK * BASE_ACTION_DIM, -1)
    .to(torch.bfloat16)
)
# Shape: (batch_size, 8*7, hidden_dim)
```

### 3.2 L1 Regression 模式

```python
# 1. Action Head 预测 (输入 7 维 tokens，输出 8 维 actions)
predicted_actions = action_head.module.predict_action(actions_hidden_states)
# Shape: (batch_size, 8, 8) - (batch, num_actions, action_dim)

# 2. Ground Truth (从数据加载中获得的 8D actions)
ground_truth_actions = batch["actions"]
# Shape: (batch_size, 8, 8)

# 3. 计算 L1 Loss
action_loss = torch.nn.functional.l1_loss(predicted_actions, ground_truth_actions)
```

**Action Head 内部** (`prismatic/models/action_heads.py`):
```python
class L1RegressionActionHead:
    def predict_action(self, actions_hidden_states):
        # actions_hidden_states: (batch, 8*7, hidden_dim)
        
        # Rearrange
        rearranged = rearrange(
            actions_hidden_states, 
            "b (n d) h -> b (n h) d", 
            d=BASE_ACTION_DIM  # 7
        )
        
        # MLP 预测
        action = self.model(rearranged)  # Shape: (batch, 8, 8)
        
        # 对第 8 维应用 sigmoid（EOS flag 约束到 [0,1]）
        base_actions = action[..., :BASE_ACTION_DIM]  # (batch, 8, 7)
        eos_flag = torch.sigmoid(action[..., BASE_ACTION_DIM:])  # (batch, 8, 1)
        action = torch.cat([base_actions, eos_flag], dim=-1)  # (batch, 8, 8)
        
        return action
```

### 3.3 Diffusion 模式

```python
# 1. 准备 noisy actions (包含 8 维)
noisy_actions = torch.randn_like(batch["actions"])  # Shape: (batch, 8, 8)

# 2. Action Head 预测 noise
predicted_noise = action_head.module.predict_noise(
    actions_hidden_states, 
    noisy_actions, 
    timesteps
)
# Shape: (batch, 8, 8)

# 3. Ground Truth Noise
# (基于 diffusion 公式计算)
ground_truth_noise = ...  # Shape: (batch, 8, 8)

# 4. 计算 Diffusion Loss
diffusion_loss = torch.nn.functional.mse_loss(predicted_noise, ground_truth_noise)
```

---

## 4. 关键要点

### 4.1 维度对应关系

| 阶段 | Token 维度 | Hidden States 维度 | Action 预测维度 | Ground Truth 维度 |
|------|-----------|-------------------|----------------|------------------|
| Tokenization | 7 | - | - | - |
| VLM Forward | 7 tokens | 8×7 = 56 | - | - |
| Action Head | - | 56 | 8 | 8 |
| Loss Calculation | - | - | 8 | 8 |

### 4.2 EOS Flag 处理

1. **数据加载**: 根据 `is_substep_end` 在 action chunk 中标记 `1.0`
2. **Tokenization**: 跳过 EOS 维度，只 tokenize 前 7 维
3. **Action Head**: 
   - 输入: 7 维 token 的 hidden states
   - 输出: 8 维 action (前 7 维 + sigmoid(第 8 维))
4. **Loss**: 直接对 8 维 predicted 和 8 维 ground truth 计算 L1/MSE

### 4.3 数值稳定性

- **问题**: 第 8 维如果无约束，可能输出很大的值，导致 L1 loss 爆炸和 NaN
- **解决**: 对 Action Head 输出的第 8 维应用 `torch.sigmoid`，约束到 `[0, 1]`
- **匹配**: Ground truth 的 EOS flag 也是 `0.0` 或 `1.0`，完美对应

---

## 5. 完整数据流示例

**假设**: timestep=75, action_chunk_size=8, timestep=80 是 substep 结束

```python
# Step 1: 数据加载 (datasets_substep.py)
base_actions = [
    [x75, y75, z75, rx75, ry75, rz75, g75],  # timestep 75
    [x76, y76, z76, rx76, ry76, rz76, g76],  # timestep 76
    ...
    [x82, y82, z82, rx82, ry82, rz82, g82],  # timestep 82
]

eos_flags = [
    [0.0],  # 75
    [0.0],  # 76
    [0.0],  # 77
    [0.0],  # 78
    [0.0],  # 79
    [1.0],  # 80 ← is_substep_end
    [0.0],  # 81
    [0.0],  # 82
]

actions_8d = np.concatenate([base_actions, eos_flags], axis=1)  # (8, 8)

# Step 2: Tokenization
# 只对前 7 维进行 tokenization
action_tokens = tokenize(actions_8d[:, :7])  # 8 actions × 7 dims = 56 tokens

# Step 3: VLM Forward
hidden_states = vla.forward(image, prompt, action_tokens)  # (56, hidden_dim)

# Step 4: Action Head
predicted_8d = action_head(hidden_states)  # (8, 8)
# predicted_8d[:, :7] - 前 7 维 action 预测
# predicted_8d[:, 7] - EOS flag 预测 (sigmoid 后，范围 [0,1])

# Step 5: Loss
loss = L1_loss(predicted_8d, actions_8d)
# 如果 timestep=80 的 EOS 预测接近 1.0，loss 小
# 如果 timestep=80 的 EOS 预测接近 0.0，loss 大
# → 模型学习到在 substep 结束点预测高 EOS 值
```

---

## 6. 代码位置索引

| 功能 | 文件 | 关键代码行 |
|------|------|-----------|
| 标签加载 | `datasets_substep.py` | `__init__` 中的 JSON 加载 |
| EOS flag 标注 | `datasets_substep.py` | `__call__` 中的 `for i in range(num_actions)` 循环 |
| Tokenization | `datasets_substep.py` | `action_tokenizer(current_action[:BASE_ACTION_DIM])` |
| Hidden States 提取 | `finetune.py` | `.reshape(batch_size, NUM_ACTIONS_CHUNK * BASE_ACTION_DIM, -1)` |
| Action 预测 | `action_heads.py` | `predict_action` / `predict_noise` 方法 |
| Sigmoid 激活 | `action_heads.py` | `torch.sigmoid(action[..., BASE_ACTION_DIM:])` |
| Loss 计算 | `finetune.py` | `torch.nn.functional.l1_loss(predicted_actions, ground_truth_actions)` |

---

## 7. 常见问题

### Q1: 为什么不对 EOS 维度进行 tokenization？
**A**: VLM 的 token embedding 是为 7 维动作设计的。EOS flag 是额外的监督信号，通过 Action Head 直接从 7 维 token 的 hidden states 预测出来。

### Q2: 如果 action chunk 中有多个 substep 结束怎么办？
**A**: 当前实现只标记第一个 `is_substep_end=True` 的位置（`break` 语句）。如需支持多个，删除 `break`。

### Q3: EOS flag 的 ground truth 如何生成？
**A**: 直接从 `substep_labels_output.json` 中读取 `is_substep_end` 字段，转换为 `1.0`（True）或 `0.0`（False）。

### Q4: 为什么要用 sigmoid？
**A**: 
1. 将 EOS 预测约束到 `[0, 1]`，匹配 ground truth 的范围
2. 防止数值溢出导致 NaN loss
3. 提供清晰的语义（概率解释）

### Q5: 训练时模型如何学习 EOS？
**A**: 通过 L1/MSE loss，模型学习到：
- 在 `is_substep_end=True` 的位置，预测接近 1.0 的值
- 在其他位置，预测接近 0.0 的值
- 模型通过视觉和语言特征学习判断何时 substep 完成

