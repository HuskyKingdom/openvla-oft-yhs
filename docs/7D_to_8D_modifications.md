# 7D 到 8D 动作空间修改总结

## 概述

本文档详细记录了将动作空间从 7 维扩展到 8 维（第 8 维作为 EOS flag）的所有代码修改，以及如何回退到原始的 7 维实现。

**修改目的**：在动作空间中添加一个连续的 EOS（End of Substep）标志位，用于在推理时动态检测 substep 边界。

---

## 修改文件清单

以下 7 个核心文件进行了修改：

1. `prismatic/vla/constants.py` - 全局常量定义
2. `prismatic/vla/datasets/datasets_substep.py` - 数据加载和预处理
3. `prismatic/models/action_heads.py` - 动作预测头
4. `prismatic/extern/hf/modeling_prismatic.py` - 核心模型推理逻辑
5. `vla-scripts/finetune.py` - 训练前向传播
6. `experiments/robot/openvla_utils.py` - VLA 推理工具函数
7. `experiments/robot/libero/run_libero_pro_eval_substep.py` - 评估脚本

---

## 详细修改内容

### 1. `prismatic/vla/constants.py`

**修改位置**: 文件顶部常量定义区域

**8D 版本的修改**:
```python
# 添加基础动作维度常量
BASE_ACTION_DIM = 7  # Original 7-dimensional action space

# 修改 LIBERO 常量
LIBERO_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 8,
    "ACTION_DIM": 8,  # 从 7 改为 8
    "PROPRIO_DIM": 8,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

# 修改 ALOHA 常量
ALOHA_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 100,
    "ACTION_DIM": 8,  # 从 7 改为 8
    "PROPRIO_DIM": 14,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

# 修改 BRIDGE 常量
BRIDGE_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 1,
    "ACTION_DIM": 8,  # 从 7 改为 8
    "PROPRIO_DIM": 7,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}
```

**如何回退到 7D**:
```python
# 1. 删除 BASE_ACTION_DIM 常量定义（或注释掉）
# BASE_ACTION_DIM = 7

# 2. 将所有数据集常量的 ACTION_DIM 改回 7
LIBERO_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 8,
    "ACTION_DIM": 7,  # 改回 7
    "PROPRIO_DIM": 8,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

ALOHA_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 100,
    "ACTION_DIM": 7,  # 改回 7
    "PROPRIO_DIM": 14,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

BRIDGE_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 1,
    "ACTION_DIM": 7,  # 改回 7
    "PROPRIO_DIM": 7,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}
```

---

### 2. `prismatic/vla/datasets/datasets_substep.py`

**修改位置**: `SubstepRLDSBatchTransform.__call__` 方法

**8D 版本的修改**:
```python
from prismatic.vla.constants import BASE_ACTION_DIM, ACTION_DIM

# 在 __call__ 方法中添加 EOS flag 生成逻辑
base_actions = rlds_batch["action"]  # Shape: (num_actions, 7)
num_actions = base_actions.shape[0]

# 创建 EOS flags 数组
eos_flags = np.zeros((num_actions, 1), dtype=base_actions.dtype)

if self.use_substep_eos:
    # 遍历 action chunk 中的每个时间步，检查是否是 substep 结束
    for i in range(num_actions):
        future_timestep = timestep + i
        suite_name = dataset_name.replace("_no_noops", "")
        task_name = original_instruction.lower().strip().replace(" ", "_")
        episode_key = f"episode_{episode_id}"
        
        try:
            episode_data = self.substep_labels[suite_name][task_name][episode_key]
            timestep_labels = episode_data.get("timestep_labels", [])
            current_label = next((label for label in timestep_labels if label["timestep"] == future_timestep), None)
            
            if current_label and current_label.get("is_substep_end", False):
                eos_flags[i, 0] = 1.0  # 标记此 action 为 substep 结束
                break  # 只标记第一个检测到的 substep 结束
        except (KeyError, TypeError, IndexError):
            pass

# 拼接 base actions 和 EOS flag
actions = np.concatenate([base_actions, eos_flags], axis=1)  # Shape: (num_actions, 8)
current_action = actions[0]

# 只对前 7 维进行 tokenization
future_actions = actions[1:, :BASE_ACTION_DIM]
future_actions_string = ''.join(self.action_tokenizer(future_actions))
current_action_string = self.action_tokenizer(current_action[:BASE_ACTION_DIM])
action_chunk_string = current_action_string + future_actions_string

# 在返回字典中使用完整的 8D actions
return_dict = dict(
    # ... 其他字段 ...
    actions=actions,  # 传递完整的 8D actions 作为 ground truth
)
```

**如何回退到 7D**:
```python
# 1. 删除 BASE_ACTION_DIM 和 ACTION_DIM 的导入
# from prismatic.vla.constants import BASE_ACTION_DIM, ACTION_DIM

# 2. 移除 EOS flag 生成逻辑，直接使用原始 actions
actions = rlds_batch["action"]  # Shape: (num_actions, 7)
current_action = actions[0]

# 3. 直接对所有 actions 进行 tokenization（无需切片）
future_actions = actions[1:]
future_actions_string = ''.join(self.action_tokenizer(future_actions))
current_action_string = self.action_tokenizer(current_action)
action_chunk_string = current_action_string + future_actions_string

# 4. 返回原始 7D actions
return_dict = dict(
    # ... 其他字段 ...
    actions=actions,  # 传递 7D actions
)
```

---

### 3. `prismatic/models/action_heads.py`

**修改位置**: `L1RegressionActionHead` 和 `DiffusionActionHead` 类

**8D 版本的修改**:

#### L1RegressionActionHead

```python
from prismatic.vla.constants import BASE_ACTION_DIM

# __init__ 方法修改
def __init__(self, input_dim: int, hidden_dim: int, action_dim: int = 8):
    super().__init__()
    self.action_dim = action_dim
    self.model = MLPResNet(
        num_blocks=2,
        input_dim=input_dim * BASE_ACTION_DIM,  # 使用 BASE_ACTION_DIM (7)
        hidden_dim=hidden_dim,
        output_dim=action_dim  # 输出 8 维
    )

# predict_action 方法修改
def predict_action(self, actions_hidden_states):
    # actions_hidden_states shape: (batch_size, num_actions * BASE_ACTION_DIM, hidden_dim)
    rearranged_actions_hidden_states = rearrange(
        actions_hidden_states, "b (n d) h -> b (n h) d", d=BASE_ACTION_DIM
    )
    action = self.model(rearranged_actions_hidden_states)
    
    # 对第 8 维应用 sigmoid 激活
    if self.action_dim > BASE_ACTION_DIM:
        base_actions = action[..., :BASE_ACTION_DIM]
        eos_flag = torch.sigmoid(action[..., BASE_ACTION_DIM:])
        action = torch.cat([base_actions, eos_flag], dim=-1)
    
    return action
```

#### DiffusionActionHead

```python
from prismatic.vla.constants import BASE_ACTION_DIM

# __init__ 方法修改
def __init__(self, input_dim: int, hidden_dim: int, action_dim: int = 8):
    super().__init__()
    self.action_dim = action_dim
    self.noise_predictor = NoisePredictionModel(
        transformer_hidden_dim=hidden_dim * BASE_ACTION_DIM,  # 使用 BASE_ACTION_DIM
        hidden_dim=hidden_dim,
        action_dim=action_dim  # 输出 8 维
    )
```

**如何回退到 7D**:

```python
# 1. 删除 BASE_ACTION_DIM 导入
# from prismatic.vla.constants import BASE_ACTION_DIM

# 2. L1RegressionActionHead 回退
def __init__(self, input_dim: int, hidden_dim: int, action_dim: int = 7):
    super().__init__()
    self.action_dim = action_dim
    self.model = MLPResNet(
        num_blocks=2,
        input_dim=input_dim * action_dim,  # 改回 action_dim
        hidden_dim=hidden_dim,
        output_dim=action_dim
    )

def predict_action(self, actions_hidden_states):
    # 使用 action_dim 而非 BASE_ACTION_DIM
    rearranged_actions_hidden_states = rearrange(
        actions_hidden_states, "b (n d) h -> b (n h) d", d=self.action_dim
    )
    action = self.model(rearranged_actions_hidden_states)
    # 移除 sigmoid 逻辑
    return action

# 3. DiffusionActionHead 回退
def __init__(self, input_dim: int, hidden_dim: int, action_dim: int = 7):
    super().__init__()
    self.action_dim = action_dim
    self.noise_predictor = NoisePredictionModel(
        transformer_hidden_dim=hidden_dim * action_dim,  # 改回 action_dim
        hidden_dim=hidden_dim,
        action_dim=action_dim
    )
```

---

### 4. `prismatic/extern/hf/modeling_prismatic.py`

**修改位置**: 多个方法

**8D 版本的修改**:

#### `_unnormalize_actions` 方法

```python
from prismatic.vla.constants import BASE_ACTION_DIM, ACTION_DIM

def _unnormalize_actions(self, normalized_actions, mask, action_low, action_high):
    # 如果是 8D actions，只对前 7 维进行反归一化
    if normalized_actions.shape[-1] == ACTION_DIM:
        base_actions = normalized_actions[..., :BASE_ACTION_DIM]
        eos_flag = normalized_actions[..., BASE_ACTION_DIM:]  # EOS flag 保持不变
        
        unnormalized_base = np.where(
            mask,
            0.5 * (base_actions + 1) * (action_high - action_low + 1e-8) + action_low,
            base_actions,
        )
        actions = np.concatenate([unnormalized_base, eos_flag], axis=-1)
    else:
        # 原始 7D 逻辑
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low + 1e-8) + action_low,
            normalized_actions,
        )
    return actions
```

#### Hidden States 提取

```python
from prismatic.vla.constants import BASE_ACTION_DIM

# 在 _run_diffusion_prediction, _regression_or_discrete_prediction 等方法中
actions_hidden_states = last_hidden_states[
    :,
    NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES + NUM_PROMPT_TOKENS + BASE_ACTION_DIM * NUM_ACTIONS_CHUNK,
    :,
]
```

#### Diffusion 预测中的 Sigmoid

```python
# 在 _run_diffusion_prediction 方法的返回前
from prismatic.vla.constants import BASE_ACTION_DIM

if curr_noisy_actions.shape[-1] > BASE_ACTION_DIM:
    base_actions = curr_noisy_actions[..., :BASE_ACTION_DIM]
    eos_flag = torch.sigmoid(curr_noisy_actions[..., BASE_ACTION_DIM:])
    curr_noisy_actions = torch.cat([base_actions, eos_flag], dim=-1)

return curr_noisy_actions.float().cpu().detach().numpy(), actions_hidden_states
```

#### EOS 检测逻辑

```python
# 在 predict_action 方法中
from prismatic.vla.constants import BASE_ACTION_DIM

if return_eos_info and normalized_actions is not None:
    if normalized_actions.shape[-1] > BASE_ACTION_DIM:
        eos_threshold = 0.5
        eos_flags_predicted = normalized_actions[..., BASE_ACTION_DIM:]
        eos_mask = (eos_flags_predicted > eos_threshold).squeeze(-1)
        
        if eos_mask.any():
            first_eos_pos_in_chunk = eos_mask[0].nonzero(as_tuple=True)[0]
            if len(first_eos_pos_in_chunk) > 0:
                eos_position = first_eos_pos_in_chunk[0].item()
                has_eos = True
        
        max_eos_flag_val = eos_flags_predicted.max().item()
        max_eos_flag_pos = eos_flags_predicted.argmax().item()
        logger.info(
            f"[EOS DETECT] {'✓' if has_eos else '✗'} EOS detected. "
            f"Max EOS flag={max_eos_flag_val:.3f} at action {max_eos_flag_pos}"
        )
```

**如何回退到 7D**:

```python
# 1. 删除 BASE_ACTION_DIM 和 ACTION_DIM 导入
# from prismatic.vla.constants import BASE_ACTION_DIM, ACTION_DIM

# 2. _unnormalize_actions 回退到原始实现
def _unnormalize_actions(self, normalized_actions, mask, action_low, action_high):
    actions = np.where(
        mask,
        0.5 * (normalized_actions + 1) * (action_high - action_low + 1e-8) + action_low,
        normalized_actions,
    )
    return actions

# 3. Hidden States 提取改回使用 ACTION_DIM（此时为 7）
actions_hidden_states = last_hidden_states[
    :,
    NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES + NUM_PROMPT_TOKENS + ACTION_DIM * NUM_ACTIONS_CHUNK,
    :,
]

# 4. 移除 Diffusion 中的 Sigmoid 逻辑
# 直接返回：
return curr_noisy_actions.float().cpu().detach().numpy(), actions_hidden_states

# 5. 移除或注释掉 EOS 检测逻辑（或恢复为基于 token logits 的占位符实现）
```

---

### 5. `vla-scripts/finetune.py`

**修改位置**: `run_forward_pass` 函数

**8D 版本的修改**:

```python
from prismatic.vla.constants import BASE_ACTION_DIM

# L1 Regression 部分（约 385 行）
actions_hidden_states = (
    text_hidden_states[current_action_mask | next_actions_mask]
    .reshape(batch_size, NUM_ACTIONS_CHUNK * BASE_ACTION_DIM, -1)  # 使用 BASE_ACTION_DIM
    .to(torch.bfloat16)
)

# Diffusion 部分（约 524 行）
actions_hidden_states = (
    text_hidden_states[current_action_mask | next_actions_mask]
    .reshape(batch_size, NUM_ACTIONS_CHUNK * BASE_ACTION_DIM, -1)  # 使用 BASE_ACTION_DIM
    .to(torch.bfloat16)
)
```

**如何回退到 7D**:

```python
# 1. 删除 BASE_ACTION_DIM 导入
# from prismatic.vla.constants import BASE_ACTION_DIM

# 2. 将 reshape 中的 BASE_ACTION_DIM 改回 ACTION_DIM
actions_hidden_states = (
    text_hidden_states[current_action_mask | next_actions_mask]
    .reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)  # 改回 ACTION_DIM
    .to(torch.bfloat16)
)
```

---

### 6. `experiments/robot/openvla_utils.py`

**修改位置**: `get_vla_action` 函数

**8D 版本的修改**:

```python
from prismatic.vla.constants import BASE_ACTION_DIM

def get_vla_action(vla, processor, prompt_builder, ..., return_eos_info=False):
    # ... 模型预测 ...
    
    # 只提取前 7 维 action 用于环境执行
    actions_list = [action[i, :BASE_ACTION_DIM] for i in range(len(action))]
    
    if return_eos_info:
        return actions_list, has_eos, eos_position
    else:
        return actions_list
```

**如何回退到 7D**:

```python
# 1. 删除 BASE_ACTION_DIM 导入
# from prismatic.vla.constants import BASE_ACTION_DIM

# 2. 直接返回完整的 actions（此时已经是 7 维）
def get_vla_action(vla, processor, prompt_builder, ..., return_eos_info=False):
    # ... 模型预测 ...
    
    actions_list = [action[i] for i in range(len(action))]  # 不需要切片
    
    if return_eos_info:
        return actions_list, has_eos, eos_position
    else:
        return actions_list
```

---

### 7. `experiments/robot/libero/run_libero_pro_eval_substep.py`

**修改位置**: `run_episode` 函数和 `save_rollout_video_with_substep_info` 函数

**8D 版本的修改**:

```python
# 在 run_episode 函数中添加配置验证
if cfg.use_eos_detection and not cfg.use_substep_decomposition:
    log_message(
        f"[EOS WARNING] ⚠️ EOS detection requires substep decomposition "
        f"(--use_substep_decomposition=True). Disabling EOS detection.",
        log_file
    )
    cfg.use_eos_detection = False

# 增强 frame_substep_info
frame_substep_info = {
    # ... 原有字段 ...
    'eos_detected': False,
    'eos_position': None,
    'eos_triggered_switch': False,
}

# EOS 检测后更新 info
if has_eos and eos_position is not None:
    frame_substep_info['eos_detected'] = True
    frame_substep_info['eos_position'] = eos_position

# EOS 触发切换后更新 info
if substep_switched and cfg.use_eos_detection:
    frame_substep_info['eos_triggered_switch'] = True

# 在视频保存函数中添加 EOS 显示
def save_rollout_video_with_substep_info(...):
    # ... 其他显示代码 ...
    
    # 显示 EOS 检测状态
    if info.get('eos_detected', False):
        eos_text = f"EOS Detected at action {info['eos_position']}"
        cv2.putText(img_bgr, eos_text, (10, y_offset), font, font_scale, (0, 255, 255), font_thickness, cv2.LINE_AA)
        y_offset += line_spacing
    
    # 显示 EOS 触发的切换
    if info.get('eos_triggered_switch', False):
        switch_text = ">>> EOS-triggered substep switch <<<"
        cv2.putText(img_bgr, switch_text, (10, y_offset), font, font_scale, (255, 0, 255), font_thickness, cv2.LINE_AA)
        y_offset += line_spacing
```

**如何回退到 7D**:

```python
# 1. 移除 EOS 相关的配置验证代码块
# 2. 从 frame_substep_info 中删除 EOS 相关字段
frame_substep_info = {
    # ... 原有字段 ...
    # 删除: 'eos_detected', 'eos_position', 'eos_triggered_switch'
}

# 3. 移除 EOS 检测和切换的更新逻辑
# 4. 从视频保存函数中删除 EOS 显示代码
```

---

## 回退步骤总结

**按照以下顺序执行回退操作**：

### 步骤 1: 修改常量定义
```bash
# 编辑 prismatic/vla/constants.py
# - 删除或注释 BASE_ACTION_DIM 定义
# - 将所有 ACTION_DIM 改回 7
```

### 步骤 2: 回退数据加载逻辑
```bash
# 编辑 prismatic/vla/datasets/datasets_substep.py
# - 移除 EOS flag 生成逻辑
# - 移除 BASE_ACTION_DIM 导入和使用
# - 恢复直接使用 7D actions
```

### 步骤 3: 回退动作头
```bash
# 编辑 prismatic/models/action_heads.py
# - 将 input_dim 改回使用 action_dim
# - 移除 sigmoid 激活逻辑
# - 删除 BASE_ACTION_DIM 导入
```

### 步骤 4: 回退模型推理
```bash
# 编辑 prismatic/extern/hf/modeling_prismatic.py
# - 恢复 _unnormalize_actions 的原始实现
# - 将 hidden states 提取改回使用 ACTION_DIM
# - 移除 diffusion 中的 sigmoid
# - 移除 EOS 检测逻辑
```

### 步骤 5: 回退训练代码
```bash
# 编辑 vla-scripts/finetune.py
# - 将 reshape 中的 BASE_ACTION_DIM 改回 ACTION_DIM
```

### 步骤 6: 回退推理工具
```bash
# 编辑 experiments/robot/openvla_utils.py
# - 移除 action 切片逻辑
# - 删除 BASE_ACTION_DIM 导入
```

### 步骤 7: 回退评估脚本
```bash
# 编辑 experiments/robot/libero/run_libero_pro_eval_substep.py
# - 移除 EOS 相关的所有逻辑
```

### 步骤 8: 验证回退
```bash
# 1. 运行代码检查（确保没有引用 BASE_ACTION_DIM）
grep -r "BASE_ACTION_DIM" prismatic/ vla-scripts/ experiments/

# 2. 检查 ACTION_DIM 是否都是 7
grep -r "ACTION_DIM.*=.*8" prismatic/vla/constants.py

# 3. 重新训练和测试（使用 7D 的 checkpoint）
```

---

## 关键点总结

### 8D 实现的核心思想
1. **训练阶段**：将 7D 动作拼接 1D EOS flag，形成 8D ground truth
2. **Tokenization**：只对前 7 维进行 token 化（因为 VLM 的 token embedding 对应 7 维）
3. **Action Head**：输入是 7 维 token 的 hidden states，输出是 8 维预测（前 7 维是动作，第 8 维是 EOS）
4. **Sigmoid 激活**：对第 8 维应用 sigmoid，将其约束在 [0, 1] 范围
5. **推理阶段**：检测第 8 维是否超过阈值（0.5），用于判断 substep 结束
6. **环境执行**：只将前 7 维传递给机器人环境

### 回退到 7D 的要点
1. **移除 BASE_ACTION_DIM**：所有使用 BASE_ACTION_DIM 的地方改回 ACTION_DIM
2. **恢复 ACTION_DIM = 7**：在 constants.py 中修改
3. **移除 EOS 逻辑**：删除所有 EOS flag 生成、检测、显示代码
4. **恢复维度匹配**：确保所有 reshape、slice 操作使用正确的维度
5. **重新训练**：必须使用 7D 配置重新训练模型

---

## 注意事项

⚠️ **重要**：
- 8D 训练的 checkpoint 不能直接用于 7D 推理（维度不匹配）
- 7D 训练的 checkpoint 不能直接用于 8D 推理（缺少 EOS 维度）
- 切换维度后必须重新训练模型
- 数据预处理逻辑的改变会影响数据标签格式

📝 **建议**：
- 在修改前备份当前工作代码
- 使用版本控制（git）管理不同维度的实现
- 创建不同的 git 分支管理 7D 和 8D 版本

---

## 参考

- 原始讨论和修改记录请参考对话历史
- 关键 bug 修复：EOS flag 标注逻辑（在 `datasets_substep.py` 中）
- 数值稳定性关键：对 EOS 维度应用 sigmoid 激活











