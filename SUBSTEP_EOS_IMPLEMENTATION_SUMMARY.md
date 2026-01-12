# Substep EOS Token 实现总结

## 概述

本文档总结了为OpenVLA-OFT项目添加substep边界EOS token检测功能的完整实现。该功能允许模型在训练时学习预测substep结束位置，并在推理时基于EOS token检测进行自动substep切换。

## 功能目标

1. **训练阶段**：在substep结束位置添加EOS token，让模型学习预测substep边界
2. **推理阶段**：检测模型输出的EOS token，在EOS位置截断action sequence并自动切换到下一个substep

## 修改文件清单

### 1. `label_substeps.py` - Substep标记工具

**修改位置**：
- 第696行：`map_timesteps_to_apd_steps` 函数
- 第775-788行：`create_output_structure` 函数
- 第876-877行：`process_single_episode` 函数日志输出

**修改内容**：
1. 为每个timestep添加 `is_substep_end` 字段，标记substep的最后一个timestep
2. 在summary中添加 `substep_boundaries` 和 `num_substeps` 统计信息
3. 增强日志输出，显示substep边界信息

**关键代码**：
```python
# 为timestep添加is_substep_end标志
timestep_labels.append({
    "timestep": t,
    "action": block['type'],
    "APD_step": block['apd_step'],
    "cycle": block['cycle'],
    "is_substep_end": (t == block['end'] - 1)  # 标记substep结束
})

# 提取substep边界
substep_boundaries = [label['timestep'] for label in timestep_labels if label.get('is_substep_end', False)]
summary_updated['substep_boundaries'] = substep_boundaries
summary_updated['num_substeps'] = len(substep_boundaries)
```

---

### 2. `prismatic/vla/datasets/datasets_substep.py` - 数据加载层

**修改位置**：
- 第147-175行：`SubstepRLDSBatchTransform` 类定义
- 第74-144行：`get_substep_instruction` 函数
- 第233-242行：`__call__` 方法中的EOS插入逻辑
- 第254-258行：labels处理逻辑

**修改内容**：
1. 添加 `use_substep_eos` 参数控制EOS token插入
2. 修改 `get_substep_instruction` 返回 `(instruction, is_substep_end)` 元组
3. 在substep结束位置自动插入EOS token到action序列
4. EOS token的loss参与训练（不设为IGNORE_INDEX）

**关键代码**：
```python
@dataclass
class SubstepRLDSBatchTransform:
    # ... 其他字段 ...
    use_substep_eos: bool = False  # 新增参数

# 在substep结束位置插入EOS
if self.use_substep_eos and is_substep_end:
    eos_token = self.base_tokenizer.eos_token
    action_chunk_string = action_chunk_string + eos_token

# EOS token参与loss计算
if not self.predict_stop_token and not (self.use_substep_eos and is_substep_end):
    labels[-1] = IGNORE_INDEX
```

---

### 3. `vla-scripts/finetune_substep.py` - 训练脚本

**修改位置**：
- 第162行：`FinetuneSubstepConfig` 配置类
- 第409行：`SubstepRLDSBatchTransform` 初始化

**修改内容**：
1. 添加 `use_substep_eos: bool = True` 配置参数
2. 将配置传递给batch transform

**关键代码**：
```python
@dataclass
class FinetuneSubstepConfig:
    # ... 其他配置 ...
    use_substep_eos: bool = True  # 在substep结束位置添加EOS token

# 传递配置到batch transform
batch_transform = SubstepRLDSBatchTransform(
    # ... 其他参数 ...
    use_substep_eos=cfg.use_substep_eos,
)
```

---

### 4. `experiments/robot/robot_utils.py` - 推理接口

**修改位置**：
- 第110行：`get_action` 函数签名
- 第139行：函数调用传递 `return_eos_info` 参数

**修改内容**：
1. 添加 `return_eos_info` 参数
2. 修改返回类型支持EOS信息

**关键代码**：
```python
def get_action(
    # ... 其他参数 ...
    return_eos_info: bool = False,
) -> Union[List[np.ndarray], np.ndarray, Tuple]:
    result = get_vla_action(
        # ... 其他参数 ...
        return_eos_info=return_eos_info,
    )
    return result
```

---

### 5. `experiments/robot/openvla_utils.py` - VLA Action生成

**修改位置**：
- 第984行：`get_vla_action` 函数签名
- 第1052-1132行：所有 `predict_action` 调用路径
- 第1215-1226行：返回值处理

**修改内容**：
1. 添加 `return_eos_info` 参数
2. 在所有调用 `predict_action` 的路径中传递EOS检测参数
3. 处理新的返回值格式（包含EOS信息）
4. 初始化EOS检测变量确保所有代码路径都有值

**关键代码**：
```python
def get_vla_action(
    # ... 其他参数 ...
    return_eos_info: bool = False,
) -> Union[List[np.ndarray], Tuple[List[np.ndarray], bool, Optional[int]]]:
    
    # 初始化EOS变量
    has_eos = False
    eos_position = None
    
    # 调用predict_action时传递return_eos_info
    if return_eos_info:
        result = vla.predict_action(..., return_eos_info=True)
        if len(result) == 6:
            action, hiddens, layer_actions, mask, has_eos, eos_position = result
    
    # 返回EOS信息
    if return_eos_info:
        return actions_list, has_eos, eos_position
    else:
        return actions_list
```

---

### 6. `experiments/robot/libero/run_libero_pro_eval_substep.py` - 评估脚本

**修改位置**：
- 第206行：`GenerateConfig` 配置类
- 第571行：`run_episode` 函数状态变量
- 第641-682行：Substep切换逻辑
- 第684-734行：Action获取和EOS处理

**修改内容**：
1. 添加 `use_eos_detection: bool = False` 配置参数
2. 添加 `force_requery_after_queue` 状态追踪变量
3. 实现双路径substep切换：
   - **方式1（优先）**：EOS token检测切换
   - **方式2（兜底）**：视觉相似度切换
4. 在action获取时处理EOS检测结果，截断action sequence

**关键代码**：
```python
@dataclass
class GenerateConfig:
    # ... 其他配置 ...
    use_eos_detection: bool = False  # 启用EOS检测进行substep切换

# 状态追踪
force_requery_after_queue = False  # EOS检测后等待queue清空的标志

# 双路径切换逻辑
# 方式1: EOS切换（优先级高）
if cfg.use_eos_detection and force_requery_after_queue and len(action_queue) == 0:
    substep_manager.advance_substep()
    force_requery_after_queue = False
    should_requery = True

# 方式2: 视觉相似度切换（兜底）
if not substep_switched and substep_manager.should_switch_substep(img):
    substep_manager.advance_substep()

# Action获取与EOS处理
if cfg.use_eos_detection:
    actions, has_eos, eos_position = get_action(..., return_eos_info=True)
    
    if has_eos and eos_position is not None:
        actions = actions[:eos_position+1]  # 截断到EOS位置
        force_requery_after_queue = True    # 标记等待切换
```

---

### 7. `prismatic/extern/hf/modeling_prismatic.py` - 模型核心

**修改位置**：
- 第1337行：`predict_action` 函数签名
- 第1215-1325行：`_regression_or_discrete_prediction` 方法
- 第1460-1491行：EOS检测逻辑
- 第1500-1505行：返回值处理

**修改内容**：
1. 添加 `return_eos_info` 参数
2. 修改 `_regression_or_discrete_prediction` 返回 `action_logits`
3. 实现EOS token检测逻辑：
   - 从tokenizer获取EOS token ID
   - 在action logits中搜索EOS token
   - 将token位置映射到action timestep
4. 修改返回值包含EOS检测信息

**关键代码**：
```python
def predict_action(
    # ... 其他参数 ...
    return_eos_info: bool = False,
):
    # ... 模型前向传播 ...
    
    # EOS检测逻辑
    if return_eos_info and action_logits is not None:
        eos_token_id = self.language_model.config.eos_token_id
        
        if eos_token_id is not None:
            # 获取每个位置最可能的token
            predicted_token_ids = action_logits.argmax(dim=-1)
            
            # 检查是否预测了EOS
            eos_mask = (predicted_token_ids == eos_token_id)
            
            if eos_mask.any():
                # 找到第一个EOS位置并映射到action timestep
                first_eos_token_pos = eos_mask[0].nonzero()[0][0].item()
                eos_position = first_eos_token_pos // ACTION_DIM
                has_eos = True
    
    if return_eos_info:
        return actions, hiddens, layer_actions, mask, has_eos, eos_position
    else:
        return actions, hiddens, layer_actions, mask
```

---

## 数据流程

### 训练阶段流程

```
1. label_substeps.py 生成标记
   └─> 输出包含 is_substep_end 标志的JSON文件
   
2. SubstepRLDSDataset 加载数据
   └─> 读取 substep_labels_output.json
   
3. SubstepRLDSBatchTransform 处理每个batch
   └─> 检测 is_substep_end=True
   └─> 在action序列末尾插入 EOS token
   └─> 模型学习预测 EOS token
   
4. 模型训练
   └─> EOS token的loss参与训练
```

### 推理阶段流程

```
1. 环境observation
   └─> 准备模型输入
   
2. vla.predict_action() 生成actions
   └─> 在离散token模式下生成action logits
   └─> 检测logits中是否包含EOS token
   └─> 返回 (actions, has_eos, eos_position)
   
3. EOS处理
   └─> 如果检测到EOS：
       ├─> 截断actions到eos_position
       └─> 设置 force_requery_after_queue = True
   
4. Action执行
   └─> 执行truncated actions直到queue清空
   
5. Substep切换
   └─> queue清空 + force_requery_after_queue = True
   └─> 触发 substep_manager.advance_substep()
   └─> 重新query下一个substep
```

## 使用方法

### 1. 生成Substep标记

```bash
python label_substeps.py \
    --apd_path APD_plans.json \
    --rlds_data_dir /path/to/modified_libero_rlds \
    --output_path substep_labels_output.json \
    --suites libero_spatial_no_noops libero_object_no_noops \
    --max_episodes 50
```

输出JSON将包含 `is_substep_end` 字段和 `substep_boundaries` 统计。

### 2. 训练模型（启用EOS训练）

```bash
python vla-scripts/finetune_substep.py \
    --vla_path openvla/openvla-7b \
    --dataset_name libero_goal_no_noops \
    --substep_labels_path substep_labels_output.json \
    --use_substep_eos=True  # 启用EOS token训练
```

### 3. 推理评估（启用EOS检测切换）

```bash
python experiments/robot/libero/run_libero_pro_eval_substep.py \
    --pretrained_checkpoint /path/to/checkpoint \
    --task_suite_name libero_spatial \
    --use_eos_detection=True  # 启用EOS检测进行substep切换
```

## 重要限制和注意事项

### 1. EOS检测模式限制

- **仅支持离散token模式**：EOS检测只在 `action_head is None` 时有效
- **L1回归模式**：当使用 `action_head`（L1回归）时，模型输出连续动作而非token，无法检测EOS

### 2. 训练模式要求

- 必须使用 `use_substep_eos=True` 训练模型，模型才能学习预测EOS
- 训练数据必须包含 `is_substep_end` 标记（通过 `label_substeps.py` 生成）

### 3. 双路径切换机制

- **EOS切换（优先）**：基于模型预测，更直接准确
- **视觉相似度切换（兜底）**：当EOS检测失败时使用，保证系统鲁棒性
- 两种方式可以同时启用，EOS切换优先级更高

## 输出格式变更

### label_substeps.py 输出格式

**修改前**：
```json
{
  "timestep_labels": [
    {"timestep": 0, "action": "pick", "APD_step": "...", "cycle": 0}
  ]
}
```

**修改后**：
```json
{
  "timestep_labels": [
    {
      "timestep": 0,
      "action": "pick",
      "APD_step": "...",
      "cycle": 0,
      "is_substep_end": false
    },
    {
      "timestep": 45,
      "action": "pick",
      "APD_step": "...",
      "cycle": 0,
      "is_substep_end": true
    }
  ],
  "summary": {
    "substep_boundaries": [45, 120],
    "num_substeps": 2
  }
}
```

## 测试建议

### 1. 验证EOS token插入

在训练时检查数据加载器是否正确插入EOS token：
- 在 `SubstepRLDSBatchTransform.__call__` 中添加调试输出
- 确认 `is_substep_end=True` 的位置都有EOS token

### 2. 验证EOS检测

在推理时检查EOS检测是否工作：
- 启用日志输出，查看 `has_eos` 和 `eos_position` 的值
- 确认在substep边界处正确检测到EOS

### 3. 验证Substep切换

在评估时检查切换行为：
- 确认EOS检测后正确截断actions
- 确认substep切换发生在正确时机
- 对比EOS切换和视觉切换的效果

## 代码质量检查

所有修改已通过以下检查：
- ✅ Linter检查：无新增错误
- ✅ 类型提示：保持类型一致性
- ✅ 向后兼容：所有新增功能都有配置开关
- ✅ 错误处理：包含fallback逻辑

## 后续改进方向

1. **支持L1回归模式的EOS检测**：
   - 可能需要使用hidden states或其他信号
   - 或者训练一个专门的EOS分类器

2. **EOS检测置信度**：
   - 不仅检查argmax，还可以检查EOS token的概率
   - 设置置信度阈值以提高准确性

3. **多EOS处理**：
   - 如果在一个action chunk中检测到多个EOS，选择最合理的一个

4. **训练增强**：
   - 可以添加EOS预测的专门loss项
   - 或者使用Focal Loss处理EOS token的不平衡问题

## 总结

本次实现完整地添加了substep边界EOS token检测功能，包括：
- ✅ 数据标记（`label_substeps.py`）
- ✅ 数据加载（`datasets_substep.py`）
- ✅ 训练配置（`finetune_substep.py`）
- ✅ 模型推理（`modeling_prismatic.py`）
- ✅ 推理接口（`robot_utils.py`, `openvla_utils.py`）
- ✅ 评估脚本（`run_libero_pro_eval_substep.py`）

所有修改都保持了代码的模块化和可配置性，便于后续维护和扩展。

