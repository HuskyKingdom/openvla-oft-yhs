# EOS Token 机制详解

## 什么是 EOS Token？

**EOS (End of Sequence)** 是一个特殊的token，用于标记序列的结束。在自然语言处理中，EOS token通常用于：
- 标记句子的结束
- 标记生成任务的完成
- 告诉模型"这里应该停止生成"

在我们的substep场景中，**EOS token标记substep的结束**，告诉模型"这个substep已经完成，可以切换到下一个substep了"。

## EOS Token 的具体形式

### 1. 文本形式
EOS token在文本中通常是一个特殊字符串，例如：
- `"<|endoftext|>"` (GPT系列)
- `"</s>"` (LLaMA系列)
- `"<eos>"` (某些模型)

### 2. Token ID形式
在模型内部，EOS token被编码为一个整数ID，例如：
- `tokenizer.eos_token_id = 2` (LLaMA-2)
- `tokenizer.eos_token_id = 128009` (某些模型)

## 训练阶段：如何插入 EOS

### 流程概览

```
1. label_substeps.py 标记数据
   └─> 识别substep边界，设置 is_substep_end=True
   
2. SubstepRLDSBatchTransform 处理数据
   └─> 检测 is_substep_end=True
   └─> 在action序列末尾插入 EOS token
   
3. 模型训练
   └─> 学习预测 EOS token
```

### 具体实现

**位置**：`prismatic/vla/datasets/datasets_substep.py` 第242-247行

```python
# 获取action token序列
action_chunk_string = current_action_string + future_actions_string

# 如果这是substep的结束位置，插入EOS token
if self.use_substep_eos and is_substep_end:
    # 获取tokenizer的EOS token（文本形式，如"</s>"）
    eos_token = self.base_tokenizer.eos_token
    # 追加到action序列末尾
    action_chunk_string = action_chunk_string + eos_token
```

### 训练数据示例

**原始action序列**（假设8个actions，每个action用7个token表示）：
```
[action_token_1, action_token_2, ..., action_token_56]
```

**在substep结束位置插入EOS后**：
```
[action_token_1, action_token_2, ..., action_token_56, EOS_token]
```

**模型学习目标**：
- 在substep结束位置，模型应该预测EOS token
- EOS token的loss参与训练，模型会学习"什么时候应该输出EOS"

## 推理阶段：如何检测 EOS

### 流程概览

```
1. 模型生成action logits
   └─> 对每个token位置，输出vocab_size维度的概率分布
   
2. 查找EOS token
   └─> 检查每个位置最可能的token是否是EOS
   
3. 映射到action timestep
   └─> 将token位置转换为action timestep
   
4. 截断action sequence
   └─> 只保留EOS之前的actions
```

### 具体实现

**位置**：`prismatic/extern/hf/modeling_prismatic.py` 第1460-1491行

#### 步骤1：获取EOS Token ID

```python
# 从模型配置或tokenizer获取EOS token的ID
eos_token_id = self.language_model.config.eos_token_id
if eos_token_id is None:
    # 备用方法：从tokenizer获取
    eos_token_id = self.language_model.tokenizer.eos_token_id
```

#### 步骤2：模型生成Logits

模型前向传播后，得到action位置的logits：
```python
# action_logits shape: (batch_size, seq_len, vocab_size)
# 例如：(1, 56, 32000)
# - batch_size=1
# - seq_len=56 (8个actions × 7个tokens/action)
# - vocab_size=32000 (词汇表大小)
```

#### 步骤3：查找EOS Token

```python
# 对每个位置，找到最可能的token（argmax）
predicted_token_ids = action_logits.argmax(dim=-1)  
# shape: (1, 56)
# 例如: [1234, 5678, ..., 2, 1234, ...]
#                              ↑
#                            EOS token ID

# 检查哪些位置预测了EOS
eos_mask = (predicted_token_ids == eos_token_id)
# shape: (1, 56)
# 例如: [False, False, ..., True, False, ...]
#                              ↑
#                            EOS位置
```

#### 步骤4：映射到Action Timestep

```python
if eos_mask.any():  # 如果检测到EOS
    # 找到第一个EOS的token位置
    first_eos_token_pos = eos_mask[0].nonzero()[0][0].item()
    # 例如: first_eos_token_pos = 42
    
    # 将token位置转换为action timestep
    # 每个action有ACTION_DIM个tokens（通常是7）
    eos_position = first_eos_token_pos // ACTION_DIM
    # 例如: 42 // 7 = 6
    # 表示第6个action（0-indexed）之后是EOS
```

### 完整示例

假设：
- `NUM_ACTIONS_CHUNK = 8`（生成8个actions）
- `ACTION_DIM = 7`（每个action用7个tokens表示）
- `vocab_size = 32000`
- `eos_token_id = 2`

**模型输出logits**：
```
action_logits shape: (1, 56, 32000)
```

**每个位置的预测token**：
```
predicted_token_ids = [1234, 5678, 8901, ..., 2, 1234, ...]
                                              ↑
                                           位置42预测了EOS
```

**检测结果**：
```python
eos_mask = [False, False, ..., True, False, ...]
                              ↑
                           位置42
                           
first_eos_token_pos = 42
eos_position = 42 // 7 = 6  # 第6个action之后
```

**截断actions**：
```python
# 原始actions: [action_0, action_1, ..., action_7]  (8个)
# 截断后:      [action_0, action_1, ..., action_6]  (7个，包含EOS位置)
actions = actions[:eos_position+1]  # [:7]
```

## 为什么需要映射Token位置到Action Timestep？

### 问题

模型生成的是**token序列**，但我们需要的是**action序列**。两者之间的关系是：
- 每个action由多个tokens表示（通常是7个）
- 如果EOS出现在某个action的中间token位置，我们需要知道是哪个action

### 解决方案

使用整数除法将token位置映射到action timestep：
```python
action_timestep = token_position // ACTION_DIM
```

### 示例

假设 `ACTION_DIM = 7`：

| Token位置 | Action Timestep | 说明 |
|----------|----------------|------|
| 0-6      | 0              | 第1个action的tokens |
| 7-13     | 1              | 第2个action的tokens |
| 14-20    | 2              | 第3个action的tokens |
| ...      | ...            | ... |
| 42       | 6              | 第7个action的某个token位置 |

如果EOS出现在token位置42，我们截断到action timestep 6，即保留前7个actions（0-6）。

## 推理时的完整流程

### 代码调用链

```
1. run_libero_pro_eval_substep.py
   └─> get_action(..., return_eos_info=True)
   
2. robot_utils.py
   └─> get_vla_action(..., return_eos_info=True)
   
3. openvla_utils.py
   └─> vla.predict_action(..., return_eos_info=True)
   
4. modeling_prismatic.py
   └─> 模型前向传播
   └─> 检测EOS token
   └─> 返回 (actions, has_eos, eos_position)
   
5. 回到评估脚本
   └─> 如果has_eos=True，截断actions
   └─> 设置force_requery_after_queue=True
```

### 实际使用示例

**位置**：`experiments/robot/libero/run_libero_pro_eval_substep.py` 第684-734行

```python
# 1. 调用模型获取actions和EOS信息
if cfg.use_eos_detection:
    actions, has_eos, eos_position = get_action(
        cfg, model, observation, current_instruction,
        processor=processor,
        action_head=action_head,
        # ... 其他参数 ...
        return_eos_info=True,  # 请求EOS检测
    )
    
    # 2. 如果检测到EOS，截断actions
    if has_eos and eos_position is not None:
        actions = actions[:eos_position+1]  # 只保留EOS之前的actions
        log_message(
            f"[EOS] Detected at position {eos_position}, "
            f"truncated to {len(actions)} actions",
            log_file
        )
        # 3. 标记需要在queue清空后切换substep
        force_requery_after_queue = True

# 4. 将actions加入队列
action_queue.extend(actions)

# 5. 执行actions直到queue清空
# 6. 如果force_requery_after_queue=True，切换substep
```

## 关键要点总结

### 1. EOS是什么？
- **文本形式**：特殊字符串（如`"</s>"`）
- **Token ID**：整数（如`2`）
- **作用**：标记substep结束

### 2. 训练时如何插入？
- 在`is_substep_end=True`的位置
- 将EOS token追加到action序列末尾
- 模型学习预测EOS

### 3. 推理时如何检测？
- 模型生成logits（概率分布）
- 查找argmax是否为EOS token ID
- 将token位置映射到action timestep
- 截断actions到EOS位置

### 4. 为什么需要映射？
- 模型生成的是tokens，我们需要的是actions
- 每个action由多个tokens表示
- 需要知道EOS对应哪个action

### 5. 检测的准确性
- 依赖于模型是否学会了预测EOS
- 如果模型训练时没有EOS，推理时无法检测
- 需要`use_substep_eos=True`训练模型

## 调试技巧

### 1. 检查EOS Token ID

```python
# 在推理时打印EOS token ID
print(f"EOS token ID: {tokenizer.eos_token_id}")
print(f"EOS token text: {tokenizer.eos_token}")
```

### 2. 检查Logits中的EOS概率

```python
# 在modeling_prismatic.py中添加调试代码
if return_eos_info and action_logits is not None:
    # 获取EOS token的概率（不仅仅是argmax）
    eos_probs = torch.softmax(action_logits, dim=-1)[:, :, eos_token_id]
    print(f"EOS probabilities: {eos_probs}")
    print(f"Max EOS prob: {eos_probs.max()}")
```

### 3. 检查检测结果

```python
# 在评估脚本中添加日志
if has_eos:
    print(f"✓ EOS detected at action timestep {eos_position}")
    print(f"  Original actions: {len(original_actions)}")
    print(f"  Truncated actions: {len(actions)}")
else:
    print("✗ No EOS detected")
```

## 常见问题

### Q1: 为什么检测不到EOS？
**A**: 可能原因：
1. 模型训练时没有使用`use_substep_eos=True`
2. 使用的是L1回归模式（`action_head is not None`），无法检测token
3. EOS token ID配置错误

### Q2: EOS检测的准确性如何？
**A**: 取决于：
1. 模型训练质量
2. 训练数据中EOS标记的准确性
3. 模型是否真正学会了预测EOS

### Q3: 如果检测到多个EOS怎么办？
**A**: 当前实现只使用第一个EOS位置。可以改进为：
- 选择最合理的EOS位置
- 或者使用EOS概率最高的位置

### Q4: L1回归模式能检测EOS吗？
**A**: **不能**。L1回归模式直接输出连续动作，不生成token，因此无法检测EOS。只有离散token模式才能检测。

## 总结

EOS token机制的核心是：
1. **训练时**：在substep边界插入EOS，让模型学习
2. **推理时**：检测模型是否预测了EOS，判断substep是否完成
3. **切换时**：基于EOS检测结果，自动切换到下一个substep

这是一个**基于模型预测的substep边界检测方法**，比纯视觉相似度方法更直接、更准确。

