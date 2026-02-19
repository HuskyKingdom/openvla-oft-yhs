# Substep Instruction Fine-tuning & Inference 实现总结

## 1. 训练部分 (`finetune_substep.py`)

### 1.1 核心思路

将原始 per-episode 任务指令替换为 per-timestep substep 指令，配合 EOS 分类头实现子步骤边界感知训练。

### 1.2 数据流

```
substep_labels.json
       ↓
SubstepRLDSDataset
       ↓  (替换 task_description → 当前 timestep 的 substep 指令)
SubstepRLDSBatchTransform
       ↓  (use_substep_eos=True 时在子步骤边界插入 EOS token)
batch["input_ids"] + batch["eos_labels"]
```

- `batch["eos_labels"]`：shape `(B, NUM_ACTIONS_CHUNK, 1)`，标注每个 action step 是否为子步骤结束

### 1.3 模型结构

| 组件 | 说明 |
|------|------|
| VLA (LoRA) | 主干，提取 `actions_hidden_states` |
| `L1RegressionActionHead` / `DiffusionActionHead` | 动作预测头 |
| `EOSClassificationHead` | 新增：从 `actions_hidden_states` 预测每步是否为 EOS |

`EOSClassificationHead` 输入 `actions_hidden_states: (B, NUM_ACTIONS_CHUNK * ACTION_DIM, D)`，输出 `eos_logits: (B, NUM_ACTIONS_CHUNK, 1)`。

### 1.4 损失函数

```
total_loss = action_loss + lambda_eos * eos_loss
```

**EOS Loss 两种模式：**

- **Focal Loss**（推荐，应对极端不平衡）：
  `FL = -alpha_t * (1 - p_t)^gamma * log(p_t)`，默认 `alpha=0.25, gamma=2.0`

- **Weighted BCE**：
  - 全局固定权重：`pos_weight=20.0`（针对 1:800 量级不平衡）
  - 动态权重：按 batch 内正负比例计算（不适合极端不平衡）

关键梯度设计：EOS loss 通过 `actions_hidden_states`（有梯度连接）回传到 VLA 主干，实现真正的端到端 fine-tuning。

### 1.5 训练稳定性措施

- **梯度裁剪**：`max_grad_norm=0.8`，防止高 `pos_weight` 时梯度爆炸
- **NaN/Inf 检测**：检测到非有限 loss 时跳过该 batch
- **EOS 分布预采样**：训练前采样 50 个 batch 估算 EOS=1 比例，给出权重建议

### 1.6 Checkpoint 保存

标准组件（VLA、action head 等）由 `save_training_checkpoint` 处理；  
EOS head 额外单独保存为 `eos_head--{step}_checkpoint.pt`。

---

## 2. 推理部分 (`run_libero_pro_eval_substep.py`)

### 2.1 核心思路

推理时动态切换指令：使用 `SubstepManager` 维护当前子步骤状态，通过 EOS 检测或视觉相似度判断何时切换到下一子步骤。

### 2.2 子步骤切换机制（二选一）

**Method 1：EOS 检测（`use_eos_detection=True`，优先级更高）**

```
get_action(..., return_eos_info=True, eos_head=eos_head)
       ↓
(actions, has_eos, eos_position)
       ↓
如果 has_eos：截断 actions 到 eos_position+1
              设置 force_requery_after_queue=True
       ↓
action_queue 消耗完毕后 → advance_substep() → 以新指令重新查询
```

**Method 2：视觉相似度（`use_eos_detection=False`）**

```
SigCLIP 计算当前帧与 substep expected_effect 的相似度
       ↓
超过 substep_completion_threshold → advance_substep()
清空 action_queue，以新指令重新查询
```

### 2.3 EOS Head 加载

```python
# 搜索顺序
eos_head--latest_checkpoint.pt
eos_head--{step}_checkpoint.pt  (按 step 降序)
```

加载后移除 DDP `module.` 前缀，设为 `eval()` 模式。

### 2.4 指令动态替换

```
substep_manager.get_current_instruction()
       ↓  (返回当前子步骤的 subgoal 文本)
get_action(cfg, model, observation, current_instruction, ...)
```

当 `use_substep_decomposition=False` 时，回退到原始任务描述。

### 2.5 子步骤分解（LLM + SigCLIP）

| 组件 | 用途 |
|------|------|
| Qwen2.5 / 任意 CausalLM | 将任务描述分解为有序子步骤列表 |
| SigLIP / open_clip | 计算图像与预期效果文本的相似度 |

`SubstepManager` 封装两者，对外提供 `should_switch_substep(img)` 和 `advance_substep()` 接口。

### 2.6 整体推理循环

```
reset env
    ↓
初始化 SubstepManager（LLM 分解任务）
    ↓
while t < max_steps:
    准备观测
    判断是否需要 requery（queue 空 / 子步骤切换）
    get_action(current_instruction)  →  EOS 检测  →  action_queue
    popleft() 执行动作
    检查子步骤完成条件（EOS 或视觉相似度）
    done → break
```

---

## 3. 关键参数速查

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `use_substep_eos` | `True` | 训练时插入 EOS token |
| `use_eos_classification` | `True` | 训练 EOS 分类头 |
| `lambda_eos` | `1.0` | EOS loss 权重 |
| `eos_use_focal_loss` | `True` | 使用 Focal Loss |
| `eos_pos_weight` | `20.0` | BCE 正样本权重 |
| `max_grad_norm` | `0.8` | 梯度裁剪上限 |
| `use_eos_detection` | `False` | 推理时启用 EOS 切换 |
| `eos_threshold` | `0.03` | EOS 触发概率阈值 |
| `substep_completion_threshold` | `0.25` | 视觉相似度切换阈值 |

