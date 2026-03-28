# OFT 代码适配说明：支持 Vanilla OpenVLA 及 Autoregressive Discrete Token 模型评估

## 背景

OFT（OpenVLA Fine-Tuning）codebase 的推理架构与原版 Vanilla OpenVLA 存在根本性不兼容，导致直接用该 codebase 评估 `openvla/openvla-7b-finetuned-*` 系列 HuggingFace checkpoint 时 SR 为 0。本文档记录了为解决此问题所做的全部必要改动。

---

## 根本原因分析

### 问题 1：推理方式根本不同

| | OFT 推理（`predict_action`） | Vanilla OpenVLA 推理 |
|---|---|---|
| 方式 | 在输入后追加 `NUM_ACTIONS_CHUNK × ACTION_DIM = 56` 个 placeholder token，**一次 forward pass** 并行提取所有 action logit | 调用 `.generate()`，**逐 token autoregressive** 生成 7 个 action token |
| 适用模型 | OFT fine-tuned 模型 | 原版 OpenVLA 训练的模型 |

OFT 的 `predict_action` 修改了 `modeling_prismatic.py`，训练与推理均采用 single-pass parallel 方式。将此方式用于 vanilla OpenVLA 权重时，模型从未见过末尾追加 56 个 placeholder token 的输入格式，logit 完全错误。

### 问题 2：action embedding 清零机制

OFT 在训练时（`forward()`）将所有 action token 的 embedding 清零，使所有 action 从相同的 context（prompt + vision）独立并行预测。这与 vanilla OpenVLA 的 autoregressive 训练方式不同。

### 问题 3：输入图像数量

`GenerateConfig` 中 `num_images_in_input` 默认值为 **2**（主摄像头 + wrist 摄像头），而 vanilla OpenVLA 只用 1 张图像训练。多余的 wrist 图像会导致输入维度不匹配。

### 问题 4：action 返回 shape 处理

修复 vanilla 推理路径后，`get_vla_action` 末尾的 `actions_list` 构建代码假设 `action` shape 为 `[chunk_size, 7]`，但 vanilla OpenVLA 返回 shape `[7]`（单步 action），导致将 7 个 action 维度误拆成 7 个标量。

---

## 改动详情

### 改动 1：`experiments/robot/openvla_utils.py`

**位置**：`get_vla_action()` 函数

#### 1a. 修复 `action_head is None` 时的推理路径

**原代码**（走 OFT 的 `predict_action`，parallel single-pass，对 vanilla 完全错误）：
```python
if action_head is None:
    action, _, _, _ = vla.predict_action(**inputs, unnorm_key=cfg.unnorm_key, do_sample=False)
```

**改后**（走原版 autoregressive `generate()`）：
```python
if action_head is None:
    # 原版 OpenVLA 的 token 29871 检查（与训练时输入格式保持一致）
    vanilla_input_ids = inputs["input_ids"]
    if not torch.all(vanilla_input_ids[:, -1] == 29871):
        vanilla_input_ids = torch.cat(
            (vanilla_input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(vanilla_input_ids.device)),
            dim=1,
        )
    vanilla_inputs = {**inputs, "input_ids": vanilla_input_ids}

    action_dim = vla.get_action_dim(cfg.unnorm_key)
    generated_ids = vla.generate(**vanilla_inputs, max_new_tokens=action_dim, do_sample=False)
    predicted_action_token_ids = generated_ids[0, -action_dim:].cpu().numpy()
    discretized_actions = vla.vocab_size - predicted_action_token_ids
    discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=vla.bin_centers.shape[0] - 1)
    normalized_actions = vla.bin_centers[discretized_actions]
    action = vla._unnormalize_actions(normalized_actions, cfg.unnorm_key)
    has_eos, eos_position = False, None
```

#### 1b. 修复 `actions_list` 构建时的 shape 处理

**原代码**（shape `[7]` 会被错误拆成 7 个标量）：
```python
actions_list = [action[i] for i in range(len(action))]
```

**改后**：
```python
if isinstance(action, np.ndarray) and action.ndim == 1:
    actions_list = [action]  # vanilla OpenVLA 单步 action，包成列表
else:
    actions_list = [action[i] for i in range(len(action))]
```

---

### 改动 2：`prismatic/extern/hf/modeling_prismatic.py`

#### 2a. 修复 `forward()` 在 `labels=None` 时 crash

**问题**：`generate()` 调用时 `labels=None`，但 OFT 的 multimodal forward path 强制调用 `_process_action_masks(labels)` → crash。

**改后**（在 `forward()` multimodal 路径的 action mask 计算处）：
```python
# 原代码（labels=None 时 crash）：
all_actions_mask = self._process_action_masks(labels)

# 改后：
if labels is not None:
    all_actions_mask = self._process_action_masks(labels)
else:
    # generate() 调用时 labels=None，输入中没有 action token，mask 全为 False
    all_actions_mask = torch.zeros(
        input_embeddings.shape[:2], dtype=torch.bool, device=input_embeddings.device
    )
```

#### 2b. 新增 `zero_action_embeddings` 参数，解耦训练模式

**目的**：使 `use_l1_regression=False` 训练时不再清零 action embedding，实现真正的 autoregressive 56-token chunk 训练（training/eval 一致）。`use_l1_regression=True` 行为完全不变。

**`forward()` 函数签名新增参数**：
```python
def forward(
    self,
    ...
    use_film: bool = False,
    zero_action_embeddings: bool = True,  # 新增，默认 True 保持向后兼容
) -> ...:
```

**清零逻辑改动**：
```python
# 原代码（所有非 diffusion 情况均清零）：
else:
    all_actions_mask = all_actions_mask.unsqueeze(-1)
    input_embeddings = input_embeddings * ~all_actions_mask

# 改后（只在需要时清零）：
elif zero_action_embeddings:
    # L1 regression / OFT parallel 模式：清零，并行预测
    all_actions_mask = all_actions_mask.unsqueeze(-1)
    input_embeddings = input_embeddings * ~all_actions_mask
# else: autoregressive discrete-token 模式，保留 embedding，标准 causal LM
```

---

### 改动 3：`vla-scripts/finetune.py` 和 `vla-scripts/finetune_substep.py`

在两个训练脚本的 `vla(...)` forward 调用处，传入 `zero_action_embeddings` 参数：

```python
output = vla(
    ...
    use_film=use_film,
    zero_action_embeddings=(use_l1_regression or use_diffusion),  # 新增
)
```

**逻辑**：
- `use_l1_regression=True`：`zero_action_embeddings=True`，清零，OFT parallel 预测，行为完全不变 ✓
- `use_l1_regression=False`：`zero_action_embeddings=False`，不清零，autoregressive chunk 训练 ✓

---

### 改动 4：`auto_eval_nv40_openvla_pro.sh`

新增 `NUM_IMAGES_IN_INPUT=1` 变量，并在所有 python 命令中加入 `--num_images_in_input $NUM_IMAGES_IN_INPUT`。

**原因**：vanilla OpenVLA 只用 1 张图像训练，`num_images_in_input` 默认值为 2 会将 wrist 图像拼接进输入，造成维度不匹配。

---

## 改动后各场景的推理路径

| 模型类型 | `use_l1_regression` | 推理路径 | 训练/推理一致 |
|---|---|---|---|
| Vanilla HF checkpoint（`openvla-7b-finetuned-*`）| False | autoregressive `generate()` | ✓ |
| OFT fine-tuned（L1 regression）| True | OFT `predict_action`（parallel single-pass）| ✓（未改动）|
| 新训练：autoregressive discrete token chunk | False | autoregressive `generate()` | ✓ |

---

## 注意事项

- `auto_eval_nv40_openvla_pro.sh` 用于评估 vanilla / autoregressive discrete token 模型（`use_l1_regression=False`）
- `auto_eval_nv40_pro.sh` 用于评估 OFT L1 regression + substep 模型（`use_l1_regression=True`），两者不可混用
- 评估自训练 checkpoint 时需修改脚本中的 `CKPT_*` 路径和 `--unnorm_key` 为训练时的 dataset name
- `generate()` 每次生成 7 个 token（1 步 action），未利用 chunk 中的后续 7 步；如需 open-loop 执行完整 8 步 chunk，需将 `max_new_tokens` 改为 56 并相应解析
