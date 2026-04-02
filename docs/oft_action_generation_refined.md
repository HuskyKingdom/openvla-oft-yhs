# 兼容性改动速查：让 Vanilla OpenVLA HF Checkpoint 在 OFT 仓库跑起来

详细分析见 `oft_vanilla_eval_adaptation.md`，本文是简版速查。

---

## 根本问题

OFT 推理方式与 vanilla OpenVLA 完全不同：

| | OFT | Vanilla OpenVLA |
|---|---|---|
| 推理 | 追加 56 个 placeholder token，**一次 forward** 并行输出所有 action | 调用 `.generate()`，**逐 token autoregressive** 生成 7 个 action token |
| 训练 | action embedding 清零，并行预测 | 标准 causal LM |

直接用 OFT codebase 跑 vanilla checkpoint → SR = 0。

---

## 改动一览（共 5 处）

### 1. `experiments/robot/openvla_utils.py`

**改动 1a**：`action_head is None` 时改走 autoregressive `generate()`，不再调 OFT 的 `predict_action`。

```python
# 修改前
action, _, _, _ = vla.predict_action(**inputs, unnorm_key=cfg.unnorm_key, do_sample=False)

# 修改后（autoregressive 推理）
action_dim = vla.get_action_dim(cfg.unnorm_key)
generated_ids = vla.generate(**vanilla_inputs, max_new_tokens=action_dim, do_sample=False)
# ... 解码 token → 连续 action
```

**改动 1b**：修复 `actions_list` shape 错误（vanilla 返回 `[7]`，原代码误拆成 7 个标量）。

```python
# 修改后
if isinstance(action, np.ndarray) and action.ndim == 1:
    actions_list = [action]   # 单步 action，整体包成列表
else:
    actions_list = [action[i] for i in range(len(action))]
```

---

### 2. `prismatic/extern/hf/modeling_prismatic.py`

**改动 2a**：`generate()` 调用时 `labels=None`，原代码无条件调 `_process_action_masks(labels)` 会 crash，加 guard：

```python
if labels is not None:
    all_actions_mask = self._process_action_masks(labels)
else:
    all_actions_mask = torch.zeros(input_embeddings.shape[:2], dtype=torch.bool, device=input_embeddings.device)
```

**改动 2b**：新增 `zero_action_embeddings: bool = True` 参数，解耦训练模式：

```python
elif zero_action_embeddings:
    # OFT parallel 模式：清零 action embedding
    input_embeddings = input_embeddings * ~all_actions_mask
# else: autoregressive 模式，保留 embedding，标准 causal LM
```

---

### 3. `vla-scripts/finetune.py` 和 `finetune_substep.py`

forward 调用处传入 `zero_action_embeddings`：

```python
output = vla(
    ...
    zero_action_embeddings=(use_l1_regression or use_diffusion),
)
```

- `use_l1_regression=True` → 清零（OFT 原有行为不变）
- `use_l1_regression=False` → 不清零（真正的 autoregressive chunk 训练）

另：`finetune_substep.py` 增加 `action_head = None` 初始化，修复 `use_l1_regression=False` 时的 `UnboundLocalError`。

---

### 4. `auto_eval_nv40_openvla_pro.sh`

```bash
NUM_IMAGES_IN_INPUT=1   # vanilla OpenVLA 只用 1 张图，OFT 默认 2 会报错
```

所有 python 命令加 `--num_images_in_input $NUM_IMAGES_IN_INPUT`。

---

## 改动后推理路径

| 模型 | `use_l1_regression` | eval 脚本 | 推理方式 |
|---|---|---|---|
| Vanilla HF checkpoint | False | `auto_eval_nv40_openvla_pro.sh` | autoregressive `generate()` ✓ |
| OFT L1 fine-tuned | True | `auto_eval_nv40_pro.sh` | OFT parallel single-pass ✓ |
| 自训练 autoregressive chunk | False | `auto_eval_nv40_openvla_pro.sh` | autoregressive `generate()` ✓ |
