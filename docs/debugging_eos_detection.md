# EOS Detection 调试指南

## 问题现象
测试时日志显示：
```
[EOS DEBUG] has_eos=False, eos_position=None
[EOS] ✗ No EOS detected
```

## 诊断步骤

### 步骤 1：检查 EOS checkpoint 是否存在并被正确训练

运行诊断脚本：
```bash
python debug_eos_detection.py <your_checkpoint_path>
```

**期望输出**：
- ✓ 找到 EOS checkpoint 文件
- ✓ 权重统计显示 "Trained"（mean 或 std 不接近 0）
- ✓ 最后一层 bias 应该是一个较大的负数（如 -5 到 -10），表示模型学习到了大部分样本是 EOS=0

**如果失败**：
- ❌ 找不到 checkpoint → 训练时未启用 `use_eos_classification=True`
- ❌ 权重看起来像随机初始化 → EOS head 没有被训练或训练不足
- ❌ 最后一层 bias ≈ 0 → 模型没有学到有用的模式

### 步骤 2：运行带详细调试的推理

我已经在推理代码中添加了详细的调试输出。重新运行测试：

```bash
python experiments/robot/libero/run_libero_pro_eval_substep.py \
    --pretrained_checkpoint <your_checkpoint> \
    --task_suite_name libero_object \
    --use_substep_decomposition=True \
    --use_eos_detection=True \
    --eos_threshold=0.5 \
    --save_video=True
```

**检查新的调试输出**：
```
[EOS DEBUG] Loading checkpoint: ...
[EOS DEBUG] State dict keys: [...]
[EOS DEBUG]   model.0.weight: mean=X.XXXX, std=X.XXXX
[EOS DEBUG] After loading: ...
[EOS INFO] Config: hidden_dim=1024, dropout=0.1

[EOS PROBS] [0.0234 0.0189 0.0156 0.0145 0.0198 0.0176 0.0201 0.0189]
[EOS RANGE] min=0.0145, max=0.0234, mean=0.0186
[EOS THRESHOLD] 0.5
[EOS NOT DETECTED] ✗ All probs below threshold 0.5
```

### 步骤 3：分析 EOS probabilities

根据 `[EOS PROBS]` 输出判断：

#### 情况 A：概率都很低 (< 0.1)
```
[EOS PROBS] [0.0234 0.0189 0.0156 0.0145 0.0198 0.0176 0.0201 0.0189]
[EOS RANGE] min=0.0145, max=0.0234, mean=0.0186
```

**原因**：
1. 训练数据中 EOS=1 的样本太少（如 1:800 的极端不平衡）
2. 模型学习到了：几乎所有位置都是 EOS=0

**解决方案**：
- 降低 threshold：`--eos_threshold=0.02`（根据 mean 值调整）
- 或者重新训练，使用更高的 `eos_pos_weight`（如 100-200）

#### 情况 B：概率都接近 0.5
```
[EOS PROBS] [0.48 0.52 0.49 0.51 0.47 0.50 0.49 0.52]
[EOS RANGE] min=0.47, max=0.52, mean=0.50
```

**原因**：
- EOS head 没有被训练（权重仍然接近随机初始化）
- 或者训练配置有问题（如 `lambda_eos=0` 或梯度没有回传）

**解决方案**：
- 检查训练日志中的 `eos_loss` 和 `eos_accuracy`
- 确认训练时使用了 `use_eos_classification=True`
- 重新训练，确保 `lambda_eos > 0`（默认 1.0）

#### 情况 C：部分位置概率较高
```
[EOS PROBS] [0.02 0.15 0.78 0.03 0.05 0.02 0.03 0.04]
[EOS RANGE] min=0.02, max=0.78, mean=0.14
[EOS DETECTED] ✓ Position 2, prob=0.78
```

**原因**：
- ✓ 模型正常工作！
- 检测到了 substep 边界

### 步骤 4：检查训练配置

如果 EOS 概率异常，检查训练时的配置：

```bash
# 查看训练日志中的 EOS 配置
cat logs/<training_run>.log | grep -i "eos"
```

**必须确认**：
- ✓ `use_eos_classification=True`
- ✓ `use_substep_eos=True`（在 substep labels 中标注了 EOS）
- ✓ `lambda_eos=1.0`（或其他正值）
- ✓ `eos_use_focal_loss=True` 或 `eos_pos_weight=50.0`（处理不平衡）

### 步骤 5：检查训练过程

查看训练时的 EOS 指标：

```bash
# 在 WandB 或本地日志中查找
grep "eos_" <training_log_file>
```

**期望指标**：
- `eos_loss` 应该逐渐下降（从 0.7 降到 0.1-0.3）
- `eos_accuracy` 应该逐渐上升（到 90%+）
- `eos_recall` 对于极端不平衡可能较低（10-30%），这是正常的
- `eos_precision` 应该较高（70%+）

**如果 `eos_loss` 不下降**：
- 检查是否使用了 Focal Loss 或高 pos_weight
- 检查 `lambda_eos` 是否太小
- 检查梯度是否被正确回传

## 常见问题和解决方案

### 问题 1：训练时没有启用 EOS classification

**症状**：找不到 `eos_head--*.pt` checkpoint

**解决**：重新训练，确保：
```bash
python vla-scripts/finetune_substep.py \
    --use_eos_classification=True \
    --use_substep_eos=True \
    --eos_use_focal_loss=True \
    --eos_pos_weight=50.0 \
    ...
```

### 问题 2：EOS labels 没有正确生成

**症状**：训练日志显示 `eos_ratio=0.0` 或 `eos_no_labels=1.0`

**解决**：
1. 检查 `substep_labels_output.json` 中是否有 substep 信息
2. 确保 `SubstepRLDSDataset` 正确加载了 substep labels
3. 重新运行 `label_substeps.py` 生成 labels

### 问题 3：极端类别不平衡导致模型不学习

**症状**：`eos_ratio < 0.001`（如 1:800），训练后所有预测都是 0

**解决**：
- 使用 Focal Loss：`--eos_use_focal_loss=True`
- 或使用更高的 pos_weight：`--eos_pos_weight=100.0` 或更高
- 增加 `lambda_eos=2.0`（提高 EOS loss 的权重）
- 推理时降低 threshold：`--eos_threshold=0.02`

### 问题 4：配置不匹配

**症状**：加载 checkpoint 时出错或性能异常

**解决**：确保训练和推理时的配置一致：
- `eos_hidden_dim=1024`（默认）
- `eos_dropout=0.1`（默认）
- `ACTION_DIM` 和 `NUM_ACTIONS_CHUNK` 常量一致

## 快速检查清单

- [ ] Checkpoint 中存在 `eos_head--*.pt` 文件
- [ ] EOS head 权重不是随机初始化
- [ ] 最后一层 bias 是较大负数（如 -5）
- [ ] 训练时启用了 `use_eos_classification=True`
- [ ] Substep labels JSON 文件存在且包含 substep 信息
- [ ] 训练日志显示 `eos_loss` 下降和 `eos_accuracy` 上升
- [ ] 使用了适当的类别平衡策略（Focal Loss 或高 pos_weight）
- [ ] 推理时的配置与训练时一致

## 下一步

根据诊断结果：
1. **如果 EOS head 未训练**：重新训练模型
2. **如果概率太低**：降低 threshold 或重新训练with更高的 pos_weight
3. **如果正常工作**：享受 EOS-based substep switching 🎉

