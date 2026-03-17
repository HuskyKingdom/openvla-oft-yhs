# Substep-Aware RL Post-Training 项目总结

## 一、项目背景与两个代码库

### 1.1 OpenVLA-OFT (openvla-oft-yhs)

OpenVLA-OFT 是一个基于 OpenVLA-7B 的视觉-语言-动作模型（VLA），使用 LoRA 微调。

**核心架构：**
- Vision backbone → Projector → LLM (Llama 2) → Action Head
- Action Head: L1RegressionActionHead（连续动作预测）
- 可选: DiffusionActionHead, EOSClassificationHead
- 动作空间: 7D (3 平移 + 3 旋转 + 1 夹爪), chunk size = 8

**Substep 训练 (`finetune_substep.py`)：**
- 将 per-episode 任务指令替换为 per-timestep substep 指令
- substep 指令来自 `APD_plans.json`（LLM 预生成的任务分解）
- 通过 `substep_labels_output.json` 在训练数据中标注每个 timestep 属于哪个 substep
- 可选训练 EOS 分类头，用于检测 substep 边界
- 损失: `total_loss = action_loss + lambda_eos * eos_loss`

**推理阶段 (`run_libero_pro_eval_substep.py`)：**
- `SubstepManager` 管理 substep 状态
- 两种 substep 切换方式：EOS head 预测 / SigCLIP 视觉相似度
- 实时切换指令，用当前 substep subgoal 作为 VLA prompt

**指令忽视问题 (`Javas/prev_research.md`)：**
- 模型过度依赖视觉，忽视语言指令
- 空指令仍可完成任务（Long: 85.8%, Spatial: 82.6%）
- 根因：VLM 直接用于 VLA 是 naive 的，模型学到了视觉 shortcut

### 1.2 SimpleVLA-RL

SimpleVLA-RL 是一个基于 verl 框架的 VLA 强化学习训练系统，支持 OpenVLA 和 OpenVLA-OFT。

**训练流程：**
```
采样 rollout → 环境交互 → 获取 reward → 计算 advantage → PPO 策略更新
```

**Reward 系统 (`verl/trainer/main_ppo.py`)：**
- `RobRewardManager` 类管理 reward 计算
- `verify()`: 从 rollout 数据提取 `complete` (环境任务成功/失败) → 二值 score (0/1)
- `__call__()`: 将 reward 放在 trajectory 最后一个有效 step 的 token 位置
- 奖励公式: `reward = reward_coef(5) × gt_scores`

**Rollout (`verl/workers/rollout/rob_rollout.py`)：**
- `RobHFRollout` 类管理 rollout
- LIBERO 使用 multiprocessing (Process + Queue) 并行 env 交互
- 每步: VLA 生成 8 步动作 → 发送给 env worker → 收集结果
- Prompt 格式: `"In: What action should the robot take to {instruction}?\nOut:"`

**Advantage 估计 (`verl/trainer/ppo/core_algos.py`)：**
- 支持 GRPO, GAE, RLOO, REINFORCE++, ReMax
- LIBERO 默认使用 GRPO: 同 prompt 多个 trajectory 做组内归一化，无需 Critic

**Reward → Policy Update 完整链条：**
```
env.step() → complete (0/1)
  → verify() → acc score
  → __call__() → gt_scores (放在 finish_step 位置)
  → token_level_scores = 5 × gt_scores
  → apply_kl_penalty (可选)
  → compute_grpo_outcome_advantage → 组内归一化 advantage
  → compute_policy_loss → PPO clip loss → 梯度更新
```

---

## 二、问题诊断

### 2.1 现有 RL 的局限

SimpleVLA-RL 的 reward 是**任务级别二值稀疏信号**，只看任务是否成功，**不感知指令**。
一个忽视指令但视觉过拟合的模型仍能获得高 reward。

### 2.2 Substep SFT 的局限

Substep SFT 通过细粒度指令迫使模型关注语言，但：
- 仍是模仿学习，无负反馈
- 模型从未因"忽视指令"被惩罚
- 泛化受限于 APD 标注的固定 substep

### 2.3 RL Post-Training 的价值

RL 可以提供 SFT 无法提供的信号：
- 负反馈: 忽视指令时给予惩罚
- 探索: 发现 expert 之外的策略
- 对抗: 主动测试指令敏感性

---

## 三、新增工作: Instruction-Grounded RL (IG-RL)

### 3.1 总体设计

在 SimpleVLA-RL 框架上改造两个部分:
1. **Rollout**: 使用 substep 指令 + SigCLIP 动态切换
2. **Reward**: 新增 R_contrastive 直接奖惩指令遵循行为

Reward 公式:
```
R_total = w1 × R_task + w2 × R_contrastive
         (w1=5)        (w2 可调)
```

Advantage 估计: 保持 GRPO 不变。

### 3.2 新增文件

#### `verl/utils/substep_reward.py` （新建）

三个核心类:

| 类 | 功能 |
|---|---|
| `APDPlanManager` | 加载 `APD_plans.json`，按 suite+instruction 查 plan，采样错误 substep |
| `SigCLIPRewardModel` | 加载 SigLIP/open_clip，编码文本/图像，计算相似度 |
| `SubstepTracker` | 单 episode substep 追踪：预计算 text embedding，判断完成并推进 |

Suite 映射:
- `libero_spatial` → `spatial`
- `libero_object` → `object`
- `libero_goal` → `goal`
- `libero_10` → `long`

### 3.3 修改文件

#### `verl/workers/rollout/rob_rollout.py`

**`__init__` 新增初始化:**
- 当 `use_substep_rl=True` 时加载 APDPlanManager 和 SigCLIPRewardModel
- 默认 False，不影响原有功能

**`_generate_minibatch_libero` 修改:**

原始流程:
```
每步: 用 episode 级 task_description → VLA → actions → env
```

新增 substep 流程 (仅训练时, `use_substep_this_batch=True`):
```
初始化: 为每个 episode 创建 SubstepTracker (从 APD plan)

每步:
  1. 用当前 substep subgoal 替代 task_description 作为 prompt
  2. VLA → actions (正确指令)
  3. 发送 actions 到 env (非阻塞)
  4. 每 K=16 env steps: 对比前向推理
     - 从同 suite 其他任务采样错误 substep 指令
     - 用错误指令 + 相同 observation → VLA → wrong_actions
     - 计算 contrastive_score = min(||a - a_wrong|| / sqrt(N), 1.0)
  5. 收集 env 结果
  6. 用 SigCLIP 判断 substep 完成 (相似度 > threshold → 推进到下一 substep)
```

**`_prepare_output_batch` 修改:**
- 输出 `contrastive_score` (各步的平均归一化动作差异)

#### `verl/trainer/main_ppo.py`

**`RobRewardManager.__call__` 新增:**
```python
contrastive_coef = getattr(self.config.verifier, 'contrastive_reward_coef', 0)
if 'contrastive_score' in data.batch and contrastive_coef != 0:
    # 将 contrastive_score 放在 trajectory 结尾位置
    # reward_tensor += contrastive_coef × contrastive_reward
```

新增 WandB metrics:
- `contrastive`: contrastive reward 的 token-level 均值
- `contrastive_raw_mean`: 原始 contrastive score 均值

#### `verl/trainer/config/ppo_trainer.yaml`

新增配置项:
```yaml
rollout:
  use_substep_rl: false              # 总开关
  apd_plans_path: null               # APD_plans.json 路径
  sigclip_model_path: "timm/ViT-B-16-SigLIP-256"
  substep_completion_threshold: 0.25 # SigCLIP 切换阈值
  contrastive_sample_interval: 16    # 每 16 env steps 做一次对比

verifier:
  contrastive_reward_coef: 0         # R_contrastive 权重
```

#### `examples/run_openvla_oft_substep_rl_libero.sh` （新建）

基于原始 run script，新增:
- Substep RL 全部参数
- `CONTRASTIVE_REWARD_COEF=2` 环境变量控制
- `APD_PLANS_PATH` 环境变量
- KL coef = 0.01 (可选，可设为 0)

### 3.4 向后兼容性

所有修改通过以下默认值保证不影响原有功能:
- `use_substep_rl: false` → 不加载 APD/SigCLIP
- `contrastive_reward_coef: 0` → 不计算对比 reward
- Robotwin 路径完全不受影响

### 3.5 计算开销

| 组件 | 开销 |
|------|------|
| SigCLIP substep 切换检查 | ~5-10ms/帧，每步一次 |
| 对比前向推理 | 每 16 env steps 一次额外 VLA 前向，约 50% rollout 推理开销 |
| APD 加载 | 一次性 JSON 加载 |

---

## 四、实施步骤

1. **Phase 0 (基线验证)**: `use_substep_rl=True, contrastive_reward_coef=0` → 只用 substep 指令 rollout + R_task
2. **Phase 1 (对比 reward)**: `contrastive_reward_coef=2` → 加入 R_contrastive
3. **Phase 2 (评估)**: LIBERO SR + 指令扰动测试 (空指令、错误指令、同义替换)

---

## 五、文件路径速查

| 文件 | 路径 | 状态 |
|------|------|------|
| APD 计划 | `openvla-oft-yhs/APD_plans.json` | 已有 |
| Substep 标签 | `openvla-oft-yhs/substep_labels_output.json` | 已有 |
| Substep SFT | `openvla-oft-yhs/vla-scripts/finetune_substep.py` | 已有 |
| Substep 推理 | `openvla-oft-yhs/experiments/robot/libero/run_libero_pro_eval_substep.py` | 已有 |
| SubstepManager | `openvla-oft-yhs/experiments/robot/libero/substep_manager.py` | 已有 |
| **Substep reward 工具** | `SimpleVLA-RL/verl/utils/substep_reward.py` | **新建** |
| **Rollout (改)** | `SimpleVLA-RL/verl/workers/rollout/rob_rollout.py` | **修改** |
| **Reward (改)** | `SimpleVLA-RL/verl/trainer/main_ppo.py` | **修改** |
| **Config (改)** | `SimpleVLA-RL/verl/trainer/config/ppo_trainer.yaml` | **修改** |
| **Run script** | `SimpleVLA-RL/examples/run_openvla_oft_substep_rl_libero.sh` | **新建** |
