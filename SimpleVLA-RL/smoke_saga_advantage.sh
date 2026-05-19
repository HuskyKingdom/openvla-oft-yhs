#!/usr/bin/env bash
# smoke_saga_advantage.sh — Quick SAGA training smoke test for advantage logging.
#
# Runs a short SAGA RL training (20 steps, validate every 4, save every 10),
# which writes saga_advantage_log.jsonl to the experiment output directory.
# After training, calls tools/plot_saga_advantages.py to produce a PNG.
#
# Usage (from SimpleVLA-RL root):
#   export SFT_MODEL_PATH="/path/to/apd_discrete_ckpt"
#   export APD_PLANS_PATH="/path/to/APD_plans_scaled.json"
#   bash smoke_saga_advantage.sh              # submits sbatch on AMD
#
# Override experiment name to avoid overwriting a full run:
#   EXPERIMENT_NAME=saga-smoke bash smoke_saga_advantage.sh

# ── Required paths ──────────────────────────────────────────────────────────
export SFT_MODEL_PATH="${SFT_MODEL_PATH:-/work1/chunyilee/yuhang/openvla-oft-yhs/landmarked_ckpoints/oft_plus_discrete}"
export CKPT_PATH="${CKPT_PATH:-./exp_out}"
export APD_PLANS_PATH="${APD_PLANS_PATH:-/work1/chunyilee/yuhang/openvla-oft-yhs/APD_plans_scaled.json}"
export DATASET_NAME="${DATASET_NAME:-libero_4_task_suites}"

# ── Experiment identity ──────────────────────────────────────────────────────
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-saga-smoke-adv-$(date +%m%d-%H%M)}"
export PROJECT_NAME="${PROJECT_NAME:-openvla-oft-rl}"

# ── Hardware ─────────────────────────────────────────────────────────────────
export NUM_GPUS="${NUM_GPUS:-8}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# ── Smoke test: very short run ────────────────────────────────────────────────
# 20 total steps — enough to see whether advantages are non-trivial.
export TRAINER_TOTAL_EPOCHS="20"
export TRAINER_TEST_FREQ="4"         # validate every 4 steps
export TRAINER_SAVE_FREQ="10"        # checkpoint at step 10 and 20
export TRAINER_INITIAL_GLOBAL_STEPS="${TRAINER_INITIAL_GLOBAL_STEPS:-0}"

# ── SAGA config ───────────────────────────────────────────────────────────────
export ADV_ESTIMATOR="saga"
export USE_AUTOREGRESSIVE="False"

export SWAP_OBJECTS="True"
export SWAP_DISTANCE_START="0.08"
export SWAP_DISTANCE_END="0.20"      # smaller max for smoke (fewer rerolls)
export SWAP_CURRICULUM_STEPS="10"    # reach max quickly in 10 steps

export DIST_REWARD_COEF="0.0"
export DIST_REWARD_SIGMA="0.05"

# ── Accuracy filter — DISABLED for smoke test ────────────────────────────────
# The model starts with ~0% success rate on libero_4_task_suites.
# With filter_accuracy=True and lower_bound=0.1, all rollout groups are
# rejected (acc=0 < 0.1) so valid_batch never fills and no training step runs.
# Disabling the filter lets cold-start rollouts through; GRPO normalises within
# each group so all-fail groups produce 0-advantage (no false gradient).
export FILTER_ACCURACY="False"
export ACCURACY_LOWER_BOUND="0.0"
export ACCURACY_UPPER_BOUND="1.0"

# ── Batch sizes (smaller for smoke speed) ────────────────────────────────────
export DATA_TRAIN_BATCH_SIZE="8"
export ACTOR_PPO_MINI_BATCH_SIZE="32"
export ACTOR_TRAJ_MINI_BATCH_SIZE="8"

# ── Submit training ───────────────────────────────────────────────────────────
echo "=== SAGA Smoke Test: advantage logging ==="
echo "  Experiment : $EXPERIMENT_NAME"
echo "  Steps      : $TRAINER_TOTAL_EPOCHS"
echo "  SFT model  : $SFT_MODEL_PATH"
echo "  APD plans  : $APD_PLANS_PATH"
echo "  Output     : $CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME"
echo ""

JOB_ID=$(sbatch --parsable examples/run_saga.sh)
echo "Submitted SLURM job $JOB_ID"
echo ""
echo "After the job completes, visualize advantages with:"
echo ""
echo "  python tools/plot_saga_advantages.py \\"
echo "    $CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME/saga_advantage_log.jsonl \\"
echo "    -o $CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME/saga_adv.png"
echo ""
echo "Or run the post-plot step directly (blocks until job done):"
echo "  srun --dependency=afterok:$JOB_ID --ntasks=1 \\"
echo "    python tools/plot_saga_advantages.py \\"
echo "    $CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME/saga_advantage_log.jsonl \\"
echo "    -o $CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME/saga_adv.png"
