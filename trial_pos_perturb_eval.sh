#!/bin/bash
# Quick trial: compare evaluations WITH vs WITHOUT position perturbation (use_swap).
# Runs libero_10 only × 2 conditions, 1 trial per task (minimal smoke test).

echo "=== Trial: Pos Perturbation Comparison ==================================="
FILE_PATH="experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml"
PRETRAINED_CHECKPOINT="ckpt/SimpleVLA-RL/exp_out/openvla-oft-rl/grpo-liberospatial-run1/actor/global_step_24"
TASK_LABEL_PREFIX="trial_pos_perturb_rl_step24"
USE_EOS_DETECTION=False
EVAL_SCRIPT="experiments/robot/libero/run_libero_pro_eval_substep.py"
USE_PROPRIO=False
USE_L1_REGRESSION=False
USE_SUBSTEP_DECOMPOSITION=False
USE_BDDL_LANGUAGE=True
AUTO_REGRESSION=False
NUM_IMAGES_IN_INPUT=1
SUBSTEP_COMPLETION_THRESHOLD=0.03

NUM_TRIALS=1     # 1 episode
SUITE=libero_10
TASK_ID=0        # task index to evaluate (0-based); change to select a different task
SAVE_VIDEO=True  # set to True to save videos

# ---------------------------------------------------------------------------
# Round 1: WITHOUT pos perturbation  (use_language baseline)
# ---------------------------------------------------------------------------
echo ""
echo "--- Round 1: WITHOUT pos perturbation (use_language baseline) ------------"
bash experiments/robot/libero/LIBERO-PRO/reset_eval_config.sh $FILE_PATH
sed -i 's/use_language: false/use_language: true/' $FILE_PATH

LOG_NOPERTURB=$(python $EVAL_SCRIPT \
  --pretrained_checkpoint $PRETRAINED_CHECKPOINT \
  --substep_completion_threshold $SUBSTEP_COMPLETION_THRESHOLD \
  --task_suite_name $SUITE \
  --e_decoding False --save_video $SAVE_VIDEO \
  --use_substep_decomposition $USE_SUBSTEP_DECOMPOSITION \
  --num_trials_per_task $NUM_TRIALS \
  --evaluation_config_path $FILE_PATH \
  --unnorm_key $SUITE \
  --task_label ${TASK_LABEL_PREFIX}_${SUITE}_noperturb_lan \
  --use_eos_detection $USE_EOS_DETECTION \
  --use_proprio $USE_PROPRIO \
  --use_l1_regression $USE_L1_REGRESSION \
  --use_bddl_language $USE_BDDL_LANGUAGE \
  --auto_regression $AUTO_REGRESSION \
  --num_images_in_input $NUM_IMAGES_IN_INPUT \
  --single_task_id $TASK_ID 2>&1)
echo "$LOG_NOPERTURB"

# ---------------------------------------------------------------------------
# Round 2: WITH pos perturbation  (use_environment)
# ---------------------------------------------------------------------------
echo ""
echo "--- Round 2: WITH pos perturbation (use_swap) ----------------------------"
bash experiments/robot/libero/LIBERO-PRO/reset_eval_config.sh $FILE_PATH
sed -i 's/use_swap: false/use_swap: true/' $FILE_PATH

LOG_PERTURB=$(python $EVAL_SCRIPT \
  --pretrained_checkpoint $PRETRAINED_CHECKPOINT \
  --substep_completion_threshold $SUBSTEP_COMPLETION_THRESHOLD \
  --task_suite_name $SUITE \
  --e_decoding False --save_video $SAVE_VIDEO \
  --use_substep_decomposition $USE_SUBSTEP_DECOMPOSITION \
  --num_trials_per_task $NUM_TRIALS \
  --evaluation_config_path $FILE_PATH \
  --unnorm_key $SUITE \
  --task_label ${TASK_LABEL_PREFIX}_${SUITE}_posperturb_swap \
  --use_eos_detection $USE_EOS_DETECTION \
  --use_proprio $USE_PROPRIO \
  --use_l1_regression $USE_L1_REGRESSION \
  --use_bddl_language False \
  --auto_regression $AUTO_REGRESSION \
  --num_images_in_input $NUM_IMAGES_IN_INPUT \
  --single_task_id $TASK_ID 2>&1)
echo "$LOG_PERTURB"

# Reset config flags at the end
bash experiments/robot/libero/LIBERO-PRO/reset_eval_config.sh $FILE_PATH

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "=========================================================================="
echo "=== SUMMARY =============================================================="
echo "=========================================================================="
echo ""
echo "[WITHOUT pos perturbation] last result lines:"
echo "$LOG_NOPERTURB" | grep -E "Success|success|SR|score|result" | tail -5
echo ""
echo "[WITH pos perturbation]    last result lines:"
echo "$LOG_PERTURB"   | grep -E "Success|success|SR|score|result" | tail -5
echo ""
echo "=== Trial complete. Check experiments/logs/ for full logs. ==============="
