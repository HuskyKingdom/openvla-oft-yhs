#!/bin/bash
echo "Running Pick-Correct-Rate Evaluation (all 4 task suites) ------------------------------"

# ============================================================
# 用户配置区：修改这里的变量即可
# ============================================================
PRETRAINED_CHECKPOINT="ckpt/ckpoints/saga_h100_49"
PERTURBATION_MODE="task"                  # none | lan | task | swap | object

EVALUATION_CONFIG_PATH="experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml"
NUM_TRIALS=50

USE_L1_REGRESSION=False
USE_PROPRIO=False
USE_BDDL_LANGUAGE=True
NUM_IMAGES_IN_INPUT=1

PROX_THRESH=0.06
GRIP_THRESH=0.030
LIFT_THRESH=0.015
# ============================================================

for TASK_SUITE_NAME in libero_object libero_goal libero_10; do
  TASK_LABEL="pick_rate_${TASK_SUITE_NAME}_${PERTURBATION_MODE}"
  echo ""
  echo ">>> Task suite: $TASK_SUITE_NAME"

  python experiments/robot/libero/eval_pick_rate.py \
    --pretrained_checkpoint $PRETRAINED_CHECKPOINT \
    --task_suite_name $TASK_SUITE_NAME \
    --perturbation_mode $PERTURBATION_MODE \
    --evaluation_config_path $EVALUATION_CONFIG_PATH \
    --unnorm_key $TASK_SUITE_NAME \
    --num_trials_per_task $NUM_TRIALS \
    --use_l1_regression $USE_L1_REGRESSION \
    --use_proprio $USE_PROPRIO \
    --use_bddl_language $USE_BDDL_LANGUAGE \
    --num_images_in_input $NUM_IMAGES_IN_INPUT \
    --prox_thresh $PROX_THRESH \
    --grip_thresh $GRIP_THRESH \
    --lift_thresh $LIFT_THRESH \
    --task_label $TASK_LABEL

  echo "Done: experiments/logs/${TASK_LABEL}.json"
done

echo ""
echo "All 4 suites finished."
