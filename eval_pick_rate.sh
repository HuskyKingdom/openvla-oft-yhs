#!/bin/bash
echo "Running Pick-Correct-Rate Evaluation ------------------------------"

# ============================================================
# 用户配置区：修改这里的变量即可
# ============================================================
PRETRAINED_CHECKPOINT="ckpt/ckpoints/saga_h100_49"
TASK_SUITE_NAME="libero_spatial"         # libero_spatial | libero_object | libero_goal | libero_10
PERTURBATION_MODE="task"                  # none | lan | task | swap | object
TASK_LABEL="pick_rate_${TASK_SUITE_NAME}_${PERTURBATION_MODE}"

EVALUATION_CONFIG_PATH="experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml"
UNNORM_KEY="${TASK_SUITE_NAME}"
NUM_TRIALS_PER_TASK=50

USE_L1_REGRESSION=False
USE_PROPRIO=False
USE_BDDL_LANGUAGE=True
NUM_IMAGES_IN_INPUT=1

# Pick 检测阈值（一般不需要改）
PROX_THRESH=0.06
GRIP_THRESH=0.030
LIFT_THRESH=0.015
# ============================================================

python experiments/robot/libero/eval_pick_rate.py \
  --pretrained_checkpoint $PRETRAINED_CHECKPOINT \
  --task_suite_name $TASK_SUITE_NAME \
  --perturbation_mode $PERTURBATION_MODE \
  --evaluation_config_path $EVALUATION_CONFIG_PATH \
  --unnorm_key $UNNORM_KEY \
  --num_trials_per_task $NUM_TRIALS_PER_TASK \
  --use_l1_regression $USE_L1_REGRESSION \
  --use_proprio $USE_PROPRIO \
  --use_bddl_language $USE_BDDL_LANGUAGE \
  --num_images_in_input $NUM_IMAGES_IN_INPUT \
  --prox_thresh $PROX_THRESH \
  --grip_thresh $GRIP_THRESH \
  --lift_thresh $LIFT_THRESH \
  --task_label $TASK_LABEL

echo "Done. Results in experiments/logs/${TASK_LABEL}.json"
