#!/bin/bash
# InfoBot Comprehensive Evaluation Script for NV Server
# Runs all 4 LIBERO task suites and logs results

cd /home/yuhang/Warehouse/Yuhangworkspace/openvla-oft-yhs

# Activate environment
source /home/yuhang/Warehouse/Yuhangworkspace/miniconda3/etc/profile.d/conda.sh
conda activate vla_pro
export PYTHONPATH=$PYTHONPATH:/home/yuhang/Warehouse/Yuhangworkspace/openvla-oft-yhs/experiments/robot/libero/LIBERO-PRO

# Config
CKPT_DIR="/home/yuhang/Warehouse/Yuhangworkspace/openvla-oft-yhs/ckpt/ckpoints/openvla-7b+libero_4_task_suites_no_noops+b8+lr-0.0002+lora-r32+infobot-cross_attn-beta0.1--image_aug--substep--infobot_v2_stable--95000_chkpt"
RUN_LABEL="infobot_95k_full_eval"
LOG_FILE="/tmp/infobot_full_eval_$(date +%Y%m%d_%H%M%S).log"
RESULTS_FILE="/tmp/infobot_results_$(date +%Y%m%d_%H%M%S).txt"

echo "========================================" | tee -a $RESULTS_FILE
echo "InfoBot Full Evaluation - $(date)" | tee -a $RESULTS_FILE
echo "Checkpoint: $CKPT_DIR" | tee -a $RESULTS_FILE
echo "========================================" | tee -a $RESULTS_FILE

# Common args
COMMON_ARGS="--model_family openvla \
  --vla_path openvla/openvla-7b \
  --pretrained_checkpoint $CKPT_DIR \
  --use_infobot True \
  --infobot_bottleneck_type cross_attn \
  --infobot_bottleneck_dim 256 \
  --infobot_num_tokens 8 \
  --use_l1_regression True \
  --num_images_in_input 2 \
  --center_crop True \
  --use_proprio False \
  --num_trials_per_task 20 \
  --seed 7"

# Task 1: LIBERO-Spatial
echo "" | tee -a $RESULTS_FILE
echo "=== Task 1: LIBERO-Spatial ===" | tee -a $RESULTS_FILE
echo "Start: $(date)" | tee -a $RESULTS_FILE
python experiments/robot/libero/run_libero_pro_eval_javas.py \
  $COMMON_ARGS \
  --task_suite_name libero_spatial \
  --run_id_note "${RUN_LABEL}_spatial" 2>&1 | tee -a $LOG_FILE
echo "End: $(date)" | tee -a $RESULTS_FILE

# Extract result
echo "Results:" | tee -a $RESULTS_FILE
grep -E "Overall success rate|Final results" $LOG_FILE | tail -3 | tee -a $RESULTS_FILE

# Task 2: LIBERO-Object
echo "" | tee -a $RESULTS_FILE
echo "=== Task 2: LIBERO-Object ===" | tee -a $RESULTS_FILE
echo "Start: $(date)" | tee -a $RESULTS_FILE
python experiments/robot/libero/run_libero_pro_eval_javas.py \
  $COMMON_ARGS \
  --task_suite_name libero_object \
  --run_id_note "${RUN_LABEL}_object" 2>&1 | tee -a $LOG_FILE
echo "End: $(date)" | tee -a $RESULTS_FILE

echo "Results:" | tee -a $RESULTS_FILE
grep -E "Overall success rate|Final results" $LOG_FILE | tail -3 | tee -a $RESULTS_FILE

# Task 3: LIBERO-Goal
echo "" | tee -a $RESULTS_FILE
echo "=== Task 3: LIBERO-Goal ===" | tee -a $RESULTS_FILE
echo "Start: $(date)" | tee -a $RESULTS_FILE
python experiments/robot/libero/run_libero_pro_eval_javas.py \
  $COMMON_ARGS \
  --task_suite_name libero_goal \
  --run_id_note "${RUN_LABEL}_goal" 2>&1 | tee -a $LOG_FILE
echo "End: $(date)" | tee -a $RESULTS_FILE

echo "Results:" | tee -a $RESULTS_FILE
grep -E "Overall success rate|Final results" $LOG_FILE | tail -3 | tee -a $RESULTS_FILE

# Task 4: LIBERO-Long (10 task)
echo "" | tee -a $RESULTS_FILE
echo "=== Task 4: LIBERO-Long ===" | tee -a $RESULTS_FILE
echo "Start: $(date)" | tee -a $RESULTS_FILE
python experiments/robot/libero/run_libero_pro_eval_javas.py \
  $COMMON_ARGS \
  --task_suite_name libero_10 \
  --run_id_note "${RUN_LABEL}_10" 2>&1 | tee -a $LOG_FILE
echo "End: $(date)" | tee -a $RESULTS_FILE

echo "Results:" | tee -a $RESULTS_FILE
grep -E "Overall success rate|Final results" $LOG_FILE | tail -3 | tee -a $RESULTS_FILE

# Summary
echo "" | tee -a $RESULTS_FILE
echo "========================================" | tee -a $RESULTS_FILE
echo "Evaluation Complete - $(date)" | tee -a $RESULTS_FILE
echo "Log file: $LOG_FILE" | tee -a $RESULTS_FILE
echo "Results file: $RESULTS_FILE" | tee -a $RESULTS_FILE
echo "========================================" | tee -a $RESULTS_FILE

# Display summary
echo ""
echo "=== SUMMARY ==="
grep -E "Task [0-9]|Overall success rate" $RESULTS_FILE
