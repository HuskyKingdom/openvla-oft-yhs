#!/bin/bash
# InfoBot Full Evaluation Script - Reference Format

cd /home/yuhang/Warehouse/Yuhangworkspace/openvla-oft-yhs

# Setup
source /home/yuhang/Warehouse/Yuhangworkspace/miniconda3/etc/profile.d/conda.sh
conda activate vla_pro
export PYTHONPATH=$PYTHONPATH:/home/yuhang/Warehouse/Yuhangworkspace/openvla-oft-yhs/experiments/robot/libero/LIBERO-PRO

# Config
CKPT="/home/yuhang/Warehouse/Yuhangworkspace/openvla-oft-yhs/ckpt/ckpoints/openvla-7b+libero_4_task_suites_no_noops+b8+lr-0.0002+lora-r32+infobot-cross_attn-beta0.1--image_aug--substep--infobot_v2_stable--95000_chkpt"
LABEL="infobot_95k"

# Common args
COMMON="--model_family openvla \
  --vla_path openvla/openvla-7b \
  --pretrained_checkpoint $CKPT \
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

echo "========================================"
echo "InfoBot Full Evaluation"
echo "Checkpoint: $CKPT"
echo "Start: $(date)"
echo "========================================"

# 1. LIBERO-Spatial
echo ""
echo "=== [1/4] LIBERO-Spatial ==="
python experiments/robot/libero/run_libero_pro_eval_javas.py \
  $COMMON --task_suite_name libero_spatial --run_id_note "${LABEL}_spatial"

# 2. LIBERO-Object
echo ""
echo "=== [2/4] LIBERO-Object ==="
python experiments/robot/libero/run_libero_pro_eval_javas.py \
  $COMMON --task_suite_name libero_object --run_id_note "${LABEL}_object"

# 3. LIBERO-Goal
echo ""
echo "=== [3/4] LIBERO-Goal ==="
python experiments/robot/libero/run_libero_pro_eval_javas.py \
  $COMMON --task_suite_name libero_goal --run_id_note "${LABEL}_goal"

# 4. LIBERO-10 (Long)
echo ""
echo "=== [4/4] LIBERO-Long ==="
python experiments/robot/libero/run_libero_pro_eval_javas.py \
  $COMMON --task_suite_name libero_10 --run_id_note "${LABEL}_10"

echo ""
echo "========================================"
echo "Evaluation Complete: $(date)"
echo "========================================"

# Summary
echo ""
echo "=== Results Summary ==="
for suite in spatial object goal 10; do
    LOG=$(ls -t experiments/logs/EVAL-libero_${suite}*${LABEL}_${suite}*.txt 2>/dev/null | head -1)
    if [ -f "$LOG" ]; then
        RATE=$(grep "Overall success rate" "$LOG" | tail -1 | awk '{print $NF}')
        echo "libero_${suite}: ${RATE:-N/A}"
    fi
done
