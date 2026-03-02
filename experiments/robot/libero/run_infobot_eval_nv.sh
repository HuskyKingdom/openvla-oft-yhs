#!/bin/bash
# InfoBot Evaluation Script for NV Server

# Source bash profile to get aliases and conda
source ~/.bashrc

# Work function
work() {
    cd /home/yuhang/Warehouse/Yuhangworkspace/openvla-oft-yhs
    source /home/yuhang/anaconda3/bin/activate vla
}

work
conda activate vla_pro
export PYTHONPATH=$PYTHONPATH:/home/yuhang/Warehouse/Yuhangworkspace/openvla-oft-yhs/experiments/robot/libero/LIBERO-PRO

# Run InfoBot evaluation on LIBERO-Spatial
echo "Starting InfoBot evaluation..."

python experiments/robot/libero/run_libero_pro_eval_javas.py \
  --model_family openvla \
  --vla_path openvla/openvla-7b \
  --pretrained_checkpoint /home/yuhang/Warehouse/Yuhangworkspace/openvla-oft-yhs/ckpt/ckpoints/openvla-7b+libero_4_task_suites_no_noops+b8+lr-0.0002+lora-r32+infobot-cross_attn-beta0.1--image_aug--substep--infobot_v2_stable--95000_chkpt \
  --use_infobot True \
  --infobot_bottleneck_type cross_attn \
  --infobot_bottleneck_dim 256 \
  --infobot_num_tokens 8 \
  --use_l1_regression True \
  --num_images_in_input 2 \
  --use_proprio True \
  --center_crop True \
  --task_suite_name libero_spatial \
  --num_trials_per_task 20 \
  --seed 7 \
  --run_id_note "infobot_95k_test"

echo "Evaluation complete!"
