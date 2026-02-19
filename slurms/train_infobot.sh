#!/bin/bash
#SBATCH --job-name=infobot_vla
#SBATCH --time=12:00:00
#SBATCH --partition=mi3508xl
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task 128

cd /work1/chunyilee/yuhang/openvla-oft-yhs
export WANDB_API_KEY=0bdbd99b1136358467ed2d03e9a6ba5a5b2a11a8
export HF_HOME=/work1/chunyilee/yuhang/

# Activate conda environment
source /work1/chunyilee/yuhang/miniconda3/bin/activate vla

# InfoBot-VLA Training
# Information Bottleneck Constrained VLA for addressing H(L|V) ≈ 0 problem

FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE" torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/finetune_infobot.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /work1/chunyilee/yuhang/modified_libero_rlds \
  --dataset_name libero_4_task_suites_no_noops \
  --substep_labels_path /work1/chunyilee/yuhang/openvla-oft-yhs/substep_labels_output.json \
  --run_root_dir /work1/chunyilee/yuhang/openvla-oft-yhs/ckpoints \
  --use_infobot True \
  --bottleneck_type cross_attn \
  --bottleneck_dim 256 \
  --num_bottleneck_tokens 8 \
  --use_l1_regression True \
  --use_lora True \
  --lora_rank 32 \
  --batch_size 8 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 100000 \
  --max_steps 200000 \
  --save_freq 10000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --num_images_in_input 2 \
  --use_proprio True \
  --wandb_entity "yhscode-university-of-liverpool" \
  --wandb_project "yhscode-university-of-liverpool" \
  --run_id_note infobot_base_no_mi
