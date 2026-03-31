#!/bin/bash
#SBATCH --job-name=openvla_so101
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=polar4,polar3,polar,batch_block1,grizzly,batch_block2,batch_block3
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=4:00:00
#SBATCH --output=logs/so101_training_%j.out
#SBATCH --error=logs/so101_training_%j.err

# ========================================
# ORD Cluster Training Script for SO101 Poker Yellow Task
# ========================================

# Environment setup
export CACHE_DIR="/lustre/fsw/portfolios/edgeai/users/chrislin/cache"
export WORKSPACE="/lustre/fsw/portfolios/edgeai/users/chrislin/projects/openvla-oft-yhs"
export DATA_ROOT="/lustre/fsw/portfolios/edgeai/users/chrislin/datasets/so101_rlds"
export CHECKPOINT_DIR="/lustre/fsw/portfolios/edgeai/users/chrislin/checkpoints/openvla-so101"
export SUBSTEP_LABELS="/lustre/fsw/portfolios/edgeai/users/chrislin/projects/openvla-oft-yhs/substep_labels_so101.json"

# WandB configuration (update with your credentials)
export WANDB_API_KEY="e66e164e1c3b7f8c38fcea72427dafb0f4b35b80"  # TODO: Update if needed
export WANDB_ENTITY="crlc112358"  # TODO: Update to your WandB entity
export WANDB_PROJECT="openvla-so101-poker"

# Docker image
DOCKER_IMAGE="christianlin0420/openvla-apd:latest"

echo "========================================="
echo "Starting SO101 Poker Yellow Task Training"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: 8"
echo "Data: $DATA_ROOT"
echo "Checkpoints: $CHECKPOINT_DIR"
echo ""

# Create checkpoint and log directories
mkdir -p $CHECKPOINT_DIR
mkdir -p logs

# -----------------------------------------------------------------------
# To resume training from a checkpoint, set:
#   RESUME_CHECKPOINT=<path to checkpoint dir, e.g. .../run_id--50000_chkpt>
#   RESUME_STEP=50000
# and add to the torchrun command below:
#   --resume True --resume_step ${RESUME_STEP} --vla_path ${RESUME_CHECKPOINT}
# The checkpoint must contain optimizer_state.pt, scheduler_state.pt, and
# wandb_run_id.txt (saved automatically since this feature was added).
# -----------------------------------------------------------------------

# Launch training
# PYTHONPATH=/workspace set via --container-env so `from experiments.robot...` resolves
srun \
    --container-image="$DOCKER_IMAGE" \
    --container-mounts=/lustre:/lustre,${WORKSPACE}:/workspace \
    --container-env=TORCH_HOME=${CACHE_DIR}/torch,HF_HOME=${CACHE_DIR}/huggingface,PYTHONNOUSERSITE=1,PYTHONPATH=/workspace,WANDB_API_KEY=${WANDB_API_KEY},WANDB_ENTITY=${WANDB_ENTITY},WANDB_PROJECT=${WANDB_PROJECT} \
    bash -c "\
        if [ ! -d ${DATA_ROOT}/so101_poker_yellow/1.0.0 ]; then \
            echo '=== Building TFDS dataset (one-time) ===' && \
            python /workspace/scripts/build_so101_tfds.py \
                --tfrecord_path ${DATA_ROOT}/so101_poker_yellow.tfrecord \
                --output_dir ${DATA_ROOT}; \
        fi && \
        torchrun \
            --standalone \
            --nnodes 1 \
            --nproc-per-node 8 \
            /workspace/vla-scripts/finetune_substep.py \
                --vla_path openvla/openvla-7b \
                --data_root_dir ${DATA_ROOT} \
                --dataset_name so101_poker_yellow \
                --substep_labels_path ${SUBSTEP_LABELS} \
                --run_root_dir ${CHECKPOINT_DIR} \
                --use_l1_regression True \
                --use_lora True \
                --lora_rank 32 \
                --batch_size 8 \
                --learning_rate 5e-4 \
                --lr_warmup_steps 0 \
                --num_steps_before_decay 100000 \
                --max_steps 150005 \
                --save_freq 50000 \
                --save_latest_checkpoint_only False \
                --image_aug True \
                --num_images_in_input 2 \
                --use_proprio True \
                --gradient_checkpointing False \
                --wandb_entity ${WANDB_ENTITY} \
                --wandb_project ${WANDB_PROJECT} \
                --run_id_note so101_poker_yellow_apd_substep"

echo ""
echo "========================================="
echo "Training Completed"
echo "========================================="
echo "Checkpoints saved to: $CHECKPOINT_DIR"
echo "Logs saved to: logs/so101_training_${SLURM_JOB_ID}.out"
