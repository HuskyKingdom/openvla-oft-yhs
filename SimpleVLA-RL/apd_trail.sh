export SFT_MODEL_PATH="/work1/chunyilee/yuhang/openvla-oft-yhs/landmarked_ckpoints/apd_discrete_200k"
export CKPT_PATH="./exp_out"
export APD_PLANS_PATH="/work1/chunyilee/yuhang/openvla-oft-yhs/APD_plans_scaled.json"
export CONTRASTIVE_REWARD_COEF=2
export DATASET_NAME="libero_spatial"
export EXPERIMENT_NAME="grpo-apd-liberospatial-run1"
export PROJECT_NAME="openvla-oft-rl"
export NUM_GPUS=8
export HIP_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
# Use autoregressive generation (for SFT models trained with use_l1_regression=False)
export USE_AUTOREGRESSIVE="False"

# Lower batch size so training triggers with fewer filtered rollouts.
# Current model has ~0% success rate, only ~6-12 prompts per epoch produce
# mixed results. batch_size=8 requires only 8*4=32 filtered rollouts to start.
export DATA_TRAIN_BATCH_SIZE=8
export ACTOR_PPO_MINI_BATCH_SIZE=32   # must be <= DATA_TRAIN_BATCH_SIZE * n_samples (8*4=32)
export ACTOR_TRAJ_MINI_BATCH_SIZE=8   # must be <= DATA_TRAIN_BATCH_SIZE

sbatch examples/run_openvla_oft_substep_rl_libero.sh
