export SFT_MODEL_PATH="/work1/chunyilee/yuhang/openvla-oft-yhs/landmarked_ckpoints/simplevla-sft-spatial-all"
export CKPT_PATH="./exp_out"          
export APD_PLANS_PATH="/work1/chunyilee/yuhang/openvla-oft-yhs/APD_plans_scaled.json"
export CONTRASTIVE_REWARD_COEF=2
export DATASET_NAME="libero_spatial"        
export EXPERIMENT_NAME="grpo-apd-liberospatial-run1"  
export PROJECT_NAME="openvla-oft-rl"
export NUM_GPUS=8
export HIP_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

sbatch examples/run_openvla_oft_rl_libero.sh
