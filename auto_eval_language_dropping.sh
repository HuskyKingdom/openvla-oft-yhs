#!/bin/bash
echo "Running Language Dropping Experiments (§4.1) ------------------------------"
FILE_PATH="experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml"
CKPT_APD="/home/yuhang/Warehouse/Yuhangworkspace/openvla-oft-yhs/ckpt/ckpoints/openvla-7b+libero_4_task_suites_no_noops+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--substep--substep_vla_scalling--150000_chkpt/"
CKPT_OFT="ckpt/ckpoints/openvla-7b+libero_4_task_suites_no_noops+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--oft_plus--150000_chkpt"
EVAL_SCRIPT="experiments/robot/libero/run_libero_pro_eval_substep.py"
NUM_TRIALS_PER_TASK=20
# APD model flags
APD_USE_EOS_DETECTION=True
APD_USE_PROPRIO=False
APD_USE_L1_REGRESSION=False
APD_USE_SUBSTEP_DECOMPOSITION=True
APD_USE_BDDL_LANGUAGE=True
APD_AUTO_REGRESSION=False
APD_NUM_IMAGES_IN_INPUT=1
APD_SUBSTEP_COMPLETION_THRESHOLD=0.07
# OFT model flags
OFT_USE_EOS_DETECTION=False
OFT_USE_PROPRIO=True
OFT_USE_L1_REGRESSION=True
OFT_USE_SUBSTEP_DECOMPOSITION=False
OFT_USE_BDDL_LANGUAGE=True
OFT_AUTO_REGRESSION=False
OFT_NUM_IMAGES_IN_INPUT=2
OFT_SUBSTEP_COMPLETION_THRESHOLD=0.07


mkdir -p ckpts

# Reset all evaluation config flags to false before starting
bash experiments/robot/libero/LIBERO-PRO/reset_eval_config.sh $FILE_PATH


# # ===========================================================================
# # APD MODEL — NULL LANGUAGE (null_instruction True, language dropped)
# # ===========================================================================
# echo "APD: NULL LANGUAGE (Language Dropping) ------------------------------"

# # swap
# sed -i 's/use_swap: false/use_swap: true/' $FILE_PATH

# python $EVAL_SCRIPT \
#   --pretrained_checkpoint $CKPT_APD --substep_completion_threshold $APD_SUBSTEP_COMPLETION_THRESHOLD \
#   --task_suite_name libero_spatial --e_decoding False --save_video False --use_substep_decomposition $APD_USE_SUBSTEP_DECOMPOSITION \
#   --num_trials_per_task $NUM_TRIALS_PER_TASK --evaluation_config_path $FILE_PATH --unnorm_key libero_spatial \
#   --task_label scalled_spatial_swap_nulllang --use_eos_detection $APD_USE_EOS_DETECTION --use_proprio $APD_USE_PROPRIO \
#   --use_l1_regression $APD_USE_L1_REGRESSION --use_bddl_language $APD_USE_BDDL_LANGUAGE --auto_regression $APD_AUTO_REGRESSION \
#   --num_images_in_input $APD_NUM_IMAGES_IN_INPUT --null_instruction True \
#   2>&1 | tee ckpts/scalled_spatial_swap_nulllang.txt

# python $EVAL_SCRIPT \
#   --pretrained_checkpoint $CKPT_APD --substep_completion_threshold $APD_SUBSTEP_COMPLETION_THRESHOLD \
#   --task_suite_name libero_10 --e_decoding False --save_video False --use_substep_decomposition $APD_USE_SUBSTEP_DECOMPOSITION \
#   --num_trials_per_task $NUM_TRIALS_PER_TASK --evaluation_config_path $FILE_PATH --unnorm_key libero_10 \
#   --task_label scalled_10_swap_nulllang --use_eos_detection $APD_USE_EOS_DETECTION --use_proprio $APD_USE_PROPRIO \
#   --use_l1_regression $APD_USE_L1_REGRESSION --use_bddl_language $APD_USE_BDDL_LANGUAGE --auto_regression $APD_AUTO_REGRESSION \
#   --num_images_in_input $APD_NUM_IMAGES_IN_INPUT --null_instruction True \
#   2>&1 | tee ckpts/scalled_10_swap_nulllang.txt

# # object
# sed -i 's/use_swap: true/use_swap: false/' $FILE_PATH
# sed -i 's/use_object: false/use_object: true/' $FILE_PATH

# python $EVAL_SCRIPT \
#   --pretrained_checkpoint $CKPT_APD --substep_completion_threshold $APD_SUBSTEP_COMPLETION_THRESHOLD \
#   --task_suite_name libero_spatial --e_decoding False --save_video False --use_substep_decomposition $APD_USE_SUBSTEP_DECOMPOSITION \
#   --num_trials_per_task $NUM_TRIALS_PER_TASK --evaluation_config_path $FILE_PATH --unnorm_key libero_spatial \
#   --task_label scalled_spatial_object_nulllang --use_eos_detection $APD_USE_EOS_DETECTION --use_proprio $APD_USE_PROPRIO \
#   --use_l1_regression $APD_USE_L1_REGRESSION --use_bddl_language $APD_USE_BDDL_LANGUAGE --auto_regression $APD_AUTO_REGRESSION \
#   --num_images_in_input $APD_NUM_IMAGES_IN_INPUT --null_instruction True \
#   2>&1 | tee ckpts/scalled_spatial_object_nulllang.txt

# python $EVAL_SCRIPT \
#   --pretrained_checkpoint $CKPT_APD --substep_completion_threshold $APD_SUBSTEP_COMPLETION_THRESHOLD \
#   --task_suite_name libero_10 --e_decoding False --save_video False --use_substep_decomposition $APD_USE_SUBSTEP_DECOMPOSITION \
#   --num_trials_per_task $NUM_TRIALS_PER_TASK --evaluation_config_path $FILE_PATH --unnorm_key libero_10 \
#   --task_label scalled_10_object_nulllang --use_eos_detection $APD_USE_EOS_DETECTION --use_proprio $APD_USE_PROPRIO \
#   --use_l1_regression $APD_USE_L1_REGRESSION --use_bddl_language $APD_USE_BDDL_LANGUAGE --auto_regression $APD_AUTO_REGRESSION \
#   --num_images_in_input $APD_NUM_IMAGES_IN_INPUT --null_instruction True \
#   2>&1 | tee ckpts/scalled_10_object_nulllang.txt

# # lan
# sed -i 's/use_object: true/use_object: false/' $FILE_PATH
# sed -i 's/use_language: false/use_language: true/' $FILE_PATH

# python $EVAL_SCRIPT \
#   --pretrained_checkpoint $CKPT_APD --substep_completion_threshold $APD_SUBSTEP_COMPLETION_THRESHOLD \
#   --task_suite_name libero_spatial --e_decoding False --save_video False --use_substep_decomposition $APD_USE_SUBSTEP_DECOMPOSITION \
#   --num_trials_per_task $NUM_TRIALS_PER_TASK --evaluation_config_path $FILE_PATH --unnorm_key libero_spatial \
#   --task_label scalled_spatial_lan_nulllang --use_eos_detection $APD_USE_EOS_DETECTION --use_proprio $APD_USE_PROPRIO \
#   --use_l1_regression $APD_USE_L1_REGRESSION --use_bddl_language $APD_USE_BDDL_LANGUAGE --auto_regression $APD_AUTO_REGRESSION \
#   --num_images_in_input $APD_NUM_IMAGES_IN_INPUT --null_instruction True \
#   2>&1 | tee ckpts/scalled_spatial_lan_nulllang.txt

# python $EVAL_SCRIPT \
#   --pretrained_checkpoint $CKPT_APD --substep_completion_threshold $APD_SUBSTEP_COMPLETION_THRESHOLD \
#   --task_suite_name libero_10 --e_decoding False --save_video False --use_substep_decomposition $APD_USE_SUBSTEP_DECOMPOSITION \
#   --num_trials_per_task $NUM_TRIALS_PER_TASK --evaluation_config_path $FILE_PATH --unnorm_key libero_10 \
#   --task_label scalled_10_lan_nulllang --use_eos_detection $APD_USE_EOS_DETECTION --use_proprio $APD_USE_PROPRIO \
#   --use_l1_regression $APD_USE_L1_REGRESSION --use_bddl_language $APD_USE_BDDL_LANGUAGE --auto_regression $APD_AUTO_REGRESSION \
#   --num_images_in_input $APD_NUM_IMAGES_IN_INPUT --null_instruction True \
#   2>&1 | tee ckpts/scalled_10_lan_nulllang.txt

# # task
# sed -i 's/use_language: true/use_language: false/' $FILE_PATH
# sed -i 's/use_task: false/use_task: true/' $FILE_PATH

# python $EVAL_SCRIPT \
#   --pretrained_checkpoint $CKPT_APD --substep_completion_threshold $APD_SUBSTEP_COMPLETION_THRESHOLD \
#   --task_suite_name libero_spatial --e_decoding False --save_video False --use_substep_decomposition $APD_USE_SUBSTEP_DECOMPOSITION \
#   --num_trials_per_task $NUM_TRIALS_PER_TASK --evaluation_config_path $FILE_PATH --unnorm_key libero_spatial \
#   --task_label scalled_spatial_task_nulllang --use_eos_detection $APD_USE_EOS_DETECTION --use_proprio $APD_USE_PROPRIO \
#   --use_l1_regression $APD_USE_L1_REGRESSION --use_bddl_language $APD_USE_BDDL_LANGUAGE --auto_regression $APD_AUTO_REGRESSION \
#   --num_images_in_input $APD_NUM_IMAGES_IN_INPUT --null_instruction True \
#   2>&1 | tee ckpts/scalled_spatial_task_nulllang.txt

# python $EVAL_SCRIPT \
#   --pretrained_checkpoint $CKPT_APD --substep_completion_threshold $APD_SUBSTEP_COMPLETION_THRESHOLD \
#   --task_suite_name libero_10 --e_decoding False --save_video False --use_substep_decomposition $APD_USE_SUBSTEP_DECOMPOSITION \
#   --num_trials_per_task $NUM_TRIALS_PER_TASK --evaluation_config_path $FILE_PATH --unnorm_key libero_10 \
#   --task_label scalled_10_task_nulllang --use_eos_detection $APD_USE_EOS_DETECTION --use_proprio $APD_USE_PROPRIO \
#   --use_l1_regression $APD_USE_L1_REGRESSION --use_bddl_language $APD_USE_BDDL_LANGUAGE --auto_regression $APD_AUTO_REGRESSION \
#   --num_images_in_input $APD_NUM_IMAGES_IN_INPUT --null_instruction True \
#   2>&1 | tee ckpts/scalled_10_task_nulllang.txt

# sed -i 's/use_task: true/use_task: false/' $FILE_PATH


# ===========================================================================
# OFT BASELINE — NULL LANGUAGE (null_instruction True, language dropped)
# ===========================================================================
echo "OFT Baseline: NULL LANGUAGE (Language Dropping) ------------------------------"

# lan
sed -i 's/use_language: false/use_language: true/' $FILE_PATH

python $EVAL_SCRIPT \
  --pretrained_checkpoint $CKPT_OFT --substep_completion_threshold $OFT_SUBSTEP_COMPLETION_THRESHOLD \
  --task_suite_name libero_spatial --e_decoding False --save_video False --use_substep_decomposition $OFT_USE_SUBSTEP_DECOMPOSITION \
  --num_trials_per_task $NUM_TRIALS_PER_TASK --evaluation_config_path $FILE_PATH --unnorm_key libero_spatial \
  --task_label oft_spatial_lan_nulllang --use_eos_detection $OFT_USE_EOS_DETECTION --use_proprio $OFT_USE_PROPRIO \
  --use_l1_regression $OFT_USE_L1_REGRESSION --use_bddl_language $OFT_USE_BDDL_LANGUAGE --auto_regression $OFT_AUTO_REGRESSION \
  --num_images_in_input $OFT_NUM_IMAGES_IN_INPUT --null_instruction True \
  2>&1 | tee ckpts/oft_spatial_lan_nulllang.txt

python $EVAL_SCRIPT \
  --pretrained_checkpoint $CKPT_OFT --substep_completion_threshold $OFT_SUBSTEP_COMPLETION_THRESHOLD \
  --task_suite_name libero_10 --e_decoding False --save_video False --use_substep_decomposition $OFT_USE_SUBSTEP_DECOMPOSITION \
  --num_trials_per_task $NUM_TRIALS_PER_TASK --evaluation_config_path $FILE_PATH --unnorm_key libero_10 \
  --task_label oft_10_lan_nulllang --use_eos_detection $OFT_USE_EOS_DETECTION --use_proprio $OFT_USE_PROPRIO \
  --use_l1_regression $OFT_USE_L1_REGRESSION --use_bddl_language $OFT_USE_BDDL_LANGUAGE --auto_regression $OFT_AUTO_REGRESSION \
  --num_images_in_input $OFT_NUM_IMAGES_IN_INPUT --null_instruction True \
  2>&1 | tee ckpts/oft_10_lan_nulllang.txt

# task
sed -i 's/use_language: true/use_language: false/' $FILE_PATH
sed -i 's/use_task: false/use_task: true/' $FILE_PATH

python $EVAL_SCRIPT \
  --pretrained_checkpoint $CKPT_OFT --substep_completion_threshold $OFT_SUBSTEP_COMPLETION_THRESHOLD \
  --task_suite_name libero_spatial --e_decoding False --save_video False --use_substep_decomposition $OFT_USE_SUBSTEP_DECOMPOSITION \
  --num_trials_per_task $NUM_TRIALS_PER_TASK --evaluation_config_path $FILE_PATH --unnorm_key libero_spatial \
  --task_label oft_spatial_task_nulllang --use_eos_detection $OFT_USE_EOS_DETECTION --use_proprio $OFT_USE_PROPRIO \
  --use_l1_regression $OFT_USE_L1_REGRESSION --use_bddl_language $OFT_USE_BDDL_LANGUAGE --auto_regression $OFT_AUTO_REGRESSION \
  --num_images_in_input $OFT_NUM_IMAGES_IN_INPUT --null_instruction True \
  2>&1 | tee ckpts/oft_spatial_task_nulllang.txt

python $EVAL_SCRIPT \
  --pretrained_checkpoint $CKPT_OFT --substep_completion_threshold $OFT_SUBSTEP_COMPLETION_THRESHOLD \
  --task_suite_name libero_10 --e_decoding False --save_video False --use_substep_decomposition $OFT_USE_SUBSTEP_DECOMPOSITION \
  --num_trials_per_task $NUM_TRIALS_PER_TASK --evaluation_config_path $FILE_PATH --unnorm_key libero_10 \
  --task_label oft_10_task_nulllang --use_eos_detection $OFT_USE_EOS_DETECTION --use_proprio $OFT_USE_PROPRIO \
  --use_l1_regression $OFT_USE_L1_REGRESSION --use_bddl_language $OFT_USE_BDDL_LANGUAGE --auto_regression $OFT_AUTO_REGRESSION \
  --num_images_in_input $OFT_NUM_IMAGES_IN_INPUT --null_instruction True \
  2>&1 | tee ckpts/oft_10_task_nulllang.txt

# object
sed -i 's/use_task: true/use_task: false/' $FILE_PATH
sed -i 's/use_object: false/use_object: true/' $FILE_PATH

python $EVAL_SCRIPT \
  --pretrained_checkpoint $CKPT_OFT --substep_completion_threshold $OFT_SUBSTEP_COMPLETION_THRESHOLD \
  --task_suite_name libero_spatial --e_decoding False --save_video False --use_substep_decomposition $OFT_USE_SUBSTEP_DECOMPOSITION \
  --num_trials_per_task $NUM_TRIALS_PER_TASK --evaluation_config_path $FILE_PATH --unnorm_key libero_spatial \
  --task_label oft_spatial_object_nulllang --use_eos_detection $OFT_USE_EOS_DETECTION --use_proprio $OFT_USE_PROPRIO \
  --use_l1_regression $OFT_USE_L1_REGRESSION --use_bddl_language $OFT_USE_BDDL_LANGUAGE --auto_regression $OFT_AUTO_REGRESSION \
  --num_images_in_input $OFT_NUM_IMAGES_IN_INPUT --null_instruction True \
  2>&1 | tee ckpts/oft_spatial_object_nulllang.txt

python $EVAL_SCRIPT \
  --pretrained_checkpoint $CKPT_OFT --substep_completion_threshold $OFT_SUBSTEP_COMPLETION_THRESHOLD \
  --task_suite_name libero_10 --e_decoding False --save_video False --use_substep_decomposition $OFT_USE_SUBSTEP_DECOMPOSITION \
  --num_trials_per_task $NUM_TRIALS_PER_TASK --evaluation_config_path $FILE_PATH --unnorm_key libero_10 \
  --task_label oft_10_object_nulllang --use_eos_detection $OFT_USE_EOS_DETECTION --use_proprio $OFT_USE_PROPRIO \
  --use_l1_regression $OFT_USE_L1_REGRESSION --use_bddl_language $OFT_USE_BDDL_LANGUAGE --auto_regression $OFT_AUTO_REGRESSION \
  --num_images_in_input $OFT_NUM_IMAGES_IN_INPUT --null_instruction True \
  2>&1 | tee ckpts/oft_10_object_nulllang.txt

# pos (swap)
sed -i 's/use_object: true/use_object: false/' $FILE_PATH
sed -i 's/use_swap: false/use_swap: true/' $FILE_PATH

python $EVAL_SCRIPT \
  --pretrained_checkpoint $CKPT_OFT --substep_completion_threshold $OFT_SUBSTEP_COMPLETION_THRESHOLD \
  --task_suite_name libero_spatial --e_decoding False --save_video False --use_substep_decomposition $OFT_USE_SUBSTEP_DECOMPOSITION \
  --num_trials_per_task $NUM_TRIALS_PER_TASK --evaluation_config_path $FILE_PATH --unnorm_key libero_spatial \
  --task_label oft_spatial_swap_nulllang --use_eos_detection $OFT_USE_EOS_DETECTION --use_proprio $OFT_USE_PROPRIO \
  --use_l1_regression $OFT_USE_L1_REGRESSION --use_bddl_language $OFT_USE_BDDL_LANGUAGE --auto_regression $OFT_AUTO_REGRESSION \
  --num_images_in_input $OFT_NUM_IMAGES_IN_INPUT --null_instruction True \
  2>&1 | tee ckpts/oft_spatial_swap_nulllang.txt

python $EVAL_SCRIPT \
  --pretrained_checkpoint $CKPT_OFT --substep_completion_threshold $OFT_SUBSTEP_COMPLETION_THRESHOLD \
  --task_suite_name libero_10 --e_decoding False --save_video False --use_substep_decomposition $OFT_USE_SUBSTEP_DECOMPOSITION \
  --num_trials_per_task $NUM_TRIALS_PER_TASK --evaluation_config_path $FILE_PATH --unnorm_key libero_10 \
  --task_label oft_10_swap_nulllang --use_eos_detection $OFT_USE_EOS_DETECTION --use_proprio $OFT_USE_PROPRIO \
  --use_l1_regression $OFT_USE_L1_REGRESSION --use_bddl_language $OFT_USE_BDDL_LANGUAGE --auto_regression $OFT_AUTO_REGRESSION \
  --num_images_in_input $OFT_NUM_IMAGES_IN_INPUT --null_instruction True \
  2>&1 | tee ckpts/oft_10_swap_nulllang.txt

sed -i 's/use_swap: true/use_swap: false/' $FILE_PATH

echo "Language Dropping experiments done. Results in ckpts/."
