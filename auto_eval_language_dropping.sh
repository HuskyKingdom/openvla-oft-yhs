#!/bin/bash
echo "Running Language Dropping Experiments (§4.1) ------------------------------"
FILE_PATH="experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml"
CKPT_APD="ckpt/ckpoints/openvla-7b+libero_4_task_suites_no_noops+b8+lr-0.0002+lora-r32+infobot-cross_attn-beta0.1--image_aug--substep--infobot_v2_stable--200000_chkpt"
CKPT_OFT="moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10"

mkdir -p ckpts

# ===========================================================================
# APD MODEL — WITH LANGUAGE (null_instruction False, baseline condition)
# ===========================================================================
echo "APD: WITH LANGUAGE ------------------------------"

# swap
sed -i 's/use_environment: false/use_environment: false/' $FILE_PATH
sed -i 's/use_environment: true/use_environment: false/' $FILE_PATH
sed -i 's/use_swap: false/use_swap: true/' $FILE_PATH

python experiments/robot/libero/run_libero_pro_eval_substep.py \
  --pretrained_checkpoint $CKPT_APD --substep_completion_threshold 0.07 \
  --task_suite_name libero_object --e_decoding False --save_video False --use_substep_decomposition True \
  --num_trials_per_task 50 --evaluation_config_path $FILE_PATH --unnorm_key libero_object \
  --task_label apd_object_swap_withlang --use_eos_detection True --null_instruction False \
  2>&1 | tee ckpts/apd_object_swap_withlang.txt

python experiments/robot/libero/run_libero_pro_eval_substep.py \
  --pretrained_checkpoint $CKPT_APD --substep_completion_threshold 0.07 \
  --task_suite_name libero_goal --e_decoding False --save_video False --use_substep_decomposition True \
  --num_trials_per_task 50 --evaluation_config_path $FILE_PATH --unnorm_key libero_goal \
  --task_label apd_goal_swap_withlang --use_eos_detection True --null_instruction False \
  2>&1 | tee ckpts/apd_goal_swap_withlang.txt

# object
sed -i 's/use_swap: true/use_swap: false/' $FILE_PATH
sed -i 's/use_object: false/use_object: true/' $FILE_PATH

python experiments/robot/libero/run_libero_pro_eval_substep.py \
  --pretrained_checkpoint $CKPT_APD --substep_completion_threshold 0.07 \
  --task_suite_name libero_object --e_decoding False --save_video False --use_substep_decomposition True \
  --num_trials_per_task 50 --evaluation_config_path $FILE_PATH --unnorm_key libero_object \
  --task_label apd_object_object_withlang --use_eos_detection True --null_instruction False \
  2>&1 | tee ckpts/apd_object_object_withlang.txt

python experiments/robot/libero/run_libero_pro_eval_substep.py \
  --pretrained_checkpoint $CKPT_APD --substep_completion_threshold 0.07 \
  --task_suite_name libero_goal --e_decoding False --save_video False --use_substep_decomposition True \
  --num_trials_per_task 50 --evaluation_config_path $FILE_PATH --unnorm_key libero_goal \
  --task_label apd_goal_object_withlang --use_eos_detection True --null_instruction False \
  2>&1 | tee ckpts/apd_goal_object_withlang.txt

# lan
sed -i 's/use_object: true/use_object: false/' $FILE_PATH
sed -i 's/use_language: false/use_language: true/' $FILE_PATH

python experiments/robot/libero/run_libero_pro_eval_substep.py \
  --pretrained_checkpoint $CKPT_APD --substep_completion_threshold 0.07 \
  --task_suite_name libero_object --e_decoding False --save_video False --use_substep_decomposition True \
  --num_trials_per_task 50 --evaluation_config_path $FILE_PATH --unnorm_key libero_object \
  --task_label apd_object_lan_withlang --use_eos_detection True --null_instruction False \
  2>&1 | tee ckpts/apd_object_lan_withlang.txt

python experiments/robot/libero/run_libero_pro_eval_substep.py \
  --pretrained_checkpoint $CKPT_APD --substep_completion_threshold 0.07 \
  --task_suite_name libero_goal --e_decoding False --save_video False --use_substep_decomposition True \
  --num_trials_per_task 50 --evaluation_config_path $FILE_PATH --unnorm_key libero_goal \
  --task_label apd_goal_lan_withlang --use_eos_detection True --null_instruction False \
  2>&1 | tee ckpts/apd_goal_lan_withlang.txt

# task
sed -i 's/use_language: true/use_language: false/' $FILE_PATH
sed -i 's/use_task: false/use_task: true/' $FILE_PATH

python experiments/robot/libero/run_libero_pro_eval_substep.py \
  --pretrained_checkpoint $CKPT_APD --substep_completion_threshold 0.07 \
  --task_suite_name libero_object --e_decoding False --save_video False --use_substep_decomposition True \
  --num_trials_per_task 50 --evaluation_config_path $FILE_PATH --unnorm_key libero_object \
  --task_label apd_object_task_withlang --use_eos_detection True --null_instruction False \
  2>&1 | tee ckpts/apd_object_task_withlang.txt

python experiments/robot/libero/run_libero_pro_eval_substep.py \
  --pretrained_checkpoint $CKPT_APD --substep_completion_threshold 0.07 \
  --task_suite_name libero_goal --e_decoding False --save_video False --use_substep_decomposition True \
  --num_trials_per_task 50 --evaluation_config_path $FILE_PATH --unnorm_key libero_goal \
  --task_label apd_goal_task_withlang --use_eos_detection True --null_instruction False \
  2>&1 | tee ckpts/apd_goal_task_withlang.txt

sed -i 's/use_task: true/use_task: false/' $FILE_PATH


# ===========================================================================
# APD MODEL — NULL LANGUAGE (null_instruction True, language dropped)
# ===========================================================================
echo "APD: NULL LANGUAGE (Language Dropping) ------------------------------"

# swap
sed -i 's/use_swap: false/use_swap: true/' $FILE_PATH

python experiments/robot/libero/run_libero_pro_eval_substep.py \
  --pretrained_checkpoint $CKPT_APD --substep_completion_threshold 0.07 \
  --task_suite_name libero_object --e_decoding False --save_video False --use_substep_decomposition True \
  --num_trials_per_task 50 --evaluation_config_path $FILE_PATH --unnorm_key libero_object \
  --task_label apd_object_swap_nulllang --use_eos_detection True --null_instruction True \
  2>&1 | tee ckpts/apd_object_swap_nulllang.txt

python experiments/robot/libero/run_libero_pro_eval_substep.py \
  --pretrained_checkpoint $CKPT_APD --substep_completion_threshold 0.07 \
  --task_suite_name libero_goal --e_decoding False --save_video False --use_substep_decomposition True \
  --num_trials_per_task 50 --evaluation_config_path $FILE_PATH --unnorm_key libero_goal \
  --task_label apd_goal_swap_nulllang --use_eos_detection True --null_instruction True \
  2>&1 | tee ckpts/apd_goal_swap_nulllang.txt

# object
sed -i 's/use_swap: true/use_swap: false/' $FILE_PATH
sed -i 's/use_object: false/use_object: true/' $FILE_PATH

python experiments/robot/libero/run_libero_pro_eval_substep.py \
  --pretrained_checkpoint $CKPT_APD --substep_completion_threshold 0.07 \
  --task_suite_name libero_object --e_decoding False --save_video False --use_substep_decomposition True \
  --num_trials_per_task 50 --evaluation_config_path $FILE_PATH --unnorm_key libero_object \
  --task_label apd_object_object_nulllang --use_eos_detection True --null_instruction True \
  2>&1 | tee ckpts/apd_object_object_nulllang.txt

python experiments/robot/libero/run_libero_pro_eval_substep.py \
  --pretrained_checkpoint $CKPT_APD --substep_completion_threshold 0.07 \
  --task_suite_name libero_goal --e_decoding False --save_video False --use_substep_decomposition True \
  --num_trials_per_task 50 --evaluation_config_path $FILE_PATH --unnorm_key libero_goal \
  --task_label apd_goal_object_nulllang --use_eos_detection True --null_instruction True \
  2>&1 | tee ckpts/apd_goal_object_nulllang.txt

# lan
sed -i 's/use_object: true/use_object: false/' $FILE_PATH
sed -i 's/use_language: false/use_language: true/' $FILE_PATH

python experiments/robot/libero/run_libero_pro_eval_substep.py \
  --pretrained_checkpoint $CKPT_APD --substep_completion_threshold 0.07 \
  --task_suite_name libero_object --e_decoding False --save_video False --use_substep_decomposition True \
  --num_trials_per_task 50 --evaluation_config_path $FILE_PATH --unnorm_key libero_object \
  --task_label apd_object_lan_nulllang --use_eos_detection True --null_instruction True \
  2>&1 | tee ckpts/apd_object_lan_nulllang.txt

python experiments/robot/libero/run_libero_pro_eval_substep.py \
  --pretrained_checkpoint $CKPT_APD --substep_completion_threshold 0.07 \
  --task_suite_name libero_goal --e_decoding False --save_video False --use_substep_decomposition True \
  --num_trials_per_task 50 --evaluation_config_path $FILE_PATH --unnorm_key libero_goal \
  --task_label apd_goal_lan_nulllang --use_eos_detection True --null_instruction True \
  2>&1 | tee ckpts/apd_goal_lan_nulllang.txt

# task
sed -i 's/use_language: true/use_language: false/' $FILE_PATH
sed -i 's/use_task: false/use_task: true/' $FILE_PATH

python experiments/robot/libero/run_libero_pro_eval_substep.py \
  --pretrained_checkpoint $CKPT_APD --substep_completion_threshold 0.07 \
  --task_suite_name libero_object --e_decoding False --save_video False --use_substep_decomposition True \
  --num_trials_per_task 50 --evaluation_config_path $FILE_PATH --unnorm_key libero_object \
  --task_label apd_object_task_nulllang --use_eos_detection True --null_instruction True \
  2>&1 | tee ckpts/apd_object_task_nulllang.txt

python experiments/robot/libero/run_libero_pro_eval_substep.py \
  --pretrained_checkpoint $CKPT_APD --substep_completion_threshold 0.07 \
  --task_suite_name libero_goal --e_decoding False --save_video False --use_substep_decomposition True \
  --num_trials_per_task 50 --evaluation_config_path $FILE_PATH --unnorm_key libero_goal \
  --task_label apd_goal_task_nulllang --use_eos_detection True --null_instruction True \
  2>&1 | tee ckpts/apd_goal_task_nulllang.txt

sed -i 's/use_task: true/use_task: false/' $FILE_PATH


# ===========================================================================
# OFT BASELINE — WITH LANGUAGE (null_instruction False)
# ===========================================================================
echo "OFT Baseline: WITH LANGUAGE ------------------------------"

# swap
sed -i 's/use_swap: false/use_swap: true/' $FILE_PATH

python experiments/robot/libero/run_libero_pro_eval_substep.py \
  --pretrained_checkpoint $CKPT_OFT \
  --task_suite_name libero_object --e_decoding False --save_video False \
  --num_trials_per_task 50 --evaluation_config_path $FILE_PATH --unnorm_key libero_object \
  --task_label oft_object_swap_withlang --null_instruction False \
  2>&1 | tee ckpts/oft_object_swap_withlang.txt

python experiments/robot/libero/run_libero_pro_eval_substep.py \
  --pretrained_checkpoint $CKPT_OFT \
  --task_suite_name libero_goal --e_decoding False --save_video False \
  --num_trials_per_task 50 --evaluation_config_path $FILE_PATH --unnorm_key libero_goal \
  --task_label oft_goal_swap_withlang --null_instruction False \
  2>&1 | tee ckpts/oft_goal_swap_withlang.txt

# object
sed -i 's/use_swap: true/use_swap: false/' $FILE_PATH
sed -i 's/use_object: false/use_object: true/' $FILE_PATH

python experiments/robot/libero/run_libero_pro_eval_substep.py \
  --pretrained_checkpoint $CKPT_OFT \
  --task_suite_name libero_object --e_decoding False --save_video False \
  --num_trials_per_task 50 --evaluation_config_path $FILE_PATH --unnorm_key libero_object \
  --task_label oft_object_object_withlang --null_instruction False \
  2>&1 | tee ckpts/oft_object_object_withlang.txt

python experiments/robot/libero/run_libero_pro_eval_substep.py \
  --pretrained_checkpoint $CKPT_OFT \
  --task_suite_name libero_goal --e_decoding False --save_video False \
  --num_trials_per_task 50 --evaluation_config_path $FILE_PATH --unnorm_key libero_goal \
  --task_label oft_goal_object_withlang --null_instruction False \
  2>&1 | tee ckpts/oft_goal_object_withlang.txt

# lan
sed -i 's/use_object: true/use_object: false/' $FILE_PATH
sed -i 's/use_language: false/use_language: true/' $FILE_PATH

python experiments/robot/libero/run_libero_pro_eval_substep.py \
  --pretrained_checkpoint $CKPT_OFT \
  --task_suite_name libero_object --e_decoding False --save_video False \
  --num_trials_per_task 50 --evaluation_config_path $FILE_PATH --unnorm_key libero_object \
  --task_label oft_object_lan_withlang --null_instruction False \
  2>&1 | tee ckpts/oft_object_lan_withlang.txt

python experiments/robot/libero/run_libero_pro_eval_substep.py \
  --pretrained_checkpoint $CKPT_OFT \
  --task_suite_name libero_goal --e_decoding False --save_video False \
  --num_trials_per_task 50 --evaluation_config_path $FILE_PATH --unnorm_key libero_goal \
  --task_label oft_goal_lan_withlang --null_instruction False \
  2>&1 | tee ckpts/oft_goal_lan_withlang.txt

# task
sed -i 's/use_language: true/use_language: false/' $FILE_PATH
sed -i 's/use_task: false/use_task: true/' $FILE_PATH

python experiments/robot/libero/run_libero_pro_eval_substep.py \
  --pretrained_checkpoint $CKPT_OFT \
  --task_suite_name libero_object --e_decoding False --save_video False \
  --num_trials_per_task 50 --evaluation_config_path $FILE_PATH --unnorm_key libero_object \
  --task_label oft_object_task_withlang --null_instruction False \
  2>&1 | tee ckpts/oft_object_task_withlang.txt

python experiments/robot/libero/run_libero_pro_eval_substep.py \
  --pretrained_checkpoint $CKPT_OFT \
  --task_suite_name libero_goal --e_decoding False --save_video False \
  --num_trials_per_task 50 --evaluation_config_path $FILE_PATH --unnorm_key libero_goal \
  --task_label oft_goal_task_withlang --null_instruction False \
  2>&1 | tee ckpts/oft_goal_task_withlang.txt

sed -i 's/use_task: true/use_task: false/' $FILE_PATH


# ===========================================================================
# OFT BASELINE — NULL LANGUAGE (null_instruction True, language dropped)
# ===========================================================================
echo "OFT Baseline: NULL LANGUAGE (Language Dropping) ------------------------------"

# swap
sed -i 's/use_swap: false/use_swap: true/' $FILE_PATH

python experiments/robot/libero/run_libero_pro_eval_substep.py \
  --pretrained_checkpoint $CKPT_OFT \
  --task_suite_name libero_object --e_decoding False --save_video False \
  --num_trials_per_task 50 --evaluation_config_path $FILE_PATH --unnorm_key libero_object \
  --task_label oft_object_swap_nulllang --null_instruction True \
  2>&1 | tee ckpts/oft_object_swap_nulllang.txt

python experiments/robot/libero/run_libero_pro_eval_substep.py \
  --pretrained_checkpoint $CKPT_OFT \
  --task_suite_name libero_goal --e_decoding False --save_video False \
  --num_trials_per_task 50 --evaluation_config_path $FILE_PATH --unnorm_key libero_goal \
  --task_label oft_goal_swap_nulllang --null_instruction True \
  2>&1 | tee ckpts/oft_goal_swap_nulllang.txt

# object
sed -i 's/use_swap: true/use_swap: false/' $FILE_PATH
sed -i 's/use_object: false/use_object: true/' $FILE_PATH

python experiments/robot/libero/run_libero_pro_eval_substep.py \
  --pretrained_checkpoint $CKPT_OFT \
  --task_suite_name libero_object --e_decoding False --save_video False \
  --num_trials_per_task 50 --evaluation_config_path $FILE_PATH --unnorm_key libero_object \
  --task_label oft_object_object_nulllang --null_instruction True \
  2>&1 | tee ckpts/oft_object_object_nulllang.txt

python experiments/robot/libero/run_libero_pro_eval_substep.py \
  --pretrained_checkpoint $CKPT_OFT \
  --task_suite_name libero_goal --e_decoding False --save_video False \
  --num_trials_per_task 50 --evaluation_config_path $FILE_PATH --unnorm_key libero_goal \
  --task_label oft_goal_object_nulllang --null_instruction True \
  2>&1 | tee ckpts/oft_goal_object_nulllang.txt

# lan
sed -i 's/use_object: true/use_object: false/' $FILE_PATH
sed -i 's/use_language: false/use_language: true/' $FILE_PATH

python experiments/robot/libero/run_libero_pro_eval_substep.py \
  --pretrained_checkpoint $CKPT_OFT \
  --task_suite_name libero_object --e_decoding False --save_video False \
  --num_trials_per_task 50 --evaluation_config_path $FILE_PATH --unnorm_key libero_object \
  --task_label oft_object_lan_nulllang --null_instruction True \
  2>&1 | tee ckpts/oft_object_lan_nulllang.txt

python experiments/robot/libero/run_libero_pro_eval_substep.py \
  --pretrained_checkpoint $CKPT_OFT \
  --task_suite_name libero_goal --e_decoding False --save_video False \
  --num_trials_per_task 50 --evaluation_config_path $FILE_PATH --unnorm_key libero_goal \
  --task_label oft_goal_lan_nulllang --null_instruction True \
  2>&1 | tee ckpts/oft_goal_lan_nulllang.txt

# task
sed -i 's/use_language: true/use_language: false/' $FILE_PATH
sed -i 's/use_task: false/use_task: true/' $FILE_PATH

python experiments/robot/libero/run_libero_pro_eval_substep.py \
  --pretrained_checkpoint $CKPT_OFT \
  --task_suite_name libero_object --e_decoding False --save_video False \
  --num_trials_per_task 50 --evaluation_config_path $FILE_PATH --unnorm_key libero_object \
  --task_label oft_object_task_nulllang --null_instruction True \
  2>&1 | tee ckpts/oft_object_task_nulllang.txt

python experiments/robot/libero/run_libero_pro_eval_substep.py \
  --pretrained_checkpoint $CKPT_OFT \
  --task_suite_name libero_goal --e_decoding False --save_video False \
  --num_trials_per_task 50 --evaluation_config_path $FILE_PATH --unnorm_key libero_goal \
  --task_label oft_goal_task_nulllang --null_instruction True \
  2>&1 | tee ckpts/oft_goal_task_nulllang.txt

sed -i 's/use_task: true/use_task: false/' $FILE_PATH

echo "Language Dropping experiments done. Results in ckpts/."

