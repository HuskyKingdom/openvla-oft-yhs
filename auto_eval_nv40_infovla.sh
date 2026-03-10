#!/bin/bash
echo "Running Evaluations Automatically ------------------------------"
FARM_USER=sgyson10
FILE_PATH="experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml"

echo "Evaluating InfoBot-VLA ------------------------------"

CKPT_PATH="ckpt/ckpoints/openvla-7b+libero_4_task_suites_no_noops+b8+lr-0.0002+lora-r32+infobot-cross_attn-beta0.1--image_aug--substep--infobot_v2_stable--200000_chkpt"

# raw
# python experiments/robot/libero/run_libero_pro_eval_javas.py \
#   --pretrained_checkpoint $CKPT_PATH \
#   --use_infobot True --infobot_bottleneck_type cross_attn --infobot_bottleneck_dim 256 --infobot_num_tokens 8 --use_l1_regression True \
#   --task_suite_name libero_object --e_decoding False --save_video False \
#   --num_trials_per_task 50 --unnorm_key libero_object --task_label infobot_object_raw

# env
# sed -i 's/use_environment: false/use_environment: true/' experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml
# ... (omitted if not in original active flow)

# swap
sed -i 's/use_environment: true/use_environment: false/' experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml
sed -i 's/use_swap: false/use_swap: true/' experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml

python experiments/robot/libero/run_libero_pro_eval_javas.py  \
  --pretrained_checkpoint $CKPT_PATH \
  --use_infobot True --infobot_bottleneck_type cross_attn --infobot_bottleneck_dim 256 --infobot_num_tokens 8 --use_l1_regression True \
  --task_suite_name libero_object --e_decoding False --save_video False \
  --num_trials_per_task 50 --evaluation_config_path experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml  --unnorm_key libero_object --task_label infobot_object_swap

python experiments/robot/libero/run_libero_pro_eval_javas.py \
  --pretrained_checkpoint $CKPT_PATH \
  --use_infobot True --infobot_bottleneck_type cross_attn --infobot_bottleneck_dim 256 --infobot_num_tokens 8 --use_l1_regression True \
  --task_suite_name libero_goal --e_decoding False --save_video False \
  --num_trials_per_task 50 --evaluation_config_path experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml  --unnorm_key libero_goal --task_label infobot_goal_swap


# object
sed -i 's/use_swap: true/use_swap: false/' experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml
sed -i 's/use_object: false/use_object: true/' experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml

python experiments/robot/libero/run_libero_pro_eval_javas.py \
  --pretrained_checkpoint $CKPT_PATH \
  --use_infobot True --infobot_bottleneck_type cross_attn --infobot_bottleneck_dim 256 --infobot_num_tokens 8 --use_l1_regression True \
  --task_suite_name libero_object --e_decoding False --save_video False \
  --num_trials_per_task 50 --evaluation_config_path experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml  --unnorm_key libero_object --task_label infobot_object_object

python experiments/robot/libero/run_libero_pro_eval_javas.py \
  --pretrained_checkpoint $CKPT_PATH \
  --use_infobot True --infobot_bottleneck_type cross_attn --infobot_bottleneck_dim 256 --infobot_num_tokens 8 --use_l1_regression True \
  --task_suite_name libero_goal --e_decoding False --save_video False \
  --num_trials_per_task 50 --evaluation_config_path experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml  --unnorm_key libero_goal --task_label infobot_goal_object


# lan
sed -i 's/use_object: true/use_object: false/' experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml
sed -i 's/use_language: false/use_language: true/' experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml

python experiments/robot/libero/run_libero_pro_eval_javas.py \
  --pretrained_checkpoint $CKPT_PATH \
  --use_infobot True --infobot_bottleneck_type cross_attn --infobot_bottleneck_dim 256 --infobot_num_tokens 8 --use_l1_regression True \
  --task_suite_name libero_object --e_decoding False --save_video False \
  --num_trials_per_task 50 --evaluation_config_path experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml  --unnorm_key libero_object --task_label infobot_object_lan

python experiments/robot/libero/run_libero_pro_eval_javas.py \
  --pretrained_checkpoint $CKPT_PATH \
  --use_infobot True --infobot_bottleneck_type cross_attn --infobot_bottleneck_dim 256 --infobot_num_tokens 8 --use_l1_regression True \
  --task_suite_name libero_goal --e_decoding False --save_video False \
  --num_trials_per_task 50 --evaluation_config_path experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml  --unnorm_key libero_goal --task_label infobot_goal_lan


# task
sed -i 's/use_language: true/use_language: false/' experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml
sed -i 's/use_task: false/use_task: true/' experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml

python experiments/robot/libero/run_libero_pro_eval_javas.py \
  --pretrained_checkpoint $CKPT_PATH \
  --use_infobot True --infobot_bottleneck_type cross_attn --infobot_bottleneck_dim 256 --infobot_num_tokens 8 --use_l1_regression True \
  --task_suite_name libero_object --e_decoding False --save_video False \
  --num_trials_per_task 50 --evaluation_config_path experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml  --unnorm_key libero_object --task_label infobot_object_task

python experiments/robot/libero/run_libero_pro_eval_javas.py \
  --pretrained_checkpoint $CKPT_PATH \
  --use_infobot True --infobot_bottleneck_type cross_attn --infobot_bottleneck_dim 256 --infobot_num_tokens 8 --use_l1_regression True \
  --task_suite_name libero_goal --e_decoding False --save_video False \
  --num_trials_per_task 50 --evaluation_config_path experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml  --unnorm_key libero_goal --task_label infobot_goal_task

sed -i 's/use_task: true/use_task: false/' experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml
