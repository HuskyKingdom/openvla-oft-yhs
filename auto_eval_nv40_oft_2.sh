#!/bin/bash


echo "Running Evaluations Automatically ------------------------------"
FARM_USER=sgyson10
FILE_PATH="experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml"




echo "Evaluating OFT ------------------------------"

raw
python experiments/robot/libero/run_libero_pro_eval.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10  \
  --task_suite_name libero_object --e_decoding False --save_video False \
  --num_trials_per_task 50  --unnorm_key libero_object --task_label oft_object_raw

python experiments/robot/libero/run_libero_pro_eval.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10  \
  --task_suite_name libero_goal --e_decoding False --save_video False \
  --num_trials_per_task 50  --unnorm_key libero_goal --task_label oft_goal_raw

# env
sed -i 's/use_environment: false/use_environment: true/' experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml

python experiments/robot/libero/run_libero_pro_eval.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10  \
  --task_suite_name libero_object --e_decoding False --save_video False \
  --num_trials_per_task 50 --evaluation_config_path experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml  --unnorm_key libero_object --task_label oft_object_env

python experiments/robot/libero/run_libero_pro_eval.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10  \
  --task_suite_name libero_goal --e_decoding False --save_video False \
  --num_trials_per_task 50 --evaluation_config_path experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml  --unnorm_key libero_goal --task_label oft_goal_env

# swap
sed -i 's/use_environment: true/use_environment: false/' experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml
sed -i 's/use_swap: false/use_swap: true/' experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml


python experiments/robot/libero/run_libero_pro_eval.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10  \
  --task_suite_name libero_object --e_decoding False --save_video False \
  --num_trials_per_task 50 --evaluation_config_path experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml  --unnorm_key libero_object --task_label oft_object_swap

python experiments/robot/libero/run_libero_pro_eval.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10  \
  --task_suite_name libero_goal --e_decoding False --save_video False \
  --num_trials_per_task 50 --evaluation_config_path experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml  --unnorm_key libero_goal --task_label oft_goal_swap


# object
sed -i 's/use_swap: true/use_swap: false/' experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml
sed -i 's/use_object: false/use_object: true/' experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml


python experiments/robot/libero/run_libero_pro_eval.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10  \
  --task_suite_name libero_object --e_decoding False --save_video False \
  --num_trials_per_task 50 --evaluation_config_path experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml  --unnorm_key libero_object --task_label oft_object_object

python experiments/robot/libero/run_libero_pro_eval.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10  \
  --task_suite_name libero_goal --e_decoding False --save_video False \
  --num_trials_per_task 50 --evaluation_config_path experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml  --unnorm_key libero_goal --task_label oft_goal_object


# lan
sed -i 's/use_object: true/use_object: false/' experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml
sed -i 's/use_language: false/use_language: true/' experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml

python experiments/robot/libero/run_libero_pro_eval.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10  \
  --task_suite_name libero_object --e_decoding False --save_video False \
  --num_trials_per_task 50 --evaluation_config_path experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml  --unnorm_key libero_object --task_label oft_object_lan

python experiments/robot/libero/run_libero_pro_eval.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10  \
  --task_suite_name libero_goal --e_decoding False --save_video False \
  --num_trials_per_task 50 --evaluation_config_path experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml  --unnorm_key libero_goal --task_label oft_goal_lan


# task
sed -i 's/use_language: true/use_language: false/' experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml
sed -i 's/use_task: false/use_task: true/' experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml

python experiments/robot/libero/run_libero_pro_eval.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10  \
  --task_suite_name libero_object --e_decoding False --save_video False \
  --num_trials_per_task 50 --evaluation_config_path experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml  --unnorm_key libero_object --task_label oft_object_task

python experiments/robot/libero/run_libero_pro_eval.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10  \
  --task_suite_name libero_goal --e_decoding False --save_video False \
  --num_trials_per_task 50 --evaluation_config_path experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml  --unnorm_key libero_goal --task_label oft_goal_task

sed -i 's/use_task: true/use_task: false/' experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml
