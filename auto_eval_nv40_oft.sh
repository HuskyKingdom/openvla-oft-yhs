#!/bin/bash


echo "Running Evaluations Automatically ------------------------------"
FARM_USER=sgyson10
FILE_PATH="experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml"




echo "Evaluating OFT ------------------------------"

# raw
python experiments/robot/libero/run_libero_pro_eval.py \
  --pretrained_checkpoint Sylvest/openvla-7b-oft-finetuned-libero-plus-mixdata  \
  --task_suite_name libero_spatial --e_decoding False --save_video False \
  --task_label empty_true --cus_task empty --remove_wrap True \
  --num_trials_per_task 50  --unnorm_key libero_spatial --task_label oft_raw

# env
sed -i 's/use_environment: false/use_environment: true/' experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml

python experiments/robot/libero/run_libero_pro_eval.py \
  --pretrained_checkpoint Sylvest/openvla-7b-oft-finetuned-libero-plus-mixdata  \
  --task_suite_name libero_spatial --e_decoding False --save_video False \
  --task_label empty_true --cus_task empty --remove_wrap True \
  --num_trials_per_task 50 --evaluation_config_path experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml  --unnorm_key libero_spatial --task_label oft_env


# swap
sed -i 's/use_environment: true/use_environment: flase/' experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml
sed -i 's/use_swap: false/use_swap: true/' experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml


python experiments/robot/libero/run_libero_pro_eval.py \
  --pretrained_checkpoint Sylvest/openvla-7b-oft-finetuned-libero-plus-mixdata  \
  --task_suite_name libero_spatial --e_decoding False --save_video False \
  --task_label empty_true --cus_task empty --remove_wrap True \
  --num_trials_per_task 50 --evaluation_config_path experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml  --unnorm_key libero_spatial --task_label oft_swap


# object
sed -i 's/use_swap: true/use_swap: false/' experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml
sed -i 's/use_object: false/use_object: true/' experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml


python experiments/robot/libero/run_libero_pro_eval.py \
  --pretrained_checkpoint Sylvest/openvla-7b-oft-finetuned-libero-plus-mixdata  \
  --task_suite_name libero_spatial --e_decoding False --save_video False \
  --task_label empty_true --cus_task empty --remove_wrap True \
  --num_trials_per_task 50 --evaluation_config_path experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml  --unnorm_key libero_spatial --task_label oft_object


# lan
sed -i 's/use_object: true/use_object: false/' experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml
sed -i 's/use_lan: false/use_lan: true/' experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml

python experiments/robot/libero/run_libero_pro_eval.py \
  --pretrained_checkpoint Sylvest/openvla-7b-oft-finetuned-libero-plus-mixdata  \
  --task_suite_name libero_spatial --e_decoding False --save_video False \
  --task_label empty_true --cus_task empty --remove_wrap True \
  --num_trials_per_task 50 --evaluation_config_path experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml  --unnorm_key libero_spatial --task_label oft_lan


# task
sed -i 's/use_lan: true/use_lan: false/' experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml
sed -i 's/use_task: false/use_task: true/' experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml

python experiments/robot/libero/run_libero_pro_eval.py \
  --pretrained_checkpoint Sylvest/openvla-7b-oft-finetuned-libero-plus-mixdata  \
  --task_suite_name libero_spatial --e_decoding False --save_video False \
  --task_label empty_true --cus_task empty --remove_wrap True \
  --num_trials_per_task 50 --evaluation_config_path experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml  --unnorm_key libero_spatial --task_label oft_task

sed -i 's/use_task: true/use_task: false/' experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml
