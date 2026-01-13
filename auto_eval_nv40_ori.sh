#!/bin/bash


echo "Running Evaluations Automatically ------------------------------"
FARM_USER=sgyson10
FILE_PATH="experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml"




echo "Evaluating OpenVLA ------------------------------"


# swap
sed -i 's/use_environment: true/use_environment: false/' experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml
sed -i 's/use_swap: false/use_swap: true/' experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml


python experiments/robot/libero/run_libero_eval.py   \
  --model_family openvla   --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial   \
  --task_suite_name libero_spatial   --center_crop True --task_label ori_spatial_swap 


python experiments/robot/libero/run_libero_eval.py   \
  --model_family openvla   --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-10   \
  --task_suite_name libero_10   --center_crop True --task_label ori_10_swap 

# object
sed -i 's/use_swap: true/use_swap: false/' experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml
sed -i 's/use_object: false/use_object: true/' experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml


python experiments/robot/libero/run_libero_eval.py   \
  --model_family openvla   --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial   \
  --task_suite_name libero_spatial   --center_crop True --task_label ori_spatial_object 


python experiments/robot/libero/run_libero_eval.py   \
  --model_family openvla   --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-10   \
  --task_suite_name libero_10   --center_crop True --task_label ori_10_object 


# lan
sed -i 's/use_object: true/use_object: false/' experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml
sed -i 's/use_language: false/use_language: true/' experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml

python experiments/robot/libero/run_libero_eval.py   \
  --model_family openvla   --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial   \
  --task_suite_name libero_spatial   --center_crop True --task_label ori_spatial_lan 


python experiments/robot/libero/run_libero_eval.py   \
  --model_family openvla   --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-10   \
  --task_suite_name libero_10   --center_crop True --task_label ori_10_lan 

# task
sed -i 's/use_language: true/use_language: false/' experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml
sed -i 's/use_task: false/use_task: true/' experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml

python experiments/robot/libero/run_libero_eval.py   \
  --model_family openvla   --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial   \
  --task_suite_name libero_spatial   --center_crop True --task_label ori_spatial_task 


python experiments/robot/libero/run_libero_eval.py   \
  --model_family openvla   --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-10   \
  --task_suite_name libero_10   --center_crop True --task_label ori_10_task 


sed -i 's/use_task: true/use_task: false/' experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml
