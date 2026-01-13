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
  --task_suite_name libero_spatial   --center_crop True --task_label ori_spatial_swap  --evaluation_config_path /home/yuhang/Warehouse/Yuhangworkspace/openvla-oft-yhs/experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml


python experiments/robot/libero/run_libero_eval.py   \
  --model_family openvla   --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-10   \
  --task_suite_name libero_10   --center_crop True --task_label ori_10_swap  --evaluation_config_path /home/yuhang/Warehouse/Yuhangworkspace/openvla-oft-yhs/experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml

# object
sed -i 's/use_swap: true/use_swap: false/' experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml
sed -i 's/use_object: false/use_object: true/' experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml


python experiments/robot/libero/run_libero_eval.py   \
  --model_family openvla   --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial   \
  --task_suite_name libero_spatial   --center_crop True --task_label ori_spatial_object  --evaluation_config_path /home/yuhang/Warehouse/Yuhangworkspace/openvla-oft-yhs/experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml


python experiments/robot/libero/run_libero_eval.py   \
  --model_family openvla   --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-10   \
  --task_suite_name libero_10   --center_crop True --task_label ori_10_object  --evaluation_config_path /home/yuhang/Warehouse/Yuhangworkspace/openvla-oft-yhs/experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml


# lan
sed -i 's/use_object: true/use_object: false/' experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml
sed -i 's/use_language: false/use_language: true/' experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml

python experiments/robot/libero/run_libero_eval.py   \
  --model_family openvla   --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial   \
  --task_suite_name libero_spatial   --center_crop True --task_label ori_spatial_lan  --evaluation_config_path /home/yuhang/Warehouse/Yuhangworkspace/openvla-oft-yhs/experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml


python experiments/robot/libero/run_libero_eval.py   \
  --model_family openvla   --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-10   \
  --task_suite_name libero_10   --center_crop True --task_label ori_10_lan  --evaluation_config_path /home/yuhang/Warehouse/Yuhangworkspace/openvla-oft-yhs/experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml

# task
sed -i 's/use_language: true/use_language: false/' experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml
sed -i 's/use_task: false/use_task: true/' experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml

python experiments/robot/libero/run_libero_eval.py   \
  --model_family openvla   --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial   \
  --task_suite_name libero_spatial   --center_crop True --task_label ori_spatial_task  --evaluation_config_path /home/yuhang/Warehouse/Yuhangworkspace/openvla-oft-yhs/experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml


python experiments/robot/libero/run_libero_eval.py   \
  --model_family openvla   --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-10   \
  --task_suite_name libero_10   --center_crop True --task_label ori_10_task  --evaluation_config_path /home/yuhang/Warehouse/Yuhangworkspace/openvla-oft-yhs/experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml


sed -i 's/use_task: true/use_task: false/' experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml
