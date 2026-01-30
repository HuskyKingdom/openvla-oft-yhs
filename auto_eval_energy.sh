#!/bin/bash

# without energy
echo "Running Evaluations Automatically ------------------------------"
FARM_USER=sgyson10


echo "Evaluating spatial ------------------------------"
echo N | python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10 \
    --task_suite_name libero_spatial --e_decoding True --task_label w_energy_spatial_0.3_scale \
    --energy_path /home/yuhang/Warehouse/Yuhangworkspace/openvla-oft-yhs/ckpt/ckpoints/energy_scale_100k/energy_model--100000_checkpoint.pt

echo N | python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10 \
    --task_suite_name libero_spatial --e_decoding True --task_label w_energy_spatial_0.3_vision_only \
    --energy_path /home/yuhang/Warehouse/Yuhangworkspace/openvla-oft-yhs/ckpt/ckpoints/energy_vision_only/energy_model--100000_checkpoint.pt

echo "Evaluating object ------------------------------"
echo N | python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10 \
    --task_suite_name libero_object --e_decoding True --task_label w_energy_object_0.3_scale \
    --energy_path /home/yuhang/Warehouse/Yuhangworkspace/openvla-oft-yhs/ckpt/ckpoints/energy_scale_100k/energy_model--100000_checkpoint.pt
echo N | python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10 \
    --task_suite_name libero_object --e_decoding True --task_label w_energy_object_0.3_vision_only \
    --energy_path /home/yuhang/Warehouse/Yuhangworkspace/openvla-oft-yhs/ckpt/ckpoints/energy_vision_only/energy_model--100000_checkpoint.pt

echo "Evaluating goal ------------------------------"
echo N | python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10 \
    --task_suite_name libero_goal --e_decoding True --task_label w_energy_goal_0.3_scale \
    --energy_path /home/yuhang/Warehouse/Yuhangworkspace/openvla-oft-yhs/ckpt/ckpoints/energy_scale_100k/energy_model--100000_checkpoint.pt

echo N | python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10 \
    --task_suite_name libero_goal --e_decoding True --task_label w_energy_goal_0.3_vision_only \
    --energy_path /home/yuhang/Warehouse/Yuhangworkspace/openvla-oft-yhs/ckpt/ckpoints/energy_vision_only/energy_model--100000_checkpoint.pt
echo "Evaluating long ------------------------------"
echo N | python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10 \
    --task_suite_name libero_10 --e_decoding True --task_label w_energy_10_0.3_scale \
    --energy_path /home/yuhang/Warehouse/Yuhangworkspace/openvla-oft-yhs/ckpt/ckpoints/energy_scale_100k/energy_model--100000_checkpoint.pt
echo N | python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10 \
    --task_suite_name libero_10 --e_decoding True --task_label w_energy_10_0.3_vision_only \
    --energy_path /home/yuhang/Warehouse/Yuhangworkspace/openvla-oft-yhs/ckpt/ckpoints/energy_vision_only/energy_model--100000_checkpoint.pt

