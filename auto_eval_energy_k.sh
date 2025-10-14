#!/bin/bash

# without energy
echo "Running Evaluations Automatically ------------------------------"
FARM_USER=sgyson10


echo "Evaluating long ------------------------------"
echo N | python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint /mnt/nfs/$FARM_USER/openvla-oft-yhs/ckpoints/pre-trained_universal \
    --task_suite_name libero_long --e_decoding True --energy_k 1 --task_label energy_k1_long


echo N | python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint /mnt/nfs/$FARM_USER/openvla-oft-yhs/ckpoints/pre-trained_universal \
    --task_suite_name libero_long --e_decoding True --energy_k 2 --task_label energy_k2_long


echo N | python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint /mnt/nfs/$FARM_USER/openvla-oft-yhs/ckpoints/pre-trained_universal \
    --task_suite_name libero_long --e_decoding True --energy_k 3 --task_label energy_k3_long


echo N | python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint /mnt/nfs/$FARM_USER/openvla-oft-yhs/ckpoints/pre-trained_universal \
    --task_suite_name libero_long --e_decoding True --energy_k 4 --task_label energy_k4_long




# echo "Evaluating spatial ------------------------------"
# echo N | python experiments/robot/libero/run_libero_eval.py \
#     --pretrained_checkpoint /mnt/nfs/$FARM_USER/openvla-oft-yhs/ckpoints/pre-trained_universal \
#     --task_suite_name libero_spatial --e_decoding True --energy_k 1 --task_label energy_k1_spatial


# echo N | python experiments/robot/libero/run_libero_eval.py \
#     --pretrained_checkpoint /mnt/nfs/$FARM_USER/openvla-oft-yhs/ckpoints/pre-trained_universal \
#     --task_suite_name libero_spatial --e_decoding True --energy_k 2 --task_label energy_k2_spatial


# echo N | python experiments/robot/libero/run_libero_eval.py \
#     --pretrained_checkpoint /mnt/nfs/$FARM_USER/openvla-oft-yhs/ckpoints/pre-trained_universal \
#     --task_suite_name libero_spatial --e_decoding True --energy_k 3 --task_label energy_k3_spatial


# echo N | python experiments/robot/libero/run_libero_eval.py \
#     --pretrained_checkpoint /mnt/nfs/$FARM_USER/openvla-oft-yhs/ckpoints/pre-trained_universal \
#     --task_suite_name libero_spatial --e_decoding True --energy_k 4 --task_label energy_k4_spatial

