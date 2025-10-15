#!/bin/bash

# without energy
echo "Running Evaluations Automatically ------------------------------"
FARM_USER=sgyson10



echo "Evaluating goal ------------------------------"
echo N | python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint /mnt/nfs/$FARM_USER/openvla-oft-yhs/ckpoints/pre-trained_universal \
    --task_suite_name libero_goal --e_decoding True --energy_k 1 --task_label energy_k1_goal


echo N | python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint /mnt/nfs/$FARM_USER/openvla-oft-yhs/ckpoints/pre-trained_universal \
    --task_suite_name libero_goal --e_decoding True --energy_k 2 --task_label energy_k2_goal


echo N | python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint /mnt/nfs/$FARM_USER/openvla-oft-yhs/ckpoints/pre-trained_universal \
    --task_suite_name libero_goal --e_decoding True --energy_k 3 --task_label energy_k3_goal


echo N | python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint /mnt/nfs/$FARM_USER/openvla-oft-yhs/ckpoints/pre-trained_universal \
    --task_suite_name libero_goal --e_decoding True --energy_k 4 --task_label energy_k4_goal



echo "Evaluating object ------------------------------"
echo N | python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint /mnt/nfs/$FARM_USER/openvla-oft-yhs/ckpoints/pre-trained_universal \
    --task_suite_name libero_object --e_decoding True --energy_k 1 --task_label energy_k1_object


echo N | python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint /mnt/nfs/$FARM_USER/openvla-oft-yhs/ckpoints/pre-trained_universal \
    --task_suite_name libero_object --e_decoding True --energy_k 2 --task_label energy_k2_object


echo N | python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint /mnt/nfs/$FARM_USER/openvla-oft-yhs/ckpoints/pre-trained_universal \
    --task_suite_name libero_object --e_decoding True --energy_k 3 --task_label energy_k3_object


echo N | python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint /mnt/nfs/$FARM_USER/openvla-oft-yhs/ckpoints/pre-trained_universal \
    --task_suite_name libero_object --e_decoding True --energy_k 4 --task_label energy_k4_object


