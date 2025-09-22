#!/bin/bash

# without energy
echo "Running Evaluations Automatically ------------------------------"
FARM_USER=sgyson10


echo "Evaluating spatial ------------------------------"
echo N | python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint /mnt/nfs/$FARM_USER/openvla-oft-yhs/ckpoints/pre-trained_universal \
    --task_suite_name libero_spatial --e_decoding True --task_label w_energy_spatial

echo "Evaluating object ------------------------------"
echo N | python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint /mnt/nfs/$FARM_USER/openvla-oft-yhs/ckpoints/pre-trained_universal \
    --task_suite_name libero_object --e_decoding True --task_label w_energy_object

echo "Evaluating goal ------------------------------"
echo N | python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint /mnt/nfs/$FARM_USER/openvla-oft-yhs/ckpoints/pre-trained_universal \
    --task_suite_name libero_goal --e_decoding True --task_label w_energy_goal

echo "Evaluating long ------------------------------"
echo N | python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint /mnt/nfs/$FARM_USER/openvla-oft-yhs/ckpoints/pre-trained_universal \
    --task_suite_name libero_10 --e_decoding True --task_label w_energy_10


