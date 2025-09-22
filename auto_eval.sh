#!/bin/bash

# without energy
echo "Setting env... ------------------------------"
FARM_USER=sgyson10

echo N | python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint /mnt/nfs/$FARM_USER/openvla-oft-yhs/ckpoints/pre-trained_universal \
    --task_suite_name libero_10 --e_decoding False --task_label wo_energy_spatial

