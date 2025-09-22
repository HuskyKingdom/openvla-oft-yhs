#!/bin/bash

# without energy
echo "Setting env... ------------------------------"
FARM_USER=sgyson10
pip uninstall -y opencv-python; \
pip install opencv-python-headless==4.11.0.86; \
apt-get update && apt-get install -y \
libegl1-mesa-dev libgles2-mesa-dev \
libosmesa6 libosmesa6-dev \
libgl1-mesa-glx; \
export MUJOCO_GL=osmesa; \
export PYOPENGL_PLATFORM=osmesa; \
export NUMBA_DISABLE_JIT=1; \
git config --global --add safe.directory /mnt/nfs/$FARM_USER/openvla-oft-yhs; \
cd /mnt/nfs/$FARM_USER/openvla-oft-yhs && git pull; \
cd /mnt/nfs/$FARM_USER/openvla-oft-yhs && pip install -e .; \



cd /mnt/nfs/$FARM_USER/openvla-oft-yhs && echo N | python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint /mnt/nfs/$FARM_USER/openvla-oft-yhs/ckpoints/pre-trained_universal \
    --task_suite_name libero_10 --e_decoding False --task_label wo_energy_spatial

