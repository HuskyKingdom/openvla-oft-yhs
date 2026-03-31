#!/bin/bash
# ORD Cluster Interactive Session Script for OpenVLA-APD Training
# Usage: bash scripts/ord_interactive.sh

# Cluster configuration
ACCOUNT="edgeai_tao-ptm_image-foundation-model-clip"
PARTITION="interactive_singlenode"
CACHE_DIR="/lustre/fsw/portfolios/edgeai/users/chrislin/cache"
WORKSPACE="/lustre/fsw/portfolios/edgeai/users/chrislin/projects/openvla-oft-yhs"

# Docker configuration
DOCKER_IMAGE="christianlin0420/openvla-apd:latest"

echo "========================================="
echo "ORD Interactive Session for OpenVLA-APD"
echo "========================================="
echo "Account: $ACCOUNT"
echo "Partition: $PARTITION"
echo "Docker Image: $DOCKER_IMAGE"
echo "Workspace: $WORKSPACE"
echo ""
echo "Requesting 1 GPU for 4 hours..."
echo ""

srun \
    --account=$ACCOUNT \
    --partition=$PARTITION \
    --job-name "openvla_apd_interactive" \
    --gpus 8 \
    --ntasks-per-node 8 \
    --time 04:00:00 \
    --container-image="$DOCKER_IMAGE" \
    --container-mounts=/lustre:/lustre,${WORKSPACE}:/workspace \
    --container-env=TORCH_HOME=${CACHE_DIR}/torch,HF_HOME=${CACHE_DIR}/huggingface,PYTHONNOUSERSITE=1,PYTHONPATH=/workspace \
    --pty /bin/bash

echo ""
echo "Interactive session ended."
