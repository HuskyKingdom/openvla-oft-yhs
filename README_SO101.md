# SO101 Poker Yellow Task - Quick Start Guide

Complete workflow for training and deploying OpenVLA-APD on real SO follower robot.

## Overview

This guide provides a quick reference for the entire pipeline:
1. Download real-robot dataset from HuggingFace
2. Convert to RLDS format
3. **Generate APD substep labels** (critical for APD training)
4. Train OpenVLA-APD model with substep supervision
5. Deploy to real robot via LeRobot

**Dataset**: `christian0420/so101-poker-yellow-task` (11 episodes, 5,455 frames)
**Robot**: SO Follower (6-DOF + gripper)
**Task**: Pick and place yellow poker chips

## Quick Start

### 1. Dataset Preparation (20-40 minutes)

```bash
# Launch interactive session
bash scripts/ord_interactive.sh

# Download dataset
python scripts/download_so101_dataset.py \
    --output_dir /lustre/fsw/portfolios/edgeai/users/chrislin/datasets/so101_raw \
    --verify

# Convert to RLDS format
python scripts/convert_lerobot_to_rlds.py \
    --input_dir /lustre/fsw/portfolios/edgeai/users/chrislin/datasets/so101_raw \
    --output_dir /lustre/fsw/portfolios/edgeai/users/chrislin/datasets/so101_rlds \
    --dataset_name so101_poker_yellow

# Generate APD substep labels (CRITICAL for APD training)
python scripts/label_substeps_so101.py \
    --apd_path APD_plans_so101.json \
    --rlds_data_dir /lustre/fsw/portfolios/edgeai/users/chrislin/datasets/so101_rlds \
    --output_path substep_labels_so101.json
```

**Expected Output**: `✓ RLDS dataset saved` with 11 episodes, `✓ Substep labels saved`

### 2. Training (24-30 hours)

```bash
# Update WandB credentials in scripts/ord_sbatch_so101.sh
# Then submit training job
mkdir -p logs
sbatch scripts/ord_sbatch_so101.sh

# Monitor training
tail -f logs/so101_training_<JOB_ID>.out

# Or check WandB dashboard
# https://wandb.ai/your-entity/openvla-so101-poker
```

**Checkpoints saved at**: 50k, 100k, 150k steps
**Final checkpoint**: `/lustre/fsw/portfolios/edgeai/users/chrislin/checkpoints/openvla-so101/`

### 3. Upload to HuggingFace (5-10 minutes)

```bash
# Upload trained checkpoint
python scripts/upload_checkpoint_to_hf.py \
    --checkpoint_dir /lustre/fsw/portfolios/edgeai/users/chrislin/checkpoints/openvla-so101/openvla-7b+so101_poker_yellow+...--150005_chkpt \
    --repo_id christian0420/openvla-so101-poker-yellow \
    --token hf_YOUR_TOKEN
```

**Model available at**: `https://huggingface.co/christian0420/openvla-so101-poker-yellow`

### 4. Deploy to Robot (Setup ~1 hour, then continuous use)

```bash
# On robot control computer
pip install lerobot transformers torch

# Download checkpoint
huggingface-cli download christian0420/openvla-so101-poker-yellow \
    --local-dir ./checkpoints/openvla-so101

# Configure robot (see deployment guide)
# Then run autonomous trials
python deploy_robot.py --checkpoint ./checkpoints/openvla-so101
```

See [Deployment Guide](docs/SO101_LEROBOT_DEPLOYMENT.md) for complete setup.

## File Structure

```
openvla-oft-yhs/
├── scripts/
│   ├── ord_interactive.sh              # Interactive GPU session
│   ├── ord_sbatch_so101.sh             # Batch training job (APD w/ substeps)
│   ├── download_so101_dataset.py       # Download from HuggingFace
│   ├── convert_lerobot_to_rlds.py      # LeRobot → RLDS conversion
│   ├── label_substeps_so101.py         # APD substep labeling
│   └── upload_checkpoint_to_hf.py      # Upload to HuggingFace
├── docs/
│   ├── SO101_TRAINING_GUIDE.md         # Detailed training guide
│   └── SO101_LEROBOT_DEPLOYMENT.md     # Detailed deployment guide
├── APD_plans_so101.json                # APD task decomposition
└── README_SO101.md                     # This file
```

## Key Configuration

### Training Parameters
```yaml
Model: openvla/openvla-7b
Training Mode: APD (Actor-Policy Distillation) with substep labels
LoRA Rank: 32
Batch Size: 8 × 8 GPUs = 64
Learning Rate: 5e-4
Max Steps: 150,005
Cameras: 2 (top + wrist)
Proprio: Enabled (6D state)
Substep Supervision: Yes (move/pick/place decomposition)
```

**What are APD Substeps?**
APD decomposes tasks into fine-grained substeps (move → pick → move → place), allowing the model to learn more granular control with different language instructions for each phase. This improves performance on complex manipulation tasks.

### Dataset Specifications
```yaml
Episodes: 11
Total Frames: 5,455
FPS: 30
Action Dim: 6 (joints + gripper)
Proprio Dim: 6 (joint positions)
Top Camera: 1920×1080
Wrist Camera: 1280×720
```

## Detailed Guides

For more information, see the comprehensive guides:

- **[Training Guide](docs/SO101_TRAINING_GUIDE.md)**: Complete training workflow
  - Dataset preparation details
  - Training configuration options
  - Monitoring and troubleshooting
  - Performance expectations

- **[Deployment Guide](docs/SO101_LEROBOT_DEPLOYMENT.md)**: Real robot deployment
  - Hardware setup and calibration
  - Policy configuration
  - Safety protocols
  - Performance tuning

## Common Commands

### Check Training Progress
```bash
# Job status
squeue -u $USER

# Live logs
tail -f logs/so101_training_<JOB_ID>.out

# Check GPU usage
srun --account=edgeai_tao-ptm_image-foundation-model-clip --partition=interactive --gpus=1 --pty nvidia-smi
```

### Test Checkpoint
```python
from transformers import AutoModelForVision2Seq, AutoProcessor

checkpoint = "/path/to/checkpoint"
model = AutoModelForVision2Seq.from_pretrained(checkpoint, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
print("✓ Checkpoint loaded successfully!")
```

### Verify Dataset
```bash
python -c "
from prismatic.vla.datasets import RLDSDataset
dataset = RLDSDataset(
    '/lustre/fsw/portfolios/edgeai/users/chrislin/datasets/so101_rlds',
    'so101_poker_yellow',
    None,
    (224, 224),
)
print(f'Dataset loaded: {len(dataset)} samples')
"
```

## Troubleshooting

### Dataset Issues
- **Download fails**: Check HuggingFace token, network connection
- **Conversion fails**: Verify input directory has downloaded data
- **Wrong format**: Ensure LeRobot dataset, not RLDS

### Training Issues
- **OOM errors**: Reduce batch_size in `scripts/ord_sbatch_so101.sh`
- **Job won't start**: Check partition availability with `sinfo`
- **Loss is NaN**: Check dataset statistics, reduce learning rate

### Deployment Issues
- **Checkpoint won't load**: Verify all files present, check CUDA version
- **Actions too aggressive**: Reduce `max_action_rate` in robot config
- **Cameras not working**: Check USB connections, verify calibration

See detailed troubleshooting in the [Training Guide](docs/SO101_TRAINING_GUIDE.md#troubleshooting) and [Deployment Guide](docs/SO101_LEROBOT_DEPLOYMENT.md#troubleshooting).

## Performance Expectations

### Training
- **Time**: 24-30 hours (8× A100 GPUs)
- **Loss**: Should converge to ~0.05
- **L1 Loss**: Target < 0.1

### Deployment
- **Inference**: 10-15 Hz on RTX 4090
- **Success Rate**: TBD (requires real robot evaluation)
- **Latency**: <100ms per action

## Citation

If you use this work, please cite:

```bibtex
@misc{openvla-so101-2024,
  title={OpenVLA Fine-tuned on SO101 Poker Yellow Task},
  author={Christian Lin},
  year={2024},
  publisher={HuggingFace},
  url={https://huggingface.co/christian0420/openvla-so101-poker-yellow}
}
```

## Resources

- **HuggingFace Model**: https://huggingface.co/christian0420/openvla-so101-poker-yellow
- **HuggingFace Dataset**: https://huggingface.co/datasets/christian0420/so101-poker-yellow-task
- **OpenVLA**: https://github.com/openvla/openvla
- **LeRobot**: https://github.com/huggingface/lerobot
- **APD Framework**: https://github.com/APD-VLA

## Support

For questions or issues:
1. Check troubleshooting sections in detailed guides
2. Review WandB logs for training issues
3. Test with interactive session first
4. Contact: christian0420 on HuggingFace

---

**Quick Links**:
- [Training Guide](docs/SO101_TRAINING_GUIDE.md)
- [Deployment Guide](docs/SO101_LEROBOT_DEPLOYMENT.md)
- [Scripts](scripts/)
- [Checkpoints on HuggingFace](https://huggingface.co/christian0420/openvla-so101-poker-yellow)
