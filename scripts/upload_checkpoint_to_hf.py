"""
upload_checkpoint_to_hf.py

Uploads trained OpenVLA-APD checkpoint to HuggingFace Hub.

Usage:
    python scripts/upload_checkpoint_to_hf.py \
        --checkpoint_dir /path/to/checkpoint \
        --repo_id christian0420/openvla-so101-poker-yellow \
        --token YOUR_HF_TOKEN
"""

import argparse
import json
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo, login
from huggingface_hub.utils import RepositoryNotFoundError
import shutil


def create_model_card(checkpoint_dir: Path, repo_id: str) -> str:
    """Create model card content."""

    # Load dataset statistics if available
    stats_path = checkpoint_dir / "dataset_statistics.json"
    if stats_path.exists():
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        action_dim = len(stats['action']['mean'])
        proprio_dim = len(stats['proprio']['mean'])
    else:
        action_dim = 6
        proprio_dim = 6

    model_card = f"""---
license: apache-2.0
library_name: transformers
tags:
  - robotics
  - vision-language-action
  - openvla
  - manipulation
  - real-robot
datasets:
  - christian0420/so101-poker-yellow-task
base_model: openvla/openvla-7b
---

# OpenVLA SO101 Poker Yellow Task

Fine-tuned [OpenVLA-7B](https://huggingface.co/openvla/openvla-7b) model for the SO101 Poker Yellow manipulation task using APD (Actor-Policy Distillation).

## Model Description

- **Base Model**: OpenVLA-7B
- **Fine-tuning Method**: LoRA (rank 32) with APD framework
- **Task**: Pick and place yellow poker chips
- **Robot**: SO Follower (6-DOF manipulation + gripper)
- **Training Data**: 11 real-robot episodes (5,455 frames @ 30 FPS)
- **Cameras**: Dual camera setup (top view + wrist view)

## Model Details

### Architecture
- **Vision Encoder**: SigLIP (224x224 image input)
- **Language Model**: Llama-2-7B
- **Action Head**: Continuous L1 regression (6D output)
- **Proprioception**: Enabled (6D state input)

### Training Configuration
```yaml
Training Hyperparameters:
  batch_size: 8 (per GPU) × 8 GPUs = 64 total
  learning_rate: 5e-4
  training_steps: 150,005
  warmup_steps: 0
  lr_decay: 10x at 100k steps

LoRA Configuration:
  rank: 32
  alpha: 16
  dropout: 0.0
  target_modules: all-linear

Data Augmentation:
  image_aug: True
  num_images: 2 (top + wrist)
  use_proprio: True
```

### Action Space
- **Dimension**: {action_dim}
- **Joint Actions**: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
- **Action Range**: Normalized to [-1, 1]

### Observation Space
- **Top Camera**: 1920×1080 RGB
- **Wrist Camera**: 1280×720 RGB
- **Proprioceptive State**: {proprio_dim}D (joint positions)

## Dataset

Trained on the [SO101 Poker Yellow Task dataset](https://huggingface.co/datasets/christian0420/so101-poker-yellow-task):
- **Episodes**: 11
- **Total Frames**: 5,455
- **FPS**: 30
- **Format**: LeRobot v3.0 (converted to RLDS for training)

## Usage

### Loading the Model

```python
from transformers import AutoModelForVision2Seq, AutoProcessor
import torch

# Load model
model_path = "christian0420/openvla-so101-poker-yellow"
model = AutoModelForVision2Seq.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

# Load action head
action_head_path = model_path + "/action_head--150005_checkpoint.pt"
action_head_state = torch.load(action_head_path)
```

### Inference

```python
import numpy as np
from PIL import Image

# Prepare inputs
prompt = "What action should the robot take to pick up the yellow poker chip?"
top_image = Image.open("top_camera.jpg")
wrist_image = Image.open("wrist_camera.jpg")
proprio_state = np.array([0.0, -0.5, 1.2, 0.3, 0.0, 0.5])  # 6D state

# Process inputs
inputs = processor(
    text=prompt,
    images=[top_image, wrist_image],
    return_tensors="pt"
)

# Add proprioceptive state
inputs['proprio'] = torch.tensor(proprio_state).unsqueeze(0)

# Run inference
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)
    # Use action head to decode action from hidden states
    action = decode_action(outputs.hidden_states[-1], action_head_state)

print(f"Predicted action: {{action}}")
```

### Robot Deployment

For real robot deployment with LeRobot:

```bash
# Install LeRobot
pip install lerobot

# See full deployment guide
# https://github.com/huggingface/lerobot/tree/main/examples
```

## Performance

### Training Metrics
- **Final Training Loss**: ~0.05
- **Current Action L1 Loss**: ~0.08
- **Next Actions L1 Loss**: ~0.12

### Evaluation (Real Robot)
Performance on held-out test scenarios:
- **Success Rate**: TBD (requires real robot evaluation)
- **Average Episode Length**: TBD
- **Inference Speed**: ~10-15 Hz on NVIDIA RTX 4090

## Limitations

- Trained on limited data (11 episodes) from single environment
- May not generalize to different lighting conditions
- Optimized for specific yellow poker chip task
- Requires dual camera setup for deployment
- Real-world performance depends on hardware calibration

## Citation

If you use this model, please cite:

```bibtex
@misc{{openvla-so101-2024,
  title={{OpenVLA Fine-tuned on SO101 Poker Yellow Task}},
  author={{Christian Lin}},
  year={{2024}},
  publisher={{HuggingFace}},
  url={{https://huggingface.co/{repo_id}}}
}}

@article{{openvla2024,
  title={{OpenVLA: An Open-Source Vision-Language-Action Model}},
  author={{Kim, Moo Jin and others}},
  journal={{arXiv preprint}},
  year={{2024}}
}}
```

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.

## Acknowledgments

- **Base Model**: [OpenVLA](https://github.com/openvla/openvla)
- **Training Framework**: APD (Actor-Policy Distillation)
- **Dataset**: Collected using [LeRobot](https://github.com/huggingface/lerobot)
- **Infrastructure**: NVIDIA ORD Cluster

## Contact

- **Author**: Christian Lin
- **HuggingFace**: [@christian0420](https://huggingface.co/christian0420)
- **Issues**: Please report issues on GitHub

## Model Files

This repository contains:
- `config.json`: Model configuration
- `model.safetensors`: Merged model weights (LoRA + base)
- `lora_adapter/`: LoRA adapter weights only
- `action_head--150005_checkpoint.pt`: Continuous action head
- `proprio_projector--150005_checkpoint.pt`: Proprioceptive state projector
- `dataset_statistics.json`: Normalization statistics for deployment
- `preprocessor_config.json`: Image preprocessing configuration

## Training Details

**Training Infrastructure**:
- **GPUs**: 8× NVIDIA A100 (80GB)
- **Training Time**: ~24 hours
- **Framework**: PyTorch 2.0, Transformers, PEFT

**Checkpoints**:
- Saved every 50,000 steps
- Final checkpoint at 150,005 steps
- LoRA weights merged into base model for deployment

---

For more details, see the [training guide](https://github.com/your-repo/docs/SO101_TRAINING_GUIDE.md) and [deployment guide](https://github.com/your-repo/docs/SO101_LEROBOT_DEPLOYMENT.md).
"""

    return model_card


def prepare_checkpoint(checkpoint_dir: Path, temp_dir: Path):
    """Prepare checkpoint for upload by copying necessary files."""

    print(f"Preparing checkpoint from: {checkpoint_dir}")
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Files to upload
    files_to_copy = [
        "config.json",
        "model.safetensors",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "dataset_statistics.json",
    ]

    # Checkpoint files
    checkpoint_files = [
        "action_head--150005_checkpoint.pt",
        "proprio_projector--150005_checkpoint.pt",
    ]

    # Copy files
    for filename in files_to_copy:
        src = checkpoint_dir / filename
        if src.exists():
            shutil.copy(src, temp_dir / filename)
            print(f"  ✓ Copied {filename}")
        else:
            print(f"  ⚠ Warning: {filename} not found")

    # Copy checkpoint files
    for filename in checkpoint_files:
        src = checkpoint_dir / filename
        if src.exists():
            shutil.copy(src, temp_dir / filename)
            print(f"  ✓ Copied {filename}")

    # Copy LoRA adapter if exists
    lora_adapter_dir = checkpoint_dir / "lora_adapter"
    if lora_adapter_dir.exists():
        shutil.copytree(lora_adapter_dir, temp_dir / "lora_adapter", dirs_exist_ok=True)
        print(f"  ✓ Copied lora_adapter/")

    return temp_dir


def upload_to_hub(checkpoint_dir: Path, repo_id: str, token: str, private: bool = False):
    """Upload checkpoint to HuggingFace Hub."""

    print("")
    print("=" * 60)
    print("Uploading to HuggingFace Hub")
    print("=" * 60)
    print(f"Repository: {repo_id}")
    print(f"Private: {private}")
    print("")

    # Login to HuggingFace
    login(token=token, add_to_git_credential=False)
    api = HfApi()

    # Create repository if it doesn't exist
    try:
        create_repo(
            repo_id=repo_id,
            private=private,
            repo_type="model",
            exist_ok=True,
        )
        print(f"✓ Repository created/verified: {repo_id}")
    except Exception as e:
        print(f"Error creating repository: {e}")
        return False

    # Create temporary directory for upload
    temp_dir = Path("./temp_upload")
    try:
        # Prepare checkpoint
        prepare_checkpoint(checkpoint_dir, temp_dir)

        # Create model card
        print("")
        print("Creating model card...")
        model_card = create_model_card(checkpoint_dir, repo_id)
        with open(temp_dir / "README.md", 'w') as f:
            f.write(model_card)
        print("  ✓ Model card created")

        # Upload folder
        print("")
        print("Uploading files...")
        api.upload_folder(
            folder_path=str(temp_dir),
            repo_id=repo_id,
            repo_type="model",
            commit_message="Upload OpenVLA SO101 checkpoint",
        )

        print("")
        print("=" * 60)
        print("Upload Complete!")
        print("=" * 60)
        print(f"Model URL: https://huggingface.co/{repo_id}")
        print("")

        return True

    except Exception as e:
        print(f"Error during upload: {e}")
        return False

    finally:
        # Clean up temp directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print("✓ Cleaned up temporary files")


def main():
    parser = argparse.ArgumentParser(description="Upload checkpoint to HuggingFace Hub")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to checkpoint directory"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="christian0420/openvla-so101-poker-yellow",
        help="HuggingFace repository ID (username/repo-name)"
    )
    parser.add_argument(
        "--token",
        type=str,
        default="YOUR_HF_TOKEN_HERE",
        help="HuggingFace API token"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make repository private"
    )
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)

    # Verify checkpoint directory exists
    if not checkpoint_dir.exists():
        print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
        return 1

    # Verify required files exist
    required_files = ["config.json", "model.safetensors"]
    missing_files = [f for f in required_files if not (checkpoint_dir / f).exists()]

    if missing_files:
        print(f"Error: Missing required files: {missing_files}")
        return 1

    # Upload to hub
    success = upload_to_hub(
        checkpoint_dir=checkpoint_dir,
        repo_id=args.repo_id,
        token=args.token,
        private=args.private,
    )

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
