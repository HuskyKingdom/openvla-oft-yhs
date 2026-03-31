# SO101 Poker Yellow Task — Complete Pipeline Guide

End-to-end reference for training and deploying OpenVLA-APD on the SO101 real-robot dataset,
covering data collection through robot deployment.

**Dataset**: `christian0420/so101-poker-yellow-task` (11 episodes, 5,455 frames @ 30 FPS)
**Robot**: SO Follower (6-DOF arm + gripper)
**Task**: Dual pick-and-place — poker box then yellow box into white box
**Model**: OpenVLA-7B fine-tuned with APD (Actor-Policy Distillation) and substep supervision

---

## Table of Contents

1. [Overview](#1-overview)
2. [Prerequisites](#2-prerequisites)
3. [Data Collection with LeRobot](#3-data-collection-with-lerobot)
4. [Uploading Dataset to HuggingFace](#4-uploading-dataset-to-huggingface)
5. [Downloading the Dataset](#5-downloading-the-dataset)
6. [Data Conversion: LeRobot → RLDS](#6-data-conversion-lerobot--rlds)
7. [Conversion Validation](#7-conversion-validation)
8. [APD Substep Labeling](#8-apd-substep-labeling)
9. [Training](#9-training)
10. [Checkpoint Upload](#10-checkpoint-upload)
11. [Evaluation and Robot Deployment](#11-evaluation-and-robot-deployment)
12. [Troubleshooting](#12-troubleshooting)
13. [Performance Reference](#13-performance-reference)

---

## 1. Overview

### Pipeline at a Glance

```
LeRobot Data Collection
        ↓
HuggingFace Dataset Upload
        ↓
Download (download_so101_dataset.py)
        ↓
LeRobot → RLDS Conversion (convert_lerobot_to_rlds.py)
        ↓
Conversion Validation
        ↓
APD Substep Labeling (label_substeps_so101.py)
        ↓  [optional]
Substep Visualization (visualize_substeps_so101.py)
        ↓
Training (vla-scripts/finetune_substep.py via ord_sbatch_so101.sh)
        ↓
Checkpoint Upload (upload_checkpoint_to_hf.py)
        ↓
Robot Deployment via LeRobot
```

### Dataset Specifications

| Property | Value |
|----------|-------|
| Episodes | 11 |
| Total Frames | 5,455 |
| FPS | 30 |
| Action Dimension | 6 (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper) |
| Proprio Dimension | 6 (joint positions) |
| Top Camera | 1920×1080 RGB |
| Wrist Camera | 1280×720 RGB |
| Format | LeRobot v3.0 (Parquet + MP4) |

### Key Configuration

```yaml
Model: openvla/openvla-7b
Training Mode: APD with substep supervision
LoRA Rank: 32
Batch Size: 8 × 8 GPUs = 64
Learning Rate: 5e-4
Max Steps: 150,005
Cameras: 2 (top + wrist)
Proprio: Enabled (6D state)
Substep Supervision: Yes (move/pick/place decomposition)
```

### What Are APD Substeps?

APD (Actor-Policy Distillation) decomposes tasks into fine-grained substeps
(move → pick → move → place), training the model with a different language instruction
for each phase. For the dual pick-place task, this gives 8 substeps across 2 cycles:

```
Cycle 1 — Poker Box:
  t=0+      "Move gripper above the poker box"         (approach)
  t=pick1   "Pick up the poker box"                    (grasp + lift)
  t=...     "Move the poker box to the white box"      (carry, merged into place block)
  t=place1  "Place the poker box in the white box"     (release + retract)

Cycle 2 — Yellow Box:
  t=...     "Move gripper above the yellow box"
  t=pick2   "Pick up the yellow box"
  t=...     "Move the yellow box to the white box"
  t=place2  "Place the yellow box in the white box"
```

---

## 2. Prerequisites

### Cluster Access

- ORD cluster with SLURM
- Account: `edgeai_tao-ptm_image-foundation-model-clip`
- Partition: `batch` (training) or `interactive` (testing)

### Storage Paths

```bash
WORKSPACE=/lustre/fsw/portfolios/edgeai/users/chrislin/projects/openvla-oft-yhs
DATA_DIR=/lustre/fsw/portfolios/edgeai/users/chrislin/datasets
CHECKPOINT_DIR=/lustre/fsw/portfolios/edgeai/users/chrislin/checkpoints
CACHE_DIR=/lustre/fsw/portfolios/edgeai/users/chrislin/cache
```

### Docker Image

Image: `christianlin0420/openvla-apd:latest` (pre-built, available on ORD cluster)

### Hardware (for robot deployment)

- SO Follower robot arm (operational, calibrated)
- Top camera: 1920×1080, 30 FPS
- Wrist camera: 1280×720, 30 FPS
- Control computer with NVIDIA GPU (8 GB+ VRAM)

### Software (for robot deployment)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install lerobot
pip install transformers accelerate peft
pip install opencv-python pyrealsense2
```

---

## 3. Data Collection with LeRobot

### Install LeRobot

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e .
python -c "import lerobot; print(lerobot.__version__)"
```

### Configure Robot

Create `configs/robot/so_follower.yaml`:

```yaml
robot_type: manipulator

motors:
  shoulder_pan:   {index: 0, drive_mode: position}
  shoulder_lift:  {index: 1, drive_mode: position}
  elbow_flex:     {index: 2, drive_mode: position}
  wrist_flex:     {index: 3, drive_mode: position}
  wrist_roll:     {index: 4, drive_mode: position}
  gripper:        {index: 5, drive_mode: position}

cameras:
  top:    {index: 0, resolution: [1920, 1080], fps: 30}
  wrist:  {index: 1, resolution: [1280, 720],  fps: 30}

control_frequency: 10  # Hz
max_action_rate: 0.5   # rad/s

position_limits:
  shoulder_pan:  [-3.14, 3.14]
  shoulder_lift: [-2.0, 2.0]
  elbow_flex:    [-2.5, 2.5]
  wrist_flex:    [-2.0, 2.0]
  wrist_roll:    [-3.14, 3.14]
  gripper:       [0.0, 1.0]    # 0=open, 1=closed
```

### Record Episodes

Use the LeRobot `record` command (teleoperation) to collect demonstrations:

```bash
# Record a new episode
python lerobot/scripts/control_robot.py record \
    --robot-path configs/robot/so_follower.yaml \
    --fps 30 \
    --repo-id christian0420/so101-poker-yellow-task \
    --tags so101 poker real_robot \
    --warmup-time-s 5 \
    --episode-time-s 30 \
    --reset-time-s 10 \
    --num-episodes 11
```

**Expected structure after recording**:

```
so101_raw/
├── meta/
│   ├── info.json                                       # fps, feature specs
│   ├── episodes/chunk-000/episode_chunk-000.parquet    # per-episode metadata
│   └── tasks.jsonl
├── data/chunk-000/
│   └── episode_000000.parquet  ...                     # frame-level data
└── videos/
    ├── observation.images.top/chunk-000/file-000.mp4
    └── observation.images.wrist/chunk-000/file-000.mp4
```

---

## 4. Uploading Dataset to HuggingFace

After recording, push the dataset to HuggingFace for sharing and reproducibility:

```bash
# Push LeRobot dataset to HuggingFace Hub
huggingface-cli upload christian0420/so101-poker-yellow-task \
    /path/to/so101_raw \
    --repo-type dataset
```

Or use the Python API:

```python
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="/path/to/so101_raw",
    repo_id="christian0420/so101-poker-yellow-task",
    repo_type="dataset",
)
```

---

## 5. Downloading the Dataset

Use `scripts/download_so101_dataset.py` to download from HuggingFace:

```bash
# Launch interactive container session
bash scripts/ord_interactive.sh

# Download dataset
python scripts/download_so101_dataset.py \
    --output_dir /lustre/fsw/portfolios/edgeai/users/chrislin/datasets/so101_raw \
    --verify
```

**Arguments**:

| Argument | Default | Description |
|----------|---------|-------------|
| `--output_dir` | `.../so101_raw` | Local output directory |
| `--verify` | False | Verify integrity after download |

**Expected output**:

```
Dataset: christian0420/so101-poker-yellow-task
Total samples: 5455
Episodes: 11
✓ Sample count matches expected: 5455 frames
```

**Verified sample structure**:

```
action                  shape=(6,)   dtype=float32
observation.state       shape=(6,)   dtype=float32
observation.images.top  1080×1920×3  uint8
observation.images.wrist 720×1280×3  uint8
timestamp, frame_index, episode_index
```

> **Note**: `scripts/download_so101_dataset.py` uses a hardcoded HuggingFace token.
> Replace it with your own token via `HF_TOKEN` in the script or by running
> `huggingface-cli login` before executing.

---

## 6. Data Conversion: LeRobot → RLDS

APD training requires RLDS/TFRecord format. Use `scripts/convert_lerobot_to_rlds.py`:

```bash
# Standard conversion (OpenCV for video decoding)
python scripts/convert_lerobot_to_rlds.py \
    --input_dir  /lustre/fsw/portfolios/edgeai/users/chrislin/datasets/so101_raw \
    --output_dir /lustre/fsw/portfolios/edgeai/users/chrislin/datasets/so101_rlds \
    --dataset_name so101_poker_yellow

# If OpenCV cannot read AV1 codec (empty or corrupt frames), use PyAV:
python scripts/convert_lerobot_to_rlds.py \
    --input_dir  /lustre/fsw/portfolios/edgeai/users/chrislin/datasets/so101_raw \
    --output_dir /lustre/fsw/portfolios/edgeai/users/chrislin/datasets/so101_rlds \
    --dataset_name so101_poker_yellow \
    --use_pyav_video
```

**All arguments**:

| Argument | Default | Description |
|----------|---------|-------------|
| `--input_dir` | `.../so101_raw` | LeRobot dataset root (meta/ data/ videos/) |
| `--repo_id` | `christian0420/so101-poker-yellow-task` | HF repo to download if meta/info.json missing |
| `--output_dir` | `.../so101_rlds` | Output directory for TFRecord |
| `--dataset_name` | `so101_poker_yellow` | Base name for output TFRecord file |
| `--default_instruction` | `"Pick up the yellow poker chip..."` | Fallback if episode metadata has no task text |
| `--use_pyav_video` | False | Force PyAV decoder (AV1 compatibility) |

**What the script does**:

1. Reads `meta/info.json` to get FPS and feature specs; downloads from HF if missing
2. Loads all frame rows from `data/**/*.parquet`
3. Groups rows by `episode_index`
4. For each episode: decodes top + wrist camera frames from the corresponding MP4 using
   frame timestamps (`from_timestamp + frame_index / fps`)
5. Computes action and proprio statistics (mean, std, min, max, q01, q99)
6. Writes one TFRecord file with all timesteps in order, setting `is_last=True` on the
   final step of each episode

**Expected output**:

```
so101_rlds/
├── so101_poker_yellow.tfrecord   # all 5455 timesteps, episodes delimited by is_last
└── dataset_statistics.json       # action + proprio normalization stats
```

```
Found 11 episodes
Action dimension: 6
Proprio dimension: 6
RLDS dataset saved to: .../so101_poker_yellow.tfrecord
Dataset statistics saved to: .../dataset_statistics.json
```

### TFRecord Schema

Each timestep is one TFRecord Example with these features:

| Feature | Type | Description |
|---------|------|-------------|
| `observation/image_primary` | bytes | Top camera frame (raw uint8 buffer) |
| `observation/image_wrist` | bytes | Wrist camera frame (raw uint8 buffer) |
| `observation/proprio` | float32[] | 6D joint positions |
| `action` | float32[] | 6D joint actions |
| `language_instruction` | bytes | Episode-level task description |
| `is_first` | int64 | 1 for first timestep of episode |
| `is_last` | int64 | 1 for last timestep of episode |
| `is_terminal` | int64 | 1 for last timestep of episode |

### Dataset Statistics

`dataset_statistics.json` stores normalization parameters used at training and inference:

```json
{
  "action": {
    "mean": [0.012, -0.023, ...],
    "std":  [0.134,  0.201, ...],
    "max":  [...], "min": [...], "q01": [...], "q99": [...]
  },
  "proprio": {
    "mean": [...], "std": [...], ...
  }
}
```

---

## 7. Conversion Validation

### Quick Python Check

```python
from prismatic.vla.datasets import RLDSDataset

dataset = RLDSDataset(
    data_root_dir="/lustre/fsw/portfolios/edgeai/users/chrislin/datasets/so101_rlds",
    dataset_name="so101_poker_yellow",
    batch_transform=None,
    resize_resolution=(224, 224),
)
print(f"Dataset loaded: {len(dataset)} samples")
```

### TFRecord Sanity Check

```bash
python -c "
import tensorflow as tf

ds = tf.data.TFRecordDataset(
    '/lustre/fsw/portfolios/edgeai/users/chrislin/datasets/so101_rlds/so101_poker_yellow.tfrecord'
)
total = sum(1 for _ in ds)
print(f'Total records: {total}')   # Expected: 5455

# Verify episode boundaries
is_last_count = 0
for raw in ds:
    ex = tf.train.Example()
    ex.ParseFromString(raw.numpy())
    if ex.features.feature['is_last'].int64_list.value[0]:
        is_last_count += 1
print(f'Episodes (is_last=1): {is_last_count}')  # Expected: 11
"
```

### Visual Validation with `visualize_substeps_so101.py`

After generating substep labels (see §8), render annotated MP4s to inspect label quality:

```bash
pip install av imageio[ffmpeg]

python scripts/visualize_substeps_so101.py \
    --labels_path substep_labels_so101.json \
    --data_dir /lustre/fsw/portfolios/edgeai/users/chrislin/datasets/so101_raw \
    --output_dir /lustre/fsw/portfolios/edgeai/users/chrislin/datasets/so101_viz \
    --episodes 0 1 2   # omit to render all episodes
```

Each output video is a side-by-side (top + wrist) MP4 with a HUD overlay showing:
- Timestep number
- Action type (PICK/PLACE, colour-coded)
- APD substep description
- Prep-move description (carry/approach phase)
- EOS marker at block boundaries
- Cycle number

---

## 8. APD Substep Labeling

### APD Plan Structure (`APD_plans_so101.json`)

The APD plan defines the task decomposition for training. The dual pick-place task uses
8 substeps across 2 cycles:

```json
[
  {
    "suite": "so101_poker_yellow",
    "instruction": {
      "raw": "pick up poker box and yellow box and place them in the white box",
      "plan": [
        {"action_type": "move",  "description": "Move gripper above the poker box",      "cycle": 1},
        {"action_type": "pick",  "description": "Pick up the poker box",                  "cycle": 1},
        {"action_type": "move",  "description": "Move the poker box to the white box",    "cycle": 1},
        {"action_type": "place", "description": "Place the poker box in the white box",   "cycle": 1},
        {"action_type": "move",  "description": "Move gripper above the yellow box",      "cycle": 2},
        {"action_type": "pick",  "description": "Pick up the yellow box",                 "cycle": 2},
        {"action_type": "move",  "description": "Move the yellow box to the white box",   "cycle": 2},
        {"action_type": "place", "description": "Place the yellow box in the white box",  "cycle": 2}
      ]
    }
  }
]
```

Verify plan has 8 steps:

```bash
cat APD_plans_so101.json | python -c "
import json, sys
data = json.load(sys.stdin)
print('Steps:', len(data[0]['instruction']['plan']))  # Expected: 8
"
```

### Generating Substep Labels

#### Basic Usage (no LLM, fast)

```bash
python scripts/label_substeps_so101.py \
    --apd_path APD_plans_so101.json \
    --rlds_data_dir /lustre/fsw/portfolios/edgeai/users/chrislin/datasets/so101_rlds \
    --output_path substep_labels_so101.json \
    --debug
```

**Expected output** (with `--debug`, key lines):

```
SO101 Substep Labeling (LIBERO-style Block-Based)
APD Plans: APD_plans_so101.json  RLDS Data: .../so101_rlds  Scale Mode: False
[INFO] APD plan has 8 steps
[INFO] Processing dataset: so101_poker_yellow
[INFO] Episode 0: 495 timesteps
[DEBUG]   Found 2 pick moments: [50, 180]
[DEBUG]   Found 2 place moments: [120, 250]
[DEBUG]   Pick block (cycle 1): [0, 80) core=50 | Pick up the poker box
[DEBUG]   Place block (cycle 1): [80, 130) core=120 | Place the poker box in the white box
...
[INFO] Processed 11 episodes
Substep Labeling Complete!
Output saved to: substep_labels_so101.json
Total episodes labeled: 11
Total timesteps:  5455
Labeled timesteps: 5455 (100.0% coverage)
```

#### Advanced Usage (LLM language diversity scaling)

Generates 5 paraphrase variants per substep description using the Claude API, then
assigns a random variant to each timestep. This increases language diversity in
training data, helping the model generalise to varied instructions.

```bash
export ANTHROPIC_API_KEY="your-key-here"

python scripts/label_substeps_so101.py \
    --apd_path APD_plans_so101.json \
    --rlds_data_dir /lustre/fsw/portfolios/edgeai/users/chrislin/datasets/so101_rlds \
    --output_path substep_labels_so101_scaled.json \
    --scale \
    --debug
```

**Requirements**: `pip install anthropic` and `ANTHROPIC_API_KEY` set.
**Cost**: ~$0.10 for 11 episodes (~$0.01 per 8 substeps).
**Time**: ~1–2 minutes overhead for LLM calls.

| Mode | Speed | Diversity | Cost |
|------|-------|-----------|------|
| Basic | ~10 s | Low (same description per substep) | Free |
| `--scale` | ~1–2 min | High (5 variants per substep) | ~$0.10 |

**Recommendation**: Use basic for quick testing, `--scale` for final training dataset.

### All Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--apd_path` | `APD_plans_so101.json` | Path to APD plans JSON |
| `--rlds_data_dir` | `.../so101_rlds` | RLDS dataset directory |
| `--output_path` | `substep_labels_so101.json` | Output JSON path |
| `--scale` | False | Enable LLM rephrasing |
| `--debug` | False | Verbose debug logging |

### Output Format

```json
{
  "so101_poker_yellow_episode_0": {
    "instruction": "pick up poker box and yellow box and place them in the white box",
    "total_timesteps": 495,
    "labeled_timesteps": 495,
    "timestep_labels": [
      {
        "timestep": 0,
        "action": "pick",
        "APD_step": "Pick up the poker box",
        "APD_prep_step": "Move gripper above the poker box",
        "cycle": 0,
        "is_substep_end": false
      },
      ...
    ]
  }
}
```

**Output fields**:

| Field | Description |
|-------|-------------|
| `action` | `pick` or `place` (move steps are merged into pick/place blocks) |
| `APD_step` | Primary substep description (pick or place action) |
| `APD_prep_step` | Move prep description (approach/carry phase context); **conditional** — only present when the APD plan has a `move` step immediately preceding the pick/place |
| `cycle` | 0-based cycle index (0 = poker box, 1 = yellow box); note: APD plan uses 1-based `cycle` field, but the output JSON uses 0-based |
| `is_substep_end` | `true` on the last timestep of each block (EOS marker) |

### Block-Based Labeling Algorithm

`label_substeps_so101.py` mirrors the LIBERO (`label_substeps.py`) block-based logic:

```
Episode → detect_gripper_transitions() → label_actions() → label_substeps_blocks()
                                                                  ↓
                                              Non-overlapping Pick/Place blocks
```

**Step 1 — Gripper detection** (windowed relative change, σ=2, window=8):
- Closing transition (before_avg > after_avg + 0.1) → pick moment
- Opening transition (after_avg > before_avg + 0.04) → place moment
- Minimum 20-step gap between consecutive detections

**Step 2 — `label_actions()`** (intermediate, mirrors LIBERO):
- All timesteps initialised as "move"
- Pick regions: `[pick_t - 50, pick_t + 30]`
- Place regions: `[place_t - 100, place_t + 80]` (overwrites pick in overlaps)

**Step 3 — Block construction** (non-overlapping, mirrors LIBERO):

```python
# num_cycles = max(len(pick_moments), len(place_moments))
for cycle_idx in range(num_cycles):
    apd_cycle = cycle_idx + 1   # APD plan uses 1-based cycle; output label uses 0-based cycle_idx

    # Pick block
    if cycle_idx < len(pick_moments):
        pick_t     = pick_moments[cycle_idx]
        pick_start = blocks[-1]['end'] if blocks else 0   # continuous, no gaps
        pick_end   = min(T, pick_t + config["pick_expand_forward"])   # default 30
        # Clip by next pick's approach phase reservation
        if cycle_idx + 1 < len(pick_moments):
            next_pick_t = pick_moments[cycle_idx + 1]
            pick_end = min(pick_end, next_pick_t - config["pick_expand_backward"])  # default 50

    # Place block
    if cycle_idx < len(place_moments):
        place_t     = place_moments[cycle_idx]
        place_start = blocks[-1]['end'] if blocks else 0   # continuous from pick block
        place_end   = min(T, place_t + config["place_expand_forward"])   # default 80
        # Same clip logic
        if cycle_idx + 1 < len(pick_moments):
            next_pick_t = pick_moments[cycle_idx + 1]
            place_end = min(place_end, next_pick_t - config["pick_expand_backward"])  # default 50
```

**Why block-end clipping matters** — without clipping the place block would overrun
into the next cycle's approach phase, leaving the approach timesteps unlabelled:

```
Example: T=300, pick1=50, pick2=180, place1=120
  Without clipping: place block ends at min(200, 180) = 180
                    → next pick block starts at 180, missing t=130–179 (approach)
  With clipping:    place block ends at min(200, 180-50) = 130
                    → next pick block starts at 130, capturing full approach ✓
```

**Expansion parameters** (identical to LIBERO defaults):

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `pick_expand_backward` | 50 | Approach phase reservation for block-end clipping |
| `pick_expand_forward` | 30 | Captures lift phase after grasp |
| `place_expand_backward` | 100 | Used in `label_actions()` intermediate labeling |
| `place_expand_forward` | 80 | Captures retract phase after release |

**Fallback behaviour**: If only 1 pick/place is detected instead of 2, the script
silently uses `num_cycles = max(len(pick_moments), len(place_moments))` and only
iterates over the detected cycles — no warning is emitted. Use `--debug` to inspect
the pick/place moments detected per episode and confirm cycle count.

### SO101 vs LIBERO Differences

| Aspect | LIBERO | SO101 | Notes |
|--------|--------|-------|-------|
| APD key field | `step['subgoal']` | `step['description']` | Different schema |
| APD matching | Keyword-based (`pick/grasp/lift`) | `action_type` + `cycle` fields | SO101 has explicit types |
| Move prep | `apd_prep_step` optional | `APD_prep_step` conditional (when preceding move step found in APD plan) | SO101 APD has explicit move steps |
| Cycles | Implicit by position | Explicit `cycle` field (1-based in APD) | SO101 APD schema |
| Gripper shape | `(T, 2)` → `[:, 0]` | `(T,)` 1D from `action[-1]` | Different format |

---

## 9. Training

### Configuration Parameters

```python
# Model
vla_path = "openvla/openvla-7b"
lora_rank = 32

# Dataset
dataset_name = "so101_poker_yellow"
num_images_in_input = 2   # top + wrist cameras
use_proprio = True

# Training
batch_size = 8            # per GPU × 8 GPUs = 64 total
learning_rate = 5e-4
max_steps = 150005
save_freq = 50000

# Optimization
use_l1_regression = True
image_aug = True
gradient_checkpointing = False  # Set True to reduce VRAM ~30% at ~20% slower backward

# Resume
resume = False            # Set True to continue from a checkpoint
resume_step = None        # Step number to resume from (e.g. 50000)
```

### Gradient Checkpointing

Enable `--gradient_checkpointing True` to trade compute for memory (~30% VRAM reduction,
~20% slower backward pass). Useful when hitting OOM errors with full-resolution images
or large batch sizes. Must be set at launch — cannot be toggled mid-run.

The implementation calls `gradient_checkpointing_enable()` and `enable_input_require_grads()`
on the base model **before** LoRA wrapping, which is required for LoRA gradients to flow
correctly through checkpointed segments.

### WandB Setup

Update credentials in `scripts/ord_sbatch_so101.sh`:

```bash
export WANDB_API_KEY="your-wandb-key"
export WANDB_ENTITY="your-wandb-entity"
export WANDB_PROJECT="openvla-so101-poker"
```

### Option A: Interactive Session (Quick Test / Debug)

```bash
bash scripts/ord_interactive.sh

# Inside container:
cd /workspace
python vla-scripts/finetune_substep.py \
    --vla_path openvla/openvla-7b \
    --data_root_dir /lustre/fsw/portfolios/edgeai/users/chrislin/datasets/so101_rlds \
    --dataset_name so101_poker_yellow \
    --substep_labels_path substep_labels_so101.json \
    --batch_size 2 \
    --max_steps 100 \
    --num_images_in_input 2 \
    --use_proprio True
```

### Option B: SLURM Batch Job (Full Training)

```bash
mkdir -p logs
sbatch scripts/ord_sbatch_so101.sh
```

Make sure `substep_labels_path` in `scripts/ord_sbatch_so101.sh` points to your
generated labels file (scaled or basic).

**Monitor job**:

```bash
squeue -u $USER
tail -f logs/so101_training_<JOB_ID>.out
tail -f logs/so101_training_<JOB_ID>.err
```

**Expected timeline**:
- ~6–8 hours per 50k steps
- Total: 24–30 hours (150k steps, 8× A100 GPUs)

### Option C: Resume Training from Checkpoint

Each saved checkpoint now contains three extra files alongside the model weights:

```
<run_id>--50000_chkpt/
├── model.safetensors          # merged LoRA weights
├── lora_adapter/
├── action_head--50000_checkpoint.pt
├── proprio_projector--50000_checkpoint.pt
├── eos_head--50000_checkpoint.pt
├── dataset_statistics.json
├── optimizer_state.pt         # AdamW momentum buffers + step count
├── scheduler_state.pt         # MultiStepLR last_epoch counter
└── wandb_run_id.txt           # WandB run ID for same-run logging
```

To continue training from step 50000:

```bash
RESUME_CHECKPOINT=/lustre/fsw/portfolios/edgeai/users/chrislin/checkpoints/openvla-so101/<run_id>--50000_chkpt
RESUME_STEP=50000

sbatch scripts/ord_sbatch_so101.sh \
    # or edit the script to add:
    # --resume True
    # --resume_step ${RESUME_STEP}
    # --vla_path ${RESUME_CHECKPOINT}
```

Or launch directly:

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 8 \
    vla-scripts/finetune_substep.py \
        --vla_path ${RESUME_CHECKPOINT} \
        --resume True \
        --resume_step ${RESUME_STEP} \
        --data_root_dir /lustre/fsw/portfolios/edgeai/users/chrislin/datasets/so101_rlds \
        --dataset_name so101_poker_yellow \
        --substep_labels_path substep_labels_so101.json \
        --run_root_dir /lustre/fsw/portfolios/edgeai/users/chrislin/checkpoints/openvla-so101 \
        --max_steps 150005 \
        --save_freq 50000 \
        --wandb_entity ${WANDB_ENTITY} \
        --wandb_project ${WANDB_PROJECT}
```

**What gets restored on resume**:

| Component | Restored? | Notes |
|-----------|-----------|-------|
| Model weights (LoRA + heads) | Yes | via `init_module()` loading `*_checkpoint.pt` files |
| Optimizer state (AdamW moments) | Yes | via `optimizer_state.pt` |
| LR scheduler state | Yes | via `scheduler_state.pt` — `last_epoch` continues correctly |
| WandB run | Yes | `wandb_run_id.txt` → `wandb.init(id=..., resume="allow")` |
| Log step offset | Yes | `log_step = resume_step + gradient_step_idx` |

> **Note**: Checkpoints saved *before* this feature was added will not have
> `optimizer_state.pt`, `scheduler_state.pt`, or `wandb_run_id.txt`. The script
> prints a `[WARNING]` in that case and starts a fresh optimizer/WandB run.

### Monitoring (WandB)

**Key metrics**:

| Metric | Healthy Sign |
|--------|-------------|
| `VLA Train/Loss` | Decreases steadily |
| `VLA Train/Curr Action L1 Loss` | Converges to < 0.1 |
| `VLA Train/Next Actions L1 Loss` | Converges to < 0.15 |
| `VLA Train/Learning Rate` | Decays 10× at 100k steps |

Watch for NaN loss or sudden spikes — these usually indicate a normalization issue
or too-high learning rate.

### Checkpoint Structure

Checkpoints saved to:

```
/lustre/fsw/portfolios/edgeai/users/chrislin/checkpoints/openvla-so101/
├── openvla-7b+so101_poker_yellow+b8+lr-0.0005+lora-r32--50000_chkpt/
│   ├── config.json
│   ├── model.safetensors          # merged LoRA + base model
│   ├── lora_adapter/              # LoRA weights only
│   ├── action_head--50000_checkpoint.pt
│   ├── proprio_projector--50000_checkpoint.pt
│   └── dataset_statistics.json
├── ...--100000_chkpt/
└── ...--150005_chkpt/
```

**Quick checkpoint load test**:

```python
from transformers import AutoModelForVision2Seq, AutoProcessor
import torch

checkpoint = "/path/to/checkpoint"
model = AutoModelForVision2Seq.from_pretrained(
    checkpoint, torch_dtype=torch.bfloat16, trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
print("Checkpoint loaded successfully!")
```

---

## 10. Checkpoint Upload

Use `scripts/upload_checkpoint_to_hf.py` to share the trained checkpoint:

```bash
python scripts/upload_checkpoint_to_hf.py \
    --checkpoint_dir /lustre/fsw/portfolios/edgeai/users/chrislin/checkpoints/openvla-so101/openvla-7b+so101_poker_yellow+...--150005_chkpt \
    --repo_id christian0420/openvla-so101-poker-yellow \
    --token YOUR_HF_TOKEN

# Make private:
python scripts/upload_checkpoint_to_hf.py \
    --checkpoint_dir ... \
    --repo_id christian0420/openvla-so101-poker-yellow \
    --token YOUR_HF_TOKEN \
    --private
```

> **Note**: Do not hardcode HF tokens in scripts. Pass via `--token` argument or set
> `HUGGING_FACE_HUB_TOKEN` environment variable.

**What gets uploaded**:
- `config.json`, `model.safetensors`, `preprocessor_config.json`
- `tokenizer_config.json`, `tokenizer.json`, `special_tokens_map.json`
- `dataset_statistics.json`
- `action_head--150005_checkpoint.pt`, `proprio_projector--150005_checkpoint.pt`
- `lora_adapter/` directory (LoRA-only weights)
- Auto-generated `README.md` (model card)

**Model available at**: `https://huggingface.co/christian0420/openvla-so101-poker-yellow`

---

## 11. Evaluation and Robot Deployment

### Download Checkpoint

```bash
# Using CLI
huggingface-cli download christian0420/openvla-so101-poker-yellow \
    --local-dir ./checkpoints/openvla-so101

# Using Python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="christian0420/openvla-so101-poker-yellow",
    local_dir="./checkpoints/openvla-so101"
)
```

**Verify checkpoint files**:

```
checkpoints/openvla-so101/
├── config.json
├── model.safetensors
├── preprocessor_config.json
├── dataset_statistics.json
├── action_head--150005_checkpoint.pt
└── proprio_projector--150005_checkpoint.pt
```

### Camera Calibration

```python
from lerobot.common.cameras import CameraManager

camera_manager = CameraManager(
    cameras=["top", "wrist"],
    config_path="configs/robot/so_follower.yaml"
)
top_img, wrist_img = camera_manager.capture()
print(f"Top camera: {top_img.shape}")    # (1080, 1920, 3)
print(f"Wrist camera: {wrist_img.shape}")  # (720, 1280, 3)
camera_manager.save_calibration("calibration/so_follower_cameras.json")
```

### Policy Wrapper (`policies/openvla_policy.py`)

```python
import torch, numpy as np, json
from transformers import AutoModelForVision2Seq, AutoProcessor
from lerobot.common.policies.policy import Policy


class OpenVLAPolicy(Policy):
    def __init__(self, checkpoint_path, device="cuda"):
        super().__init__()
        self.device = device
        self.processor = AutoProcessor.from_pretrained(checkpoint_path, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            checkpoint_path, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).to(device)
        self.action_head_state = torch.load(
            f"{checkpoint_path}/action_head--150005_checkpoint.pt", map_location=device
        )
        with open(f"{checkpoint_path}/dataset_statistics.json") as f:
            stats = json.load(f)
        self.action_mean = np.array(stats["action"]["mean"])
        self.action_std  = np.array(stats["action"]["std"])
        self.proprio_mean = np.array(stats["proprio"]["mean"])
        self.proprio_std  = np.array(stats["proprio"]["std"])
        self.instruction = "Pick up the yellow poker chip and place it in the target location"

    def predict_action(self, observation):
        top_img   = observation["top_image"]
        wrist_img = observation["wrist_image"]
        proprio   = observation["proprio"]
        proprio_norm = (proprio - self.proprio_mean) / (self.proprio_std + 1e-8)
        prompt = f"What action should the robot take to {self.instruction.lower()}?"
        inputs = self.processor(text=prompt, images=[top_img, wrist_img], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        inputs["proprio"] = torch.tensor(proprio_norm, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(**inputs, output_hidden_states=True)
            predicted_action = self._decode_action(outputs.hidden_states[-1])
        action = predicted_action.cpu().numpy()
        action = action * self.action_std + self.action_mean
        return np.clip(action, -1.0, 1.0)

    def _decode_action(self, hidden_states):
        return hidden_states[:, -1, :6]

    def reset(self):
        pass
```

### Safety Checklist (Before Running on Robot)

- [ ] Emergency stop button accessible
- [ ] Robot workspace clear of obstacles
- [ ] Camera feeds verified
- [ ] Robot homed and calibrated
- [ ] Software kill switch configured
- [ ] Test in manual mode first

### Manual Mode Test

```python
from lerobot import LeRobotEnv
from policies.openvla_policy import OpenVLAPolicy

env = LeRobotEnv(
    robot_config="configs/robot/so_follower.yaml",
    camera_config="calibration/so_follower_cameras.json",
    control_mode="manual",
)
policy = OpenVLAPolicy(checkpoint_path="./checkpoints/openvla-so101")
obs = env.reset()
for step in range(100):
    action = policy.predict_action(obs)
    obs, reward, done, info = env.step(action)
    if done:
        break
env.close()
```

### Autonomous Evaluation

```python
env = LeRobotEnv(
    robot_config="configs/robot/so_follower.yaml",
    camera_config="calibration/so_follower_cameras.json",
    control_mode="autonomous",
)
num_trials = 10
successes = 0

for trial in range(num_trials):
    obs = env.reset()
    for step in range(200):
        action = policy.predict_action(obs)
        obs, reward, done, info = env.step(action)
        if done:
            if info.get("success", False):
                successes += 1
            break

print(f"Success Rate: {successes}/{num_trials} ({100*successes/num_trials:.1f}%)")
env.close()
```

### Metrics to Track

**Success**: task success rate, average episode length, average reward per episode.

**Safety**: collision count, joint limit violations, emergency stops required.

**Performance**: inference latency (ms), control frequency (Hz), action smoothness.

---

## 12. Troubleshooting

### Data Download

| Problem | Solution |
|---------|----------|
| Download fails | Check HF token; run `huggingface-cli login` |
| Slow download | Check network; use `--resume-download` flag |
| Wrong sample count | Dataset may have changed; check HF dataset page |

### Data Conversion

| Problem | Solution |
|---------|----------|
| Empty/corrupt video frames | Use `--use_pyav_video` flag (AV1 codec) |
| `FileNotFoundError` for meta/info.json | Pass `--repo_id` so script downloads automatically |
| Image dimension mismatch | Verify `num_images_in_input=2` in training config |

### Substep Labeling

| Problem | Solution |
|---------|----------|
| Only 1 cycle labeled (silent) | Run with `--debug` and check "Found N pick moments" lines; tune `relative_threshold` in CONFIG; verify dataset has 2 pick-place cycles |
| "LLM rephrasing failed" | Check `ANTHROPIC_API_KEY`; script falls back to original descriptions |
| Substep boundaries misaligned | Tune `pick_expand_backward/forward` in CONFIG |
| Dataset not found | Check `dataset_name` matches TFRecord filename |

Adjust CONFIG in `label_substeps_so101.py` if detection is off:

```python
CONFIG = {
    "gripper_threshold": 0.5,         # Legacy absolute threshold (not used with relative detection)
    "relative_threshold": 0.1,        # Try 0.05–0.15
    "pick_expand_backward": 50,       # Try 30–70
    "pick_expand_forward": 30,        # Try 20–40
    "place_expand_backward": 100,     # Try 60–120
    "place_expand_forward": 80,       # Try 50–100
    "dataset_name": "so101_poker_yellow",
}
```

### Training

| Problem | Solution |
|---------|----------|
| CUDA out of memory | Reduce `batch_size` (try 4 or 2); add `gradient_accumulation_steps`; enable `--gradient_checkpointing True` (~30% VRAM reduction) |
| Loss is NaN | Check normalization stats; reduce `learning_rate` (try 1e-4); enable gradient clipping |
| Checkpoint saving fails | Check disk space on `/lustre`; verify write permissions |
| Job stuck in queue | Check `sinfo`; reduce GPU request; try different partition |
| Container mount errors | Verify `/lustre` paths exist; use absolute paths |

### Robot Deployment

| Problem | Solution |
|---------|----------|
| Camera feed drops/lags | Check USB bandwidth; reduce resolution/FPS; check cables |
| Actions too jerky | Reduce `max_action_rate`; enable smoothing; lower control frequency |
| Robot doesn't move | Check action scaling/denormalization; verify e-stop is off; print raw action values |
| Inference too slow (< 10 Hz) | Use GPU; reduce image resolution; enable `torch.compile()`; use bfloat16 |
| Out of memory | Reduce to batch size 1; use `torch.no_grad()`; clear CUDA cache between steps |

---

## 13. Performance Reference

### Training

| Metric | Value |
|--------|-------|
| GPUs | 8× NVIDIA A100 (80 GB) |
| Steps per second | ~2–3 |
| Time per 50k checkpoint | ~6–8 hours |
| Total time (150k steps) | ~24–30 hours |
| Target action L1 loss | < 0.1 |
| Final training loss | ~0.05 |

### Deployment

| Metric | Value |
|--------|-------|
| Inference speed | ~10–15 Hz on RTX 4090 |
| Latency per action | < 100 ms |
| Success rate | TBD (requires real-robot evaluation) |

---

## File Reference

```
openvla-oft-yhs/
├── scripts/
│   ├── ord_interactive.sh              # Launch interactive GPU session
│   ├── ord_sbatch_so101.sh             # Submit full training job
│   ├── download_so101_dataset.py       # Download dataset from HuggingFace
│   ├── convert_lerobot_to_rlds.py      # Convert LeRobot v3 → RLDS TFRecord
│   ├── label_substeps_so101.py         # Generate APD substep labels
│   ├── visualize_substeps_so101.py     # Render annotated validation videos
│   └── upload_checkpoint_to_hf.py      # Upload checkpoint to HuggingFace
├── docs/
│   └── SO101_COMPLETE_GUIDE.md         # This file
├── APD_plans_so101.json                # APD task decomposition (8 substeps)
├── substep_labels_so101.json           # Generated substep labels
└── README_SO101.md                     # Quick-start reference
```

## Resources

- HuggingFace Model: `christian0420/openvla-so101-poker-yellow`
- HuggingFace Dataset: `christian0420/so101-poker-yellow-task`
- OpenVLA: https://github.com/openvla/openvla
- LeRobot: https://github.com/huggingface/lerobot
- APD Framework: https://github.com/APD-VLA
