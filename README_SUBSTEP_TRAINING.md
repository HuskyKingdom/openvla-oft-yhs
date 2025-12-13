# Per-Timestep Substep Instruction Training

This document explains how to use the new substep-aware training pipeline for OpenVLA.

## Overview

The substep training pipeline extends the original OpenVLA fine-tuning to support **per-timestep** substep instructions instead of per-episode task instructions. This enables more fine-grained language supervision at each timestep during training.

## Key Features

- **Per-timestep instruction replacement**: Each timestep gets its own APD_step instruction
- **Episode tracking**: Preserves episode IDs through the data pipeline
- **Compatible with original training**: Uses same LoRA setup, supports all original features
- **Fallback handling**: Gracefully falls back to original instruction if substep not found

## Quick Start

### 1. Prepare Substep Labels

Ensure you have a substep labels JSON file (e.g., `substep_labels_output.json`) with the following structure:

```json
{
  "libero_goal": {
    "put_the_cream_cheese_in_the_bowl": {
      "episode_5": {
        "instruction": "put the cream cheese in the bowl",
        "total_timesteps": 112,
        "timestep_labels": [
          {
            "timestep": 0,
            "action": "pick",
            "APD_step": "Pick up the cream cheese with the left arm",
            "cycle": 0
          },
          {
            "timestep": 1,
            "action": "pick",
            "APD_step": "Pick up the cream cheese with the left arm",
            "cycle": 0
          }
        ]
      }
    }
  }
}
```

### 2. Verify Installation (Optional)

Test that all modules import correctly:

```bash
python test_substep_imports.py
```

If successful, you should see:
```
✓ All core substep training modules imported successfully!
```

### 3. Run Training

```bash
python vla-scripts/finetune_substep.py \
    --vla_path openvla/openvla-7b \
    --data_root_dir datasets/rlds \
    --dataset_name libero_goal_no_noops \
    --substep_labels_path substep_labels_output.json \
    --run_root_dir runs \
    --batch_size 8 \
    --learning_rate 5e-4 \
    --max_steps 200000 \
    --save_freq 10000 \
    --use_lora True \
    --lora_rank 32 \
    --image_aug True \
    --wandb_entity your-wandb-entity \
    --wandb_project your-wandb-project
```

### 4. All Configuration Options

The training script supports all parameters from the original `finetune.py`, plus:

- `--substep_labels_path`: Path to substep labels JSON file (required)

All other parameters work identically to the original training script.

## Architecture

### New Components

1. **transforms_substep.py**: Extended RLDS transforms that add episode ID tracking
2. **dataset_substep.py**: Modified dataset loader that preserves episode metadata
3. **materialize_substep.py**: Extended OXE dataset factory with episode tracking
4. **datasets_substep.py**: Substep-aware batch transform and dataset classes
5. **finetune_substep.py**: Main training script with substep support

### Data Flow

```
RLDS Episode (with episode_metadata)
    ↓
libero_dataset_transform_with_episode_id (adds episode_id field)
    ↓
restructure_with_episode_id (preserves episode_id for each frame)
    ↓
Frame-level data (observation, action, task, dataset_name, episode_id, timestep)
    ↓
SubstepRLDSBatchTransform (queries and replaces instruction)
    ↓
Training batch (with per-timestep substep instruction)
```

### Episode ID Tracking

Episode IDs are tracked using a global counter that increments for each trajectory:
- Counter resets when dataset is initialized
- Episode IDs are sequential integers starting from 0
- IDs match the `episode_N` format in the JSON labels

### Instruction Replacement Logic

For each timestep, the system:
1. Extracts: `dataset_name`, `task_instruction`, `episode_id`, `timestep`
2. Strips `_no_noops` suffix from dataset name to get suite name
3. Converts task instruction to underscore format
4. Queries substep labels: `labels[suite][task][episode_N][timestep]`
5. Returns `APD_step` if found, otherwise returns original instruction

## Supported Datasets

Currently, episode tracking is implemented for:
- `libero_spatial_no_noops`
- `libero_object_no_noops`
- `libero_goal_no_noops`
- `libero_10_no_noops`
- `libero_4_task_suites_no_noops`

Other datasets will fall back to standard transforms (episode_id = 0).

## Troubleshooting

### Issue: "cannot import name 'ImageTransform'" or "cannot import name 'make_interleaved_dataset'"
**Solution**: These import errors have been fixed in the latest version. Make sure you're using the corrected files:
- `datasets_substep.py` (ImageTransform import)
- `datasets_substep.py` (make_interleaved_dataset import)

If you still see these errors, run the test script to diagnose:
```bash
python test_substep_imports.py
```

### Issue: "Substep labels file not found"
**Solution**: Check that the path to `substep_labels_output.json` is correct

### Issue: Instructions not being replaced
**Possible causes**:
1. Suite name mismatch (check that dataset name matches JSON structure)
2. Task name mismatch (ensure task names use underscores correctly)
3. Episode ID mismatch (verify episode numbering in JSON)
4. Timestep not found (check timestep_labels in JSON)

**Debug**: Look for `"Replaced instruction at episode=X, timestep=Y"` messages in logs

### Issue: "Could not find substep" warnings
**Expected behavior**: The system falls back to original task instruction when substeps are not found. This is normal for:
- Episodes not in the JSON file
- Timesteps without explicit labels
- Tasks with different naming

## Performance Notes

- Substep labels are loaded into memory at dataset initialization (minimal overhead)
- Instruction lookup is O(1) dictionary access (no performance impact)
- Episode tracking adds minimal computational overhead
- Training speed should be nearly identical to original training

## Extending to Other Datasets

To add episode tracking for other datasets:

1. Add transform function in `transforms_substep.py`:
```python
def your_dataset_transform_with_episode_id(trajectory):
    trajectory = your_original_transform(trajectory)
    episode_id = get_and_increment_episode_id()
    traj_len = tf.shape(trajectory["action"])[0]
    trajectory["episode_id"] = tf.repeat(episode_id, traj_len)
    return trajectory
```

2. Register in `OXE_STANDARDIZATION_TRANSFORMS_WITH_EPISODE_ID`:
```python
"your_dataset_name": your_dataset_transform_with_episode_id,
```

3. Create substep labels JSON with matching structure

## Comparison with Original Training

| Feature | Original | Substep |
|---------|----------|---------|
| Instruction granularity | Per-episode | Per-timestep |
| Episode tracking | No | Yes |
| Fallback behavior | N/A | Uses original instruction |
| Training speed | Baseline | ~Same |
| LoRA support | Yes | Yes |
| All other features | Yes | Yes |

## Citation

If you use this substep training approach in your research, please cite:

```bibtex
@article{openvla2024,
  title={OpenVLA: An Open-Source Vision-Language-Action Model},
  author={...},
  journal={arXiv preprint},
  year={2024}
}
```

## License

Same as the original OpenVLA repository.

