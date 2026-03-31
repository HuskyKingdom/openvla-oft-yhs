"""
label_substeps_so101.py

Automatically labels timesteps in SO101 RLDS episodes with substep information
using the same block-based logic as label_substeps.py (LIBERO approach).

Pipeline (mirrors label_substeps.py):
  RLDS Episode → Gripper Detection → label_actions() → label_substeps_blocks()
                                                              ↓
                                              Non-overlapping Pick/Place blocks
                                              with Move steps merged in and
                                              APD prep descriptions attached

Key design decisions (same as label_substeps.py):
- Relative change gripper detection (robust to different object sizes)
- label_actions() intermediate step labels move/pick/place per timestep
- Non-overlapping block assignment: pick_start = blocks[-1]['end'] or 0
- Block end boundary: min(pick_t + forward, next_pick_t - pick_expand_backward)
  This correctly reserves the approach phase for the next cycle's pick block
- Move APD steps merged INTO pick/place blocks as apd_prep_step (not separate)
- EOS (is_substep_end) markers for substep-aware training
- Optional LLM-based language diversity scaling

Usage:
    # Basic usage
    python scripts/label_substeps_so101.py \
        --apd_path APD_plans_so101.json \
        --rlds_data_dir /path/to/so101_rlds \
        --output_path substep_labels_so101.json

    # With LLM scaling for language diversity
    export ANTHROPIC_API_KEY="your-key"
    python scripts/label_substeps_so101.py \
        --apd_path APD_plans_so101.json \
        --rlds_data_dir /path/to/so101_rlds \
        --output_path substep_labels_so101_scaled.json \
        --scale \
        --debug

Author: Adapted for SO101 Dual Pick-Place Task (aligned with label_substeps.py logic)
"""

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from scipy.ndimage import gaussian_filter1d

# Disable GPU for data loading
tf.config.set_visible_devices([], 'GPU')

# Configuration for SO Follower (6-DOF robot)
# Forward/backward expansion values mirror label_substeps.py defaults
CONFIG = {
    "gripper_threshold": 0.5,            # Absolute threshold (legacy, not used with relative detection)
    "relative_threshold": 0.1,           # Relative change threshold for gripper detection
    "pick_expand_backward": 50,          # Pick backward expansion: reserves approach phase for block start calc
    "pick_expand_forward": 30,           # Pick forward expansion: captures lift phase after grasp
    "place_expand_backward": 100,        # Place backward expansion: reserves carry phase for block start calc
    "place_expand_forward": 80,          # Place forward expansion: captures retract phase after release
    "dataset_name": "so101_poker_yellow"
}

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Loading
# ============================================================================

def load_apd_plans(apd_path: str) -> Dict[str, Dict[str, List[Dict]]]:
    """Load APD plans and organize for easy querying."""
    logger.info(f"Loading APD plans from {apd_path}")

    with open(apd_path, 'r', encoding='utf-8') as f:
        apd_data = json.load(f)

    # Organize by suite and instruction
    plans_by_suite = {}

    for item in apd_data:
        suite = item['suite']
        instruction = item['instruction']['raw'].lower().strip()
        plan = item['instruction']['plan']

        if suite not in plans_by_suite:
            plans_by_suite[suite] = {}

        plans_by_suite[suite][instruction] = plan

    # Log statistics
    for suite, instructions in plans_by_suite.items():
        logger.info(f"  Suite '{suite}': {len(instructions)} instructions")

    return plans_by_suite


def load_rlds_dataset(
    data_dir: str, dataset_name: str
) -> Tuple[Optional[Union[tf.data.Dataset, Iterator]], bool]:
    """
    Load RLDS data.

    Returns:
        (dataset_or_none, is_flat_tfrecord)
        When is_flat_tfrecord is True, each TFRecord row is one timestep; group with
        iter_tfrecord_episodes() before extract_episode_data().
    """
    tfrecord_path = os.path.join(data_dir, f"{dataset_name}.tfrecord")

    if os.path.exists(tfrecord_path):
        logger.info(f"Loading TFRecord dataset: {tfrecord_path}")
        return load_tfrecord_dataset(tfrecord_path), True

    dataset_path = os.path.join(data_dir, dataset_name, "1.0.0")

    if os.path.exists(dataset_path):
        try:
            builder = tfds.builder_from_directory(dataset_path)
            dataset = builder.as_dataset(split="train")
            logger.info(f"Loaded TFDS dataset: {dataset_name}")
            return dataset, False
        except Exception as e:
            logger.error(f"Failed to load TFDS dataset: {e}")

    logger.error(f"Dataset not found at {data_dir}")
    return None, False


def load_tfrecord_dataset(tfrecord_path: str) -> tf.data.Dataset:
    """Load dataset from TFRecord format."""

    def _parse_example(example_proto):
        """Parse TFRecord example."""
        feature_description = {
            'observation/image_primary': tf.io.FixedLenFeature([], tf.string),
            'observation/image_wrist': tf.io.FixedLenFeature([], tf.string),
            'observation/proprio': tf.io.VarLenFeature(tf.float32),
            'action': tf.io.VarLenFeature(tf.float32),
            'language_instruction': tf.io.FixedLenFeature([], tf.string),
            'is_first': tf.io.FixedLenFeature([], tf.int64),
            'is_last': tf.io.FixedLenFeature([], tf.int64),
            'is_terminal': tf.io.FixedLenFeature([], tf.int64),
        }

        parsed = tf.io.parse_single_example(example_proto, feature_description)

        # Convert sparse tensors to dense
        parsed['observation/proprio'] = tf.sparse.to_dense(parsed['observation/proprio'])
        parsed['action'] = tf.sparse.to_dense(parsed['action'])

        return parsed

    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(_parse_example)

    return dataset


def iter_tfrecord_episodes(dataset: tf.data.Dataset) -> Iterator[Dict[str, Any]]:
    """
    Group flat TFRecord timesteps (one Example per step) into RLDS-style episodes.

    convert_lerobot_to_rlds.py sets is_last on the final step of each episode.
    """
    steps_buffer: List[Dict[str, Any]] = []
    for example in dataset:
        step = {
            "action": example["action"],
            "observation": {"proprio": example["observation/proprio"]},
            "language_instruction": example["language_instruction"],
        }
        steps_buffer.append(step)
        is_last = bool(example["is_last"].numpy())
        if is_last:
            yield {"steps": steps_buffer}
            steps_buffer = []
    if steps_buffer:
        logger.warning(
            "Trailing timesteps without is_last=True (%d steps); emitting final partial episode",
            len(steps_buffer),
        )
        yield {"steps": steps_buffer}


def extract_episode_data(episode: Dict) -> Dict[str, Any]:
    """
    Extract key data from RLDS episode.

    For SO follower (6-DOF):
    - Action: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
    - Proprio: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
    """
    steps = episode['steps']

    actions = []
    proprios = []
    gripper_states = []
    language_instruction = ""

    for step in steps:
        # Action (6D for SO follower)
        action = step['action'].numpy()
        actions.append(action)

        # Proprioceptive state (6D)
        if 'proprio' in step['observation']:
            proprio = step['observation']['proprio'].numpy()
        elif 'state' in step['observation']:
            proprio = step['observation']['state'].numpy()
        else:
            proprio = action.copy()  # Fallback
        proprios.append(proprio)

        # Gripper state (last dimension)
        gripper_state = action[-1] if len(action) >= 6 else 0.0
        gripper_states.append(gripper_state)

        # Language instruction
        if 'language_instruction' in step:
            language_instruction = step['language_instruction'].numpy().decode('utf-8')

    return {
        "actions": np.array(actions),
        "proprios": np.array(proprios),
        "gripper_states": np.array(gripper_states),
        "language_instruction": language_instruction.lower().strip(),
        "length": len(actions)
    }


# ============================================================================
# Gripper Detection (LIBERO-style Relative Change) — mirrors label_substeps.py
# ============================================================================

def detect_gripper_transitions(
    gripper_states: np.ndarray,
    relative_threshold: float = 0.1,
    use_relative: bool = True
) -> Tuple[List[int], List[int]]:
    """
    Detect gripper open/close transitions using relative change detection.

    Mirrors label_substeps.py detect_gripper_transitions():
    - Uses windowed comparison (window=8) to detect significant changes
    - Relative threshold makes detection robust to different object sizes
    - Closing (before > after + threshold) → pick moment
    - Opening (after > before + threshold*0.4) → place moment
    - Minimum 20-step gap between consecutive detections

    Args:
        gripper_states: (T,) gripper state array
        relative_threshold: Minimum change to count as transition
        use_relative: If True, use relative change detection

    Returns:
        (pick_moments, place_moments)
        - pick_moments: List of gripper closing moments (open→close)
        - place_moments: List of gripper opening moments (close→open)
    """
    T = len(gripper_states)

    pick_moments = []
    place_moments = []

    if use_relative:
        # Smooth gripper values to reduce noise
        gripper_smooth = gaussian_filter1d(gripper_states, sigma=2.0)

        # Use windowed comparison (same as label_substeps.py: window=8)
        window = 8

        for t in range(window, T - window):
            before_window = gripper_smooth[t-window:t]
            after_window = gripper_smooth[t+1:t+window+1]

            before_avg = np.mean(before_window)
            after_avg = np.mean(after_window)

            # Detect significant closing (before > after): gripper closing = pick
            if before_avg > after_avg + relative_threshold:
                if not pick_moments or t - pick_moments[-1] > 20:
                    pick_moments.append(t)
                    logger.debug(f"    Pick at t={t}: before={before_avg:.4f}, after={after_avg:.4f}")

            # Detect significant opening (after > before): gripper opening = place
            elif after_avg > before_avg + relative_threshold * 0.4:
                if not place_moments or t - place_moments[-1] > 20:
                    place_moments.append(t)
                    logger.debug(f"    Place at t={t}: before={before_avg:.4f}, after={after_avg:.4f}")
    else:
        # Legacy: absolute threshold (not recommended)
        threshold = 0.5
        is_closed = gripper_states > threshold

        for t in range(1, T):
            if not is_closed[t-1] and is_closed[t]:
                pick_moments.append(t)
            elif is_closed[t-1] and not is_closed[t]:
                place_moments.append(t)

    return pick_moments, place_moments


# ============================================================================
# Intermediate Action Labeling — mirrors label_substeps.py label_actions()
# ============================================================================

def label_actions(
    episode_data: Dict[str, Any],
    config: Dict
) -> Tuple[List[str], Dict]:
    """
    Intermediate step: label each timestep as move/pick/place.

    Mirrors label_substeps.py label_actions():
    1. Detect gripper transitions
    2. Initialize all timesteps as "move"
    3. Label pick regions: [pick_t - backward, pick_t + forward]
    4. Label place regions (overwrites pick in overlaps): [place_t - backward, place_t + forward]

    This intermediate labeling is used for analysis and to identify move segments
    that get merged into pick/place blocks in the final output.

    Args:
        episode_data: Episode data dict with gripper_states
        config: Configuration dict with expand parameters

    Returns:
        (action_labels, summary)
        - action_labels: List[str] of length T, each "move"/"pick"/"place"
        - summary: Statistics dict with pick/place moments and segments
    """
    gripper_states = episode_data["gripper_states"]
    T = len(gripper_states)

    logger.debug(f"  Gripper range: min={gripper_states.min():.4f}, max={gripper_states.max():.4f}")

    pick_moments, place_moments = detect_gripper_transitions(
        gripper_states,
        config.get("relative_threshold", 0.1),
        use_relative=True
    )

    logger.debug(f"  Found {len(pick_moments)} pick moments: {pick_moments}")
    logger.debug(f"  Found {len(place_moments)} place moments: {place_moments}")

    # Initialize all as move
    action_labels = ["move"] * T

    # Label pick regions
    pick_segments = []
    for pick_t in pick_moments:
        start_t = max(0, pick_t - config.get("pick_expand_backward", 50))
        end_t = min(T, pick_t + config.get("pick_expand_forward", 30))
        for t in range(start_t, end_t):
            action_labels[t] = "pick"
        pick_segments.append([start_t, end_t])
        logger.debug(f"  Pick labeled: t={start_t} to {end_t} (core={pick_t})")

    # Label place regions (place overwrites pick in overlapping areas)
    place_segments = []
    for place_t in place_moments:
        start_t = max(0, place_t - config.get("place_expand_backward", 100))
        end_t = min(T, place_t + config.get("place_expand_forward", 80))
        for t in range(start_t, end_t):
            action_labels[t] = "place"
        place_segments.append([start_t, end_t])
        logger.debug(f"  Place labeled: t={start_t} to {end_t} (core={place_t})")

    action_counts = {
        "move": action_labels.count("move"),
        "pick": action_labels.count("pick"),
        "place": action_labels.count("place")
    }

    summary = {
        "pick_moments": pick_moments,
        "place_moments": place_moments,
        "num_pick_place_cycles": min(len(pick_moments), len(place_moments)),
        "pick_segments": pick_segments,
        "place_segments": place_segments,
        "action_counts": action_counts
    }

    return action_labels, summary


# ============================================================================
# LLM-based Language Diversity Scaling
# ============================================================================

def rephrase_with_llm(original_description: str, num_variants: int = 5) -> List[str]:
    """
    Use LLM to generate diverse rephrasing of action descriptions.

    Args:
        original_description: Original action description
        num_variants: Number of variants to generate

    Returns:
        List of rephrased descriptions including the original
    """
    try:
        import anthropic

        client = anthropic.Anthropic()

        prompt = f"""Generate {num_variants - 1} diverse paraphrases of this robot action instruction:

Original: "{original_description}"

Requirements:
- Keep the same meaning and action type
- Use varied vocabulary and sentence structures
- Maintain clarity and conciseness
- Each variant should be on a new line
- Do not number or bullet the variants

Generate {num_variants - 1} paraphrases:"""

        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )

        # Parse response
        variants = [original_description]
        response_text = message.content[0].text.strip()

        for line in response_text.split('\n'):
            line = line.strip()
            # Remove numbering if present
            if line and not line[0].isdigit():
                variants.append(line)
            elif line and '. ' in line[:5]:
                variants.append(line.split('. ', 1)[1])

        # Limit to requested number
        variants = variants[:num_variants]

        logger.debug(f"Generated {len(variants)} variants for: {original_description}")
        return variants

    except Exception as e:
        logger.warning(f"LLM rephrasing failed: {e}, using original description")
        return [original_description]


# ============================================================================
# Block-Based Substep Labeling — mirrors label_substeps.py map_timesteps_to_apd_steps()
# ============================================================================

def _extract_apd_steps_by_cycle(apd_plan: List[Dict]) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Extract pick/place/move steps from SO101 APD plan, organized by cycle.

    SO101 APD format uses action_type ('move'/'pick'/'place') and cycle (1/2).
    Move steps are merged into pick/place blocks as prep descriptions.

    Returns:
        (apd_pick_steps, apd_place_steps, apd_pre_pick_moves, apd_pre_place_moves)
        Each dict maps cycle_number -> step dict
    """
    apd_pick_steps = {}    # cycle -> pick step
    apd_place_steps = {}   # cycle -> place step
    apd_pre_pick_moves = {}   # cycle -> move step immediately before pick
    apd_pre_place_moves = {}  # cycle -> move step immediately before place

    for i, step in enumerate(apd_plan):
        action_type = step.get("action_type", "").lower()
        cycle = step.get("cycle", i)

        if action_type == "pick":
            apd_pick_steps[cycle] = step
            # Find the move step immediately before this pick in the plan
            for j in range(i - 1, -1, -1):
                if apd_plan[j].get("action_type", "").lower() == "move":
                    apd_pre_pick_moves[cycle] = apd_plan[j]
                    break

        elif action_type == "place":
            apd_place_steps[cycle] = step
            # Find the move step immediately before this place in the plan
            for j in range(i - 1, -1, -1):
                if apd_plan[j].get("action_type", "").lower() == "move":
                    apd_pre_place_moves[cycle] = apd_plan[j]
                    break

    return apd_pick_steps, apd_place_steps, apd_pre_pick_moves, apd_pre_place_moves


def _detect_gripper_crossings(gripper_states: np.ndarray) -> Tuple[List[int], List[int]]:
    """
    Detect gripper open/close crossings using midpoint threshold on smoothed signal.

    SO101 gripper convention:
      - Low value (~1)  = closed / neutral
      - High value (~27) = open wide

    Returns:
        (open_moments, close_moments)
        open_moments:  timesteps where gripper crosses above midpoint (opening)
        close_moments: timesteps where gripper crosses below midpoint (closing)
    """
    T = len(gripper_states)
    window = 10
    gs_smooth = np.convolve(gripper_states, np.ones(window) / window, mode="same")
    midpoint = (gripper_states.max() + gripper_states.min()) / 2.0
    above = gs_smooth > midpoint

    open_moments: List[int] = []
    close_moments: List[int] = []
    for t in range(1, T):
        if above[t] and not above[t - 1]:
            open_moments.append(t)
        elif not above[t] and above[t - 1]:
            close_moments.append(t)

    return open_moments, close_moments


def label_substeps_blocks(
    episode_data: Dict[str, Any],
    apd_plan: List[Dict],
    config: Dict,
    scale: bool = False
) -> List[Dict]:
    """
    Label each timestep with exactly one APD plan step.

    The APD plan defines a full sequence for the dual pick-place task::

        cycle 1: move → pick → move → place
        cycle 2: move → pick → move → place   (8 steps total)

    Each episode executes this plan **once**. This function creates one
    block per APD step and tiles them over [0, T) with no gaps, no
    overlaps, so ``labeled_timesteps == total_timesteps``.

    Labeling mechanism
    ------------------
    1. Smooth the gripper state signal and find midpoint crossings:

       - **open_moment**  – gripper crosses upward (opens wide)
       - **close_moment** – gripper crosses downward (closes on object)

    2. Map crossings to the APD plan using the physical gripper pattern::

           start (closed) → open (prepare) → CLOSE (= PICK grasp)
                          → OPEN (= PLACE release) → close (carry/idle)
                          → open (prepare) → CLOSE (= PICK grasp)
                          → OPEN (= PLACE release) → close (end)

       So:  close_moment[0] = pick-1 core,  open_moment[1] = place-1 core,
            close_moment[1] = pick-2 core,  open_moment[2] = place-2 core
       (with open_moment[0] = the pre-pick-1 opening, not a place)

    3. Derive move cores as midpoints between neighbouring pick/place cores.

    4. Compute non-overlapping block boundaries via midpoints between
       successive cores, tiling exactly [0, T).
    """
    gripper_states = episode_data["gripper_states"]
    T = len(gripper_states)

    open_moments, close_moments = _detect_gripper_crossings(gripper_states)

    logger.debug(f"  Gripper open  moments: {open_moments}")
    logger.debug(f"  Gripper close moments: {close_moments}")

    # ── Map crossings to the 2 pick and 2 place events ──────────────
    #
    # Physical pattern (start with gripper closed at ~1):
    #   open[0]  = opens to prepare for pick-1
    #   close[0] = closes on object = PICK 1
    #   open[1]  = opens to release = PLACE 1
    #   close[1] = closes (may be idle / prepare for pick-2)
    #   open[2]  = opens to prepare for pick-2  (if present)
    #   close[2] = closes on object = PICK 2    (if present, else use close[1])
    #   open[3]  = opens to release = PLACE 2   (if present, else use open[2])
    #   ... last close = return to neutral
    #
    # We need 4 core moments: pick1, place1, pick2, place2

    def _safe_get(lst, idx, fallback):
        return lst[idx] if idx < len(lst) else fallback

    # pick  = gripper closing on object (close moments)
    # place = gripper opening to release (open moments after the first one)
    pick1_core = _safe_get(close_moments, 0, T // 4)
    place1_core = _safe_get(open_moments, 1, pick1_core + (T - pick1_core) // 3)

    if len(close_moments) >= 3 and len(open_moments) >= 3:
        pick2_core = _safe_get(close_moments, 2, place1_core + 30)
        place2_core = _safe_get(open_moments, 3, pick2_core + 30) if len(open_moments) >= 4 \
            else _safe_get(open_moments, 2, pick2_core + 30)
    elif len(close_moments) >= 2:
        pick2_core = close_moments[-1]
        place2_core = open_moments[-1] if open_moments and open_moments[-1] > pick2_core \
            else min(pick2_core + 30, T - 1)
    else:
        pick2_core = min(place1_core + (T - place1_core) // 3, T - 1)
        place2_core = min(pick2_core + (T - pick2_core) // 2, T - 1)

    # Enforce strict ordering
    place1_core = max(place1_core, pick1_core + 1)
    pick2_core = max(pick2_core, place1_core + 1)
    place2_core = max(place2_core, pick2_core + 1)

    pick_cores = [pick1_core, pick2_core]
    place_cores = [place1_core, place2_core]

    logger.debug(f"  Assigned pick  cores: {pick_cores}")
    logger.debug(f"  Assigned place cores: {place_cores}")

    # ── Build the ordered 8-step list with core moments ─────────────

    pick_iter = 0
    place_iter = 0
    prev_core = 0
    ordered_steps: List[Dict[str, Any]] = []

    for step_idx, step in enumerate(apd_plan):
        action_type = step.get("action_type", "").lower()
        cycle = step.get("cycle", 1)
        description = step.get("description", "")

        if action_type == "pick":
            core = pick_cores[min(pick_iter, len(pick_cores) - 1)]
            pick_iter += 1
        elif action_type == "place":
            core = place_cores[min(place_iter, len(place_cores) - 1)]
            place_iter += 1
        elif action_type == "move":
            next_core = None
            for fs in apd_plan[step_idx + 1:]:
                ft = fs.get("action_type", "").lower()
                if ft == "pick":
                    next_core = pick_cores[min(pick_iter, len(pick_cores) - 1)]
                    break
                elif ft == "place":
                    next_core = place_cores[min(place_iter, len(place_cores) - 1)]
                    break
            if next_core is not None and next_core > prev_core:
                core = (prev_core + next_core) // 2
            else:
                core = min(prev_core + 15, T - 1)
        else:
            core = min(prev_core + 15, T - 1)

        core = max(0, min(core, T - 1))
        if ordered_steps and core <= ordered_steps[-1]["core"]:
            core = min(ordered_steps[-1]["core"] + 1, T - 1)

        ordered_steps.append({
            "action_type": action_type,
            "cycle": cycle,
            "description": description,
            "core": core,
        })
        prev_core = core

    logger.debug(f"  Ordered APD steps ({len(ordered_steps)}):")
    for s in ordered_steps:
        logger.debug(
            f"    core={s['core']} {s['action_type']} cycle={s['cycle']} | {s['description']}"
        )

    # ── Compute non-overlapping boundaries via midpoints ────────────
    boundaries = [0]
    for i in range(len(ordered_steps) - 1):
        mid = (ordered_steps[i]["core"] + ordered_steps[i + 1]["core"]) // 2
        mid = max(boundaries[-1] + 1, mid)
        boundaries.append(min(mid, T))
    boundaries.append(T)

    # ── Generate diverse phrasings if scale=True ────────────────────
    apd_variants: Dict[Tuple, List[str]] = {}
    if scale:
        logger.info("Generating diverse phrasings with LLM...")
        for step in apd_plan:
            key = (step.get("cycle", 0), step.get("action_type", ""))
            if key not in apd_variants:
                apd_variants[key] = rephrase_with_llm(step["description"], num_variants=5)
    else:
        for step in apd_plan:
            key = (step.get("cycle", 0), step.get("action_type", ""))
            if key not in apd_variants:
                apd_variants[key] = [step["description"]]

    # ── Build blocks ────────────────────────────────────────────────
    blocks = []
    for idx, step_info in enumerate(ordered_steps):
        blk_start = boundaries[idx]
        blk_end = boundaries[idx + 1]
        if blk_start >= blk_end:
            continue

        action_type = step_info["action_type"]
        apd_cycle = step_info["cycle"]
        description = step_info["description"]
        core = step_info["core"]

        key = (apd_cycle, action_type)
        apd_step = random.choice(apd_variants.get(key, [description])) if scale else description
        cycle_idx = apd_cycle - 1

        blocks.append({
            "start": blk_start,
            "end": blk_end,
            "type": action_type,
            "cycle": cycle_idx,
            "apd_cycle": apd_cycle,
            "core_moment": core,
            "apd_step": apd_step,
            "apd_variants": apd_variants.get(key, [description]),
        })

        logger.debug(
            f"    {action_type.capitalize()} block (cycle {apd_cycle}): "
            f"[{blk_start}, {blk_end}) core={core} | {apd_step}"
        )

    logger.debug(f"  Total blocks: {len(blocks)}")

    # Step 4: Generate per-timestep labels with EOS markers
    timestep_labels = []

    for block in blocks:
        for t in range(block['start'], block['end']):
            # Randomly select variant if scale=True
            if scale:
                apd_step = random.choice(block['apd_variants'])
            else:
                apd_step = block['apd_step']

            label = {
                "timestep": t,
                "action": block['type'],
                "APD_step": apd_step,
                "cycle": block['cycle'],
                "is_substep_end": (t == block['end'] - 1),
            }
            timestep_labels.append(label)

    # Sort by timestep
    timestep_labels.sort(key=lambda x: x['timestep'])

    coverage_pct = len(timestep_labels) / T * 100 if T > 0 else 0
    logger.debug(
        f"  Generated {len(timestep_labels)} timestep labels "
        f"({coverage_pct:.1f}% coverage)"
    )

    return timestep_labels


# ============================================================================
# Main Processing
# ============================================================================

def process_dataset(
    rlds_data_dir: str,
    dataset_name: str,
    apd_plans: Dict[str, Dict[str, List[Dict]]],
    config: Dict,
    debug: bool = False,
    scale: bool = False
) -> Dict:
    """
    Process entire dataset and generate substep labels.

    Args:
        rlds_data_dir: Path to RLDS dataset directory
        dataset_name: Name of the dataset
        apd_plans: Loaded APD plans
        config: Configuration dictionary
        debug: Enable debug logging
        scale: If True, use LLM to generate diverse phrasings for each substep

    Returns:
        Dictionary with substep labels for each episode
    """

    logger.info(f"Processing dataset: {dataset_name}")
    if scale:
        logger.info("Scale mode enabled: Will generate diverse phrasings with LLM")

    dataset, flat_tfrecord = load_rlds_dataset(rlds_data_dir, dataset_name)
    if dataset is None:
        return {}

    if flat_tfrecord:
        episode_iter: Union[tf.data.Dataset, Iterator] = iter_tfrecord_episodes(dataset)
        logger.info("TFRecord is flat (one row per timestep); grouping by is_last into episodes")
    else:
        episode_iter = dataset

    # Get APD plan
    if dataset_name not in apd_plans:
        logger.error(f"No APD plan found for {dataset_name}")
        return {}

    # For SO101, there's typically one instruction
    instruction_plans = apd_plans[dataset_name]
    if not instruction_plans:
        logger.error(f"No instructions in APD plan for {dataset_name}")
        return {}

    # Get the first (and likely only) instruction plan
    apd_plan = list(instruction_plans.values())[0]

    logger.info(f"APD plan has {len(apd_plan)} steps")
    for step in apd_plan:
        logger.info(
            f"  cycle={step.get('cycle')} action_type={step.get('action_type')} "
            f"description={step.get('description')}"
        )

    # Process episodes
    output_data = {}
    episode_idx = 0

    for episode in episode_iter:
        try:
            # Extract episode data
            episode_data = extract_episode_data(episode)

            if debug:
                logger.info(f"Episode {episode_idx}: {episode_data['length']} timesteps")

            # Label substeps using LIBERO-style block-based approach
            timestep_labels = label_substeps_blocks(episode_data, apd_plan, config, scale=scale)

            # Store in output format (compatible with LIBERO format)
            episode_key = f"{dataset_name}_episode_{episode_idx}"
            output_data[episode_key] = {
                "instruction": episode_data["language_instruction"],
                "total_timesteps": episode_data['length'],
                "labeled_timesteps": len(timestep_labels),
                "timestep_labels": timestep_labels
            }

            episode_idx += 1

        except Exception as e:
            logger.error(f"Error processing episode {episode_idx}: {e}")
            if debug:
                import traceback
                traceback.print_exc()
            continue

    logger.info(f"Processed {episode_idx} episodes")

    return output_data


def main():
    parser = argparse.ArgumentParser(
        description="Label substeps for SO101 dataset using LIBERO-style block-based approach"
    )
    parser.add_argument(
        "--apd_path",
        type=str,
        default="APD_plans_so101.json",
        help="Path to APD plans JSON file"
    )
    parser.add_argument(
        "--rlds_data_dir",
        type=str,
        default="/lustre/fsw/portfolios/edgeai/users/chrislin/datasets/so101_rlds",
        help="Path to RLDS dataset directory"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="substep_labels_so101.json",
        help="Output path for substep labels JSON"
    )
    parser.add_argument(
        "--scale",
        action="store_true",
        default=False,
        help="Enable LLM-based rephrasing to generate diverse action descriptions"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    print("=" * 60)
    print("SO101 Substep Labeling (LIBERO-style Block-Based)")
    print("=" * 60)
    print(f"APD Plans: {args.apd_path}")
    print(f"RLDS Data: {args.rlds_data_dir}")
    print(f"Output:    {args.output_path}")
    print(f"Scale Mode: {args.scale}")
    print("")

    if args.scale:
        print("SCALE MODE ENABLED:")
        print("  - Will use Claude API to generate diverse phrasings")
        print("  - Requires ANTHROPIC_API_KEY environment variable")
        print("  - Each substep will have multiple variant descriptions")
        print("")

    print("DESIGN (mirrors label_substeps.py):")
    print("  1. detect_gripper_transitions()  - relative change, windowed, sigma=2")
    print("  2. label_actions()               - intermediate move/pick/place labeling")
    print("  3. label_substeps_blocks()       - non-overlapping block assignment")
    print("     - pick block end clipped by: next_pick_t - pick_expand_backward")
    print("     - move steps merged into pick/place blocks as APD_prep_step")
    print("     - is_substep_end EOS markers at block boundaries")
    print("")

    # Load APD plans
    apd_plans = load_apd_plans(args.apd_path)

    # Process dataset
    output_data = process_dataset(
        rlds_data_dir=args.rlds_data_dir,
        dataset_name=CONFIG["dataset_name"],
        apd_plans=apd_plans,
        config=CONFIG,
        debug=args.debug,
        scale=args.scale
    )

    # Save output
    if output_data:
        with open(args.output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print("")
        print("=" * 60)
        print("Substep Labeling Complete!")
        print("=" * 60)
        print(f"Output saved to: {args.output_path}")
        print(f"Total episodes labeled: {len(output_data)}")

        # Print statistics
        total_timesteps = sum(ep['total_timesteps'] for ep in output_data.values())
        total_labeled = sum(ep['labeled_timesteps'] for ep in output_data.values())
        coverage = total_labeled / total_timesteps * 100 if total_timesteps > 0 else 0

        print(f"Total timesteps:  {total_timesteps}")
        print(f"Labeled timesteps: {total_labeled} ({coverage:.1f}% coverage)")

        if args.scale:
            print("Each timestep has diverse language variations")

        print("")
        print("Output format (LIBERO-compatible):")
        print("  'action':        pick/place (move merged into pick/place blocks)")
        print("  'APD_step':      substep description (pick or place)")
        print("  'APD_prep_step': move prep description (approach/carry phase)")
        print("  'is_substep_end': EOS marker at last timestep of each block")
        print("")
        print("Next step: Use this file in training with --substep_labels_path")
    else:
        print("\nError: No data processed. Check logs above.")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
