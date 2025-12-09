"""
label_substeps.py

Automatically labels timesteps in LIBERO RLDS episodes with action types (move/pick/place)
based on gripper state transitions and end-effector motion patterns.

Usage:
    python label_substeps.py \
        --apd_path APD_plans.json \
        --rlds_data_dir /path/to/modified_libero_rlds \
        --output_path substep_labels_output.json \
        --suites libero_spatial_no_noops libero_object_no_noops \
        --max_episodes 50

Author: Generated for OpenVLA-OFT LIBERO research
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from scipy.ndimage import gaussian_filter1d

# Disable GPU for data loading
tf.config.set_visible_devices([], 'GPU')

# Global configuration
CONFIG = {
    "gripper_threshold": 0.025,          # Gripper open threshold (adjusted for LIBERO data range 0.002-0.04)
    "pick_expand_backward": 30,          # Pick backward expansion max steps
    "pick_expand_forward": 20,           # Pick forward expansion max steps
    "place_expand_backward": 20,         # Place backward expansion max steps
    "place_expand_forward": 15,          # Place forward expansion max steps
    "z_descent_threshold": -0.005,       # Z-axis descent threshold
    "z_ascent_threshold": 0.01,          # Z-axis ascent threshold
    "movement_threshold": 0.05,          # Movement threshold
    "suite_names": [
        "libero_spatial_no_noops",
        "libero_object_no_noops",
        "libero_goal_no_noops",
        "libero_10_no_noops"
    ]
}

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# ============================================================================
# Module 2.1: Data Loading
# ============================================================================

def load_apd_plans(apd_path: str) -> Dict[str, Dict[str, List[Dict]]]:
    """
    Load APD plans and organize for easy querying.
    
    Args:
        apd_path: Path to APD_plans.json file
    
    Returns:
        {
            "suite_name": {
                "instruction_text": [plan_steps]
            }
        }
    """
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


def load_rlds_dataset(data_dir: str, suite_name: str) -> Optional[tf.data.Dataset]:
    """
    Load RLDS dataset for specified suite.
    
    Args:
        data_dir: RLDS dataset root directory
        suite_name: Task suite name (e.g., "libero_spatial_no_noops")
    
    Returns:
        TensorFlow Dataset object containing all episodes, or None if not found
    """
    dataset_path = os.path.join(data_dir, suite_name) + "/1.0.0/"
    
    if not os.path.exists(dataset_path):
        logger.warning(f"Dataset not found: {dataset_path}")
        return None
    
    try:
        # Load using tensorflow_datasets builder
        builder = tfds.builder_from_directory(dataset_path)
        dataset = builder.as_dataset(split='train')
        
        logger.info(f"Loaded dataset: {suite_name}")
        return dataset
    
    except Exception as e:
        logger.error(f"Failed to load dataset {suite_name}: {e}")
        return None


def extract_episode_data(episode: Dict) -> Dict[str, Any]:
    """
    Extract key data from RLDS episode.
    
    Args:
        episode: RLDS episode dictionary
    
    Returns:
        {
            "observation": {...},
            "action": (T, 7),
            "ee_states": (T, 6),
            "gripper_states": (T, 2),
            "language_instruction": str,
        }
    """
    steps = episode['steps']
    
    # Extract arrays from steps
    actions = []
    ee_states = []
    gripper_states = []
    language_instruction = ""
    
    # Debug: Check episode keys on first call
    if len(actions) == 0:
        logger.debug(f"Episode keys: {list(episode.keys())}")
    
    for step_idx, step in enumerate(steps):
        # Action
        action = step['action'].numpy()
        actions.append(action)
        
        # End-effector state (assuming it's in observation)
        if 'EEF_state' in step['observation']:
            ee_state = step['observation']['EEF_state'].numpy()
        elif 'state' in step['observation']:
            # For some datasets, ee_state is first 6 dims of state
            state = step['observation']['state'].numpy()
            ee_state = state[:6] if len(state) >= 6 else state
        else:
            # Fallback: try to construct from action (less accurate)
            ee_state = action[:6] if len(action) >= 6 else np.zeros(6)
        ee_states.append(ee_state)
        
        # Gripper state
        if 'gripper_state' in step['observation']:
            gripper_state = step['observation']['gripper_state'].numpy()
        elif 'state' in step['observation']:
            # Gripper state is typically last 2 dims
            state = step['observation']['state'].numpy()
            gripper_state = state[-2:] if len(state) >= 2 else np.zeros(2)
        else:
            # Fallback: use last action dim as gripper
            gripper_state = np.array([action[-1], action[-1]]) if len(action) > 0 else np.zeros(2)
        gripper_states.append(gripper_state)
        
        # Get language instruction from first step if available
        if step_idx == 0:
            # Debug: log observation keys
            logger.debug(f"Step 0 observation keys: {list(step['observation'].keys())}")
            
            if not language_instruction and 'language_instruction' in step['observation']:
                language_instruction = step['observation']['language_instruction'].numpy().decode('utf-8')
            
            # Try common alternative keys
            if not language_instruction:
                for key in ['instruction', 'task', 'task_description', 'natural_language_instruction']:
                    if key in step['observation']:
                        try:
                            language_instruction = step['observation'][key].numpy().decode('utf-8')
                            logger.debug(f"Found instruction in step['observation']['{key}']")
                            break
                        except:
                            pass
    
    # Get language instruction from episode level if not found in steps
    if not language_instruction:
        # Check episode_metadata first (LIBERO RLDS format)
        if 'episode_metadata' in episode:
            metadata = episode['episode_metadata']
            logger.debug(f"episode_metadata keys: {list(metadata.keys())}")
            
            # Try to extract from file_path
            if 'file_path' in metadata:
                try:
                    file_path = metadata['file_path'].numpy().decode('utf-8')
                    logger.debug(f"file_path: {file_path}")
                    
                    # Extract task name from file path
                    # Expected format: .../task_name_demo.hdf5
                    import os
                    filename = os.path.basename(file_path)
                    if filename.endswith('_demo.hdf5'):
                        task_name = filename[:-10]  # Remove '_demo.hdf5'
                        # Convert underscores to spaces for instruction
                        language_instruction = task_name.replace('_', ' ')
                        logger.debug(f"Extracted instruction from file_path: '{language_instruction}'")
                except Exception as e:
                    logger.debug(f"Failed to extract from file_path: {e}")
            
            # Try standard keys if file_path extraction didn't work
            if not language_instruction:
                for key in ['language_instruction', 'instruction', 'task', 'task_description', 'natural_language_instruction']:
                    if key in metadata:
                        try:
                            language_instruction = metadata[key].numpy().decode('utf-8')
                            logger.debug(f"Found instruction in episode_metadata['{key}']")
                            break
                        except:
                            pass
        
        # Try direct episode level as fallback
        if not language_instruction:
            for key in ['language_instruction', 'instruction', 'task', 'task_description', 'natural_language_instruction']:
                if key in episode:
                    try:
                        language_instruction = episode[key].numpy().decode('utf-8')
                        logger.debug(f"Found instruction in episode['{key}']")
                        break
                    except:
                        pass
    
    return {
        "action": np.array(actions),
        "ee_states": np.array(ee_states),
        "gripper_states": np.array(gripper_states),
        "language_instruction": language_instruction.lower().strip() if language_instruction else ""
    }


# ============================================================================
# Module 2.2: Action Detection
# ============================================================================

def detect_gripper_transitions(gripper_states: np.ndarray, 
                               threshold: float = 0.04,
                               use_relative: bool = True,
                               relative_threshold: float = 0.008) -> Tuple[List[int], List[int]]:
    """
    Detect gripper open/close transition moments using relative change detection.
    
    Args:
        gripper_states: (T, 2) gripper state array
        threshold: Absolute threshold for determining gripper open (legacy, not used if use_relative=True)
        use_relative: If True, use relative change detection instead of absolute threshold
        relative_threshold: Minimum change in gripper value to count as transition
    
    Returns:
        (pick_moments, place_moments)
        - pick_moments: List of gripper closing moments (open→close)
        - place_moments: List of gripper opening moments (close→open)
    """
    T = len(gripper_states)
    gripper_values = gripper_states[:, 0]
    
    pick_moments = []
    place_moments = []
    
    if use_relative:
        # Use relative change detection (better for different object sizes)
        # Smooth gripper values to reduce noise
        gripper_smooth = gaussian_filter1d(gripper_values, sigma=2)
        
        # Method: Detect transitions by comparing before/after windows
        window = 8  # Larger window for more stable detection
        
        for t in range(window, T - window):
            # Compare windows before and after this timestep
            before_window = gripper_smooth[t-window:t]
            after_window = gripper_smooth[t+1:t+window+1]
            
            before_avg = np.mean(before_window)
            after_avg = np.mean(after_window)
            current = gripper_smooth[t]
            
            # Detect significant closing (before > after)
            if before_avg > after_avg + relative_threshold:
                # Gripper closing: before was more open, after is more closed
                if not pick_moments or t - pick_moments[-1] > 20:  # Avoid duplicates
                    pick_moments.append(t)
                    logger.debug(f"    Pick detected at t={t}: before={before_avg:.4f}, after={after_avg:.4f}, change={before_avg-after_avg:.4f}")
            
            # Detect significant opening (after > before)
            # Use smaller threshold for place since opening can be gradual
            elif after_avg > before_avg + relative_threshold * 0.4:  # More sensitive (0.4x)
                # Gripper opening: before was closed, after is more open
                if not place_moments or t - place_moments[-1] > 20:  # Avoid duplicates
                    place_moments.append(t)
                    logger.debug(f"    Place detected at t={t}: before={before_avg:.4f}, after={after_avg:.4f}, change={after_avg-before_avg:.4f}")
    else:
        # Legacy: absolute threshold method
        is_gripper_open = gripper_values > threshold
        
        for t in range(1, T):
            # Open → Close = Pick
            if is_gripper_open[t-1] and not is_gripper_open[t]:
                pick_moments.append(t)
            # Close → Open = Place
            elif not is_gripper_open[t-1] and is_gripper_open[t]:
                place_moments.append(t)
    
    return pick_moments, place_moments


def expand_pick_range(ee_states: np.ndarray, 
                     pick_t: int, 
                     T: int,
                     max_backward: int = 30,
                     max_forward: int = 20) -> Tuple[int, int]:
    """
    Expand Pick action time range based on Z-axis motion.
    
    Pick process:
    1. Descend to approach object (backward expansion)
    2. Gripper closes (core moment pick_t)
    3. Lift object (forward expansion)
    
    Args:
        ee_states: (T, 6) end-effector state [x,y,z,rx,ry,rz]
        pick_t: Core moment when gripper closes
        T: Total timesteps
        max_backward: Max backward expansion steps
        max_forward: Max forward expansion steps
    
    Returns:
        (start_t, end_t) Start and end time of Pick action
    """
    z_positions = ee_states[:, 2]
    
    # Backward expansion: Find when descent starts
    start_t = max(0, pick_t - max_backward)
    for t in range(pick_t-1, max(0, pick_t-max_backward), -1):
        if t >= 5:
            # Check if Z is descending
            z_trend = z_positions[t] - z_positions[t-5]
            if z_trend < CONFIG['z_descent_threshold']:
                start_t = t - 5
                break
    
    # Forward expansion: Find when ascent stops (object lifted)
    end_t = min(T, pick_t + max_forward)
    max_z_after_pick = z_positions[pick_t]
    
    for t in range(pick_t+1, min(T, pick_t+max_forward)):
        if z_positions[t] > max_z_after_pick:
            max_z_after_pick = z_positions[t]
            end_t = t + 1
        elif z_positions[t] < max_z_after_pick - 0.01:
            # Z starts descending, stop expansion
            break
    
    return start_t, end_t


def expand_place_range(ee_states: np.ndarray,
                      place_t: int,
                      T: int,
                      max_backward: int = 20,
                      max_forward: int = 15) -> Tuple[int, int]:
    """
    Expand Place action time range based on position and Z-axis motion.
    
    Place process:
    1. Move to target position and descend (backward expansion)
    2. Gripper opens (core moment place_t)
    3. May lift up or move away (forward expansion)
    
    Args:
        ee_states: (T, 6) end-effector state
        place_t: Core moment when gripper opens
        T: Total timesteps
        max_backward: Max backward expansion steps
        max_forward: Max forward expansion steps
    
    Returns:
        (start_t, end_t) Start and end time of Place action
    """
    z_positions = ee_states[:, 2]
    positions = ee_states[:, :3]
    
    # Backward expansion: Find when descent/approach starts
    start_t = max(0, place_t - max_backward)
    for t in range(place_t-1, max(0, place_t-max_backward), -1):
        if t >= 5:
            # Check if descending or moving rapidly
            z_trend = z_positions[t] - z_positions[t-5]
            movement = np.linalg.norm(positions[t] - positions[t-5])
            
            if z_trend < CONFIG['z_descent_threshold'] or movement > CONFIG['movement_threshold']:
                start_t = t - 5
                break
    
    # Forward expansion: Find when gripper stabilizes or moves away after opening
    end_t = min(T, place_t + max_forward)
    for t in range(place_t+1, min(T, place_t+max_forward)):
        if t < T - 5:
            # Check if moving away (Z ascent or lateral movement)
            z_trend = z_positions[t+5] - z_positions[t]
            movement = np.linalg.norm(positions[t+5] - positions[t])
            
            if z_trend > CONFIG['z_ascent_threshold'] or movement > CONFIG['movement_threshold']:
                end_t = t + 5
                break
    
    return start_t, end_t


def label_actions(episode_data: Dict[str, np.ndarray],
                 gripper_threshold: float = 0.04) -> Tuple[List[str], Dict]:
    """
    Label each timestep with action type.
    
    Process:
    1. Detect all gripper transitions (may have multiple picks and places)
    2. Expand range for each pick moment and label
    3. Expand range for each place moment and label
    4. Label all remaining time as move
    5. Handle overlaps: later labels overwrite earlier ones
    
    Args:
        episode_data: Dict containing ee_states and gripper_states
        gripper_threshold: Gripper open threshold
    
    Returns:
        (action_labels, summary)
        - action_labels: List of length T, each element is "move"/"pick"/"place"
        - summary: Statistics dict
    """
    ee_states = episode_data['ee_states']
    gripper_states = episode_data['gripper_states']
    T = len(ee_states)
    
    # Debug: Print gripper state statistics
    logger.debug(f"  Gripper states range: min={gripper_states[:, 0].min():.4f}, max={gripper_states[:, 0].max():.4f}")
    logger.debug(f"  Gripper threshold: {gripper_threshold}")
    logger.debug(f"  Gripper open count: {(gripper_states[:, 0] > gripper_threshold).sum()}/{T}")
    
    # Output complete gripper sequence for analysis
    logger.debug(f"  Complete gripper sequence ({T} timesteps):")
    for t in range(0, T, 10):  # Print every 10th timestep to avoid clutter
        end_t = min(t + 10, T)
        values_str = ', '.join([f"{gripper_states[i, 0]:.4f}" for i in range(t, end_t)])
        logger.debug(f"    t={t:3d}-{end_t-1:3d}: [{values_str}]")
    
    # Detect gripper transitions using relative change detection
    pick_moments, place_moments = detect_gripper_transitions(
        gripper_states, 
        gripper_threshold,
        use_relative=True,
        relative_threshold=0.008
    )
    
    logger.debug(f"  Found {len(pick_moments)} pick moments: {pick_moments}")
    logger.debug(f"  Found {len(place_moments)} place moments: {place_moments}")
    
    # Initialize all as "move"
    action_labels = ["move"] * T
    
    # Track segments
    pick_segments = []
    place_segments = []
    
    # Label Pick regions
    for pick_t in pick_moments:
        start_t, end_t = expand_pick_range(
            ee_states, pick_t, T,
            CONFIG['pick_expand_backward'],
            CONFIG['pick_expand_forward']
        )
        
        for t in range(start_t, end_t):
            action_labels[t] = "pick"
        
        pick_segments.append([start_t, end_t])
        logger.debug(f"  Pick labeled: t={start_t} to {end_t} (core={pick_t})")
    
    # Label Place regions
    for place_t in place_moments:
        start_t, end_t = expand_place_range(
            ee_states, place_t, T,
            CONFIG['place_expand_backward'],
            CONFIG['place_expand_forward']
        )
        
        for t in range(start_t, end_t):
            action_labels[t] = "place"
        
        place_segments.append([start_t, end_t])
        logger.debug(f"  Place labeled: t={start_t} to {end_t} (core={place_t})")
    
    # Extract move segments
    move_segments = []
    in_move = False
    move_start = 0
    
    for t in range(T):
        if action_labels[t] == "move":
            if not in_move:
                move_start = t
                in_move = True
        else:
            if in_move:
                move_segments.append([move_start, t])
                in_move = False
    
    if in_move:
        move_segments.append([move_start, T])
    
    # Count actions
    action_counts = {
        "move": action_labels.count("move"),
        "pick": action_labels.count("pick"),
        "place": action_labels.count("place")
    }
    
    # Create summary
    summary = {
        "pick_moments": pick_moments,
        "place_moments": place_moments,
        "num_pick_place_cycles": min(len(pick_moments), len(place_moments)),
        "pick_segments": pick_segments,
        "place_segments": place_segments,
        "move_segments": move_segments,
        "action_counts": action_counts
    }
    
    return action_labels, summary


# ============================================================================
# Module 2.3: Output Generation
# ============================================================================

def map_timesteps_to_apd_steps(action_labels: List[str], 
                               summary: Dict,
                               apd_plan: Optional[List[Dict]],
                               instruction: str) -> List[Dict]:
    """
    Map timesteps to APD plan substeps.
    
    Args:
        action_labels: List of action types for each timestep
        summary: Summary dict with pick/place moments and segments
        apd_plan: APD plan steps (or None if not found)
        instruction: The instruction text
    
    Returns:
        List of dicts with timestep and APD_step mapping
    """
    T = len(action_labels)
    timestep_labels = []
    
    if apd_plan is None or len(apd_plan) == 0:
        # No plan found, just label with action types
        for t in range(T):
            timestep_labels.append({
                "timestep": t,
                "action": action_labels[t],
                "APD_step": "unknown"
            })
        return timestep_labels
    
    # Create action segments with their types
    segments = []
    
    # Add pick segments
    for i, (start, end) in enumerate(summary['pick_segments']):
        segments.append({
            'start': start,
            'end': end,
            'type': 'pick',
            'index': i
        })
    
    # Add place segments
    for i, (start, end) in enumerate(summary['place_segments']):
        segments.append({
            'start': start,
            'end': end,
            'type': 'place',
            'index': i
        })
    
    # Add move segments
    for i, (start, end) in enumerate(summary['move_segments']):
        segments.append({
            'start': start,
            'end': end,
            'type': 'move',
            'index': i
        })
    
    # Sort segments by start time
    segments.sort(key=lambda x: x['start'])
    
    # Map segments to APD steps
    # Typical pattern: move -> pick -> move -> place -> move
    # APD pattern: Move step -> Pick step -> Move step -> Place step -> Return step
    
    apd_step_index = 0
    
    for seg in segments:
        # Find matching APD step based on action type
        while apd_step_index < len(apd_plan):
            apd_subgoal = apd_plan[apd_step_index]['subgoal'].lower()
            
            if seg['type'] == 'move':
                if any(kw in apd_subgoal for kw in ['move', 'reach', 'return']):
                    break
            elif seg['type'] == 'pick':
                if any(kw in apd_subgoal for kw in ['pick', 'grasp', 'lift']):
                    break
            elif seg['type'] == 'place':
                if any(kw in apd_subgoal for kw in ['place', 'put', 'lower', 'release']):
                    break
            
            apd_step_index += 1
        
        # Assign APD step to this segment
        if apd_step_index < len(apd_plan):
            seg['apd_step'] = apd_plan[apd_step_index]['subgoal']
            apd_step_index += 1
        else:
            seg['apd_step'] = apd_plan[-1]['subgoal'] if apd_plan else 'unknown'
    
    # Create timestep labels
    for t in range(T):
        # Find which segment this timestep belongs to
        apd_step_text = 'unknown'
        action_type = action_labels[t]
        
        for seg in segments:
            if seg['start'] <= t < seg['end']:
                apd_step_text = seg.get('apd_step', 'unknown')
                break
        
        timestep_labels.append({
            "timestep": t,
            "action": action_type,
            "APD_step": apd_step_text
        })
    
    return timestep_labels


def match_instruction_to_plan(instruction: str, 
                             apd_plans: Dict[str, Dict]) -> Optional[List[Dict]]:
    """
    Match RLDS instruction to APD plan.
    
    Args:
        instruction: Language instruction from RLDS dataset
        apd_plans: APD plans dictionary
    
    Returns:
        Matched plan steps list, or None if not found
    """
    instruction_clean = instruction.lower().strip()
    
    # Try exact match first
    for suite, plans in apd_plans.items():
        if instruction_clean in plans:
            return plans[instruction_clean]
    
    # Try fuzzy match (simple substring matching)
    for suite, plans in apd_plans.items():
        for plan_instruction, plan in plans.items():
            if plan_instruction in instruction_clean or instruction_clean in plan_instruction:
                logger.debug(f"  Fuzzy matched: '{instruction_clean}' → '{plan_instruction}'")
                return plan
    
    logger.warning(f"  No matching plan found for: '{instruction_clean}'")
    return None


def create_output_structure(suite_name: str,
                           task_name: str,
                           episode_idx: int,
                           instruction: str,
                           action_labels: List[str],
                           summary: Dict,
                           apd_plan: Optional[List[Dict]] = None) -> Dict:
    """
    Create output data structure for single episode.
    
    Args:
        suite_name: Task suite name
        task_name: Task name
        episode_idx: Episode index
        instruction: Language instruction
        action_labels: Timestep-level action labels
        summary: Statistics summary
        apd_plan: APD plan steps (optional)
    
    Returns:
        Formatted episode data dictionary
    """
    # Map timesteps to APD steps
    timestep_labels = map_timesteps_to_apd_steps(
        action_labels, 
        summary, 
        apd_plan,
        instruction
    )
    
    return {
        "instruction": instruction,
        "total_timesteps": len(action_labels),
        "timestep_labels": timestep_labels,
        "summary": summary
    }


def save_results(output_data: Dict, output_path: str) -> None:
    """
    Save results to JSON file.
    
    Args:
        output_data: Complete labeling results
        output_path: Output file path
    """
    logger.info(f"Saving results to {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved successfully")


# ============================================================================
# Module 2.4: Main Processing
# ============================================================================

def process_single_episode(episode: Dict,
                          episode_idx: int,
                          suite_name: str,
                          apd_plans: Dict) -> Optional[Dict]:
    """
    Process single episode with complete workflow.
    
    Args:
        episode: RLDS episode data
        episode_idx: Episode index
        suite_name: Task suite name
        apd_plans: APD plans dictionary
    
    Returns:
        Labeling results for this episode, or None if failed
    """
    try:
        # Extract episode data
        episode_data = extract_episode_data(episode)
        instruction = episode_data['language_instruction']
        
        if not instruction:
            logger.warning(f"  Episode {episode_idx}: No instruction found, skipping")
            return None
        
        logger.info(f"  Episode {episode_idx}: '{instruction}'")
        
        # Find matching APD plan
        apd_plan = match_instruction_to_plan(instruction, apd_plans)
        if apd_plan:
            logger.info(f"    Found APD plan with {len(apd_plan)} steps")
        else:
            logger.warning(f"    No matching APD plan found")
        
        # Label actions
        action_labels, summary = label_actions(
            episode_data,
            CONFIG['gripper_threshold']
        )
        
        # Create output structure
        task_name = instruction.replace(' ', '_')
        result = create_output_structure(
            suite_name,
            task_name,
            episode_idx,
            instruction,
            action_labels,
            summary,
            apd_plan
        )
        
        # Log summary
        logger.info(f"    Total timesteps: {len(action_labels)}")
        logger.info(f"    Action counts: {summary['action_counts']}")
        logger.info(f"    Pick-Place cycles: {summary['num_pick_place_cycles']}")
        
        return result
    
    except Exception as e:
        logger.error(f"  Episode {episode_idx}: Error - {e}", exc_info=True)
        return None


def process_suite(rlds_data_dir: str,
                 suite_name: str,
                 apd_plans: Dict,
                 max_episodes: Optional[int] = None,
                 episode_ids: Optional[List[int]] = None) -> Dict:
    """
    Process entire task suite.
    
    Args:
        rlds_data_dir: RLDS dataset root directory
        suite_name: Task suite name
        apd_plans: APD plans dictionary
        max_episodes: Maximum episodes to process (None = all)
        episode_ids: Specific episode IDs to process (None = all)
    
    Returns:
        All labeling results for this suite
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing suite: {suite_name}")
    logger.info(f"{'='*60}")
    
    if episode_ids:
        logger.info(f"Processing specific episodes: {episode_ids}")
    
    # Load dataset
    dataset = load_rlds_dataset(rlds_data_dir, suite_name)
    if dataset is None:
        return {}
    
    # Process episodes
    suite_results = {}
    episode_count = 0
    
    for episode_idx, episode in enumerate(dataset):
        # Skip if not in specified episode_ids
        if episode_ids is not None and episode_idx not in episode_ids:
            continue
        
        # Stop if max_episodes limit reached
        if max_episodes and episode_count >= max_episodes:
            logger.info(f"Reached max_episodes limit: {max_episodes}")
            break
        
        result = process_single_episode(episode, episode_idx, suite_name, apd_plans)
        
        if result is not None:
            instruction = result['instruction']
            task_name = instruction.replace(' ', '_')
            
            # Organize by task
            if task_name not in suite_results:
                suite_results[task_name] = {}
            
            suite_results[task_name][f"episode_{episode_idx}"] = result
            episode_count += 1
    
    logger.info(f"Suite {suite_name}: Processed {episode_count} episodes")
    
    return suite_results


def main(apd_path: str,
        rlds_data_dir: str,
        output_path: str,
        suites: Optional[List[str]] = None,
        max_episodes_per_suite: Optional[int] = None,
        episode_ids: Optional[List[int]] = None) -> None:
    """
    Main function: Process all suites and generate output.
    
    Args:
        apd_path: APD_plans.json path
        rlds_data_dir: RLDS dataset root directory
        output_path: Output JSON file path
        suites: List of suites to process (None = all)
        max_episodes_per_suite: Max episodes per suite
        episode_ids: Specific episode IDs to process (None = all)
    """
    logger.info("="*60)
    logger.info("LIBERO Substep Labeling Tool")
    logger.info("="*60)
    
    # Load APD plans
    apd_plans = load_apd_plans(apd_path)
    
    # Determine which suites to process
    if suites is None:
        suites = CONFIG['suite_names']
    
    logger.info(f"\nProcessing {len(suites)} suites:")
    for suite in suites:
        logger.info(f"  - {suite}")
    
    # Process each suite
    all_results = {}
    total_episodes = 0
    
    for suite_name in suites:
        suite_results = process_suite(
            rlds_data_dir,
            suite_name,
            apd_plans,
            max_episodes_per_suite,
            episode_ids
        )
        
        if suite_results:
            # Extract suite short name (remove _no_noops suffix)
            suite_short = suite_name.replace('_no_noops', '')
            all_results[suite_short] = suite_results
            
            # Count episodes
            for task_results in suite_results.values():
                total_episodes += len(task_results)
    
    # Save results
    save_results(all_results, output_path)
    
    # Print final statistics
    logger.info("\n" + "="*60)
    logger.info("Processing Complete!")
    logger.info("="*60)
    logger.info(f"Total suites processed: {len(all_results)}")
    logger.info(f"Total episodes processed: {total_episodes}")
    logger.info(f"Output saved to: {output_path}")
    
    # Print per-suite statistics
    for suite_name, suite_results in all_results.items():
        task_count = len(suite_results)
        episode_count = sum(len(task) for task in suite_results.values())
        logger.info(f"  {suite_name}: {task_count} tasks, {episode_count} episodes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Label LIBERO episodes with move/pick/place actions"
    )
    
    parser.add_argument(
        "--apd_path",
        type=str,
        default="APD_plans.json",
        help="Path to APD_plans.json file"
    )
    
    parser.add_argument(
        "--rlds_data_dir",
        type=str,
        required=True,
        help="Path to RLDS dataset directory (e.g., /path/to/modified_libero_rlds)"
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        default="substep_labels_output.json",
        help="Path to output JSON file"
    )
    
    parser.add_argument(
        "--suites",
        type=str,
        nargs='+',
        default=None,
        help="List of suites to process (default: all)"
    )
    
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Maximum episodes to process per suite (default: all)"
    )
    
    parser.add_argument(
        "--episode_ids",
        type=int,
        nargs='+',
        default=None,
        help="Specific episode IDs to process (e.g., --episode_ids 0 5 10 15)"
    )
    
    parser.add_argument(
        "--debug",
        action='store_true',
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Set debug level if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Run main processing
    main(
        apd_path=args.apd_path,
        rlds_data_dir=args.rlds_data_dir,
        output_path=args.output_path,
        suites=args.suites,
        max_episodes_per_suite=args.max_episodes,
        episode_ids=args.episode_ids
    )

