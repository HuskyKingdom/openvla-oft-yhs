"""
label_substeps.py

Automatically labels timesteps in LIBERO RLDS episodes with action types (move/pick/place)
based on gripper state transitions and end-effector motion patterns.
Updated to support explicit 'action_type' field parsing for high-entropy subgoals.

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
    "pick_expand_backward": 50,          # Pick backward expansion max steps
    "pick_expand_forward": 30,           # Pick forward expansion max steps
    "place_expand_backward": 100,        # Place backward expansion max steps (much larger)
    "place_expand_forward": 80,          # Place forward expansion max steps (much larger)
    "z_descent_threshold": -0.002,       # Z-axis descent threshold (more sensitive)
    "z_ascent_threshold": 0.005,         # Z-axis ascent threshold (more sensitive)
    "movement_threshold": 0.03,          # Movement threshold (more sensitive)
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
    """
    steps = episode['steps']
    
    # Extract arrays from steps
    actions = []
    ee_states = []
    gripper_states = []
    language_instruction = ""
    
    if len(actions) == 0:
        logger.debug(f"Episode keys: {list(episode.keys())}")
    
    for step_idx, step in enumerate(steps):
        # Action
        action = step['action'].numpy()
        actions.append(action)
        
        # End-effector state
        if 'EEF_state' in step['observation']:
            ee_state = step['observation']['EEF_state'].numpy()
        elif 'state' in step['observation']:
            state = step['observation']['state'].numpy()
            ee_state = state[:6] if len(state) >= 6 else state
        else:
            ee_state = action[:6] if len(action) >= 6 else np.zeros(6)
        ee_states.append(ee_state)
        
        # Gripper state
        if 'gripper_state' in step['observation']:
            gripper_state = step['observation']['gripper_state'].numpy()
        elif 'state' in step['observation']:
            state = step['observation']['state'].numpy()
            gripper_state = state[-2:] if len(state) >= 2 else np.zeros(2)
        else:
            gripper_state = np.array([action[-1], action[-1]]) if len(action) > 0 else np.zeros(2)
        gripper_states.append(gripper_state)
        
        # Get language instruction from first step if available
        if step_idx == 0:
            logger.debug(f"Step 0 observation keys: {list(step['observation'].keys())}")
            
            if not language_instruction and 'language_instruction' in step['observation']:
                language_instruction = step['observation']['language_instruction'].numpy().decode('utf-8')
            
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
        if 'episode_metadata' in episode:
            metadata = episode['episode_metadata']
            
            if 'file_path' in metadata: 
                try:
                    file_path = metadata['file_path'].numpy().decode('utf-8')
                    import os
                    filename = os.path.basename(file_path)
                    if filename.endswith('_demo.hdf5'):
                        task_name = filename[:-10]
                        language_instruction = task_name.replace('_', ' ')
                except Exception as e:
                    pass
            
            if not language_instruction:
                for key in ['language_instruction', 'instruction', 'task', 'task_description', 'natural_language_instruction']:
                    if key in metadata:
                        try:
                            language_instruction = metadata[key].numpy().decode('utf-8')
                            break
                        except:
                            pass
        
        if not language_instruction:
            for key in ['language_instruction', 'instruction', 'task', 'task_description', 'natural_language_instruction']:
                if key in episode:
                    try:
                        language_instruction = episode[key].numpy().decode('utf-8')
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
    """
    T = len(gripper_states)
    gripper_values = gripper_states[:, 0]
    
    pick_moments = []
    place_moments = []
    
    if use_relative:
        gripper_smooth = gaussian_filter1d(gripper_values, sigma=2)
        window = 8  
        
        for t in range(window, T - window):
            before_window = gripper_smooth[t-window:t]
            after_window = gripper_smooth[t+1:t+window+1]
            
            before_avg = np.mean(before_window)
            after_avg = np.mean(after_window)
            
            if before_avg > after_avg + relative_threshold:
                if not pick_moments or t - pick_moments[-1] > 20: 
                    pick_moments.append(t)
            
            elif after_avg > before_avg + relative_threshold * 0.4:  
                if not place_moments or t - place_moments[-1] > 20: 
                    place_moments.append(t)
    else:
        is_gripper_open = gripper_values > threshold
        
        for t in range(1, T):
            if is_gripper_open[t-1] and not is_gripper_open[t]:
                pick_moments.append(t)
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
    """
    start_t = max(0, pick_t - max_backward)
    end_t = min(T, pick_t + max_forward)
    return start_t, end_t


def expand_place_range(ee_states: np.ndarray,
                      place_t: int,
                      T: int,
                      max_backward: int = 20,
                      max_forward: int = 15) -> Tuple[int, int]:
    """
    Expand Place action time range based on position and Z-axis motion.
    """
    start_t = max(0, place_t - max_backward)
    end_t = min(T, place_t + max_forward)
    return start_t, end_t


def label_actions(episode_data: Dict[str, np.ndarray],
                 gripper_threshold: float = 0.04) -> Tuple[List[str], Dict]:
    """
    Label each timestep with action type.
    """
    ee_states = episode_data['ee_states']
    gripper_states = episode_data['gripper_states']
    T = len(ee_states)
    
    pick_moments, place_moments = detect_gripper_transitions(
        gripper_states, 
        gripper_threshold,
        use_relative=True,
        relative_threshold=0.008
    )
    
    action_labels = ["move"] * T
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
    Map timesteps to APD plan substeps utilizing the 'action_type' field.
    """
    T = len(action_labels)
    timestep_labels = []
    
    if apd_plan is None or len(apd_plan) == 0:
        logger.warning(f"  No APD plan found, cannot map timesteps")
        return []
    
    pick_moments = summary['pick_moments']
    place_moments = summary['place_moments']
    
    blocks = []
    num_cycles = max(len(pick_moments), len(place_moments))
    
    for cycle_idx in range(num_cycles):
        # Pick block
        if cycle_idx < len(pick_moments):
            pick_t = pick_moments[cycle_idx]
            pick_start = blocks[-1]['end'] if len(blocks) > 0 else 0
            pick_end = min(T, pick_t + CONFIG['pick_expand_forward'])
            
            if cycle_idx + 1 < len(pick_moments):
                next_pick_t = pick_moments[cycle_idx + 1]
                pick_end = min(pick_end, next_pick_t - CONFIG['pick_expand_backward'])
            
            blocks.append({
                'start': pick_start,
                'end': pick_end,
                'type': 'pick',
                'cycle': cycle_idx,
                'core_moment': pick_t
            })
        
        # Place block
        if cycle_idx < len(place_moments):
            place_t = place_moments[cycle_idx]
            place_start = blocks[-1]['end'] if len(blocks) > 0 else 0
            place_end = min(T, place_t + CONFIG['place_expand_forward'])
            
            if cycle_idx + 1 < len(pick_moments):
                next_pick_t = pick_moments[cycle_idx + 1]
                place_end = min(place_end, next_pick_t - CONFIG['pick_expand_backward'])
            
            blocks.append({
                'start': place_start,
                'end': place_end,
                'type': 'place',
                'cycle': cycle_idx,
                'core_moment': place_t
            })
    
    # === NEW: Map blocks to APD steps utilizing explicit 'action_type' ===
    apd_pick_steps = []
    apd_place_steps = []
    
    for step in apd_plan:
        action_type = step.get('action_type', '').lower()
        subgoal = step.get('subgoal', '').lower()
        
        # Determine if step is pick/place via action_type or fallback to subgoal matching
        is_pick = (action_type == 'pick') or (not action_type and any(kw in subgoal for kw in ['pick', 'grasp', 'lift']))
        is_place = (action_type == 'place') or (not action_type and any(kw in subgoal for kw in ['place', 'put', 'lower', 'release']))
        
        if is_pick:
            apd_pick_steps.append(step)
        elif is_place:
            apd_place_steps.append(step)
    
    # Assign APD steps to blocks
    for block in blocks:
        if block['type'] == 'pick':
            cycle = block['cycle']
            if cycle < len(apd_pick_steps):
                # Find Move step before this Pick in APD plan
                for step in apd_plan:
                    action_type = step.get('action_type', '').lower()
                    subgoal = step.get('subgoal', '').lower()
                    
                    is_move = (action_type == 'move') or (not action_type and any(kw in subgoal for kw in ['move', 'reach', 'navigate', 'hover', 'position', 'align', 'translate', 'direct', 'shift']))
                    
                    if is_move and step['step'] < apd_pick_steps[cycle]['step']:
                        block['apd_step'] = apd_pick_steps[cycle]['subgoal']
                        block['apd_prep_step'] = step['subgoal']
                        break
                if 'apd_step' not in block:
                    block['apd_step'] = apd_pick_steps[cycle]['subgoal']
            else:
                block['apd_step'] = 'Pick (no APD match)'
        
        elif block['type'] == 'place':
            cycle = block['cycle']
            if cycle < len(apd_place_steps):
                block['apd_step'] = apd_place_steps[cycle]['subgoal']
            else:
                block['apd_step'] = 'Place (no APD match)'
    
    # Generate timestep labels
    for block in blocks:
        for t in range(block['start'], block['end']):
            timestep_labels.append({
                "timestep": t,
                "action": block['type'],
                "APD_step": block['apd_step'],
                "cycle": block['cycle'],
                "is_substep_end": (t == block['end'] - 1) 
            })
    
    timestep_labels.sort(key=lambda x: x['timestep'])
    return timestep_labels


def match_instruction_to_plan(instruction: str, 
                             apd_plans: Dict[str, Dict]) -> Optional[List[Dict]]:
    """
    Match RLDS instruction to APD plan.
    """
    instruction_clean = instruction.lower().strip()
    
    for suite, plans in apd_plans.items():
        if instruction_clean in plans:
            return plans[instruction_clean]
    
    for suite, plans in apd_plans.items():
        for plan_instruction, plan in plans.items():
            if plan_instruction in instruction_clean or instruction_clean in plan_instruction:
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
    """
    timestep_labels = map_timesteps_to_apd_steps(
        action_labels, 
        summary, 
        apd_plan,
        instruction
    )
    
    labeled_pick_count = sum(1 for label in timestep_labels if label['action'] == 'pick')
    labeled_place_count = sum(1 for label in timestep_labels if label['action'] == 'place')
    
    substep_boundaries = [label['timestep'] for label in timestep_labels if label.get('is_substep_end', False)]
    num_substeps = len(substep_boundaries)
    
    summary_updated = summary.copy()
    summary_updated['action_counts'] = {
        "pick": labeled_pick_count,
        "place": labeled_place_count,
        "move": 0 
    }
    summary_updated['labeled_timesteps'] = len(timestep_labels)
    summary_updated['unlabeled_timesteps'] = len(action_labels) - len(timestep_labels)
    summary_updated['substep_boundaries'] = substep_boundaries
    summary_updated['num_substeps'] = num_substeps
    
    return {
        "instruction": instruction,
        "total_timesteps": len(action_labels),
        "labeled_timesteps": len(timestep_labels),
        "timestep_labels": timestep_labels,
        "summary": summary_updated
    }


def save_results(output_data: Dict, output_path: str) -> None:
    """
    Save results to JSON file.
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
    """
    try:
        episode_data = extract_episode_data(episode)
        instruction = episode_data['language_instruction']
        
        if not instruction:
            logger.warning(f"  Episode {episode_idx}: No instruction found, skipping")
            return None
        
        logger.info(f"  Episode {episode_idx}: '{instruction}'")
        
        apd_plan = match_instruction_to_plan(instruction, apd_plans)
        
        action_labels, summary = label_actions(
            episode_data,
            CONFIG['gripper_threshold']
        )
        
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
        
        return result
    
    except Exception as e:
        logger.error(f"  Episode {episode_idx}: Error - {e}", exc_info=True)
        return None


def process_suite(rlds_data_dir: str,
                 suite_name: str,
                 apd_plans: Dict,
                 max_episodes: Optional[int] = None,
                 episode_ids: Optional[List[int]] = None) -> Tuple[Dict, int, int, int, int]:
    """
    Process entire task suite.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing suite: {suite_name}")
    logger.info(f"{'='*60}")
    
    dataset = load_rlds_dataset(rlds_data_dir, suite_name)
    if dataset is None:
        return {}
    
    suite_results = {}
    episode_count = 0
    all_instructions = set()  
    labeled_instructions = set()  
    total_timesteps = 0  
    total_labeled_timesteps = 0  
    
    for episode_idx, episode in enumerate(dataset):
        if episode_ids is not None and episode_idx not in episode_ids:
            continue
        
        if max_episodes and episode_count >= max_episodes:
            break
        
        try:
            episode_data = extract_episode_data(episode)
            instruction = episode_data['language_instruction']
            if instruction:
                all_instructions.add(instruction)
        except:
            pass
        
        result = process_single_episode(episode, episode_idx, suite_name, apd_plans)
        
        if result is not None:
            instruction = result['instruction']
            task_name = instruction.replace(' ', '_')
            
            labeled_instructions.add(instruction)
            total_timesteps += result['total_timesteps']
            total_labeled_timesteps += result['labeled_timesteps']
            
            if task_name not in suite_results:
                suite_results[task_name] = {}
            
            suite_results[task_name][f"episode_{episode_idx}"] = result
            episode_count += 1
    
    return suite_results, len(all_instructions), len(labeled_instructions), total_timesteps, total_labeled_timesteps


def main(apd_path: str,
        rlds_data_dir: str,
        output_path: str,
        suites: Optional[List[str]] = None,
        max_episodes_per_suite: Optional[int] = None,
        episode_ids: Optional[List[int]] = None) -> None:
    """
    Main function: Process all suites and generate output.
    """
    logger.info("="*60)
    logger.info("LIBERO Substep Labeling Tool")
    logger.info("="*60)
    
    apd_plans = load_apd_plans(apd_path)
    
    if suites is None:
        suites = CONFIG['suite_names']
    
    all_results = {}
    total_episodes = 0
    total_all_instructions = 0
    total_labeled_instructions = 0
    grand_total_timesteps = 0
    grand_total_labeled_timesteps = 0
    
    for suite_name in suites:
        suite_results, all_instr_count, labeled_instr_count, suite_timesteps, suite_labeled_timesteps = process_suite(
            rlds_data_dir,
            suite_name,
            apd_plans,
            max_episodes_per_suite,
            episode_ids
        )
        
        if suite_results:
            suite_short = suite_name.replace('_no_noops', '')
            all_results[suite_short] = suite_results
            
            for task_results in suite_results.values():
                total_episodes += len(task_results)
            
            total_all_instructions += all_instr_count
            total_labeled_instructions += labeled_instr_count
            grand_total_timesteps += suite_timesteps
            grand_total_labeled_timesteps += suite_labeled_timesteps
    
    save_results(all_results, output_path)
    
    logger.info("\n" + "="*60)
    logger.info("Processing Complete!")
    logger.info("="*60)
    logger.info(f"Total suites processed: {len(all_results)}")
    logger.info(f"Total episodes processed: {total_episodes}")
    logger.info(f"Total timesteps: {grand_total_timesteps}")
    logger.info(f"Total labeled timesteps: {grand_total_labeled_timesteps}")
    
    if total_episodes > 0:
        avg_timesteps = grand_total_timesteps / total_episodes
        avg_labeled_timesteps = grand_total_labeled_timesteps / total_episodes
        label_ratio = (grand_total_labeled_timesteps / grand_total_timesteps * 100) if grand_total_timesteps > 0 else 0
        logger.info(f"Average timesteps per episode: {avg_timesteps:.2f}")
        logger.info(f"Average labeled timesteps per episode: {avg_labeled_timesteps:.2f}")
        logger.info(f"Label coverage: {label_ratio:.2f}%")
    
    logger.info(f"Output saved to: {output_path}")

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
        help="Path to RLDS dataset directory"
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
        help="List of suites to process"
    )
    
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Maximum episodes to process per suite"
    )
    
    parser.add_argument(
        "--episode_ids",
        type=int,
        nargs='+',
        default=None,
        help="Specific episode IDs to process"
    )
    
    parser.add_argument(
        "--debug",
        action='store_true',
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    main(
        apd_path=args.apd_path,
        rlds_data_dir=args.rlds_data_dir,
        output_path=args.output_path,
        suites=args.suites,
        max_episodes_per_suite=args.max_episodes,
        episode_ids=args.episode_ids
    )