"""
inspect_instructions.py

Traverse RLDS dataset and extract all unique instructions with counts.
This script inspects the structure: Dataset -> Episode -> steps -> Step
and extracts instruction from each episode's steps.

Usage:
    python inspect_instructions.py \
        --rlds_data_dir /path/to/modified_libero_rlds \
        --suites libero_spatial_no_noops libero_object_no_noops \
        --output_json instructions_stats.json

Author: Generated for OpenVLA-OFT LIBERO research
"""

import argparse
import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import tensorflow as tf
import tensorflow_datasets as tfds

# Disable GPU for data loading
tf.config.set_visible_devices([], 'GPU')

# Default suite names
DEFAULT_SUITES = [
    "libero_spatial_no_noops",
    "libero_object_no_noops",
    "libero_goal_no_noops",
    "libero_10_no_noops"
]

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


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


def extract_instruction_from_episode(episode: Dict, episode_idx: int, print_all_fields: bool = False) -> str:
    """
    Extract instruction from episode.
    
    Follows structure: Episode -> steps -> Step[0] -> observation -> instruction
    
    Args:
        episode: RLDS episode dictionary
        episode_idx: Episode index for logging
        print_all_fields: If True, print all available fields
    
    Returns:
        Instruction string (lowercase, stripped), or empty string if not found
    """
    try:
        # Print all episode-level keys
        if print_all_fields:
            logger.info(f"\n  Episode {episode_idx} - All Fields:")
            logger.info(f"    Episode keys: {list(episode.keys())}")
        
        steps = episode['steps']
        
        # Get first step (all steps in same episode should have same instruction)
        first_step = next(iter(steps))
        
        # Print all step-level keys
        if print_all_fields:
            logger.info(f"    Step keys: {list(first_step.keys())}")
            
            if 'observation' in first_step:
                observation = first_step['observation']
                logger.info(f"    Step observation keys: {list(observation.keys())}")
                
                # Print observation values
                for key in observation.keys():
                    try:
                        value = observation[key].numpy()
                        if isinstance(value, bytes):
                            value = value.decode('utf-8')
                        # Truncate long arrays
                        if hasattr(value, 'shape'):
                            logger.info(f"      {key}: shape={value.shape}, dtype={value.dtype}")
                        else:
                            logger.info(f"      {key}: {value}")
                    except Exception as e:
                        logger.info(f"      {key}: <error reading: {e}>")
            
            if 'action' in first_step:
                try:
                    action = first_step['action'].numpy()
                    logger.info(f"    Step action: shape={action.shape}, dtype={action.dtype}")
                except:
                    pass
        
        # Print episode metadata if exists
        if print_all_fields and 'episode_metadata' in episode:
            metadata = episode['episode_metadata']
            logger.info(f"    Episode metadata keys: {list(metadata.keys())}")
            
            for key in metadata.keys():
                try:
                    value = metadata[key].numpy()
                    if isinstance(value, bytes):
                        value = value.decode('utf-8')
                    if hasattr(value, 'shape'):
                        logger.info(f"      {key}: shape={value.shape}, dtype={value.dtype}")
                    else:
                        logger.info(f"      {key}: {value}")
                except Exception as e:
                    logger.info(f"      {key}: <error reading: {e}>")
        
        instruction = ""
        
        # Try to extract from step observation
        if 'observation' in first_step:
            observation = first_step['observation']
            
            # Try common keys
            for key in ['language_instruction', 'instruction', 'task', 
                       'task_description', 'natural_language_instruction']:
                if key in observation:
                    try:
                        instruction = observation[key].numpy().decode('utf-8')
                        if instruction:
                            break
                    except:
                        pass
        
        # If not found in step, try episode metadata
        if not instruction and 'episode_metadata' in episode:
            metadata = episode['episode_metadata']
            
            # Try to extract from file_path
            if 'file_path' in metadata:
                try:
                    file_path = metadata['file_path'].numpy().decode('utf-8')
                    filename = os.path.basename(file_path)
                    if filename.endswith('_demo.hdf5'):
                        task_name = filename[:-10]  # Remove '_demo.hdf5'
                        instruction = task_name.replace('_', ' ')
                except:
                    pass
            
            # Try standard keys in metadata
            if not instruction:
                for key in ['language_instruction', 'instruction', 'task', 
                           'task_description', 'natural_language_instruction']:
                    if key in metadata:
                        try:
                            instruction = metadata[key].numpy().decode('utf-8')
                            if instruction:
                                break
                        except:
                            pass
        
        # Try episode level
        if not instruction:
            for key in ['language_instruction', 'instruction', 'task', 
                       'task_description', 'natural_language_instruction']:
                if key in episode:
                    try:
                        instruction = episode[key].numpy().decode('utf-8')
                        if instruction:
                            break
                    except:
                        pass
        
        return instruction.lower().strip() if instruction else ""
    
    except Exception as e:
        logger.debug(f"Error extracting instruction: {e}")
        return ""


def process_suite(data_dir: str, suite_name: str, print_all_fields: bool = False, 
                 max_episodes: Optional[int] = None) -> Dict[str, int]:
    """
    Process entire task suite and count instructions.
    
    Args:
        data_dir: RLDS dataset root directory
        suite_name: Task suite name
        print_all_fields: If True, print all available fields
        max_episodes: Maximum number of episodes to process (for debugging)
    
    Returns:
        Dictionary mapping instruction to count
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing suite: {suite_name}")
    logger.info(f"{'='*60}")
    
    # Load dataset
    dataset = load_rlds_dataset(data_dir, suite_name)
    if dataset is None:
        return {}
    
    # Count instructions
    instruction_counter = Counter()
    total_episodes = 0
    episodes_without_instruction = 0
    
    for episode_idx, episode in enumerate(dataset):
        # Stop if max_episodes limit reached
        if max_episodes and episode_idx >= max_episodes:
            logger.info(f"Reached max_episodes limit: {max_episodes}")
            break
        
        total_episodes += 1
        
        # Extract instruction
        instruction = extract_instruction_from_episode(episode, episode_idx, print_all_fields)
        
        if instruction:
            instruction_counter[instruction] += 1
            # Print immediately when found
            if not print_all_fields:  # Avoid duplicate output when printing all fields
                logger.info(f"  Episode {episode_idx}: '{instruction}'")
            else:
                logger.info(f"    -> Instruction: '{instruction}'\n")
        else:
            episodes_without_instruction += 1
            logger.warning(f"  Episode {episode_idx}: No instruction found")
    
    logger.info(f"Suite {suite_name} summary:")
    logger.info(f"  Total episodes: {total_episodes}")
    logger.info(f"  Episodes with instruction: {total_episodes - episodes_without_instruction}")
    logger.info(f"  Episodes without instruction: {episodes_without_instruction}")
    logger.info(f"  Unique instructions: {len(instruction_counter)}")
    
    return dict(instruction_counter)


def print_statistics(all_results: Dict[str, Dict[str, int]]) -> None:
    """
    Print formatted statistics for all suites.
    
    Args:
        all_results: Dictionary mapping suite_name to instruction counts
    """
    logger.info("\n" + "="*60)
    logger.info("INSTRUCTION STATISTICS")
    logger.info("="*60)
    
    grand_total_episodes = 0
    grand_total_unique = set()
    
    for suite_name, instruction_counts in all_results.items():
        logger.info(f"\n{suite_name}:")
        logger.info("-" * 60)
        
        # Sort by count (descending) then by instruction name
        sorted_instructions = sorted(
            instruction_counts.items(), 
            key=lambda x: (-x[1], x[0])
        )
        
        suite_total = sum(instruction_counts.values())
        grand_total_episodes += suite_total
        
        for instruction, count in sorted_instructions:
            logger.info(f"  [{count:3d}] {instruction}")
            grand_total_unique.add(instruction)
        
        logger.info(f"\n  Suite total: {len(instruction_counts)} unique instructions, {suite_total} episodes")
    
    # Grand total
    logger.info("\n" + "="*60)
    logger.info("GRAND TOTAL")
    logger.info("="*60)
    logger.info(f"Total suites: {len(all_results)}")
    logger.info(f"Total unique instructions across all suites: {len(grand_total_unique)}")
    logger.info(f"Total episodes: {grand_total_episodes}")


def main(rlds_data_dir: str,
         suites: Optional[List[str]] = None,
         output_json: Optional[str] = None,
         print_all_fields: bool = False,
         max_episodes: Optional[int] = None) -> None:
    """
    Main function: Process all suites and display statistics.
    
    Args:
        rlds_data_dir: RLDS dataset root directory
        suites: List of suites to process (None = all)
        output_json: Optional path to save results as JSON
        print_all_fields: If True, print all available fields in episodes
        max_episodes: Maximum number of episodes to process per suite (for debugging)
    """
    logger.info("="*60)
    logger.info("RLDS Instruction Inspection Tool")
    logger.info("="*60)
    
    # Determine which suites to process
    if suites is None:
        suites = DEFAULT_SUITES
    
    logger.info(f"\nProcessing {len(suites)} suites:")
    for suite in suites:
        logger.info(f"  - {suite}")
    
    if print_all_fields:
        logger.info("\n*** PRINTING ALL FIELDS MODE ***\n")
    
    if max_episodes:
        logger.info(f"\n*** Processing maximum {max_episodes} episodes per suite ***\n")
    
    # Process each suite
    all_results = {}
    
    for suite_name in suites:
        instruction_counts = process_suite(rlds_data_dir, suite_name, print_all_fields, max_episodes)
        
        if instruction_counts:
            # Remove _no_noops suffix for cleaner output
            suite_short = suite_name.replace('_no_noops', '')
            all_results[suite_short] = instruction_counts
    
    # Print statistics
    print_statistics(all_results)
    
    # Save to JSON if requested
    if output_json:
        logger.info(f"\nSaving results to {output_json}")
        
        # Create output structure with metadata
        output_data = {
            "metadata": {
                "total_suites": len(all_results),
                "total_unique_instructions": len(set(
                    inst for counts in all_results.values() for inst in counts.keys()
                )),
                "total_episodes": sum(
                    sum(counts.values()) for counts in all_results.values()
                )
            },
            "suites": all_results
        }
        
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect and count instructions in RLDS datasets"
    )
    
    parser.add_argument(
        "--rlds_data_dir",
        type=str,
        required=True,
        help="Path to RLDS dataset directory (e.g., /path/to/modified_libero_rlds)"
    )
    
    parser.add_argument(
        "--suites",
        type=str,
        nargs='+',
        default=None,
        help="List of suites to process (default: all standard suites)"
    )
    
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path to save results as JSON file"
    )
    
    parser.add_argument(
        "--print_all_fields",
        action='store_true',
        help="Print all available fields in episodes and steps"
    )
    
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Maximum number of episodes to process per suite (useful for debugging)"
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
        rlds_data_dir=args.rlds_data_dir,
        suites=args.suites,
        output_json=args.output_json,
        print_all_fields=args.print_all_fields,
        max_episodes=args.max_episodes
    )

