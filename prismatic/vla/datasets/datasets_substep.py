"""
datasets_substep.py

Extended VLA datasets that support per-timestep substep instructions.
This module provides dataset classes that can replace per-episode instructions
with per-timestep substep instructions from a JSON label file.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type

import numpy as np
import torch
from PIL import Image
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.overwatch import initialize_overwatch
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import ACTION_PROPRIO_NORMALIZATION_TYPE, NUM_ACTIONS_CHUNK
from prismatic.vla.datasets.datasets import rephrase
from prismatic.vla.datasets.rlds.oxe.materialize_substep import (
    get_oxe_dataset_kwargs_and_weights_with_episode_id,
)
from prismatic.vla.datasets.rlds.oxe.mixtures import OXE_NAMED_MIXTURES

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


def load_substep_labels(substep_labels_path: str) -> Dict:
    """
    Load substep labels from JSON file.
    
    Args:
        substep_labels_path: Path to substep_labels_output.json
        
    Returns:
        Dictionary containing substep labels organized by suite/task/episode/timestep
        
    Expected JSON structure:
        {
            "libero_goal": {
                "put_the_cream_cheese_in_the_bowl": {
                    "episode_5": {
                        "instruction": "...",
                        "total_timesteps": 112,
                        "timestep_labels": [
                            {"timestep": 0, "action": "pick", "APD_step": "..."},
                            ...
                        ]
                    }
                }
            }
        }
    """
    overwatch.info(f"Loading substep labels from: {substep_labels_path}")
    with open(substep_labels_path, 'r', encoding='utf-8') as f:
        substep_labels = json.load(f)
    
    # Log statistics
    total_suites = len(substep_labels)
    overwatch.info(f"Loaded substep labels for {total_suites} suites")
    for suite_name, tasks in substep_labels.items():
        overwatch.info(f"  Suite '{suite_name}': {len(tasks)} tasks")
    
    return substep_labels


def get_substep_instruction(
    substep_labels: Dict,
    dataset_name: str,
    task_instruction: str,
    episode_id: int,
    timestep: int,
    default_instruction: str,
) -> str:
    """
    Query substep instruction from loaded labels.
    
    Args:
        substep_labels: Loaded substep labels dictionary
        dataset_name: Dataset name (e.g., "libero_goal_no_noops")
        task_instruction: Task instruction (e.g., "put the cream cheese in the bowl")
        episode_id: Episode index (e.g., 5)
        timestep: Timestep index (e.g., 0)
        default_instruction: Default instruction if substep not found
        
    Returns:
        APD_step instruction string, or default_instruction if not found
        
    Note:
        - Strips "_no_noops" suffix from dataset_name to match JSON keys
        - Converts task_instruction spaces to underscores for matching
        - Returns default_instruction if any key is missing
    """
    try:
        # Strip "_no_noops" suffix from dataset name to get suite name
        suite_name = dataset_name.replace("_no_noops", "")
        
        # Convert task instruction to underscore format
        # "put the cream cheese in the bowl" -> "put_the_cream_cheese_in_the_bowl"
        task_name = task_instruction.lower().strip().replace(" ", "_")
        
        # Format episode key
        episode_key = f"episode_{episode_id}"
        
        # Navigate to episode data
        if suite_name not in substep_labels:
            return default_instruction
        if task_name not in substep_labels[suite_name]:
            return default_instruction
        if episode_key not in substep_labels[suite_name][task_name]:
            return default_instruction
        
        episode_data = substep_labels[suite_name][task_name][episode_key]
        timestep_labels = episode_data.get("timestep_labels", [])
        
        # Find matching timestep
        # Note: timestep_labels might not have entries for every timestep
        # Return the APD_step for the closest previous timestep
        best_match = None
        for label in timestep_labels:
            if label["timestep"] <= timestep:
                best_match = label
            else:
                break  # Assuming timestep_labels is sorted
        
        if best_match and "APD_step" in best_match:
            return best_match["APD_step"]
        
        return default_instruction
        
    except (KeyError, TypeError, IndexError) as e:
        # If anything goes wrong, return default instruction
        overwatch.debug(
            f"Could not find substep for suite={dataset_name}, task={task_instruction}, "
            f"episode={episode_id}, timestep={timestep}: {e}"
        )
        return default_instruction


@dataclass
class SubstepRLDSBatchTransform:
    """
    Extended batch transform that replaces task instructions with per-timestep substep instructions.
    
    This transform extends RLDSBatchTransform by:
    1. Loading substep labels from JSON file
    2. Querying appropriate substep instruction for each (dataset, task, episode, timestep)
    3. Replacing the original task instruction with the substep instruction
    
    Attributes:
        action_tokenizer: Tokenizer for discretizing actions
        base_tokenizer: Base tokenizer for text
        image_transform: Transform for processing images
        prompt_builder_fn: Function to build prompts
        substep_labels: Loaded substep labels dictionary
        predict_stop_token: Whether to predict stop token
        use_wrist_image: Whether to use wrist camera images
        use_proprio: Whether to use proprioceptive state
    """
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    substep_labels: Dict
    predict_stop_token: bool = True
    use_wrist_image: bool = False
    use_proprio: bool = False

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Converts an RLDS batch to the format expected by OpenVLA, with substep instruction.
        
        Args:
            rlds_batch: Dictionary containing:
                - dataset_name: str
                - action: (chunk_size, action_dim) array
                - observation: dict with image_primary, timestep, etc.
                - task: dict with language_instruction
                - episode_id: int (added by extended transforms)
                
        Returns:
            Dictionary with pixel_values, input_ids, labels, etc. ready for training
        """
        # Extract basic info
        dataset_name = rlds_batch["dataset_name"]
        current_action = rlds_batch["action"][0]
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        actions = rlds_batch["action"]
        
        # Get original task instruction
        original_instruction = rlds_batch["task"]["language_instruction"].decode().lower()
        
        # Get episode_id and timestep
        episode_id = int(rlds_batch.get("episode_id", 0))
        timestep = int(rlds_batch["observation"]["timestep"][0])
        
        # Query substep instruction
        substep_instruction = get_substep_instruction(
            self.substep_labels,
            dataset_name,
            original_instruction,
            episode_id,
            timestep,
            default_instruction=original_instruction,  # Fallback to original if not found
        )
        
        # Log if substep instruction was successfully retrieved
        if substep_instruction != original_instruction:
            overwatch.debug(
                f"Replaced instruction at episode={episode_id}, timestep={timestep}: "
                f"'{original_instruction}' -> '{substep_instruction}'"
            )
        
        # Use substep instruction instead of original
        lang = substep_instruction
        
        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        prompt_builder = self.prompt_builder_fn("openvla")
        
        # Get future action chunk
        future_actions = rlds_batch["action"][1:]
        future_actions_string = ''.join(self.action_tokenizer(future_actions))
        
        # Get action chunk string
        current_action_string = self.action_tokenizer(current_action)
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)
        
        # Build conversation with substep instruction
        conversation = [
            {"from": "human", "value": f"{rephrase()} {lang}"},
            {"from": "gpt", "value": action_chunk_string},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])
        
        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)
        
        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(img)
        
        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        from prismatic.vla.constants import IGNORE_INDEX
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX
        
        return_dict = dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels,
            dataset_name=dataset_name,
            actions=actions,
        )
        
        # Add additional inputs
        if self.use_wrist_image:
            all_wrist_pixels = []
            for k in rlds_batch["observation"].keys():
                if "wrist" in k:
                    img_wrist = Image.fromarray(rlds_batch["observation"][k][0])
                    pixel_values_wrist = self.image_transform(img_wrist)
                    all_wrist_pixels.append(pixel_values_wrist)
            return_dict["pixel_values_wrist"] = torch.cat(all_wrist_pixels, dim=0)
        
        if self.use_proprio and "proprio" in rlds_batch["observation"]:
            proprio = rlds_batch["observation"]["proprio"]
            return_dict["proprio"] = proprio
        
        return return_dict


class SubstepRLDSDataset(IterableDataset):
    """
    Extended RLDS dataset that supports per-timestep substep instructions.
    
    This dataset class:
    1. Uses extended transforms that preserve episode IDs
    2. Loads substep labels from JSON file
    3. Passes substep labels to SubstepRLDSBatchTransform for per-timestep instruction replacement
    
    Args:
        data_root_dir: Root directory containing RLDS datasets
        data_mix: Dataset name or mixture name
        batch_transform: SubstepRLDSBatchTransform instance
        substep_labels_path: Path to substep_labels_output.json
        resize_resolution: Image resize resolution (height, width)
        shuffle_buffer_size: Size of shuffle buffer
        train: Whether this is training split
        image_aug: Whether to use image augmentations
    """
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: SubstepRLDSBatchTransform,
        substep_labels_path: str,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline with substep instruction support."""
        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform
        self.substep_labels_path = substep_labels_path
        
        # Load substep labels
        self.substep_labels = load_substep_labels(substep_labels_path)
        
        # Pass substep labels to batch transform
        self.batch_transform.substep_labels = self.substep_labels
        
        # Configure RLDS Dataset(s)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]
        
        # Determine camera views based on dataset
        if "aloha" in self.data_mix:
            load_camera_views = ("primary", "left_wrist", "right_wrist")
        else:
            load_camera_views = ("primary", "wrist")
        
        # Get dataset kwargs with episode ID tracking
        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights_with_episode_id(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=load_camera_views,
            load_depth=False,
            load_proprio=True,
            load_language=True,
            action_proprio_normalization_type=ACTION_PROPRIO_NORMALIZATION_TYPE,
        )
        
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=1,  # If we wanted to feed / predict more than one step
                future_action_window_size=NUM_ACTIONS_CHUNK - 1,  # For action chunking
                skip_unlabeled=True,  # Skip trajectories without language labels
                goal_relabeling_strategy="uniform",  # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=16,  # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
        )
        
        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({
                "image_augment_kwargs": dict(
                    random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                    random_brightness=[0.2],
                    random_contrast=[0.8, 1.2],
                    random_saturation=[0.8, 1.2],
                    random_hue=[0.05],
                    augment_order=[
                        "random_resized_crop",
                        "random_brightness",
                        "random_contrast",
                        "random_saturation",
                        "random_hue",
                    ],
                )
            })
        
        # Initialize RLDS Dataset with episode tracking
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)
    
    def make_dataset(self, rlds_config):
        """
        Create the interleaved dataset with episode ID tracking.
        
        Note: This imports and uses the standard make_interleaved_dataset from OXE.
        The episode tracking is handled by the transforms specified in per_dataset_kwargs.
        """
        from prismatic.vla.datasets.rlds.oxe import make_interleaved_dataset
        
        return make_interleaved_dataset(**rlds_config)
    
    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield self.batch_transform(rlds_batch)
    
    def __len__(self) -> int:
        return self.dataset_length
    
    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")

