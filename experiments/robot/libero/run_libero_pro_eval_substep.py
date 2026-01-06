"""
run_libero_eval.py

Evaluates a trained policy in a LIBERO simulation benchmark task suite.
"""

import json
import logging
import os
import sys
import yaml
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union
import torch
import torch.nn as nn
import draccus
import numpy as np
import tqdm
import imageio
from libero.libero import benchmark

from experiments.robot.libero import perturbation

import wandb

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    DATE,
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)

from energy_model.model import EnergyModel

import cv2
from prismatic.vla.constants import NUM_ACTIONS_CHUNK


# Define task suite constants
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"
    LIBERO_GOAL_TEMP = "libero_goal_temp"
    LIBERO_SPATIAL_TEMP = "libero_spatial_temp"
    LIBERO_10_TEMP = "libero_10_temp"
    LIBERO_OBJECT_TEMP = "libero_object_temp"
    LIBERO_GOAL_LAN = "libero_goal_lan"
    LIBERO_SPATIAL_LAN = "libero_spatial_lan"
    LIBERO_10_LAN = "libero_10_lan"
    LIBERO_OBJECT_LAN = "libero_object_lan"
    LIBERO_GOAL_OBJECT = "libero_goal_object"
    LIBERO_SPATIAL_OBJECT = "libero_spatial_object"
    LIBERO_10_OBJECT = "libero_10_object"
    LIBERO_OBJECT_OBJECT = "libero_object_object"
    LIBERO_GOAL_SWAP = "libero_goal_swap"
    LIBERO_SPATIAL_SWAP = "libero_spatial_swap"
    LIBERO_10_SWAP = "libero_10_swap"
    LIBERO_OBJECT_SWAP = "libero_object_swap"
    LIBERO_GOAL_TASK = "libero_goal_task"
    LIBERO_SPATIAL_TASK = "libero_spatial_task"
    LIBERO_10_TASK = "libero_10_task"
    LIBERO_OBJECT_TASK = "libero_object_task"
    LIBERO_GOAL_ENV = "libero_goal_env"
    LIBERO_SPATIAL_ENV = "libero_spatial_env"
    LIBERO_10_ENV = "libero_10_env"
    LIBERO_OBJECT_ENV = "libero_object_env"


# Define max steps for each task suite
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,  # longest training demo has 193 steps
    TaskSuite.LIBERO_OBJECT: 280,  # longest training demo has 254 steps
    TaskSuite.LIBERO_GOAL: 300,  # longest training demo has 270 steps
    TaskSuite.LIBERO_10: 520,  # longest training demo has 505 steps
    TaskSuite.LIBERO_90: 400,  # longest training demo has 373 steps
    TaskSuite.LIBERO_GOAL_TEMP: 300,
    TaskSuite.LIBERO_SPATIAL_TEMP: 220,
    TaskSuite.LIBERO_10_TEMP: 520,
    TaskSuite.LIBERO_OBJECT_TEMP: 280,
    TaskSuite.LIBERO_GOAL_LAN: 300,
    TaskSuite.LIBERO_SPATIAL_LAN: 220,
    TaskSuite.LIBERO_10_LAN: 520,
    TaskSuite.LIBERO_OBJECT_LAN: 280,
    TaskSuite.LIBERO_GOAL_OBJECT: 300,
    TaskSuite.LIBERO_SPATIAL_OBJECT: 220,
    TaskSuite.LIBERO_10_OBJECT: 520,
    TaskSuite.LIBERO_OBJECT_OBJECT: 280,
    TaskSuite.LIBERO_GOAL_SWAP: 300,
    TaskSuite.LIBERO_SPATIAL_SWAP: 220,
    TaskSuite.LIBERO_10_SWAP: 520,
    TaskSuite.LIBERO_OBJECT_SWAP: 280,
    TaskSuite.LIBERO_GOAL_TASK: 300,
    TaskSuite.LIBERO_SPATIAL_TASK: 220,
    TaskSuite.LIBERO_10_TASK: 520,
    TaskSuite.LIBERO_OBJECT_TASK: 280,
    TaskSuite.LIBERO_GOAL_ENV: 300,
    TaskSuite.LIBERO_SPATIAL_ENV: 220,
    TaskSuite.LIBERO_10_ENV: 520,
    TaskSuite.LIBERO_OBJECT_ENV: 280,
}


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path

    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps used for training
    num_diffusion_steps_inference: int = 50          # (When `diffusion==True`) Number of diffusion steps used for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy

    lora_rank: int = 32                              # Rank of LoRA weight matrix (MAKE SURE THIS MATCHES TRAINING!)

    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL  # Task suite
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project

    seed: int = 7                                    # Random Seed (for reproducibility)
    h_decoding:bool = False
    save_video:bool = False
    e_decoding:bool = False
    task_label:str = ""
    remove_wrap:bool = False
    energy_k:int = 1
    energy_alpha:float = 0.5
    evaluation_config_path: Optional[str] = None     # Path to LIBERO-PRO evaluation config YAML file
    
    # Substep decomposition parameters
    use_substep_decomposition: bool = False          # Enable substep-based instruction decomposition
    llm_model_path: str = "Qwen/Qwen2.5-0.5B-Instruct" # Path to LLM for instruction decomposition
    sigclip_model_path: str = "timm/ViT-B-16-SigLIP-256"  # Path to SigLIP-2 model for substep completion
    substep_completion_threshold: float = 0.25       # Vision-language similarity threshold for substep completion
    substep_log_dir: str = "./experiments/logs/substeps"  # Directory for substep-specific logs

    # fmt: on


def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Validate task suite
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"


def initialize_model(cfg: GenerateConfig):
    """Initialize model and associated components."""
    # Load model
    model = get_model(cfg)

    # Load proprio projector if needed
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(
            cfg,
            model.llm_dim,
            proprio_dim=8,  # 8-dimensional proprio for LIBERO
        )

    # Load action head if needed
    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(cfg, model.llm_dim)

    # Load noisy action projector if using diffusion
    noisy_action_projector = None
    if cfg.use_diffusion:
        noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim)

    # Get OpenVLA processor if needed
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        check_unnorm_key(cfg, model)

    return model, action_head, proprio_projector, noisy_action_projector, processor


def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    """Check that the model contains the action un-normalization key."""
    # Initialize unnorm_key from cfg.unnorm_key if provided, otherwise use task_suite_name
    if cfg.unnorm_key:
        unnorm_key = cfg.unnorm_key
    else:
        unnorm_key = cfg.task_suite_name

    # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
    # with the suffix "_no_noops" in the dataset name)
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"

    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"

    # Set the unnorm_key in cfg
    cfg.unnorm_key = unnorm_key


def setup_logging(cfg: GenerateConfig):
    """Set up logging to file and optionally to wandb."""
    # Create run ID
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    # Set up local logging
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging if enabled
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    return log_file, local_log_filepath, run_id


def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def load_initial_states(cfg: GenerateConfig, task_suite, task_id: int, log_file=None):
    """Load initial states for the given task."""
    # Get default initial states
    initial_states = task_suite.get_task_init_states(task_id)

    # If using custom initial states, load them from file
    if cfg.initial_states_path != "DEFAULT":
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {cfg.initial_states_path}", log_file)
        return initial_states, all_initial_states
    else:
        log_message("Using default initial states", log_file)
        return initial_states, None


def prepare_observation(obs, resize_size):
    """Prepare observation for policy input."""
    # Get preprocessed images
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)

    # Resize images to size expected by model
    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

    # Prepare observations dict
    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }

    return observation, img  # Return both processed observation and original image for replay


def process_action(action, model_family):
    """Process action before sending to environment."""
    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
    action = normalize_gripper_action(action, binarize=True)

    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    if model_family == "openvla":
        action = invert_gripper_action(action)

    return action


def save_rollout_video_with_substep_info(
    rollout_images,
    substep_info_list,
    episode_idx,
    success,
    task_description,
    log_file=None
):
    """
    Save rollout video with substep information overlayed on each frame.
    
    Args:
        rollout_images: List of RGB images (H, W, 3) as numpy arrays
        substep_info_list: List of dicts with substep info for each frame
        episode_idx: Episode index number
        success: Whether episode succeeded
        task_description: Original task instruction
        log_file: Optional log file handle
    """
    rollout_dir = f"./rollouts/{DATE}"
    os.makedirs(rollout_dir, exist_ok=True)
    
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/{DATE_TIME}--substep_eval--episode={episode_idx}--success={success}--task={processed_task_description}.mp4"
    
    # Process each frame and add text overlay
    annotated_frames = []
    for frame_idx, img in enumerate(rollout_images):
        # Get substep info for this frame
        if frame_idx < len(substep_info_list):
            info = substep_info_list[frame_idx]
        else:
            info = substep_info_list[-1] if substep_info_list else {}
        
        # Convert to BGR for OpenCV
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Create overlay with semi-transparent background
        overlay = img_bgr.copy()
        height, width = img_bgr.shape[:2]
        
        # Calculate background box size (tighter fit)
        line_spacing_bg = 14
        num_text_lines = 9  # Enough for all lines with some wrapping
        box_height = min(int(num_text_lines * line_spacing_bg + 8), height - 10)  # Don't exceed image height
        
        # Draw semi-transparent background for text
        cv2.rectangle(overlay, (5, 5), (width - 5, box_height), (0, 0, 0), -1)
        img_bgr = cv2.addWeighted(overlay, 0.75, img_bgr, 0.25, 0)
        
        # Prepare text information
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Much smaller font for 256px images
        font_scale = 0.3
        font_thickness = 1
        line_spacing = 14
        y_offset = 12
        text_color = (255, 255, 255)  # White
        max_chars = 50  # Maximum characters per line
        
        # Helper function to wrap text
        def wrap_text(text, max_width):
            words = text.split()
            lines = []
            current_line = ""
            for word in words:
                test_line = current_line + word + " "
                if len(test_line) <= max_width:
                    current_line = test_line
                else:
                    if current_line:
                        lines.append(current_line.strip())
                    current_line = word + " "
            if current_line:
                lines.append(current_line.strip())
            return lines if lines else [text[:max_width]]
        
        # Line 1: Original instruction (with wrapping)
        task_lines = wrap_text(f"Task: {task_description}", max_chars)
        for line in task_lines[:2]:  # Max 2 lines for task
            cv2.putText(img_bgr, line, (10, y_offset), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
            y_offset += line_spacing
        
        # Line 2: Current substep with counter
        if 'current_substep' in info and info['current_substep']:
            substep_header = f"Substep {info.get('substep_idx', 0)}/{info.get('total_substeps', 0)}:"
            cv2.putText(img_bgr, substep_header, (10, y_offset), font, font_scale, (180, 180, 255), font_thickness, cv2.LINE_AA)
            y_offset += line_spacing
            
            # Substep content (wrapped)
            substep_lines = wrap_text(info['current_substep'], max_chars)
            for line in substep_lines[:2]:  # Max 2 lines
                cv2.putText(img_bgr, f"  {line}", (10, y_offset), font, font_scale, (200, 200, 255), font_thickness, cv2.LINE_AA)
                y_offset += line_spacing
        else:
            substep_text = "Substep: N/A"
            cv2.putText(img_bgr, substep_text, (10, y_offset), font, font_scale, (128, 128, 128), font_thickness, cv2.LINE_AA)
            y_offset += line_spacing
        
        # Line 3: Expected effect (wrapped)
        if 'expected_effect' in info and info['expected_effect']:
            cv2.putText(img_bgr, "Expected:", (10, y_offset), font, font_scale, (200, 200, 200), font_thickness, cv2.LINE_AA)
            y_offset += line_spacing
            
            effect_lines = wrap_text(info['expected_effect'], max_chars)
            for line in effect_lines[:2]:  # Max 2 lines
                cv2.putText(img_bgr, f"  {line}", (10, y_offset), font, font_scale, (220, 220, 220), font_thickness, cv2.LINE_AA)
                y_offset += line_spacing
        
        # Line 4: Similarity score with color coding
        if 'similarity' in info and info['similarity'] is not None:
            sim_score = info['similarity']
            threshold = info.get('threshold', 0.0)
            
            # Color code with softer, more readable colors
            if sim_score >= threshold:
                sim_color = (80, 200, 80)  # Soft green (BGR)
                status = "OK"
            else:
                sim_color = (120, 180, 220)  # Soft peach/tan (BGR)
                status = "..."
            
            sim_text = f"Sim: {sim_score:.3f}/{threshold:.2f} [{status}]"
            cv2.putText(img_bgr, sim_text, (10, y_offset), font, font_scale, sim_color, font_thickness, cv2.LINE_AA)
        y_offset += line_spacing
        
        # Line 5: Frame info
        frame_text = f"Frame: {frame_idx+1}/{len(rollout_images)}"
        cv2.putText(img_bgr, frame_text, (10, y_offset), font, font_scale, (150, 150, 150), font_thickness, cv2.LINE_AA)
        
        # Convert back to RGB for imageio
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        annotated_frames.append(img_rgb)
    
    # Write video
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for frame in annotated_frames:
        video_writer.append_data(frame)
    video_writer.close()
    
    log_msg = f"Saved annotated rollout video at: {mp4_path}"
    print(log_msg)
    if log_file is not None:
        log_file.write(log_msg + "\n")
        log_file.flush()
    
    return mp4_path


def run_episode(
    cfg: GenerateConfig,
    env,
    task_description: str,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    initial_state=None,
    log_file=None,
    head=None,
    llm_model=None,
    llm_tokenizer=None,
    sigclip_model=None,
    sigclip_processor=None,
):
    """Run a single episode in the environment."""
    # Reset environment
    env.reset()

    # Set initial state if provided
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    # Initialize action queue
    if cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        print(f"WARNING: cfg.num_open_loop_steps ({cfg.num_open_loop_steps}) does not match the NUM_ACTIONS_CHUNK "
              f"({NUM_ACTIONS_CHUNK}) constant defined in prismatic.vla.constants! For best performance (in terms of "
               "both speed and success rate), we recommend executing the full action chunk.")
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    # Initialize SubstepManager if enabled
    substep_manager = None
    if cfg.use_substep_decomposition and llm_model is not None:
        try:
            from experiments.robot.libero.substep_manager import SubstepManager
            
            substep_manager = SubstepManager(
                task_description=task_description,
                llm_model=llm_model,
                llm_tokenizer=llm_tokenizer,
                sigclip_model=sigclip_model,
                sigclip_processor=sigclip_processor,
                completion_threshold=cfg.substep_completion_threshold,
                device=model.device,
            )
            
            log_message(f"[SUBSTEP] Decomposed into {len(substep_manager.substeps)} substeps", log_file)
            for i, substep in enumerate(substep_manager.substeps):
                log_message(f"  [{i+1}] {substep['subgoal']} -> {substep['expected_effect']}", log_file)
                
        except Exception as e:
            log_message(f"[SUBSTEP] Failed to initialize SubstepManager: {e}", log_file)
            substep_manager = None

    # Setup
    t = 0
    replay_images = []
    substep_info_list = []  # Track substep info for each frame
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]

    # Drawing Utils
    actions_accum = []
    flag = 0

  


    # Run episode
    success = False
    # try:
    while t < max_steps + cfg.num_steps_wait:
        # Do nothing for the first few timesteps to let objects stabilize
        if t < cfg.num_steps_wait:
            obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
            
            # Still collect images for video during wait period
            _, img = prepare_observation(obs, resize_size)
            replay_images.append(img)
            
            # Add substep info for wait frames
            frame_info = {
                'task_description': task_description,
                'current_substep': 'Waiting for environment to stabilize...',
                'expected_effect': None,
                'similarity': None,
                'threshold': cfg.substep_completion_threshold if cfg.use_substep_decomposition else None,
                'substep_idx': 0,
                'total_substeps': len(substep_manager.substeps) if substep_manager else 0,
            }
            substep_info_list.append(frame_info)
            
            t += 1
            continue

        # Prepare observation
        observation, img = prepare_observation(obs, resize_size)
        replay_images.append(img)
        
        # Collect current substep info BEFORE any switching (for video annotation)
        frame_substep_info = {
            'task_description': task_description,
            'current_substep': None,
            'expected_effect': None,
            'similarity': None,
            'threshold': cfg.substep_completion_threshold if cfg.use_substep_decomposition else None,
            'substep_idx': 0,
            'total_substeps': 0,
        }
        
        if substep_manager is not None and len(substep_manager.substeps) > 0:
            # Always compute similarity for every frame (for video annotation)
            img_for_check = get_libero_image(obs)
            if substep_manager.current_substep_idx < len(substep_manager.substeps):
                similarity_score = substep_manager._compute_similarity(img_for_check)
                frame_substep_info['similarity'] = similarity_score
            
            # Get current substep info (before potential switching)
            progress = substep_manager.get_progress_info()
            frame_substep_info.update({
                'current_substep': progress['current_subgoal'],
                'substep_idx': progress['current_idx'] + 1,  # 1-indexed for display
                'total_substeps': progress['total'],
            })
            
            # Get expected effect for current substep
            if substep_manager.current_substep_idx < len(substep_manager.substeps):
                current_substep_data = substep_manager.substeps[substep_manager.current_substep_idx]
                frame_substep_info['expected_effect'] = current_substep_data['expected_effect']
        
        substep_info_list.append(frame_substep_info)
        
        # Check substep completion EVERY timestep (fine-grained switching)
        should_requery = len(action_queue) == 0  # Default: requery when queue is empty
        
        if substep_manager is not None:
            img_for_check = get_libero_image(obs)
            
            # Check if substep completed (even if action queue is not empty)
            if substep_manager.should_switch_substep(img_for_check):
                substep_manager.advance_substep()
                progress_info = substep_manager.get_progress_info()
                log_message(
                    f"[SUBSTEP] ✓ Switched to step {progress_info['current_idx']+1}/{progress_info['total']}: "
                    f"{progress_info['current_subgoal']}", 
                    log_file
                )
                
                # Discard remaining actions in queue (they're based on old substep)
                if len(action_queue) > 0:
                    discarded_count = len(action_queue)
                    action_queue.clear()
                    log_message(
                        f"[SUBSTEP] Discarded {discarded_count} remaining actions from old substep",
                        log_file
                    )
                
                # Force requery with new substep instruction
                should_requery = True
        
        # Query model if needed (queue empty OR substep just switched)
        if should_requery:
            # Determine which instruction to use
            current_instruction = task_description  # Default to original task description
            
            if substep_manager is not None:
                # Get current instruction from substep manager
                current_instruction = substep_manager.get_current_instruction()
            
            # Query model to get action using current instruction
            actions = get_action(
                cfg,
                model,
                observation,
                current_instruction,  # Use dynamic instruction (substep or original)
                processor=processor,
                action_head=action_head,
                proprio_projector=proprio_projector,
                noisy_action_projector=noisy_action_projector,
                use_film=cfg.use_film,
                h_head=head,
            )
            action_queue.extend(actions)
            actions_accum.append(actions)
 

        # Get action from queue
        action = action_queue.popleft()

        # Process action
        action = process_action(action, cfg.model_family)

        # Execute action in environment
        obs, reward, done, info = env.step(action.tolist())
        if done:
            success = True
            break
        t += 1

    # except Exception as e:
    #     log_message(f"Episode error: {e}", log_file) 要画轨迹多帧累积：将历史若干帧的末端投影点连线，使用第三人称agentview_image绘制，需要物理上正确的投影

    # Log substep statistics if enabled
    if substep_manager is not None:
        substep_stats = substep_manager.get_final_statistics()
        log_message(
            f"[SUBSTEP STATS] Completed {substep_stats['completed_substeps']}/{substep_stats['total_substeps']} "
            f"substeps ({substep_stats['completion_rate']*100:.1f}%), {substep_stats['total_switches']} switches",
            log_file
        )

    return success, replay_images, substep_info_list


def run_task(
    cfg: GenerateConfig,
    task_suite,
    task_id: int,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    total_episodes=0,
    total_successes=0,
    log_file=None,
):
    """Run evaluation for a single task."""
    # Initialize LLM and SigCLIP models for substep decomposition if enabled
    llm_model, llm_tokenizer = None, None
    sigclip_model, sigclip_processor = None, None
    
    if cfg.use_substep_decomposition:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, AutoProcessor
            
            log_message("[SUBSTEP] Loading LLM for instruction decomposition...", log_file)
            llm_model = AutoModelForCausalLM.from_pretrained(
                cfg.llm_model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            llm_tokenizer = AutoTokenizer.from_pretrained(
                cfg.llm_model_path,
                trust_remote_code=True,
            )
            llm_model.eval()
            log_message(f"[SUBSTEP] LLM loaded: {cfg.llm_model_path}", log_file)
            
            log_message("[SUBSTEP] Loading vision-language model for substep completion detection...", log_file)
            
            # Check if using open_clip (timm hub) model or transformers model
            if cfg.sigclip_model_path.startswith("timm/"):
                # Load open_clip model from timm hub
                from open_clip import create_model_from_pretrained, get_tokenizer
                
                # Model name format: 'hf-hub:timm/ViT-B-16-SigLIP2'
                openclip_model_name = f"hf-hub:{cfg.sigclip_model_path}"
                
                log_message(f"[SUBSTEP] Loading open_clip model: {openclip_model_name}", log_file)
                sigclip_model, sigclip_processor = create_model_from_pretrained(openclip_model_name)
                sigclip_tokenizer = get_tokenizer(openclip_model_name)
                
                sigclip_model = sigclip_model.to(model.device)
                sigclip_model.eval()
                
                # Store tokenizer in processor for unified interface
                sigclip_processor._tokenizer = sigclip_tokenizer
                sigclip_processor._is_openclip = True
                
                # Mark as open_clip model
                sigclip_model._is_openclip_model = True
                
                log_message(f"[SUBSTEP] Open-CLIP model loaded: {cfg.sigclip_model_path}", log_file)
            else:
                # Load transformers model (CLIP/SigLIP)
                sigclip_model = AutoModel.from_pretrained(cfg.sigclip_model_path)
                sigclip_processor = AutoProcessor.from_pretrained(cfg.sigclip_model_path)
                sigclip_model = sigclip_model.to(model.device)
                sigclip_model.eval()
                sigclip_model._is_openclip_model = False
                sigclip_processor._is_openclip = False
                log_message(f"[SUBSTEP] Transformers model loaded: {cfg.sigclip_model_path}", log_file)
            
        except Exception as e:
            log_message(f"[SUBSTEP] Failed to load models: {e}. Disabling substep decomposition.", log_file)
            llm_model, llm_tokenizer = None, None
            sigclip_model, sigclip_processor = None, None
    
    # Get task
    task = task_suite.get_task(task_id)

    # Get initial states
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)

    # Initialize environment and get task description
    env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)

    # Start episodes
    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        log_message(f"\nTask: {task_description}", log_file)

        # Handle initial state
        if cfg.initial_states_path == "DEFAULT":
            # Use default initial state
            initial_state = initial_states[episode_idx]
        else:
            # Get keys for fetching initial episode state from JSON
            initial_states_task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{episode_idx}"

            # Skip episode if expert demonstration failed to complete the task
            if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                log_message(f"Skipping task {task_id} episode {episode_idx} due to failed expert demo!", log_file)
                continue

            # Get initial state
            initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])

        log_message(f"Starting episode {task_episodes + 1}...", log_file)

        # loading h model
        if cfg.h_decoding:
            hnn_potential_mlp_head = nn.Sequential(
                nn.Linear(6 * 2, 64, bias=True),
                nn.ReLU(),
                nn.Linear(64,2,bias = True)
            ).to(model.device).to(torch.bfloat16)
            hnn_potential_mlp_head.load_state_dict(torch.load(cfg.pretrained_checkpoint + "/h_head--200000_checkpoint.pt"))
            hnn_potential_mlp_head.to(model.device)
        else:
            hnn_potential_mlp_head = None

        # loading energy model
        if cfg.e_decoding:
            hnn_potential_mlp_head = EnergyModel(model.llm_dim,7).to(model.device)
            hnn_potential_mlp_head.load_state_dict(torch.load("/home/aup/YuhangWorkspace/openvla-oft-yhs/ckpts/energy_refined_80000.pt"))
        else:
            hnn_potential_mlp_head = None


        # Run episode
        success, replay_images, substep_info_list = run_episode(
            cfg,
            env,
            task_description,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            initial_state,
            log_file,
            hnn_potential_mlp_head,
            llm_model,
            llm_tokenizer,
            sigclip_model,
            sigclip_processor,
        )

        # Update counters
        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1

        # Save replay video with substep annotations
        if cfg.save_video:
            if cfg.use_substep_decomposition and substep_info_list:
                # Use enhanced video with substep info
                save_rollout_video_with_substep_info(
                    replay_images,
                    substep_info_list,
                    total_episodes,
                    success=success,
                    task_description=task_description,
                    log_file=log_file
                )
            else:
                # Use original video saving (fallback)
                from experiments.robot.libero.libero_utils import save_rollout_video
                save_rollout_video(
                    replay_images, total_episodes, success=success, task_description=task_description, log_file=log_file
                )

        # Log results
        log_message(f"Success: {success}", log_file)
        log_message(f"# episodes completed so far: {total_episodes}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)

    # Log task results
    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    log_message(f"Current task success rate: {task_success_rate}", log_file)
    log_message(f"Current total success rate: {total_success_rate}", log_file)

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                f"success_rate/{task_description}": task_success_rate,
                f"num_episodes/{task_description}": task_episodes,
            }
        )

    return total_episodes, total_successes


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> float:
    """Main function to evaluate a trained policy on LIBERO benchmark tasks."""
    # Validate configuration
    validate_config(cfg)

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Handle LIBERO-PRO evaluation configuration if provided
    if cfg.evaluation_config_path:
        with open(cfg.evaluation_config_path, "r", encoding="utf-8") as f:
            evaluation_cfg = yaml.safe_load(f)

        evaluation_cfg["bddl_files_path"] = evaluation_cfg.get("bddl_files_path", "") + "/" + cfg.task_suite_name
        evaluation_cfg["task_suite_name"] = cfg.task_suite_name

        use_swap = evaluation_cfg.get("use_swap", False)
        use_object = evaluation_cfg.get("use_object", False)
        use_language = evaluation_cfg.get("use_language", False)
        use_task = evaluation_cfg.get("use_task", False)
        use_environment = evaluation_cfg.get("use_environment", False)

        # Step 1: Check if only one of the use_xxx flags is True
        if sum([use_swap, use_object, use_language, use_task, use_environment]) > 1:
            # If more than one flag is True, use the temp environment
            bddl_file_path = evaluation_cfg.get("bddl_files_path", "") + "/" + cfg.task_suite_name + "_temp/"

            init_file_path = evaluation_cfg.get("init_file_dir", "") + "/" + cfg.task_suite_name + "_temp/"

            # Check if the directories exist and the log.txt file contents match
            if not os.path.exists(bddl_file_path) or not os.path.exists(init_file_path):
                # If directories don't exist, create them and the log.txt file
                os.makedirs(init_file_path, exist_ok=True)
                os.makedirs(bddl_file_path, exist_ok=True)

                # Create the log.txt dynamically based on current flag values
                log_content = f"{use_swap},{use_object},{use_language},{use_task},{use_environment}"
                with open(os.path.join(bddl_file_path, "log.txt"), "w") as log_file:
                    log_file.write(log_content)  # Write the dynamic state to the log file

                perturbation.create_env(configs=evaluation_cfg)
            else:
                # If directories exist, check the contents of the log.txt file
                with open(os.path.join(bddl_file_path, "log.txt"), "r") as log_file:
                    log_contents = log_file.read().strip()

                # Define the expected log content based on the current flags
                expected_log = f"{use_swap},{use_object},{use_language},{use_task},{use_environment}"

                # If the log contents don't match, clean up and recreate the environment
                if log_contents != expected_log:
                    # Remove existing files in both directories
                    for folder in [bddl_file_path, init_file_path]:
                        for root, dirs, files in os.walk(folder, topdown=False):
                            for name in files:
                                os.remove(os.path.join(root, name))
                            for name in dirs:
                                os.rmdir(os.path.join(root, name))
                    # Create the environment again
                    os.makedirs(init_file_path, exist_ok=True)
                    os.makedirs(bddl_file_path, exist_ok=True)

                    # Write the updated log content based on current flags
                    with open(os.path.join(bddl_file_path, "log.txt"), "w") as log_file:
                        log_file.write(expected_log)  # Write the updated log

                    perturbation.create_env(configs=evaluation_cfg)

            # Update task_suite_name with "_temp" suffix
            cfg.task_suite_name = cfg.task_suite_name + "_temp"

        # Step 2: Handle the case when only one use_xxx flag is True
        else:
            if use_swap:
                perturb_key = "use_swap"
            elif use_object:
                perturb_key = "use_object"
            elif use_language:
                perturb_key = "use_language"
            elif use_task:
                perturb_key = "use_task"
            elif use_environment:
                perturb_key = "use_environment"
            else:
                perturb_key = None

            if perturb_key:
                init_file_path = evaluation_cfg.get("init_file_dir", "") + cfg.task_suite_name + "_" + evaluation_cfg.get(
                    "perturbation_mapping", {}).get(perturb_key, "")

                if not os.path.exists(init_file_path):
                    perturbation.create_env(configs=evaluation_cfg)

                cfg.task_suite_name = cfg.task_suite_name + "_" + evaluation_cfg.get("perturbation_mapping", {}).get(perturb_key, "")

    # Initialize model and components
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(cfg)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks

    log_message(f"Task suite: {cfg.task_suite_name}", log_file)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks)):
        total_episodes, total_successes = run_task(
            cfg,
            task_suite,
            task_id,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            total_episodes,
            total_successes,
            log_file,
        )

    # Calculate final success rate
    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    # Log final results
    log_message("Final results:", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)


    # saving results
    with open(f"/home/yuhang/Warehouse/Yuhangworkspace/openvla-oft-yhs/ckpts/{cfg.task_label}.txt", "w", encoding="utf-8") as f:
        f.write(f"{final_success_rate:.4f}")  

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": final_success_rate,
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)

    # Close log file
    if log_file:
        log_file.close()

    return final_success_rate


if __name__ == "__main__":
    eval_libero()
