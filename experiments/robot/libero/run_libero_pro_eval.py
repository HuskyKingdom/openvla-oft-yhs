"""
run_libero_eval.py

Evaluates a trained policy in a LIBERO simulation benchmark task suite.
"""

import json
import logging
import os
import re
import sys
import yaml
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union
import cv2
import imageio
import torch
import torch.nn as nn
import draccus
import numpy as np
import tqdm
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
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)

from energy_model.model import EnergyModel


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
    auto_regression: bool = False                    # If True, use autoregressive vla.generate() (56 forward passes); False = parallel predict_action (1 forward pass)
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

    # Task description source
    use_bddl_language: bool = False                  # If True, reads task description from bddl file content (:language field) instead of filename

    # Visualization
    compute_attention: bool = False                  # If True, overlays ViT CLS-attention heatmap on saved video frames
    video_dir: str = "./experiments/logs/videos"     # Directory for annotated MP4s (only used when save_video=True)
    video_fps: int = 10

    # fmt: on


# ---------------------------------------------------------------------------
# Attention / video visualization helpers (shared with eval_pick_rate.py)
# ---------------------------------------------------------------------------

def _vit_attn_ctx(model):
    """Register a forward hook on ViT blocks[-2].attn to capture CLS-attention.

    Usage:
        ctx = _vit_attn_ctx(model)
        get_action(...)          # forward pass fires the hook
        sal = ctx.get()          # [H_grid, W_grid] or None; hook is removed
    """
    class _Ctx:
        def __init__(self):
            self._cache = [None]
            self._handle = None
            cache = self._cache
            try:
                am = model.vision_backbone.featurizer.blocks[-2].attn

                def _hook(module, inp, output):
                    try:
                        x = inp[0].detach()
                        B, N, C = x.shape
                        num_heads = module.num_heads
                        head_dim = C // num_heads
                        scale = head_dim ** -0.5
                        with torch.no_grad():
                            qkv = module.qkv(x).reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
                            q, k, _ = qkv.unbind(0)
                            if getattr(module, 'q_norm', None) is not None:
                                q = module.q_norm(q)
                            if getattr(module, 'k_norm', None) is not None:
                                k = module.k_norm(k)
                            attn_w = (q.float() * scale) @ k.float().transpose(-2, -1)
                            attn_w = attn_w.softmax(dim=-1)
                        cache[0] = attn_w.cpu()
                    except Exception as e:
                        logger.debug(f"[ATTN hook] {e}")

                self._handle = am.register_forward_hook(_hook)
            except (AttributeError, IndexError) as e:
                logger.debug(f"[ATTN] hook setup failed: {e}")

        def get(self) -> Optional[np.ndarray]:
            if self._handle is not None:
                self._handle.remove()
            if self._cache[0] is None:
                return None
            try:
                attn = self._cache[0]              # [1, heads, N, N]
                attn_avg = attn[0].mean(dim=0).cpu().numpy()  # [N, N]
                N = attn_avg.shape[0]
                # Detect token layout: try prefix sizes 0,1,2,4,5 before spatial patches
                for num_prefix in [0, 1, 2, 4, 5, 8]:
                    n_spatial = N - num_prefix
                    if n_spatial <= 0:
                        continue
                    s = int(round(n_spatial ** 0.5))
                    if s * s == n_spatial:
                        if num_prefix == 0:
                            received = attn_avg.mean(axis=0)
                        else:
                            received = attn_avg[0, num_prefix:]  # CLS→spatial
                        sal = received.reshape(s, s)
                        sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
                        return sal
                return None
            except Exception as e:
                logger.debug(f"[ATTN] get() error: {e}")
                return None

    return _Ctx()


def _blend_attn(img_rgb: np.ndarray, sal: np.ndarray, alpha: float = 0.50) -> np.ndarray:
    H, W = img_rgb.shape[:2]
    sal_u8 = (sal * 255).astype(np.uint8)
    sal_r = cv2.resize(sal_u8, (W, H), interpolation=cv2.INTER_LINEAR)
    heat_bgr = cv2.applyColorMap(sal_r, cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    return np.clip(img_rgb.astype(np.float32) * (1 - alpha) + heat_rgb.astype(np.float32) * alpha, 0, 255).astype(np.uint8)


def _annotate_frame_vis(img_rgb: np.ndarray, task_description: str, step: int, success: Optional[bool]) -> np.ndarray:
    """Add info bar above the image. Height is always 3 lines (constant across episode)."""
    H, W = img_rgb.shape[:2]
    font, pad, th = cv2.FONT_HERSHEY_SIMPLEX, 4, 1
    # Auto-scale font so reference 28 chars fits width
    fs = 0.45
    while fs > 0.20:
        (tw, _), _ = cv2.getTextSize("M" * 28, font, fs, th)
        if tw <= W - 2 * pad:
            break
        fs -= 0.02
    lh = max(14, int(cv2.getTextSize("Ag", font, fs, th)[0][1] * 2.2))
    max_px = W - 2 * pad

    # word-wrap instruction to 2 lines
    words = task_description.split()
    lines_instr: List[str] = []
    cur: List[str] = []
    for w in words:
        cand = " ".join(cur + [w])
        if cv2.getTextSize(cand, font, fs, th)[0][0] <= max_px:
            cur.append(w)
        else:
            if cur:
                lines_instr.append(" ".join(cur))
            cur = [w]
    if cur:
        lines_instr.append(" ".join(cur))
    lines_instr = lines_instr[:2]
    while len(lines_instr) < 2:
        lines_instr.append("")

    if success is None:
        status_text, status_color = f"Step {step}", (130, 130, 130)
    else:
        status_text = f"Step {step}  [{'SUCCESS' if success else 'FAIL'}]"
        status_color = (100, 240, 100) if success else (255, 100, 100)

    text_lines = [
        (lines_instr[0], (220, 220, 220)),
        (lines_instr[1], (195, 195, 195)),
        (status_text,    status_color),
    ]
    bar_h = pad + lh * len(text_lines) + pad
    total_h = ((bar_h + H + 15) // 16) * 16   # pad to multiple of 16

    canvas = np.zeros((total_h, W, 3), dtype=np.uint8)
    canvas[bar_h: bar_h + H, :] = img_rgb
    for i, (text, color) in enumerate(text_lines):
        if text:
            cv2.putText(canvas, text, (pad, pad + lh * i + lh - 3), font, fs, color, th, cv2.LINE_AA)
    return canvas


def _save_vis_video(frames: List[np.ndarray], path: str, fps: int = 10) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    writer = imageio.get_writer(path, fps=fps)
    for f in frames:
        writer.append_data(f)
    writer.close()


# ---------------------------------------------------------------------------


def extract_task_from_bddl(bddl_file_path, task_suite=None):
    """Extract task language descriptions from bddl file content (:language field).

    If task_suite is provided, files are read in benchmark task order (correct).
    Otherwise falls back to sorted filename order (may mismatch benchmark order).
    """
    language_pattern = re.compile(r"\(:language\s*(.*?)\)", re.IGNORECASE | re.DOTALL)
    tasks = []
    bddl_dir = Path(bddl_file_path)

    if task_suite is not None:
        # Read in benchmark task order to avoid index mismatch
        for task_id in range(task_suite.n_tasks):
            task = task_suite.get_task(task_id)
            bddl_file = bddl_dir / task.bddl_file
            with bddl_file.open("r", encoding="utf-8") as f:
                content = f.read()
            matches = language_pattern.findall(content)
            if matches:
                tasks.append(matches[0].strip())
            else:
                tasks.append(content.split(":language")[1].split(")")[0].strip())
    else:
        # Fallback: sorted order (original behavior, may mismatch benchmark task order)
        for bddl_file in sorted(bddl_dir.glob("*.bddl")):
            with bddl_file.open("r", encoding="utf-8") as f:
                content = f.read()
            matches = language_pattern.findall(content)
            if matches:
                for lang_text in matches:
                    tasks.append(lang_text.strip())
            else:
                tasks.append(content.split(":language")[1].split(")")[0].strip())
    return tasks


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
):
    """Run a single episode in the environment."""
    env.reset()
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    if cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        print(f"WARNING: cfg.num_open_loop_steps ({cfg.num_open_loop_steps}) does not match NUM_ACTIONS_CHUNK "
              f"({NUM_ACTIONS_CHUNK}). Recommend executing the full action chunk.")
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    t = 0
    replay_images = []           # raw images (for original save_rollout_video path)
    vis_frames: List[np.ndarray] = []   # annotated+heatmap frames (for compute_attention path)
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]
    current_saliency: Optional[np.ndarray] = None

    success = False
    while t < max_steps + cfg.num_steps_wait:
        if t < cfg.num_steps_wait:
            obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
            t += 1
            continue

        observation, img = prepare_observation(obs, resize_size)
        replay_images.append(img)

        if len(action_queue) == 0:
            # Capture attention during get_action forward pass
            attn_ctx = _vit_attn_ctx(model) if cfg.compute_attention else None

            actions = get_action(
                cfg,
                model,
                observation,
                task_description,
                processor=processor,
                action_head=action_head,
                proprio_projector=proprio_projector,
                noisy_action_projector=noisy_action_projector,
                use_film=cfg.use_film,
                h_head=head,
                auto_regression=cfg.auto_regression,
            )
            action_queue.extend(actions)

            if attn_ctx is not None:
                current_saliency = attn_ctx.get()

        action = action_queue.popleft()
        action = process_action(action, cfg.model_family)
        obs, reward, done, info = env.step(action.tolist())

        # Build annotated visualization frame
        if cfg.save_video and cfg.compute_attention:
            raw = get_libero_image(obs)
            frame = _blend_attn(raw, current_saliency) if current_saliency is not None else raw.copy()
            frame = _annotate_frame_vis(frame, task_description, t, success=None)
            vis_frames.append(frame)

        if done:
            success = True
            break
        t += 1

    # Append a final annotated frame with success status
    if cfg.save_video and cfg.compute_attention and vis_frames:
        raw = get_libero_image(obs)
        frame = _blend_attn(raw, current_saliency) if current_saliency is not None else raw.copy()
        frame = _annotate_frame_vis(frame, task_description, t, success=success)
        vis_frames.append(frame)

    return success, replay_images, vis_frames


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
    task_description_override=None,
):
    """Run evaluation for a single task."""
    # Get task
    task = task_suite.get_task(task_id)

    # Get initial states
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)

    # Initialize environment and get task description
    env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
    if task_description_override is not None:
        task_description = task_description_override

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
        success, replay_images, vis_frames = run_episode(
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
        )

        # Update counters
        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1

        # Save replay video
        if cfg.save_video:
            if cfg.compute_attention and vis_frames:
                # Use annotated frames with attention heatmap
                safe_desc = task_description.lower().replace(" ", "_")[:40]
                vid_path = os.path.join(
                    cfg.video_dir,
                    f"ep{total_episodes:04d}_{safe_desc}_{'ok' if success else 'fail'}.mp4",
                )
                _save_vis_video(vis_frames, vid_path, fps=cfg.video_fps)
                log_message(f"Saved attention video: {vid_path}", log_file)
            else:
                save_rollout_video(
                    replay_images, total_episodes, success=success,
                    task_description=task_description, log_file=log_file,
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

    # Extract task descriptions from bddl file content if requested
    bddl_task_descriptions = None
    if cfg.use_bddl_language:
        bddl_dir = str(Path(task_suite.get_task_bddl_file_path(0)).parent)
        bddl_task_descriptions = extract_task_from_bddl(bddl_dir, task_suite=task_suite)
        log_message(f"Using bddl content language from: {bddl_dir}", log_file)
        assert len(bddl_task_descriptions) == num_tasks, (
            f"bddl descriptions count ({len(bddl_task_descriptions)}) != num_tasks ({num_tasks})"
        )

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
            task_description_override=bddl_task_descriptions[task_id] if bddl_task_descriptions else None,
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
