"""
eval_pick_rate.py

Measure how often a trained policy picks the correct object as specified by the
language instruction.

For each trial, the script:
  1. Parses the task instruction to identify the target "pick" object.
  2. Runs the rollout; detects the first pick event (EEF proximity + gripper
     closed + object lifted).
  3. Records whether the picked object matches the instructed object.

Reports per-task and aggregate pick-correct rates, saved to a JSON file.

Usage (on NV server, inside `screen exp`, after setting PYTHONPATH):
    python experiments/robot/libero/eval_pick_rate.py \
        --pretrained_checkpoint ckpt/ckpoints/<name> \
        --task_suite_name libero_spatial \
        --perturbation_mode task \
        --evaluation_config_path experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml \
        --unnorm_key libero_spatial \
        --num_trials_per_task 50 \
        --use_l1_regression False \
        --use_proprio False \
        --num_images_in_input 1 \
        --task_label my_pick_rate_test

Perturbation modes:
    none   – no perturbation (standard eval)
    lan    – language-only perturbation (instruction changed, scene unchanged)
    task   – task instruction swap (instruction from a different task)
    swap   – physical object swap (objects' xy positions swapped)
    object – object replacement (different object instance used)
"""

import json
import logging
import os
import re
import sys
import yaml
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import tqdm
import draccus
import torch
from libero.libero import benchmark

from experiments.robot.libero import perturbation

sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
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
from prismatic.vla.constants import NUM_ACTIONS_CHUNK


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Max steps per suite (mirrors run_libero_pro_eval_substep.py)
# ---------------------------------------------------------------------------
TASK_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    # Perturbed variants keep the same budget
    "libero_spatial_lan": 220, "libero_spatial_task": 220,
    "libero_spatial_swap": 220, "libero_spatial_object": 220,
    "libero_spatial_temp": 220,
    "libero_object_lan": 280, "libero_object_task": 280,
    "libero_object_swap": 280, "libero_object_object": 280,
    "libero_object_temp": 280,
    "libero_goal_lan": 300, "libero_goal_task": 300,
    "libero_goal_swap": 300, "libero_goal_object": 300,
    "libero_goal_temp": 300,
    "libero_10_lan": 520, "libero_10_task": 520,
    "libero_10_swap": 520, "libero_10_object": 520,
    "libero_10_temp": 520,
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class PickRateConfig:
    # Model
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = ""
    use_l1_regression: bool = True
    use_diffusion: bool = False
    use_film: bool = False
    auto_regression: bool = False
    num_images_in_input: int = 2
    use_proprio: bool = True
    center_crop: bool = True
    num_open_loop_steps: int = 8
    lora_rank: int = 32
    unnorm_key: Union[str, Path] = ""
    load_in_8bit: bool = False
    load_in_4bit: bool = False

    # Env
    task_suite_name: str = "libero_spatial"
    perturbation_mode: str = "none"        # none | lan | task | swap | object
    evaluation_config_path: Optional[str] = None
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    env_img_res: int = 256
    single_task_id: int = -1               # -1 = all tasks

    # BDDL language
    use_bddl_language: bool = False

    # Pick detection thresholds
    prox_thresh: float = 0.06             # EEF-to-object proximity (m)
    grip_thresh: float = 0.030            # gripper closed if qpos < this
    lift_thresh: float = 0.015            # object lifted above initial z by this

    # Required by get_vla_action / get_action internals (keep defaults, don't expose to user)
    remove_wrap: bool = False
    h_decoding: bool = False
    e_decoding: bool = False
    energy_alpha: float = 0.5
    num_diffusion_steps_train: int = 50
    num_diffusion_steps_inference: int = 50

    # Output
    task_label: str = ""
    local_log_dir: str = "./experiments/logs"
    seed: int = 7


# ---------------------------------------------------------------------------
# Pick detection helpers
# ---------------------------------------------------------------------------

def _build_moveable_body_map(sim) -> Dict[str, int]:
    """name → body_id for all free-jointed non-robot bodies."""
    mapping = {}
    for jnt_id, jnt_type in enumerate(sim.model.jnt_type):
        if jnt_type != 0:  # 0 = mjJNT_FREE
            continue
        body_id = sim.model.jnt_bodyid[jnt_id]
        body_name = sim.model.body_id2name(body_id)
        if "robot" in body_name.lower():
            continue
        mapping[body_name] = body_id
    return mapping


def _get_initial_z(sim, body_map: Dict[str, int]) -> Dict[str, float]:
    return {name: float(sim.data.body_xpos[bid][2]) for name, bid in body_map.items()}


def _detect_picked_object(
    obs,
    sim,
    body_map: Dict[str, int],
    initial_z: Dict[str, float],
    prox_thresh: float,
    grip_thresh: float,
    lift_thresh: float,
) -> Optional[str]:
    """Return the body name of the object being picked, or None.

    Criteria (all must hold):
      1. Gripper is closed (robot0_gripper_qpos[0] < grip_thresh).
      2. EEF is within prox_thresh of the object's current position.
      3. Object has been lifted by at least lift_thresh above its initial z.
    """
    eef_pos = obs.get("robot0_eef_pos")
    grip = obs.get("robot0_gripper_qpos")
    if eef_pos is None or grip is None:
        return None

    if float(grip[0]) >= grip_thresh:
        return None  # gripper open

    best_name, best_dist = None, float("inf")
    for name, bid in body_map.items():
        obj_pos = sim.data.body_xpos[bid]
        dist = float(np.linalg.norm(eef_pos - obj_pos))
        if dist < prox_thresh and dist < best_dist:
            init_z = initial_z.get(name, 0.0)
            if float(obj_pos[2]) > init_z + lift_thresh:
                best_dist = dist
                best_name = name
    return best_name


def _parse_target_object(instruction: str, body_map: Dict[str, int]) -> Optional[str]:
    """Identify which object the instruction asks the robot to pick.

    Strategy:
      1. Match body names (underscores→spaces) against the instruction text.
         Longer matches win (avoids "mug" matching "coffee mug_1").
      2. Return the body name with the longest natural-language match found in
         the instruction, or None if no match.
    """
    instruction_lower = instruction.lower()
    best_name, best_len = None, 0
    for name in body_map:
        natural = name.replace("_", " ").lower()
        # Remove trailing digit suffix for matching (e.g. "akita dog toy 1" → "akita dog toy")
        natural_no_digit = re.sub(r"\s*\d+$", "", natural)
        for candidate in [natural, natural_no_digit]:
            if candidate and candidate in instruction_lower and len(candidate) > best_len:
                best_len = len(candidate)
                best_name = name
    return best_name


# ---------------------------------------------------------------------------
# BDDL helpers (mirrors run_libero_pro_eval_substep.py)
# ---------------------------------------------------------------------------

def _extract_task_from_bddl(bddl_dir_str: str, task_suite) -> List[str]:
    pattern = re.compile(r"\(:language\s*(.*?)\)", re.IGNORECASE | re.DOTALL)
    tasks = []
    bddl_dir = Path(bddl_dir_str)
    for task_id in range(task_suite.n_tasks):
        task = task_suite.get_task(task_id)
        bddl_file = bddl_dir / task.bddl_file
        with bddl_file.open("r", encoding="utf-8") as f:
            content = f.read()
        matches = pattern.findall(content)
        if matches:
            tasks.append(matches[0].strip())
        else:
            tasks.append(content.split(":language")[1].split(")")[0].strip())
    return tasks


# ---------------------------------------------------------------------------
# Model initialisation (mirrors the existing eval script)
# ---------------------------------------------------------------------------

def _initialize_model(cfg: PickRateConfig):
    model = get_model(cfg)

    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)

    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(cfg, model.llm_dim)

    noisy_action_projector = None
    if cfg.use_diffusion:
        noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim)

    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        _check_unnorm_key(cfg, model)

    return model, action_head, proprio_projector, noisy_action_projector, processor


def _check_unnorm_key(cfg: PickRateConfig, model) -> None:
    key = cfg.unnorm_key or cfg.task_suite_name
    if key not in model.norm_stats and f"{key}_no_noops" in model.norm_stats:
        key = f"{key}_no_noops"
    assert key in model.norm_stats, f"unnorm_key '{key}' not in model.norm_stats!"
    cfg.unnorm_key = key


# ---------------------------------------------------------------------------
# Perturbation config setup (mirrors run_libero_pro_eval_substep.py Step 2)
# ---------------------------------------------------------------------------

def _apply_perturbation_config(cfg: PickRateConfig) -> None:
    """Modify cfg.task_suite_name based on perturbation_mode, creating env if needed."""
    if cfg.perturbation_mode == "none" or not cfg.evaluation_config_path:
        return

    mode = cfg.perturbation_mode
    assert mode in ("lan", "task", "swap", "object"), \
        f"perturbation_mode must be one of: none, lan, task, swap, object. Got: {mode}"

    with open(cfg.evaluation_config_path, "r", encoding="utf-8") as f:
        eval_cfg = yaml.safe_load(f)

    # Set only the requested flag
    for flag in ("use_language", "use_task", "use_swap", "use_object", "use_environment"):
        eval_cfg[flag] = False
    flag_key = {"lan": "use_language", "task": "use_task",
                "swap": "use_swap", "object": "use_object"}[mode]
    eval_cfg[flag_key] = True

    eval_cfg["bddl_files_path"] = eval_cfg.get("bddl_files_path", "") + "/" + cfg.task_suite_name
    eval_cfg["task_suite_name"] = cfg.task_suite_name

    perturb_suffix = eval_cfg.get("perturbation_mapping", {}).get(flag_key, mode)
    init_file_path = eval_cfg.get("init_file_dir", "") + cfg.task_suite_name + "_" + perturb_suffix

    if not os.path.exists(init_file_path):
        perturbation.create_env(configs=eval_cfg)

    cfg.task_suite_name = cfg.task_suite_name + "_" + perturb_suffix
    logger.info(f"Perturbation applied: task_suite_name → {cfg.task_suite_name}")


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------

def _prepare_observation(obs, resize_size):
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)
    img_r = resize_image_for_policy(img, resize_size)
    wrist_r = resize_image_for_policy(wrist_img, resize_size)
    observation = {
        "full_image": img_r,
        "wrist_image": wrist_r,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }
    return observation


def _process_action(action, model_family):
    action = normalize_gripper_action(action, binarize=True)
    if model_family == "openvla":
        action = invert_gripper_action(action)
    return action


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def _run_episode_pick(
    cfg: PickRateConfig,
    env,
    task_description: str,
    expected_object: Optional[str],
    model,
    resize_size,
    processor,
    action_head,
    proprio_projector,
    noisy_action_projector,
    initial_state,
) -> Tuple[bool, Optional[str], bool]:
    """Run one episode and detect the first pick.

    Returns:
        success          – environment success signal
        picked_object    – body name of first object picked (or None)
        pick_correct     – True if picked_object matches expected_object
    """
    env.reset()
    obs = env.set_init_state(initial_state)

    sim = env.sim
    body_map = _build_moveable_body_map(sim)
    initial_z = _get_initial_z(sim, body_map)

    action_queue: deque = deque(maxlen=cfg.num_open_loop_steps)
    max_steps = TASK_MAX_STEPS.get(cfg.task_suite_name, 300)

    picked_object: Optional[str] = None
    success = False
    t = 0

    while t < max_steps + cfg.num_steps_wait:
        if t < cfg.num_steps_wait:
            obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
            t += 1
            continue

        observation = _prepare_observation(obs, resize_size)

        if len(action_queue) == 0:
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
                auto_regression=cfg.auto_regression,
            )
            action_queue.extend(actions)

        action = _process_action(action_queue.popleft(), cfg.model_family)
        obs, reward, done, info = env.step(action.tolist())

        # Pick detection (only track first pick)
        if picked_object is None:
            picked_object = _detect_picked_object(
                obs, sim, body_map, initial_z,
                cfg.prox_thresh, cfg.grip_thresh, cfg.lift_thresh,
            )

        if done:
            success = True
            break
        t += 1

    pick_correct = (
        picked_object is not None
        and expected_object is not None
        and picked_object == expected_object
    )
    return success, picked_object, pick_correct


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

@draccus.wrap()
def eval_pick_rate(cfg: PickRateConfig) -> None:
    assert cfg.pretrained_checkpoint, "--pretrained_checkpoint is required"

    set_seed_everywhere(cfg.seed)

    # Apply perturbation (modifies cfg.task_suite_name in-place)
    _apply_perturbation_config(cfg)

    # Load model
    model, action_head, proprio_projector, noisy_action_projector, processor = _initialize_model(cfg)
    resize_size = get_image_resize_size(cfg)

    # Set up benchmark
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks

    # BDDL language override
    bddl_descriptions = None
    if cfg.use_bddl_language:
        bddl_dir = str(Path(task_suite.get_task_bddl_file_path(0)).parent)
        bddl_descriptions = _extract_task_from_bddl(bddl_dir, task_suite)
        assert len(bddl_descriptions) == num_tasks

    # Logging
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    label = cfg.task_label or f"pick_rate_{cfg.task_suite_name}_{DATE_TIME}"
    log_path = os.path.join(cfg.local_log_dir, f"{label}.txt")
    log_file = open(log_path, "w")

    def log(msg):
        logger.info(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"Checkpoint : {cfg.pretrained_checkpoint}")
    log(f"Task suite : {cfg.task_suite_name}")
    log(f"Perturbation: {cfg.perturbation_mode}")
    log(f"Trials/task: {cfg.num_trials_per_task}")

    # Aggregate stats
    all_results = []  # list of per-task dicts
    total_episodes = total_successes = total_picks = total_correct_picks = 0

    task_ids = [cfg.single_task_id] if cfg.single_task_id >= 0 else range(num_tasks)

    for task_id in tqdm.tqdm(task_ids):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)

        if bddl_descriptions:
            task_description = bddl_descriptions[task_id]

        # Identify the expected target object from the instruction
        # We need the body map — spin up env temporarily to get sim
        env.reset()
        body_map = _build_moveable_body_map(env.sim)
        expected_object = _parse_target_object(task_description, body_map)

        log(f"\n--- Task {task_id}: {task_description}")
        log(f"    Expected pick object: {expected_object}  (from body map: {list(body_map.keys())})")

        task_episodes = task_successes_n = task_picks = task_correct_picks = 0

        for ep_idx in tqdm.tqdm(range(cfg.num_trials_per_task), leave=False):
            initial_state = initial_states[ep_idx]

            success, picked_obj, pick_correct = _run_episode_pick(
                cfg, env, task_description, expected_object,
                model, resize_size, processor,
                action_head, proprio_projector, noisy_action_projector,
                initial_state,
            )

            task_episodes += 1
            if success:
                task_successes_n += 1
            if picked_obj is not None:
                task_picks += 1
            if pick_correct:
                task_correct_picks += 1

            log(
                f"  ep {ep_idx:3d} | success={success} | picked={picked_obj} | "
                f"expected={expected_object} | correct={pick_correct}"
            )

        env.close()

        pick_rate = task_correct_picks / task_episodes if task_episodes > 0 else 0.0
        pick_any_rate = task_picks / task_episodes if task_episodes > 0 else 0.0
        success_rate = task_successes_n / task_episodes if task_episodes > 0 else 0.0

        log(f"  Task SR: {success_rate:.3f}  |  Pick-any rate: {pick_any_rate:.3f}  |  Pick-correct rate: {pick_rate:.3f}")

        task_result = {
            "task_id": task_id,
            "task_description": task_description,
            "expected_object": expected_object,
            "episodes": task_episodes,
            "successes": task_successes_n,
            "picks": task_picks,
            "correct_picks": task_correct_picks,
            "success_rate": success_rate,
            "pick_any_rate": pick_any_rate,
            "pick_correct_rate": pick_rate,
        }
        all_results.append(task_result)

        total_episodes += task_episodes
        total_successes += task_successes_n
        total_picks += task_picks
        total_correct_picks += task_correct_picks

    # Final summary
    overall_sr = total_successes / total_episodes if total_episodes > 0 else 0.0
    overall_pick_any = total_picks / total_episodes if total_episodes > 0 else 0.0
    overall_pick_correct = total_correct_picks / total_episodes if total_episodes > 0 else 0.0

    log("\n========== FINAL RESULTS ==========")
    log(f"Total episodes   : {total_episodes}")
    log(f"Success rate     : {overall_sr:.4f} ({overall_sr*100:.1f}%)")
    log(f"Pick-any rate    : {overall_pick_any:.4f} ({overall_pick_any*100:.1f}%)")
    log(f"Pick-correct rate: {overall_pick_correct:.4f} ({overall_pick_correct*100:.1f}%)")

    # Save JSON results
    results_dict = {
        "checkpoint": str(cfg.pretrained_checkpoint),
        "task_suite_name": cfg.task_suite_name,
        "perturbation_mode": cfg.perturbation_mode,
        "num_trials_per_task": cfg.num_trials_per_task,
        "overall": {
            "success_rate": overall_sr,
            "pick_any_rate": overall_pick_any,
            "pick_correct_rate": overall_pick_correct,
        },
        "per_task": all_results,
    }

    json_path = os.path.join(cfg.local_log_dir, f"{label}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2)
    log(f"\nResults saved to: {json_path}")
    log(f"Log saved to    : {log_path}")

    log_file.close()


if __name__ == "__main__":
    eval_pick_rate()
