"""
Standalone LIBERO environment worker for multiprocessing with spawn start method.

This module intentionally imports ONLY lightweight dependencies (libero, numpy, gc)
so that spawned subprocesses do not inherit or re-import heavy libraries such as
transformers, tensorflow, or PyTorch.
"""
import gc
import os
import numpy as np


# ---------------------------------------------------------------------------
# Minimal helpers inlined from verl/utils/libero_utils.py
# (only the functions actually used by env_worker; no tensorflow dependency)
# ---------------------------------------------------------------------------

def _get_libero_env(task, model_family, resolution=256):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    task_description = task.language
    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    return env, task_description


def _get_libero_dummy_action(model_family: str):
    return [0, 0, 0, 0, 0, 0, -1]


def _normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    normalized_action = action.copy()
    orig_low, orig_high = 0.0, 1.0
    normalized_action[..., -1] = (
        2 * (normalized_action[..., -1] - orig_low) / (orig_high - orig_low) - 1
    )
    if binarize:
        normalized_action[..., -1] = np.sign(normalized_action[..., -1])
    return normalized_action


def _invert_gripper_action(action: np.ndarray) -> np.ndarray:
    inverted_action = action.copy()
    inverted_action[..., -1] = inverted_action[..., -1] * -1.0
    return inverted_action


# ---------------------------------------------------------------------------
# env_worker: the target function for each spawned LIBERO subprocess
# ---------------------------------------------------------------------------

def env_worker(task_name, task_id, trial_id, config,
               input_queue, output_queue, is_valid, global_steps, max_steps):
    """Worker process for Libero environments (spawn-safe, no heavy imports)."""
    # PyTorch 2.6+ changed torch.load default to weights_only=True, which breaks
    # libero's get_task_init_states() (stores numpy arrays). Patch before import.
    import torch
    _orig_load = torch.load
    def _patched_load(*args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return _orig_load(*args, **kwargs)
    torch.load = _patched_load

    from libero.libero import benchmark

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_name]()
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    initial_state = initial_states[trial_id]

    env = None
    while True:
        try:
            env, task_description = _get_libero_env(
                task, config.model_family, resolution=256
            )
            break
        except Exception:
            print("*** env initialization failed ***")
            if env is not None:
                try:
                    env.close()
                except Exception as e:
                    print(f"error when close the env: {e}")
            gc.collect()
            print("gc collect finish")

    env.reset()
    obs = env.set_init_state(initial_state)

    t = 0
    valid_images = []
    while t < config.num_steps_wait:
        obs, _, _, _ = env.step(_get_libero_dummy_action(config.model_family))
        t += 1

    if is_valid:
        img = obs["agentview_image"][::-1, ::-1]
        valid_images.append(img)

    output_queue.put({
        'type': 'init',
        'obs': obs,
        'task_description': task_description,
        'valid_images': valid_images.copy(),
        'task_file_name': f"{task_name}_task_{task_id}_trial_{trial_id}",
        'active': True,
        'complete': False,
        'finish_step': 0,
    })

    active = True
    complete = False
    finish_step = 0

    while True:
        action = input_queue.get()
        if action is None:
            env.close()
            output_queue.put({'type': 'terminate'})
            break

        step_images = []
        for i in range(len(action)):
            a = action[i]
            normalized_action = _normalize_gripper_action(a, binarize=True)
            inverted_action = _invert_gripper_action(normalized_action)
            obs, reward, done, info = env.step(inverted_action.tolist())

            if is_valid:
                img = obs["agentview_image"][::-1, ::-1]
                step_images.append(img)

            finish_step += 1
            if done or finish_step >= max_steps:
                active = False
                complete = done
                break

        output_queue.put({
            'type': 'step',
            'obs': obs,
            'active': active,
            'complete': complete,
            'finish_step': finish_step,
            'valid_images': step_images.copy() if is_valid else [],
        })
