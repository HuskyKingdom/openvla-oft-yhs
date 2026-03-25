"""
Standalone LIBERO environment worker for multiprocessing with spawn start method.

This module intentionally imports ONLY lightweight dependencies (libero, numpy, gc)
so that spawned subprocesses do not inherit or re-import heavy libraries such as
transformers, tensorflow, or PyTorch.

Task metadata (bddl_file, task_description, initial_state) are pre-loaded by the
parent process and passed as arguments, so this worker never needs to import torch
or libero.benchmark.
"""
import gc
import os
import numpy as np


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

def env_worker(task_bddl_file, task_description, initial_state,
               task_name, task_id, trial_id, config,
               input_queue, output_queue, is_valid, global_steps, max_steps):
    """Worker process for Libero environments (spawn-safe, no torch needed).

    Parameters
    ----------
    task_bddl_file    : str   – absolute path to the BDDL file (pre-computed by parent)
    task_description  : str   – language instruction (pre-computed by parent)
    initial_state     : array – initial robot/object state (pre-loaded by parent via torch.load)
    task_name         : str   – suite name (for task_file_name logging)
    task_id           : int
    trial_id          : int
    """
    from libero.libero.envs import OffScreenRenderEnv

    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": 256,
        "camera_widths": 256,
    }

    env = None
    while True:
        try:
            env = OffScreenRenderEnv(**env_args)
            env.seed(0)
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
