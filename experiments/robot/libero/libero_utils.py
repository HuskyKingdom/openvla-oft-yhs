"""Utils for evaluating policies in LIBERO simulation environments."""

import math
import os
from typing import List, Optional, Tuple

import cv2
import imageio
import numpy as np
import tensorflow as tf
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from experiments.robot.robot_utils import (
    DATE,
    DATE_TIME,
)


def get_libero_env(task, model_family, resolution=256):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def get_libero_dummy_action(model_family: str):
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


def get_libero_image(obs):
    """Extracts third-person image from observations and preprocesses it."""
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img


def get_libero_wrist_image(obs):
    """Extracts wrist camera image from observations and preprocesses it."""
    img = obs["robot0_eye_in_hand_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img


def save_rollout_video(rollout_images, idx, success, task_description, log_file=None):
    """Saves an MP4 replay of an episode."""
    rollout_dir = f"./rollouts/{DATE}"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/{DATE_TIME}--openvla_oft--episode={idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

import torch

def quat2axisangle_torch(quat: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Batch convert quaternions to axis-angle representation: (B,4) -> (B,3)
    quat[..., :3] = (x,y,z), quat[..., 3] = w
    """
    # Separate xyz and w
    xyz = quat[..., :3]          # (B,3)
    w   = quat[..., 3].clamp(-1.0, 1.0)  # (B,)

    # Calculate angle = 2 * arccos(w)
    angle = 2.0 * torch.acos(w)  # (B,)

    # Calculate denom = sqrt(1 - w^2), and prevent division by zero
    denom = torch.sqrt(torch.clamp(1.0 - w * w, min=0.0))  # (B,)
    # For denom < eps, we directly return zero vector
    safe_denom = denom.clone().masked_fill_(denom < eps, 1.0)

    # Axis-angle vector = xyz * angle / denom
    axis_angle = xyz * (angle / safe_denom).unsqueeze(-1)  # (B,3)

    # For positions where original denom < eps (i.e., angle ≈ 0), set to 0
    axis_angle = axis_angle.masked_fill(denom.unsqueeze(-1) < eps, 0.0)
    return axis_angle


def project_world_to_pixel(
    world_pos: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    img_width: int = 256,
    img_height: int = 256,
    apply_180_rotation: bool = True
) -> Optional[Tuple[int, int]]:
    """
    Projects a 3D point from world coordinates to pixel coordinates.
    
    Args:
        world_pos: (3,) Position in world coordinate frame
        extrinsic: (4,4) World-to-camera extrinsic matrix [R|t]
        intrinsic: (3,3) Camera intrinsic matrix
        img_width: Image width for boundary checking
        img_height: Image height for boundary checking
        apply_180_rotation: Whether to apply 180-degree rotation compensation for rotated images
    
    Returns:
        Pixel coordinates (u, v), or None if point is outside the field of view
    """
    # 1. Transform world coordinates to camera coordinates (homogeneous transformation)
    world_pos_homog = np.append(world_pos, 1.0)  # (4,)
    camera_pos = extrinsic @ world_pos_homog     # (4,)
    camera_pos = camera_pos[:3]                   # (3,) Take first 3 dimensions
    
    # 2. Check if point is behind the camera
    if camera_pos[2] <= 0:
        return None
    
    # 3. Project camera coordinates to pixel (pinhole camera model)
    pixel_homog = intrinsic @ camera_pos         # (3,)
    u = pixel_homog[0] / pixel_homog[2]
    v = pixel_homog[1] / pixel_homog[2]
    
    # 4. Check if within image bounds BEFORE rotation
    # (This ensures we validate coordinates in the original camera frame)
    if not (0 <= u < img_width and 0 <= v < img_height):
        return None
    
    # 5. Apply 180-degree rotation compensation (after boundary check)
    # LIBERO images are rotated 180 degrees via img[::-1, ::-1]
    if apply_180_rotation:
        u = img_width - 1 - u
        v = img_height - 1 - v
    
    return (int(u), int(v))


def draw_trajectory_on_frame(
    frame: np.ndarray,
    trajectory_points: List[Tuple[int, int]],
    gripper_states: List[float],
    history_length: int = 15
) -> np.ndarray:
    """
    Draws trajectory on a single frame image.
    
    Args:
        frame: (H, W, 3) RGB image
        trajectory_points: List of pixel coordinates for current and historical frames
        gripper_states: Corresponding gripper states (normalized values, -1=closed, +1=open)
        history_length: Maximum number of historical frames to draw
    
    Returns:
        Copy of the image with trajectory drawn
    """
    # Create a copy of the image
    frame_copy = frame.copy()
    
    # If no trajectory points, return as is
    if len(trajectory_points) == 0:
        return frame_copy
    
    # Only take the most recent history_length frames
    start_idx = max(0, len(trajectory_points) - history_length)
    recent_points = trajectory_points[start_idx:]
    recent_grippers = gripper_states[start_idx:]
    
    # Filter out None points to check if we have any valid points
    valid_points = [p for p in recent_points if p is not None]
    if len(valid_points) == 0:
        # No valid points to draw, return original frame
        return frame_copy
    
    # If only one point, just draw a marker
    if len(recent_points) == 1:
        if recent_points[0] is not None:
            u, v = recent_points[0]
            # Draw cross marker (yellow)
            cv2.drawMarker(frame_copy, (u, v), (255, 255, 0), cv2.MARKER_CROSS, 8, 2)
        return frame_copy
    
    # Draw connecting lines (gradient color: blue->red)
    for i in range(len(recent_points) - 1):
        if recent_points[i] is None or recent_points[i+1] is None:
            continue
        
        # Calculate gradient color (old->new: blue->red)
        t = i / (len(recent_points) - 1)  # 0 to 1
        color_b = int(255 * (1 - t))  # Blue component decreases
        color_r = int(255 * t)         # Red component increases
        color = (color_r, 0, color_b)  # BGR format
        
        # Draw line
        cv2.line(frame_copy, recent_points[i], recent_points[i+1], color, 2)
    
    # Draw circle markers at each point (based on gripper state)
    for i, (point, gripper) in enumerate(zip(recent_points, recent_grippers)):
        if point is None:
            continue
        
        u, v = point
        # Gripper closed (-1): red filled circle, open (+1): green hollow circle
        if gripper < 0:  # Closed
            cv2.circle(frame_copy, (u, v), 3, (0, 0, 255), -1)  # Red filled
        else:  # Open
            cv2.circle(frame_copy, (u, v), 3, (0, 255, 0), 2)   # Green hollow
    
    # Draw cross marker at current frame (last point, yellow, slightly larger)
    if recent_points[-1] is not None:
        u, v = recent_points[-1]
        cv2.drawMarker(frame_copy, (u, v), (255, 255, 0), cv2.MARKER_CROSS, 10, 2)
    
    return frame_copy


def draw_trajectory_on_episode(
    frames: List[np.ndarray],
    eef_positions: List[np.ndarray],
    gripper_actions: List[float],
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    history_length: int = 15
) -> List[np.ndarray]:
    """
    Draws trajectory on all frames of an episode.
    
    Args:
        frames: All original frames of the episode
        eef_positions: End-effector world coordinates corresponding to each frame
        gripper_actions: Gripper actions executed at each frame (post-processed values)
        extrinsic: (4,4) World-to-camera extrinsic matrix
        intrinsic: (3,3) Camera intrinsic matrix
        history_length: Length of history window
    
    Returns:
        List of frames with trajectory drawn
    """
    # Ensure data lengths are consistent
    assert len(frames) == len(eef_positions) == len(gripper_actions), \
        f"Data length mismatch: frames={len(frames)}, eef={len(eef_positions)}, gripper={len(gripper_actions)}"
    
    # First project all eef positions to pixel coordinates
    pixel_trajectory = []
    for eef_pos in eef_positions:
        pixel_pos = project_world_to_pixel(eef_pos, extrinsic, intrinsic, apply_180_rotation=True)
        pixel_trajectory.append(pixel_pos)
    
    # Debug: count valid projections
    valid_count = sum(1 for p in pixel_trajectory if p is not None)
    print(f"[TRAJECTORY DEBUG] Total frames: {len(frames)}, Valid projections: {valid_count}/{len(pixel_trajectory)}")
    
    if valid_count > 0:
        sample_points = [p for p in pixel_trajectory if p is not None][:5]
        print(f"[TRAJECTORY DEBUG] Sample projected points: {sample_points}")
    else:
        # Print first eef position for debugging
        if len(eef_positions) > 0:
            print(f"[TRAJECTORY DEBUG] Sample eef world position: {eef_positions[0]}")
    
    # Draw trajectory for each frame
    annotated_frames = []
    for i, frame in enumerate(frames):
        # Get trajectory points for current frame and history window
        start_idx = max(0, i + 1 - history_length)
        trajectory_window = pixel_trajectory[start_idx:i+1]
        gripper_window = gripper_actions[start_idx:i+1]
        
        # Draw trajectory
        annotated_frame = draw_trajectory_on_frame(
            frame,
            trajectory_window,
            gripper_window,
            history_length=history_length
        )
        annotated_frames.append(annotated_frame)
    
    return annotated_frames