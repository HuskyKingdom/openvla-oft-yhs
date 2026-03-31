"""
convert_lerobot_to_rlds.py

Converts LeRobot v3 format (Parquet + MP4 videos) to RLDS/TFRecord format for APD training.

HuggingFace `datasets` Arrow caches do not include video bytes; this script expects a full
local copy with `meta/`, `data/`, and `videos/` (downloaded automatically if missing).

Usage:
    python scripts/convert_lerobot_to_rlds.py \\
        --input_dir /path/to/so101_raw \\
        --output_dir /path/to/so101_rlds \\
        --dataset_name so101_poker_yellow
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pyarrow.parquet as pq
import tensorflow as tf
from datasets import load_dataset
from huggingface_hub import snapshot_download
from tqdm import tqdm


def _ensure_dataset_root(input_dir: Path, repo_id: str) -> Path:
    """Return path containing meta/info.json, data/*.parquet, and videos/."""
    info = input_dir / "meta" / "info.json"
    if info.is_file():
        return input_dir.resolve()
    print(f"No meta/info.json under {input_dir}; downloading dataset from {repo_id} ...")
    snapshot_download(
        repo_id,
        repo_type="dataset",
        local_dir=str(input_dir),
    )
    if not info.is_file():
        raise FileNotFoundError(
            f"After download, expected {info}. Check repo_id and permissions."
        )
    return input_dir.resolve()


def _load_dataset_info(root: Path) -> dict:
    with open(root / "meta" / "info.json") as f:
        return json.load(f)


def _sorted_parquet_files(root: Path, subdir: str) -> List[Path]:
    files = list((root / subdir).glob("**/*.parquet"))
    files.sort()
    return files


def load_episode_metadata(root: Path) -> Dict[int, dict]:
    """Per-episode video chunk/file indices and segment start times (LeRobot v3)."""
    paths = _sorted_parquet_files(root, "meta/episodes")
    if not paths:
        raise FileNotFoundError(f"No episode metadata under {root / 'meta/episodes'}")

    by_ep: Dict[int, dict] = {}
    for p in paths:
        table = pq.read_table(p)
        d = table.to_pydict()
        n = table.num_rows
        tasks_col = d.get("tasks")
        for i in range(n):
            epi = int(d["episode_index"][i])
            tasks = tasks_col[i] if tasks_col is not None else []
            instruction = tasks[0] if tasks else ""
            by_ep[epi] = {
                "instruction": instruction,
                "top": {
                    "chunk": int(d["videos/observation.images.top/chunk_index"][i]),
                    "file": int(d["videos/observation.images.top/file_index"][i]),
                    "from_ts": float(d["videos/observation.images.top/from_timestamp"][i]),
                },
                "wrist": {
                    "chunk": int(d["videos/observation.images.wrist/chunk_index"][i]),
                    "file": int(d["videos/observation.images.wrist/file_index"][i]),
                    "from_ts": float(d["videos/observation.images.wrist/from_timestamp"][i]),
                },
            }
    return by_ep


def load_lerobot_frame_table(root: Path):
    """Load all frame rows from data/**/*.parquet (flat LeRobot column names)."""
    data_files = _sorted_parquet_files(root, "data")
    if not data_files:
        raise FileNotFoundError(f"No parquet under {root / 'data'}")
    print(f"Loading {len(data_files)} parquet shard(s) from {root / 'data'} ...")
    return load_dataset("parquet", data_files=[str(p) for p in data_files], split="train")


def video_file_path(root: Path, video_key: str, chunk_index: int, file_index: int) -> Path:
    return root / "videos" / video_key / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.mp4"


def _read_rgb_cv2(cap: cv2.VideoCapture) -> np.ndarray:
    ok, bgr = cap.read()
    if not ok or bgr is None:
        raise RuntimeError("cv2.VideoCapture.read() failed (codec or EOF)")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _grab_frame_pyav(path: Path, time_sec: float) -> np.ndarray:
    import av

    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        seek_us = max(0, int((time_sec - 0.25) * 1_000_000))
        container.seek(seek_us, backward=True, any_frame=False)
        best = None
        best_err = 1e9
        for frame in container.decode(stream):
            ft = float(frame.time)
            err = abs(ft - time_sec)
            if err < best_err:
                best_err = err
                best = frame
            if ft > time_sec + 0.1:
                break
        if best is None:
            raise RuntimeError(f"PyAV: no frame near t={time_sec} in {path}")
        return best.to_ndarray(format="rgb24")
    finally:
        container.close()


def read_episode_camera_frames(
    path: Path,
    from_ts: float,
    fps: float,
    frame_indices: List[int],
    prefer_pyav: bool = False,
) -> List[np.ndarray]:
    """
    Decode frames at logical indices frame_indices (within-episode indices).
    Absolute time in file = from_ts + frame_index / fps (matches LeRobot timestamps).
    """
    if not frame_indices:
        return []

    times = [from_ts + fi / fps for fi in frame_indices]
    out: List[np.ndarray] = []

    if prefer_pyav:
        for t in times:
            out.append(_grab_frame_pyav(path, t))
        return out

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        for t in times:
            out.append(_grab_frame_pyav(path, t))
        return out

    try:
        expected_next: Optional[int] = None
        start_offset = int(round(from_ts * fps))
        for fi in frame_indices:
            abs_frame = start_offset + int(fi)
            if expected_next is None or abs_frame != expected_next:
                cap.set(cv2.CAP_PROP_POS_FRAMES, abs_frame)
            rgb = _read_rgb_cv2(cap)
            out.append(rgb)
            expected_next = abs_frame + 1
    except Exception:
        cap.release()
        return [_grab_frame_pyav(path, t) for t in times]

    cap.release()
    return out


def get_proprio_vector(sample: dict) -> np.ndarray:
    if "observation.state" in sample:
        return np.array(sample["observation.state"], dtype=np.float32)
    return np.array(sample["observation"]["state"], dtype=np.float32)


def group_by_episodes(dataset) -> Dict[int, List[int]]:
    episodes: Dict[int, List[int]] = {}
    for idx, sample in enumerate(tqdm(dataset, desc="Grouping episodes")):
        episode_idx = int(sample["episode_index"])
        episodes.setdefault(episode_idx, []).append(idx)
    print(f"Found {len(episodes)} episodes")
    return episodes


def compute_dataset_statistics(dataset, episodes: Dict[int, List[int]]):
    print("Computing dataset statistics...")
    all_actions: List[np.ndarray] = []
    all_proprios: List[np.ndarray] = []

    for _, indices in tqdm(episodes.items(), desc="Computing stats"):
        for idx in indices:
            sample = dataset[idx]
            action = np.array(sample["action"], dtype=np.float32)
            proprio = get_proprio_vector(sample)
            all_actions.append(action)
            all_proprios.append(proprio)

    all_actions = np.array(all_actions)
    all_proprios = np.array(all_proprios)

    return {
        "action": {
            "mean": all_actions.mean(axis=0).tolist(),
            "std": all_actions.std(axis=0).tolist(),
            "max": all_actions.max(axis=0).tolist(),
            "min": all_actions.min(axis=0).tolist(),
            "q01": np.percentile(all_actions, 1, axis=0).tolist(),
            "q99": np.percentile(all_actions, 99, axis=0).tolist(),
        },
        "proprio": {
            "mean": all_proprios.mean(axis=0).tolist(),
            "std": all_proprios.std(axis=0).tolist(),
            "max": all_proprios.max(axis=0).tolist(),
            "min": all_proprios.min(axis=0).tolist(),
            "q01": np.percentile(all_proprios, 1, axis=0).tolist(),
            "q99": np.percentile(all_proprios, 99, axis=0).tolist(),
        },
    }


def create_episode_features(
    dataset,
    episode_indices: List[int],
    episode_idx: int,
    episode_meta: Dict[int, dict],
    root: Path,
    fps: float,
    default_instruction: str,
    use_pyav_video: bool,
) -> List[dict]:
    meta = episode_meta[episode_idx]
    path_top = video_file_path(
        root, "observation.images.top", meta["top"]["chunk"], meta["top"]["file"]
    )
    path_wrist = video_file_path(
        root, "observation.images.wrist", meta["wrist"]["chunk"], meta["wrist"]["file"]
    )
    instruction = meta.get("instruction") or default_instruction
    if not instruction.strip():
        instruction = default_instruction

    frame_indices = [int(dataset[idx]["frame_index"]) for idx in episode_indices]

    tops = read_episode_camera_frames(
        path_top, meta["top"]["from_ts"], fps, frame_indices, prefer_pyav=use_pyav_video
    )
    wrists = read_episode_camera_frames(
        path_wrist, meta["wrist"]["from_ts"], fps, frame_indices, prefer_pyav=use_pyav_video
    )

    steps = []
    for i, idx in enumerate(episode_indices):
        sample = dataset[idx]
        action = np.array(sample["action"], dtype=np.float32)
        proprio = get_proprio_vector(sample)
        img_top = np.asarray(tops[i], dtype=np.uint8)
        img_wrist = np.asarray(wrists[i], dtype=np.uint8)

        steps.append(
            {
                "observation": {
                    "image_primary": img_top,
                    "image_wrist": img_wrist,
                    "proprio": proprio,
                },
                "action": action,
                "is_first": i == 0,
                "is_last": i == len(episode_indices) - 1,
                "is_terminal": i == len(episode_indices) - 1,
                "language_instruction": instruction,
            }
        )
    return steps


def create_rlds_dataset(
    dataset,
    episodes: Dict[int, List[int]],
    output_dir: Path,
    dataset_name: str,
    episode_meta: Dict[int, dict],
    root: Path,
    fps: float,
    default_instruction: str,
    use_pyav_video: bool,
):
    print("Converting to RLDS format...")
    output_dir.mkdir(parents=True, exist_ok=True)
    tfrecord_path = output_dir / f"{dataset_name}.tfrecord"

    with tf.io.TFRecordWriter(str(tfrecord_path)) as writer:
        for episode_idx in tqdm(sorted(episodes.keys()), desc="Converting episodes"):
            episode_indices = episodes[episode_idx]
            episode_steps = create_episode_features(
                dataset,
                episode_indices,
                episode_idx,
                episode_meta,
                root,
                fps,
                default_instruction,
                use_pyav_video,
            )
            for step in episode_steps:
                example = create_tf_example(step)
                writer.write(example.SerializeToString())

    print(f"RLDS dataset saved to: {tfrecord_path}")
    return tfrecord_path


def create_tf_example(step: Dict):
    def _bytes_feature(value):
        if isinstance(value, np.ndarray):
            value = value.tobytes()
        elif isinstance(value, str):
            value = value.encode("utf-8")
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_list_feature(value):
        if isinstance(value, np.ndarray):
            value = value.flatten().tolist()
        elif not isinstance(value, list):
            value = [value]
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def _int64_feature(value):
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    features = {
        "observation/image_primary": _bytes_feature(step["observation"]["image_primary"]),
        "observation/image_wrist": _bytes_feature(step["observation"]["image_wrist"]),
        "observation/proprio": _float_list_feature(step["observation"]["proprio"]),
        "action": _float_list_feature(step["action"]),
        "language_instruction": _bytes_feature(step["language_instruction"]),
        "is_first": _int64_feature(int(step["is_first"])),
        "is_last": _int64_feature(int(step["is_last"])),
        "is_terminal": _int64_feature(int(step["is_terminal"])),
    }
    return tf.train.Example(features=tf.train.Features(feature=features))


def save_dataset_statistics(stats: Dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = output_dir / "dataset_statistics.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Dataset statistics saved to: {stats_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert LeRobot v3 dataset to RLDS format")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/lustre/fsw/portfolios/edgeai/users/chrislin/datasets/so101_raw",
        help="Directory with meta/, data/, videos/ (full snapshot); downloaded if incomplete",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="christian0420/so101-poker-yellow-task",
        help="Hugging Face dataset repo when input_dir has no meta/info.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/lustre/fsw/portfolios/edgeai/users/chrislin/datasets/so101_rlds",
        help="Output directory for RLDS dataset",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="so101_poker_yellow",
        help="Base name for the output TFRecord",
    )
    parser.add_argument(
        "--default_instruction",
        type=str,
        default="Pick up the yellow poker chip and place it in the target location",
        help="Fallback language string if episode metadata has no task text",
    )
    parser.add_argument(
        "--use_pyav_video",
        action="store_true",
        help="Decode video with PyAV only (recommended if OpenCV cannot read AV1)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LeRobot to RLDS Conversion")
    print("=" * 60)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Dataset: {args.dataset_name}")
    print("")

    root = _ensure_dataset_root(input_dir, args.repo_id)
    info = _load_dataset_info(root)
    fps = float(info.get("fps", 30))
    print(f"Dataset root: {root} (fps={fps})")

    episode_meta = load_episode_metadata(root)
    dataset = load_lerobot_frame_table(root)
    print(f"Loaded {len(dataset)} samples")
    print("")

    episodes = group_by_episodes(dataset)
    stats = compute_dataset_statistics(dataset, episodes)

    print("")
    print("Dataset Statistics:")
    print(f"  Action dimension: {len(stats['action']['mean'])}")
    print(f"  Proprio dimension: {len(stats['proprio']['mean'])}")
    print("")

    save_dataset_statistics(stats, output_dir)
    create_rlds_dataset(
        dataset,
        episodes,
        output_dir,
        args.dataset_name,
        episode_meta,
        root,
        fps,
        args.default_instruction,
        args.use_pyav_video,
    )

    print("")
    print("=" * 60)
    print("Conversion Complete!")
    print("=" * 60)
    print(f"RLDS dataset: {output_dir / args.dataset_name}.tfrecord")
    print(f"Statistics: {output_dir / 'dataset_statistics.json'}")


if __name__ == "__main__":
    main()
