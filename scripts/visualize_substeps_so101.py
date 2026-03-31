"""
visualize_substeps_so101.py

Render annotated MP4 videos that overlay substep labels from
substep_labels_so101.json on the original SO101 LeRobot camera videos
for human validation.

For each episode the script produces one side-by-side video (top + wrist)
with a semi-transparent HUD showing:
  - timestep number
  - action type (PICK / PLACE) colour-coded
  - APD_step description
  - APD_prep_step (move description) when present
  - EOS marker when is_substep_end is True
  - cycle number

Usage (inside the container):
    pip install av imageio[ffmpeg]

    python scripts/visualize_substeps_so101.py \
        --labels_path substep_labels_so101.json \
        --data_dir /lustre/fsw/portfolios/edgeai/users/chrislin/datasets/so101_raw \
        --output_dir /lustre/fsw/portfolios/edgeai/users/chrislin/datasets/so101_viz \
        --episodes 0 1 2

    # or all episodes (omit --episodes)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import av
import cv2
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

ACTION_COLORS_BGR = {
    "pick": (0, 220, 0),
    "place": (0, 140, 255),
    "move": (255, 200, 0),
}
WHITE = (255, 255, 255)
GREY = (180, 180, 180)
BG = (0, 0, 0)


def _sorted_parquet_files(root: Path, subdir: str) -> List[Path]:
    files = list((root / subdir).glob("**/*.parquet"))
    files.sort()
    return files


def load_episode_metadata(root: Path) -> Dict[int, dict]:
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
            by_ep[epi] = {
                "instruction": tasks[0] if tasks else "",
                "length": int(d["length"][i]),
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


def video_file_path(root: Path, key: str, chunk: int, file: int) -> Path:
    return root / "videos" / key / f"chunk-{chunk:03d}" / f"file-{file:03d}.mp4"


def decode_episode_frames_pyav(path: Path, from_ts: float, fps: float, n_frames: int) -> List[np.ndarray]:
    container = av.open(str(path))
    stream = container.streams.video[0]

    seek_us = max(0, int((from_ts - 0.5) * 1_000_000))
    container.seek(seek_us, backward=True, any_frame=False)

    frames_by_idx: Dict[int, np.ndarray] = {}
    target_indices = set(range(n_frames))

    for frame in container.decode(stream):
        ft = float(frame.time)
        idx = round((ft - from_ts) * fps)
        if idx in target_indices and idx not in frames_by_idx:
            frames_by_idx[idx] = frame.to_ndarray(format="rgb24")
        if len(frames_by_idx) == n_frames:
            break
        if idx > n_frames + 5:
            break

    container.close()

    out: List[np.ndarray] = []
    for i in range(n_frames):
        if i in frames_by_idx:
            out.append(frames_by_idx[i])
        elif out:
            out.append(out[-1].copy())
        else:
            out.append(np.zeros((480, 640, 3), dtype=np.uint8))
    return out


def build_label_lookup(timestep_labels: List[dict], total_ts: int) -> List[List[dict]]:
    """Build a per-timestep array; each slot holds all labels for that timestep."""
    lookup: List[List[dict]] = [[] for _ in range(total_ts)]
    for lab in timestep_labels:
        t = lab["timestep"]
        if 0 <= t < total_ts:
            lookup[t].append(lab)
    return lookup


def wrap_text(text: str, max_chars: int) -> List[str]:
    words = text.split()
    lines: List[str] = []
    cur = ""
    for w in words:
        if len(cur) + len(w) + 1 <= max_chars:
            cur += (" " + w if cur else w)
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [""]


def annotate_frame(
    frame_bgr: np.ndarray,
    timestep: int,
    total_ts: int,
    labels: List[dict],
    instruction: str,
) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = max(0.35, w / 900.0)
    thick = 1
    lh = int(max(14, h / 22.0))
    pad = int(max(4, w / 80.0))

    lines: List[tuple] = []

    lines.append((f"t={timestep}/{total_ts}", WHITE))
    lines.append(("", WHITE))

    if labels:
        label = labels[0]
        apd = label.get("APD_step", "")
        action = label.get("action", "")
        color = ACTION_COLORS_BGR.get(action, WHITE)
        for seg in wrap_text(apd if apd else "(no step)", 46):
            lines.append((seg, color))
    else:
        lines.append(("(no label)", GREY))

    max_tw = 0
    for txt, _ in lines:
        if txt:
            (tw, _th), _ = cv2.getTextSize(txt, font, fs, thick)
            max_tw = max(max_tw, tw)

    box_w = max_tw + 2 * pad
    half_lh = lh // 2
    box_h = sum(half_lh if txt == "" else lh for txt, _ in lines) + 2 * pad
    bx1, by1 = pad, pad
    bx2, by2 = bx1 + box_w, by1 + box_h

    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (bx1, by1), (bx2, by2), BG, -1)
    cv2.addWeighted(overlay, 0.65, frame_bgr, 0.35, 0, frame_bgr)
    cv2.rectangle(frame_bgr, (bx1, by1), (bx2, by2), WHITE, 1)

    y = by1 + pad + lh
    for txt, color in lines:
        if txt == "":
            y += half_lh
            continue
        cv2.putText(frame_bgr, txt, (bx1 + pad, y), font, fs, color, thick, cv2.LINE_AA)
        y += lh

    return frame_bgr


def resize_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == target_h:
        return img
    new_w = int(w * target_h / h)
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)


def render_episode(
    ep_idx: int,
    ep_meta: dict,
    labels_entry: dict,
    root: Path,
    fps: float,
    output_dir: Path,
    target_h: int,
):
    total_ts = ep_meta["length"]
    n_frames = total_ts

    path_top = video_file_path(root, "observation.images.top", ep_meta["top"]["chunk"], ep_meta["top"]["file"])
    path_wrist = video_file_path(root, "observation.images.wrist", ep_meta["wrist"]["chunk"], ep_meta["wrist"]["file"])

    instruction = ep_meta.get("instruction", "")
    label_lookup = build_label_lookup(labels_entry.get("timestep_labels", []), total_ts)

    print(f"  Decoding top camera ({n_frames} frames) ...")
    tops = decode_episode_frames_pyav(path_top, ep_meta["top"]["from_ts"], fps, n_frames)
    print(f"  Decoding wrist camera ({n_frames} frames) ...")
    wrists = decode_episode_frames_pyav(path_wrist, ep_meta["wrist"]["from_ts"], fps, n_frames)

    out_path = output_dir / f"episode_{ep_idx}_annotated.mp4"
    container = av.open(str(out_path), mode="w")
    stream = container.add_stream("h264", rate=int(fps))

    first_top = resize_to_height(tops[0], target_h)
    first_wrist = resize_to_height(wrists[0], target_h)
    total_w = first_top.shape[1] + first_wrist.shape[1]
    stream.width = total_w
    stream.height = target_h
    stream.pix_fmt = "yuv420p"

    for t in tqdm(range(n_frames), desc=f"  Rendering ep {ep_idx}", leave=False):
        top_rgb = resize_to_height(tops[t], target_h)
        wrist_rgb = resize_to_height(wrists[t], target_h)

        top_bgr = cv2.cvtColor(top_rgb, cv2.COLOR_RGB2BGR)
        wrist_bgr = cv2.cvtColor(wrist_rgb, cv2.COLOR_RGB2BGR)

        labs = label_lookup[t] if t < len(label_lookup) else []
        top_bgr = annotate_frame(top_bgr, t, total_ts, labs, instruction)
        wrist_bgr = annotate_frame(wrist_bgr, t, total_ts, labs, instruction)

        combined_bgr = np.hstack([top_bgr, wrist_bgr])
        combined_rgb = cv2.cvtColor(combined_bgr, cv2.COLOR_BGR2RGB)

        av_frame = av.VideoFrame.from_ndarray(combined_rgb, format="rgb24")
        for packet in stream.encode(av_frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()

    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize SO101 substep labels on video for human validation")
    parser.add_argument("--labels_path", type=str, default="substep_labels_so101.json")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/lustre/fsw/portfolios/edgeai/users/chrislin/datasets/so101_raw",
        help="LeRobot dataset root with meta/, data/, videos/",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/lustre/fsw/portfolios/edgeai/users/chrislin/datasets/so101_viz",
    )
    parser.add_argument("--episodes", type=int, nargs="*", default=None, help="Episode indices (default: all)")
    parser.add_argument("--dataset_name", type=str, default="so101_poker_yellow")
    parser.add_argument("--target_height", type=int, default=480, help="Output frame height in pixels")
    args = parser.parse_args()

    with open(args.labels_path) as f:
        all_labels = json.load(f)

    root = Path(args.data_dir)
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Expected {info_path}. Run convert_lerobot_to_rlds.py first to download.")
    with open(info_path) as f:
        info = json.load(f)
    fps = float(info.get("fps", 30))

    ep_meta = load_episode_metadata(root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ep_indices = args.episodes if args.episodes is not None else sorted(ep_meta.keys())

    print("=" * 60)
    print("SO101 Substep Visualization")
    print("=" * 60)
    print(f"Labels: {args.labels_path}")
    print(f"Data:   {root}")
    print(f"Output: {output_dir}")
    print(f"Episodes: {ep_indices}")
    print(f"FPS: {fps}")
    print()

    for ep_idx in ep_indices:
        key = f"{args.dataset_name}_episode_{ep_idx}"
        if key not in all_labels:
            print(f"Episode {ep_idx}: no labels found (key={key}), skipping")
            continue
        if ep_idx not in ep_meta:
            print(f"Episode {ep_idx}: no video metadata, skipping")
            continue

        print(f"Episode {ep_idx}:")
        render_episode(ep_idx, ep_meta[ep_idx], all_labels[key], root, fps, output_dir, args.target_height)

    print()
    print("Done. Review the MP4s in:", output_dir)


if __name__ == "__main__":
    main()
