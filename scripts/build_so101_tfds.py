"""
build_so101_tfds.py

Builds a proper TFDS RLDS dataset from the flat SO101 TFRecord so that
``tfds.builder('so101_poker_yellow', data_dir=...)`` works with the
OpenVLA-OFT training pipeline.

Usage (inside the container):
    python scripts/build_so101_tfds.py \
        --tfrecord_path /path/to/so101_rlds/so101_poker_yellow.tfrecord \
        --output_dir /path/to/so101_rlds

After running, the output_dir will contain:
    so101_poker_yellow/1.0.0/
        so101_poker_yellow-train.tfrecord-00000-of-00001
        dataset_info.json
        features.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


class So101PokerYellowConfig(tfds.core.BuilderConfig):
    pass


class So101PokerYellow(tfds.core.GeneratorBasedBuilder):
    """SO101 dual pick-place RLDS dataset."""

    VERSION = tfds.core.Version("1.0.0")
    BUILDER_CONFIGS = [So101PokerYellowConfig(name="default", description="SO101 poker yellow task")]

    def __init__(self, tfrecord_path: str = "", **kwargs):
        self._tfrecord_path = tfrecord_path
        super().__init__(**kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description="SO101 dual pick-place: poker box then yellow box into white box",
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "image_primary": tfds.features.Image(
                                        shape=(1080, 1920, 3), dtype=np.uint8, encoding_format="jpeg",
                                    ),
                                    "image_wrist": tfds.features.Image(
                                        shape=(720, 1280, 3), dtype=np.uint8, encoding_format="jpeg",
                                    ),
                                    "proprio": tfds.features.Tensor(shape=(6,), dtype=np.float32),
                                }
                            ),
                            "action": tfds.features.Tensor(shape=(6,), dtype=np.float32),
                            "language_instruction": tfds.features.Text(),
                            "is_first": np.bool_,
                            "is_last": np.bool_,
                            "is_terminal": np.bool_,
                        }
                    ),
                }
            ),
        )

    def _split_generators(self, dl_manager):
        return {"train": self._generate_examples()}

    def _generate_examples(self):
        feature_desc = {
            "observation/image_primary": tf.io.FixedLenFeature([], tf.string),
            "observation/image_wrist": tf.io.FixedLenFeature([], tf.string),
            "observation/proprio": tf.io.VarLenFeature(tf.float32),
            "action": tf.io.VarLenFeature(tf.float32),
            "language_instruction": tf.io.FixedLenFeature([], tf.string),
            "is_first": tf.io.FixedLenFeature([], tf.int64),
            "is_last": tf.io.FixedLenFeature([], tf.int64),
            "is_terminal": tf.io.FixedLenFeature([], tf.int64),
        }

        ds = tf.data.TFRecordDataset(self._tfrecord_path)
        episodes: List[list] = []
        current: list = []

        for raw in ds:
            parsed = tf.io.parse_single_example(raw, feature_desc)
            current.append(parsed)
            if parsed["is_last"].numpy() == 1:
                episodes.append(current)
                current = []
        if current:
            episodes.append(current)

        for ep_idx, steps in enumerate(episodes):
            episode_steps = []
            for s in steps:
                img_primary_bytes = s["observation/image_primary"].numpy()
                img_wrist_bytes = s["observation/image_wrist"].numpy()
                proprio = tf.sparse.to_dense(s["observation/proprio"]).numpy()
                action = tf.sparse.to_dense(s["action"]).numpy()
                lang = s["language_instruction"].numpy().decode("utf-8")

                h_p, w_p = 1080, 1920
                h_w, w_w = 720, 1280
                try:
                    img_primary = np.frombuffer(img_primary_bytes, dtype=np.uint8).reshape(h_p, w_p, 3)
                except ValueError:
                    img_primary = np.zeros((h_p, w_p, 3), dtype=np.uint8)
                try:
                    img_wrist = np.frombuffer(img_wrist_bytes, dtype=np.uint8).reshape(h_w, w_w, 3)
                except ValueError:
                    img_wrist = np.zeros((h_w, w_w, 3), dtype=np.uint8)

                episode_steps.append(
                    {
                        "observation": {
                            "image_primary": img_primary,
                            "image_wrist": img_wrist,
                            "proprio": proprio.astype(np.float32),
                        },
                        "action": action.astype(np.float32),
                        "language_instruction": lang,
                        "is_first": bool(s["is_first"].numpy()),
                        "is_last": bool(s["is_last"].numpy()),
                        "is_terminal": bool(s["is_terminal"].numpy()),
                    }
                )

            yield ep_idx, {"steps": episode_steps}


def main():
    parser = argparse.ArgumentParser(description="Build TFDS RLDS dataset for SO101")
    parser.add_argument(
        "--tfrecord_path",
        type=str,
        default="/lustre/fsw/portfolios/edgeai/users/chrislin/datasets/so101_rlds/so101_poker_yellow.tfrecord",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/lustre/fsw/portfolios/edgeai/users/chrislin/datasets/so101_rlds",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building TFDS dataset from: {args.tfrecord_path}")
    print(f"Output: {output_dir}/so101_poker_yellow/1.0.0/")

    builder = So101PokerYellow(
        tfrecord_path=args.tfrecord_path,
        data_dir=str(output_dir),
    )
    builder.download_and_prepare()

    print("\nVerifying...")
    ds = builder.as_dataset(split="train")
    count = 0
    for ep in ds:
        n_steps = sum(1 for _ in ep["steps"])
        print(f"  Episode {count}: {n_steps} steps")
        count += 1
    print(f"Total episodes: {count}")
    print(f"\nDone. Use: tfds.builder('so101_poker_yellow', data_dir='{output_dir}')")


if __name__ == "__main__":
    main()
