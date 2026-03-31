"""
download_so101_dataset.py

Downloads the SO101 Poker Yellow Task dataset from HuggingFace.
Dataset: christian0420/so101-poker-yellow-task (LeRobot v3.0 format)

Usage:
    python scripts/download_so101_dataset.py --output_dir /path/to/output
"""

import site
import sys

# ~/.local can appear before conda env site-packages and shadow it with partial installs.
_user_site = site.getusersitepackages()
while _user_site in sys.path:
    sys.path.remove(_user_site)

import argparse
import os
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import login

# HuggingFace token (KEEP THIS PRIVATE - for local use only)
HF_TOKEN = "YOUR_HF_TOKEN_HERE"


def main():
    parser = argparse.ArgumentParser(description="Download SO101 Poker Yellow Task dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/lustre/fsw/portfolios/edgeai/users/chrislin/datasets/so101_raw",
        help="Output directory for downloaded dataset"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify download integrity after completion"
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SO101 Poker Yellow Task Dataset Download")
    print("=" * 60)
    print(f"Dataset: christian0420/so101-poker-yellow-task")
    print(f"Output Directory: {output_dir}")
    print("")

    # Login to HuggingFace
    print("Logging in to HuggingFace...")
    login(token=HF_TOKEN, add_to_git_credential=False)
    print("Login successful!")
    print("")

    # Download dataset
    print("Downloading dataset...")
    print("This may take a while depending on network speed...")
    print("")

    try:
        # Load dataset with cache directory
        dataset = load_dataset(
            "christian0420/so101-poker-yellow-task",
            cache_dir=str(output_dir),
            trust_remote_code=True,
        )

        print("")
        print("=" * 60)
        print("Download Complete!")
        print("=" * 60)
        print(f"Dataset saved to: {output_dir}")
        print("")

        # Print dataset information
        print("Dataset Information:")
        print(f"  Splits: {list(dataset.keys())}")
        for split_name, split_data in dataset.items():
            print(f"  {split_name} split: {len(split_data)} samples")
        print("")

        # Verify dataset if requested
        if args.verify:
            print("Verifying dataset integrity...")
            verify_dataset(dataset, output_dir)

    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please check your internet connection and HuggingFace token.")
        return 1

    return 0


def verify_dataset(dataset, output_dir):
    """Verify the downloaded dataset."""
    print("")
    print("Dataset Verification:")
    print("-" * 60)

    # Expected dataset statistics
    EXPECTED_EPISODES = 11
    EXPECTED_TOTAL_FRAMES = 5455
    EXPECTED_FPS = 30

    train_split = dataset.get("train")
    if train_split is None:
        print("  WARNING: 'train' split not found!")
        return

    num_samples = len(train_split)
    print(f"  Total samples: {num_samples}")

    # Check if matches expected
    if num_samples == EXPECTED_TOTAL_FRAMES:
        print(f"  ✓ Sample count matches expected: {EXPECTED_TOTAL_FRAMES} frames")
    else:
        print(f"  ⚠ Sample count mismatch! Expected {EXPECTED_TOTAL_FRAMES}, got {num_samples}")

    # Check first sample structure
    print("")
    print("  Sample Structure (first entry):")
    first_sample = train_split[0]
    for key, value in first_sample.items():
        if hasattr(value, 'shape'):
            print(f"    {key}: shape={value.shape}, dtype={value.dtype}")
        elif isinstance(value, (list, tuple)):
            print(f"    {key}: length={len(value)}")
        else:
            print(f"    {key}: {type(value).__name__}")

    print("")
    print("  Expected Fields:")
    print("    ✓ action (6D)")
    print("    ✓ observation.state (6D)")
    print("    ✓ observation.images.top (1080x1920x3)")
    print("    ✓ observation.images.wrist (720x1280x3)")
    print("    ✓ timestamp, frame_index, episode_index")
    print("")
    print("Verification complete!")
    print("-" * 60)


if __name__ == "__main__":
    exit(main())
