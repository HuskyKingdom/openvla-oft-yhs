"""
compute_negative_collision_rate.py

Compute the collision rate in randomly sampled negative samples to demonstrate data sparsity.
Statistics on the proportion of negative sample pairs that satisfy both conditions:
1. Same task (Identity)
2. Similar state (State similarity > 0.95)
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

# Try to import CLIP - support multiple implementations
try:
    import clip as openai_clip
    HAS_OPENAI_CLIP = True
except ImportError:
    HAS_OPENAI_CLIP = False
    openai_clip = None

try:
    import open_clip
    HAS_OPEN_CLIP = True
except ImportError:
    HAS_OPEN_CLIP = False
    open_clip = None

try:
    from transformers import CLIPModel, CLIPProcessor
    HAS_TRANSFORMERS_CLIP = True
except ImportError:
    HAS_TRANSFORMERS_CLIP = False
    CLIPModel = None
    CLIPProcessor = None

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from torch.nn.utils.rnn import pad_sequence
from prismatic.vla.constants import IGNORE_INDEX
from experiments.robot.openvla_utils import (
    check_model_logic_mismatch,
    model_is_on_hf_hub,
    update_auto_map,
)
from huggingface_hub import snapshot_download


@dataclass
class TaskAwareRLDSBatchTransform(RLDSBatchTransform):
    """Extended RLDSBatchTransform that preserves original task descriptions"""
    
    def __call__(self, rlds_batch: Dict) -> Dict:
        """Convert RLDS batch and preserve original task description"""
        # Call parent class method
        result = super().__call__(rlds_batch)
        
        # Extract and save original task description
        if "task" in rlds_batch and "language_instruction" in rlds_batch["task"]:
            language_instruction = rlds_batch["task"]["language_instruction"]
            if isinstance(language_instruction, bytes):
                language_instruction = language_instruction.decode('utf-8')
            result["language_instruction"] = language_instruction.lower().strip()
        else:
            result["language_instruction"] = ""
        
        return result


def load_clip_model(device: str = "cuda"):
    """
    Load CLIP model for extracting image features.
    Supports multiple CLIP implementations: OpenAI CLIP, Open CLIP, Transformers CLIP
    """
    # Try OpenAI CLIP first
    if HAS_OPENAI_CLIP:
        try:
            model, preprocess = openai_clip.load("ViT-B/32", device=device)
            model.eval()
            return model, preprocess, "openai_clip"
        except Exception as e:
            print(f"Failed to load OpenAI CLIP: {e}")
            try:
                model, preprocess = openai_clip.load("ViT-B/32", device="cpu")
                model.eval()
                return model, preprocess, "openai_clip"
            except Exception as e2:
                print(f"Failed to load OpenAI CLIP (CPU): {e2}")
    
    # Try Open CLIP
    if HAS_OPEN_CLIP:
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai", device=device
            )
            model.eval()
            return model, preprocess, "open_clip"
        except Exception as e:
            print(f"Failed to load Open CLIP: {e}")
            try:
                model, _, preprocess = open_clip.create_model_and_transforms(
                    "ViT-B-32", pretrained="openai", device="cpu"
                )
                model.eval()
                return model, preprocess, "open_clip"
            except Exception as e2:
                print(f"Failed to load Open CLIP (CPU): {e2}")
    
    # Try Transformers CLIP
    if HAS_TRANSFORMERS_CLIP:
        try:
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            model.eval()
            return model, processor, "transformers_clip"
        except Exception as e:
            print(f"Failed to load Transformers CLIP: {e}")
            try:
                model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cpu")
                processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                model.eval()
                return model, processor, "transformers_clip"
            except Exception as e2:
                print(f"Failed to load Transformers CLIP (CPU): {e2}")
    
    raise RuntimeError(
        "Unable to load CLIP model! Please install one of the following:\n"
        "  - OpenAI CLIP: pip install git+https://github.com/openai/CLIP.git\n"
        "  - Open CLIP: pip install open-clip-torch\n"
        "  - Transformers CLIP: pip install transformers (already included)"
    )


def extract_image_features(
    pixel_values: torch.Tensor,
    clip_model,
    clip_processor,
    clip_type: str,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Extract image features using CLIP (batch processing for efficiency).
    Supports multiple CLIP implementations.
    
    Args:
        pixel_values: [B, C, H, W] image tensor (ImageNet normalized)
                     May contain multiple images stacked in channel dimension (e.g., primary + wrist camera)
        clip_model: CLIP model
        clip_processor: CLIP processor/preprocessor
        clip_type: CLIP type ("openai_clip", "open_clip", "transformers_clip")
        device: device
    
    Returns:
        features: [B, D] image feature vectors
    """
    B, C, H, W = pixel_values.shape
    
    # Handle multi-image input: only use primary camera image (first 3 channels)
    # If C > 3, may be multiple images stacked or fused backbone (6 channels)
    if C != 3:
        if C > 3:
            # Only take first 3 channels (primary camera image)
            pixel_values = pixel_values[:, :3, :, :]
            C = 3
        else:
            raise ValueError(f"Unsupported image channel count: {C}, expected 3 channels (RGB)")
    
    with torch.no_grad():
        if clip_type == "openai_clip" or clip_type == "open_clip":
            # OpenAI CLIP or Open CLIP
            # Denormalize (assuming ImageNet mean and std)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            
            # Denormalize
            pixel_values_denorm = pixel_values * std + mean
            pixel_values_denorm = torch.clamp(pixel_values_denorm, 0, 1)
            
            # Resize to CLIP input size (224x224)
            if H != 224 or W != 224:
                pixel_values_denorm = F.interpolate(
                    pixel_values_denorm,
                    size=(224, 224),
                    mode='bilinear',
                    align_corners=False
                )
            
            # CLIP expected input format
            if clip_type == "openai_clip":
                # OpenAI CLIP: use specific mean and std
                clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
                clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)
                pixel_values_clip = (pixel_values_denorm - clip_mean) / clip_std
                image_features = clip_model.encode_image(pixel_values_clip)
            else:
                # Open CLIP: use processor for preprocessing
                # Convert tensor to PIL Images
                images = []
                for i in range(B):
                    img_array = (pixel_values_denorm[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    img_pil = Image.fromarray(img_array)
                    images.append(img_pil)
                
                # Use processor for preprocessing (Open CLIP's processor is a transform)
                img_tensors = torch.stack([clip_processor(img) for img in images]).to(device)
                # Open CLIP's encode_image supports normalize parameter
                image_features = clip_model.encode_image(img_tensors, normalize=True)
            
            # Normalize features (if not already normalized)
            if clip_type == "openai_clip":
                image_features = F.normalize(image_features, p=2, dim=-1)
        
        elif clip_type == "transformers_clip":
            # Transformers CLIP
            # Convert tensor to PIL Images
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            pixel_values_denorm = pixel_values * std + mean
            pixel_values_denorm = torch.clamp(pixel_values_denorm, 0, 1)
            
            images = []
            for i in range(B):
                img_array = (pixel_values_denorm[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                img_pil = Image.fromarray(img_array)
                images.append(img_pil)
            
            # Use processor for preprocessing
            inputs = clip_processor(images=images, return_tensors="pt").to(device)
            image_features = clip_model.get_image_features(**inputs)
            image_features = F.normalize(image_features, p=2, dim=-1)
        
        else:
            raise ValueError(f"Unsupported CLIP type: {clip_type}")
    
    return image_features


def compute_cosine_similarity_matrix(features: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine similarity matrix.
    
    Args:
        features: [N, D] feature matrix
    
    Returns:
        similarity_matrix: [N, N] cosine similarity matrix
    """
    # Features should already be normalized
    similarity_matrix = torch.mm(features, features.t())
    return similarity_matrix


def count_collisions(
    task_descriptions: List[str],
    similarity_matrix: torch.Tensor,
    similarity_threshold: float = 0.95
) -> Tuple[int, int]:
    """
    Count the number of colliding negative sample pairs.
    
    Args:
        task_descriptions: List of task descriptions
        similarity_matrix: State similarity matrix [N, N]
        similarity_threshold: State similarity threshold
    
    Returns:
        (collision_count, total_pairs): Number of collisions and total pairs
    """
    N = len(task_descriptions)
    collision_count = 0
    total_pairs = 0
    
    similarity_matrix_np = similarity_matrix.cpu().numpy()
    
    for i in range(N):
        for j in range(i + 1, N):  # Only traverse upper triangle to avoid duplicates
            total_pairs += 1
            
            # Condition A: Same task
            task_match = (task_descriptions[i] == task_descriptions[j])
            
            # Condition B: Similar state (similarity > threshold)
            state_similar = (similarity_matrix_np[i, j] > similarity_threshold)
            
            # Both conditions must be satisfied for a collision
            if task_match and state_similar:
                collision_count += 1
    
    return collision_count, total_pairs


def compute_collision_rate(
    data_root_dir: Path,
    dataset_name: str,
    vla_path: str,
    batch_size: int = 8,
    num_batches: int = 100,
    similarity_threshold: float = 0.95,
    device: str = "cuda"
) -> Dict:
    """
    Compute negative sample collision rate.
    
    Args:
        data_root_dir: Data root directory
        dataset_name: Dataset name
        vla_path: VLA model path
        batch_size: Batch size
        num_batches: Number of batches to process
        similarity_threshold: State similarity threshold
        device: Device
    
    Returns:
        Statistics result dictionary
    """
    print(f"Loading CLIP model...")
    clip_model, clip_processor, clip_type = load_clip_model(device)
    print(f"Successfully loaded CLIP model (type: {clip_type})")
    
    print(f"Loading VLA processor...")
    # Load VLA processor
    if model_is_on_hf_hub(vla_path):
        vla_download_path = snapshot_download(repo_id=vla_path)
        vla_path = vla_download_path
    else:
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    
    if os.getpid() == 0 or True:  # Main process
        update_auto_map(vla_path)
        check_model_logic_mismatch(vla_path)
    
    processor = AutoProcessor.from_pretrained(vla_path, trust_remote_code=True)
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    
    print(f"Loading dataset: {dataset_name}")
    # Create dataset
    batch_transform = TaskAwareRLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=False,
        use_proprio=False,
    )
    
    # Load VLA config to get image size
    vla_config = AutoConfig.from_pretrained(vla_path, trust_remote_code=True)
    image_size = tuple(vla_config.image_sizes) if hasattr(vla_config, 'image_sizes') else (224, 224)
    
    train_dataset = RLDSDataset(
        data_root_dir,
        dataset_name,
        batch_transform,
        resize_resolution=image_size,
        shuffle_buffer_size=10000,
        image_aug=False,  # No data augmentation to ensure objectivity
    )
    
    # Create custom collator to preserve language_instruction field
    class CustomCollator(PaddedCollatorForActionPrediction):
        def __call__(self, instances):
            # Extract language_instruction if present
            language_instructions = None
            if "language_instruction" in instances[0]:
                language_instructions = [instance["language_instruction"] for instance in instances]
            
            # Call parent class method
            result = super().__call__(instances)
            
            # Add language_instruction field
            if language_instructions is not None:
                result["language_instruction"] = language_instructions
            
            return result
    
    collator = CustomCollator(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right"
    )
    
    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,
    )
    
    print(f"Starting statistics, will process {num_batches} batches...")
    
    total_collisions = 0
    total_pairs = 0
    batch_collision_rates = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, total=num_batches)):
            if batch_idx >= num_batches:
                break
            
            # Extract task descriptions
            if "language_instruction" in batch:
                task_descriptions = batch["language_instruction"]
                # Ensure it's a list format
                if isinstance(task_descriptions, str):
                    task_descriptions = [task_descriptions]
                elif not isinstance(task_descriptions, list):
                    task_descriptions = list(task_descriptions)
            else:
                # If not saved, try to decode from input_ids
                # This requires knowing the prompt structure, which is complex
                # For simplicity, we assume batch_transform has already saved it
                print("Warning: batch does not contain language_instruction field, skipping this batch")
                continue
            
            # Extract image features
            pixel_values = batch["pixel_values"].to(device)
            image_features = extract_image_features(
                pixel_values, clip_model, clip_processor, clip_type, device
            )
            
            # Compute similarity matrix
            similarity_matrix = compute_cosine_similarity_matrix(image_features)
            
            # Count collisions
            collision_count, pairs_in_batch = count_collisions(
                task_descriptions,
                similarity_matrix,
                similarity_threshold
            )
            
            total_collisions += collision_count
            total_pairs += pairs_in_batch
            
            if pairs_in_batch > 0:
                batch_rate = (collision_count / pairs_in_batch) * 100
                batch_collision_rates.append(batch_rate)
            
            if (batch_idx + 1) % 10 == 0:
                current_rate = (total_collisions / total_pairs * 100) if total_pairs > 0 else 0
                print(f"Processed {batch_idx + 1} batches, current collision rate: {current_rate:.4f}%")
    
    # Compute final statistics
    final_collision_rate = (total_collisions / total_pairs * 100) if total_pairs > 0 else 0
    
    results = {
        "total_collisions": total_collisions,
        "total_pairs": total_pairs,
        "collision_rate_percent": final_collision_rate,
        "mean_batch_rate": np.mean(batch_collision_rates) if batch_collision_rates else 0,
        "std_batch_rate": np.std(batch_collision_rates) if batch_collision_rates else 0,
        "min_batch_rate": np.min(batch_collision_rates) if batch_collision_rates else 0,
        "max_batch_rate": np.max(batch_collision_rates) if batch_collision_rates else 0,
        "similarity_threshold": similarity_threshold,
        "num_batches": batch_idx + 1,
    }
    
    return results


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute negative sample collision rate")
    parser.add_argument("--data_root_dir", type=str, default="datasets/rlds",
                       help="Data root directory")
    parser.add_argument("--dataset_name", type=str, default="aloha_scoop_x_into_bowl",
                       help="Dataset name")
    parser.add_argument("--vla_path", type=str, default="openvla/openvla-7b",
                       help="VLA model path")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size")
    parser.add_argument("--num_batches", type=int, default=100,
                       help="Number of batches to process")
    parser.add_argument("--similarity_threshold", type=float, default=0.95,
                       help="State similarity threshold")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Negative Sample Collision Rate Statistics Experiment")
    print("=" * 80)
    print(f"Dataset: {args.dataset_name}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of batches: {args.num_batches}")
    print(f"Similarity threshold: {args.similarity_threshold}")
    print("=" * 80)
    
    results = compute_collision_rate(
        data_root_dir=Path(args.data_root_dir),
        dataset_name=args.dataset_name,
        vla_path=args.vla_path,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        similarity_threshold=args.similarity_threshold,
        device=args.device
    )
    
    print("\n" + "=" * 80)
    print("Statistics Results")
    print("=" * 80)
    print(f"Total collision pairs: {results['total_collisions']}")
    print(f"Total negative sample pairs: {results['total_pairs']}")
    print(f"Collision rate: {results['collision_rate_percent']:.4f}%")
    print(f"Mean batch collision rate: {results['mean_batch_rate']:.4f}%")
    print(f"Batch collision rate std: {results['std_batch_rate']:.4f}%")
    print(f"Min batch collision rate: {results['min_batch_rate']:.4f}%")
    print(f"Max batch collision rate: {results['max_batch_rate']:.4f}%")
    print("=" * 80)
    print(f"\nConclusion: Only {results['collision_rate_percent']:.4f}% of randomly sampled negative samples are collisions (False Negative)")
    print("This demonstrates data sparsity: randomly sampled negative samples are almost always safe.")
    print("=" * 80)


if __name__ == "__main__":
    main()

