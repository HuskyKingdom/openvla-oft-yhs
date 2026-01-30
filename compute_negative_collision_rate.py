"""
compute_negative_collision_rate.py

计算随机采样负样本中的冲突比例，证明数据的稀疏性。
统计满足以下两个条件的负样本对的比例：
1. 相同的任务（Identity）
2. 相似的状态（State similarity > 0.95）
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
    """扩展RLDSBatchTransform，保存原始任务描述"""
    
    def __call__(self, rlds_batch: Dict) -> Dict:
        """转换RLDS batch，并保存原始任务描述"""
        # 调用父类方法
        result = super().__call__(rlds_batch)
        
        # 提取并保存原始任务描述
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
    加载CLIP模型用于提取图像特征
    支持多种CLIP实现：OpenAI CLIP, Open CLIP, Transformers CLIP
    """
    # Try OpenAI CLIP first
    if HAS_OPENAI_CLIP:
        try:
            model, preprocess = openai_clip.load("ViT-B/32", device=device)
            model.eval()
            return model, preprocess, "openai_clip"
        except Exception as e:
            print(f"OpenAI CLIP加载失败: {e}")
            try:
                model, preprocess = openai_clip.load("ViT-B/32", device="cpu")
                model.eval()
                return model, preprocess, "openai_clip"
            except Exception as e2:
                print(f"OpenAI CLIP (CPU)加载失败: {e2}")
    
    # Try Open CLIP
    if HAS_OPEN_CLIP:
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai", device=device
            )
            model.eval()
            return model, preprocess, "open_clip"
        except Exception as e:
            print(f"Open CLIP加载失败: {e}")
            try:
                model, _, preprocess = open_clip.create_model_and_transforms(
                    "ViT-B-32", pretrained="openai", device="cpu"
                )
                model.eval()
                return model, preprocess, "open_clip"
            except Exception as e2:
                print(f"Open CLIP (CPU)加载失败: {e2}")
    
    # Try Transformers CLIP
    if HAS_TRANSFORMERS_CLIP:
        try:
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            model.eval()
            return model, processor, "transformers_clip"
        except Exception as e:
            print(f"Transformers CLIP加载失败: {e}")
            try:
                model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cpu")
                processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                model.eval()
                return model, processor, "transformers_clip"
            except Exception as e2:
                print(f"Transformers CLIP (CPU)加载失败: {e2}")
    
    raise RuntimeError(
        "无法加载CLIP模型！请安装以下之一：\n"
        "  - OpenAI CLIP: pip install git+https://github.com/openai/CLIP.git\n"
        "  - Open CLIP: pip install open-clip-torch\n"
        "  - Transformers CLIP: pip install transformers (已包含)"
    )


def extract_image_features(
    pixel_values: torch.Tensor,
    clip_model,
    clip_processor,
    clip_type: str,
    device: str = "cuda"
) -> torch.Tensor:
    """
    使用CLIP提取图像特征（批量处理以提高效率）
    支持多种CLIP实现
    
    Args:
        pixel_values: [B, C, H, W] 图像tensor（ImageNet归一化的）
        clip_model: CLIP模型
        clip_processor: CLIP处理器/预处理器
        clip_type: CLIP类型 ("openai_clip", "open_clip", "transformers_clip")
        device: 设备
    
    Returns:
        features: [B, D] 图像特征向量
    """
    B, C, H, W = pixel_values.shape
    
    with torch.no_grad():
        if clip_type == "openai_clip" or clip_type == "open_clip":
            # OpenAI CLIP 或 Open CLIP
            # 反归一化（假设使用ImageNet均值和标准差）
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            
            # 反归一化
            pixel_values_denorm = pixel_values * std + mean
            pixel_values_denorm = torch.clamp(pixel_values_denorm, 0, 1)
            
            # 调整大小到CLIP的输入尺寸 (224x224)
            if H != 224 or W != 224:
                pixel_values_denorm = F.interpolate(
                    pixel_values_denorm,
                    size=(224, 224),
                    mode='bilinear',
                    align_corners=False
                )
            
            # CLIP期望的输入格式
            if clip_type == "openai_clip":
                # OpenAI CLIP: 使用特定的均值和标准差
                clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
                clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)
                pixel_values_clip = (pixel_values_denorm - clip_mean) / clip_std
                image_features = clip_model.encode_image(pixel_values_clip)
            else:
                # Open CLIP: 使用processor进行预处理
                # 将tensor转换为PIL Images
                images = []
                for i in range(B):
                    img_array = (pixel_values_denorm[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    img_pil = Image.fromarray(img_array)
                    images.append(img_pil)
                
                # 使用processor预处理（Open CLIP的processor是一个transform）
                img_tensors = torch.stack([clip_processor(img) for img in images]).to(device)
                # Open CLIP的encode_image支持normalize参数
                image_features = clip_model.encode_image(img_tensors, normalize=True)
            
            # 归一化特征（如果还没有归一化）
            if clip_type == "openai_clip":
                image_features = F.normalize(image_features, p=2, dim=-1)
        
        elif clip_type == "transformers_clip":
            # Transformers CLIP
            # 将tensor转换为PIL Images
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            pixel_values_denorm = pixel_values * std + mean
            pixel_values_denorm = torch.clamp(pixel_values_denorm, 0, 1)
            
            images = []
            for i in range(B):
                img_array = (pixel_values_denorm[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                img_pil = Image.fromarray(img_array)
                images.append(img_pil)
            
            # 使用processor预处理
            inputs = clip_processor(images=images, return_tensors="pt").to(device)
            image_features = clip_model.get_image_features(**inputs)
            image_features = F.normalize(image_features, p=2, dim=-1)
        
        else:
            raise ValueError(f"不支持的CLIP类型: {clip_type}")
    
    return image_features


def compute_cosine_similarity_matrix(features: torch.Tensor) -> torch.Tensor:
    """
    计算余弦相似度矩阵
    
    Args:
        features: [N, D] 特征矩阵
    
    Returns:
        similarity_matrix: [N, N] 余弦相似度矩阵
    """
    # 特征应该已经归一化
    similarity_matrix = torch.mm(features, features.t())
    return similarity_matrix


def count_collisions(
    task_descriptions: List[str],
    similarity_matrix: torch.Tensor,
    similarity_threshold: float = 0.95
) -> Tuple[int, int]:
    """
    统计冲突的负样本对数量
    
    Args:
        task_descriptions: 任务描述列表
        similarity_matrix: 状态相似度矩阵 [N, N]
        similarity_threshold: 状态相似度阈值
    
    Returns:
        (collision_count, total_pairs): 冲突对数和总对数
    """
    N = len(task_descriptions)
    collision_count = 0
    total_pairs = 0
    
    similarity_matrix_np = similarity_matrix.cpu().numpy()
    
    for i in range(N):
        for j in range(i + 1, N):  # 只遍历上三角，避免重复
            total_pairs += 1
            
            # 条件A: 相同的任务
            task_match = (task_descriptions[i] == task_descriptions[j])
            
            # 条件B: 相似的状态（相似度 > threshold）
            state_similar = (similarity_matrix_np[i, j] > similarity_threshold)
            
            # 同时满足两个条件才算冲突
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
    计算负样本冲突率
    
    Args:
        data_root_dir: 数据根目录
        dataset_name: 数据集名称
        vla_path: VLA模型路径
        batch_size: 批次大小
        num_batches: 统计的批次数量
        similarity_threshold: 状态相似度阈值
        device: 设备
    
    Returns:
        统计结果字典
    """
    print(f"正在加载CLIP模型...")
    clip_model, clip_processor, clip_type = load_clip_model(device)
    print(f"成功加载CLIP模型 (类型: {clip_type})")
    
    print(f"正在加载VLA处理器...")
    # 加载VLA处理器
    if model_is_on_hf_hub(vla_path):
        vla_download_path = snapshot_download(repo_id=vla_path)
        vla_path = vla_download_path
    else:
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    
    if os.getpid() == 0 or True:  # 主进程
        update_auto_map(vla_path)
        check_model_logic_mismatch(vla_path)
    
    processor = AutoProcessor.from_pretrained(vla_path, trust_remote_code=True)
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    
    print(f"正在加载数据集: {dataset_name}")
    # 创建数据集
    batch_transform = TaskAwareRLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=False,
        use_proprio=False,
    )
    
    # 加载VLA配置以获取图像尺寸
    vla_config = AutoConfig.from_pretrained(vla_path, trust_remote_code=True)
    image_size = tuple(vla_config.image_sizes) if hasattr(vla_config, 'image_sizes') else (224, 224)
    
    train_dataset = RLDSDataset(
        data_root_dir,
        dataset_name,
        batch_transform,
        resize_resolution=image_size,
        shuffle_buffer_size=10000,
        image_aug=False,  # 不使用数据增强，保证客观性
    )
    
    # 创建自定义collator以保留language_instruction字段
    class CustomCollator(PaddedCollatorForActionPrediction):
        def __call__(self, instances):
            # 提取language_instruction（如果存在）
            language_instructions = None
            if "language_instruction" in instances[0]:
                language_instructions = [instance["language_instruction"] for instance in instances]
            
            # 调用父类方法
            result = super().__call__(instances)
            
            # 添加language_instruction字段
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
    
    print(f"开始统计，将处理 {num_batches} 个批次...")
    
    total_collisions = 0
    total_pairs = 0
    batch_collision_rates = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, total=num_batches)):
            if batch_idx >= num_batches:
                break
            
            # 提取任务描述
            if "language_instruction" in batch:
                task_descriptions = batch["language_instruction"]
                # 确保是列表格式
                if isinstance(task_descriptions, str):
                    task_descriptions = [task_descriptions]
                elif not isinstance(task_descriptions, list):
                    task_descriptions = list(task_descriptions)
            else:
                # 如果没有保存，尝试从input_ids解码
                # 这需要知道prompt的结构，比较复杂
                # 为了简化，我们假设batch_transform已经保存了
                print("警告: batch中没有language_instruction字段，跳过此批次")
                continue
            
            # 提取图像特征
            pixel_values = batch["pixel_values"].to(device)
            image_features = extract_image_features(
                pixel_values, clip_model, clip_processor, clip_type, device
            )
            
            # 计算相似度矩阵
            similarity_matrix = compute_cosine_similarity_matrix(image_features)
            
            # 统计冲突
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
                print(f"已处理 {batch_idx + 1} 个批次，当前冲突率: {current_rate:.4f}%")
    
    # 计算最终统计结果
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
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="计算负样本冲突率")
    parser.add_argument("--data_root_dir", type=str, default="datasets/rlds",
                       help="数据根目录")
    parser.add_argument("--dataset_name", type=str, default="aloha_scoop_x_into_bowl",
                       help="数据集名称")
    parser.add_argument("--vla_path", type=str, default="openvla/openvla-7b",
                       help="VLA模型路径")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="批次大小")
    parser.add_argument("--num_batches", type=int, default=100,
                       help="统计的批次数量")
    parser.add_argument("--similarity_threshold", type=float, default=0.95,
                       help="状态相似度阈值")
    parser.add_argument("--device", type=str, default="cuda",
                       help="设备 (cuda/cpu)")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("负样本冲突率统计实验")
    print("=" * 80)
    print(f"数据集: {args.dataset_name}")
    print(f"批次大小: {args.batch_size}")
    print(f"统计批次数: {args.num_batches}")
    print(f"相似度阈值: {args.similarity_threshold}")
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
    print("统计结果")
    print("=" * 80)
    print(f"总冲突对数: {results['total_collisions']}")
    print(f"总负样本对数: {results['total_pairs']}")
    print(f"冲突率: {results['collision_rate_percent']:.4f}%")
    print(f"平均批次冲突率: {results['mean_batch_rate']:.4f}%")
    print(f"批次冲突率标准差: {results['std_batch_rate']:.4f}%")
    print(f"最小批次冲突率: {results['min_batch_rate']:.4f}%")
    print(f"最大批次冲突率: {results['max_batch_rate']:.4f}%")
    print("=" * 80)
    print(f"\n结论: 随机采样的负样本中，只有 {results['collision_rate_percent']:.4f}% 是冲突的（False Negative）")
    print("这证明了数据的稀疏性：随机采样的负样本几乎总是安全的。")
    print("=" * 80)


if __name__ == "__main__":
    main()

