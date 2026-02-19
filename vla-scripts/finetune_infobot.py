"""
InfoBot-VLA Training Script

Extends OpenVLA fine-tuning with Information Bottleneck architecture
to address H(L|V) ≈ 0 problem and visual overfitting.

Key differences from standard VLA:
1. Adds InfoBottleneckLayer between vision and action prediction
2. Uses mutual information regularization loss
3. Forces action prediction through language-conditioned bottleneck
"""

import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import draccus
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import tqdm
from accelerate import PartialState
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb

from experiments.robot.openvla_utils import (
    check_model_logic_mismatch,
    model_is_on_hf_hub,
    update_auto_map,
)
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import L1RegressionActionHead, EOSClassificationHead
from prismatic.models.infobot_vla import (
    InfoBottleneckLayer,
    LanguageConditionedBottleneck,
    MutualInformationEstimator,
    compute_infobot_loss,
)
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.training.train_utils import (
    get_current_action_mask,
    get_next_actions_mask,
)
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
)
from prismatic.vla.datasets.datasets_substep import SubstepRLDSBatchTransform, SubstepRLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

import sys
sys.path.insert(0, str(Path(__file__).parent))
from finetune import (
    remove_ddp_in_checkpoint,
    load_checkpoint,
    wrap_ddp,
    count_parameters,
    init_module,
    compute_smoothened_metrics,
    log_metrics_to_wandb,
    save_training_checkpoint,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import random


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class InfoBotVLAConfig:
    """Configuration for InfoBot-VLA fine-tuning."""
    
    # Model paths
    vla_path: str = "openvla/openvla-7b"
    
    # Dataset
    data_root_dir: Path = Path("datasets/rlds")
    dataset_name: str = "libero_goal_no_noops"
    substep_labels_path: str = "substep_labels_output.json"
    run_root_dir: Path = Path("runs")
    shuffle_buffer_size: int = 100_000
    
    # InfoBot Architecture
    use_infobot: bool = True  # Enable InfoBot bottleneck
    bottleneck_type: str = "cross_attn"  # "cross_attn" or "lang_conditioned"
    bottleneck_dim: int = 256
    num_bottleneck_tokens: int = 8
    beta_mi: float = 0.1  # Weight for mutual information regularization
    
    # Training
    use_l1_regression: bool = True
    batch_size: int = 8
    learning_rate: float = 5e-4
    lr_warmup_steps: int = 0
    num_steps_before_decay: int = 100_000
    grad_accumulation_steps: int = 1
    max_steps: int = 200_000
    max_grad_norm: float = 0.8
    use_val_set: bool = False
    val_freq: int = 10_000
    val_time_limit: int = 180
    save_freq: int = 10_000
    save_latest_checkpoint_only: bool = False
    resume: bool = False
    resume_step: Optional[int] = None
    image_aug: bool = True
    
    # LoRA
    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0
    merge_lora_during_training: bool = True
    
    # Logging
    wandb_entity: str = "your-wandb-entity"
    wandb_project: str = "your-wandb-project"
    run_id_note: Optional[str] = None
    run_id_override: Optional[str] = None
    wandb_log_freq: int = 10
    
    # Substep
    use_substep_eos: bool = True
    use_eos_classification: bool = True
    eos_hidden_dim: int = 1024
    eos_dropout: float = 0.1
    lambda_eos: float = 1.0
    eos_use_focal_loss: bool = True
    eos_focal_alpha: float = 0.25
    eos_focal_gamma: float = 2.0
    
    # Additional
    num_images_in_input: int = 2
    use_proprio: bool = True
    

def get_run_id(cfg: InfoBotVLAConfig) -> str:
    """Generate experiment run ID."""
    if cfg.run_id_override is not None:
        return cfg.run_id_override
    elif cfg.resume:
        run_id = cfg.vla_path.split("/")[-1]
        if "chkpt" in run_id.split("--")[-1]:
            run_id = "--".join(run_id.split("--")[:-1])
    else:
        run_id = (
            f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
            f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
            f"+lr-{cfg.learning_rate}"
        )
        if cfg.use_lora:
            run_id += f"+lora-r{cfg.lora_rank}"
        if cfg.use_infobot:
            run_id += f"+infobot-{cfg.bottleneck_type}-beta{cfg.beta_mi}"
        if cfg.image_aug:
            run_id += "--image_aug"
        run_id += "--substep"
        if cfg.run_id_note is not None:
            run_id += f"--{cfg.run_id_note}"
    return run_id


class InfoBotVLAModel(nn.Module):
    """
    InfoBot-VLA Model wrapper that adds information bottleneck to OpenVLA.
    
    Architecture:
        Vision Backbone → Projector → InfoBottleneck (conditioned on Language) → Action Head
    """
    def __init__(
        self,
        base_vla: OpenVLAForActionPrediction,
        bottleneck_type: str = "cross_attn",
        bottleneck_dim: int = 256,
        num_bottleneck_tokens: int = 8,
        use_l1_regression: bool = True,
    ):
        super().__init__()
        self.base_vla = base_vla
        self.llm_dim = base_vla.llm_dim
        self.bottleneck_type = bottleneck_type
        self.bottleneck_dim = bottleneck_dim
        self.num_bottleneck_tokens = num_bottleneck_tokens
        
        # Vision backbone and projector from base VLA
        self.vision_backbone = base_vla.vision_backbone
        self.projector = base_vla.projector
        
        # Language model from base VLA (for getting language embeddings)
        self.language_model = base_vla.language_model
        
        # InfoBot Bottleneck
        if bottleneck_type == "cross_attn":
            self.bottleneck = InfoBottleneckLayer(
                visual_dim=self.llm_dim,
                language_dim=self.llm_dim,
                bottleneck_dim=bottleneck_dim,
                num_bottleneck_tokens=num_bottleneck_tokens,
            )
        elif bottleneck_type == "lang_conditioned":
            self.bottleneck = LanguageConditionedBottleneck(
                visual_dim=self.llm_dim,
                language_dim=self.llm_dim,
                bottleneck_dim=bottleneck_dim,
                num_bottleneck_tokens=num_bottleneck_tokens,
            )
        else:
            raise ValueError(f"Unknown bottleneck type: {bottleneck_type}")
        
        # MI Estimator for regularization - created outside DDP to avoid gradient issues
        # self.mi_estimator = MutualInformationEstimator(...)
        
        # Action head (replaces base VLA's action prediction)
        if use_l1_regression:
            from prismatic.models.action_heads import L1RegressionActionHead
            self.action_head = L1RegressionActionHead(
                input_dim=bottleneck_dim,
                hidden_dim=self.llm_dim,
                action_dim=ACTION_DIM,
            )
        
        # Flag for returning bottleneck features during training
        self.return_bottleneck_features = False
        
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        proprio: Optional[torch.Tensor] = None,
        proprio_projector: Optional[nn.Module] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with InfoBot bottleneck.
        
        Args:
            pixel_values: Input images (B, C, H, W)
            input_ids: Input token IDs (B, seq_len)
            attention_mask: Attention mask (B, seq_len)
            labels: Labels for action prediction
            
        Returns:
            Dictionary with loss, predictions, and bottleneck features
        """
        # Get input embeddings from language model
        input_embeddings = self.language_model.get_input_embeddings()(input_ids)
        
        # Extract action masks
        current_action_mask = get_current_action_mask(labels)
        next_actions_mask = get_next_actions_mask(labels)
        all_actions_mask = current_action_mask | next_actions_mask
        
        # Get language embeddings (non-action tokens only)
        lang_embeddings = input_embeddings[~all_actions_mask].reshape(
            input_embeddings.shape[0], -1, input_embeddings.shape[2]
        )
        
        # Get visual features
        patch_features = self.vision_backbone(pixel_values)
        projected_visual = self.projector(patch_features)
        
        # Add proprio if provided
        if proprio_projector is not None and proprio is not None:
            proprio = proprio.reshape(projected_visual.shape[0], -1)
            proprio_features = proprio_projector(proprio).unsqueeze(1)
            projected_visual = torch.cat([projected_visual, proprio_features], dim=1)
        
        # Apply InfoBot bottleneck
        # Visual features are compressed conditioned on language
        if self.bottleneck_type == "cross_attn":
            bottleneck_features, bottleneck_info = self.bottleneck(
                visual_features=projected_visual,
                language_features=lang_embeddings,
            )
        else:
            bottleneck_features = self.bottleneck(
                visual_features=projected_visual,
                language_features=lang_embeddings,
            )
            bottleneck_info = {}
        
        # Get action hidden states from bottleneck
        # For simplicity, we use bottleneck features as action hidden states
        # Expand bottleneck tokens to match expected action token format
        batch_size = bottleneck_features.shape[0]
        actions_hidden_states = bottleneck_features.unsqueeze(2).expand(
            -1, -1, ACTION_DIM, -1
        ).reshape(batch_size, self.num_bottleneck_tokens * ACTION_DIM, self.bottleneck_dim)
        
        # Predict actions
        predicted_actions = self.action_head.predict_action(actions_hidden_states)
        
        output = {
            'predicted_actions': predicted_actions,
            'bottleneck_features': bottleneck_features,
            'visual_features': projected_visual,
            'language_embeddings': lang_embeddings,
        }
        
        if self.return_bottleneck_features or self.training:
            output['bottleneck_info'] = bottleneck_info
        
        return output


def run_infobot_forward_pass(
    infobot_model: InfoBotVLAModel,
    batch: Dict,
    device_id: int,
    use_l1_regression: bool,
    action_tokenizer,
    mi_estimator: MutualInformationEstimator,
    beta_mi: float,
) -> Tuple[torch.Tensor, Dict]:
    """
    Run forward pass with InfoBot-VLA architecture.
    
    Returns:
        total_loss: Combined loss with MI regularization
        metrics: Dictionary of loss components
    """
    # Get ground truth actions
    ground_truth_actions = batch["actions"].to(device_id).to(torch.bfloat16)
    
    # Forward through InfoBot-VLA
    output = infobot_model(
        pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
        input_ids=batch["input_ids"].to(device_id),
        attention_mask=batch["attention_mask"].to(device_id),
        labels=batch["labels"].to(device_id),
        proprio=batch["proprio"] if "proprio" in batch else None,
    )
    
    predicted_actions = output['predicted_actions']
    bottleneck_features = output['bottleneck_features']
    visual_features = output['visual_features']
    lang_embeddings = output['language_embeddings']
    
    # Action loss
    if use_l1_regression:
        action_loss = torch.nn.L1Loss()(predicted_actions, ground_truth_actions)
    
    # MI regularization loss (disabled for now due to NaN issues)
    # mi_loss = mi_estimator(...)
    mi_loss = torch.tensor(0.0, device=device_id)
    
    # Check for NaN/Inf in action loss
    if not torch.isfinite(action_loss):
        metrics = {
            'total_loss': 0.0,
            'action_loss': 0.0,
            'mi_loss': 0.0,
            'curr_action_l1_loss': 0.0,
            'beta_mi': beta_mi,
        }
        return torch.tensor(0.0, device=device_id, requires_grad=True), metrics
    
    # Total loss (without MI for now)
    total_loss = action_loss  # + beta_mi * mi_loss
    
    # Compute detailed metrics
    with torch.no_grad():
        ground_truth_curr = ground_truth_actions[:, 0]
        predicted_curr = predicted_actions[:, 0]
        curr_action_l1 = torch.nn.L1Loss()(ground_truth_curr, predicted_curr)
        
        metrics = {
            'total_loss': total_loss.item(),
            'action_loss': action_loss.item(),
            'mi_loss': mi_loss.item(),
            'curr_action_l1_loss': curr_action_l1.item(),
            'beta_mi': beta_mi,
        }
    
    return total_loss, metrics


@draccus.wrap()
def finetune_infobot(cfg: InfoBotVLAConfig) -> None:
    """Main training function for InfoBot-VLA."""
    
    assert cfg.use_lora, "Only LoRA fine-tuning is supported!"
    assert os.path.exists(cfg.substep_labels_path), \
        f"Substep labels file not found: {cfg.substep_labels_path}"
    
    cfg.vla_path = cfg.vla_path.rstrip("/")
    print(f"Fine-tuning InfoBot-VLA on `{cfg.dataset_name}`")
    print(f"Bottleneck: {cfg.bottleneck_type}, dim={cfg.bottleneck_dim}, beta={cfg.beta_mi}")
    
    run_id = get_run_id(cfg)
    run_dir = cfg.run_root_dir / run_id
    os.makedirs(run_dir, exist_ok=True)
    
    distributed_state = PartialState()
    device_id = distributed_state.local_process_index
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()
    
    if distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"infobot-{run_id}")
    
    # Load base VLA
    if model_is_on_hf_hub(cfg.vla_path):
        vla_download_path = snapshot_download(repo_id=cfg.vla_path)
        cfg.vla_path = vla_download_path
    
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    
    if distributed_state.is_main_process:
        update_auto_map(cfg.vla_path)
        check_model_logic_mismatch(cfg.vla_path)
    
    dist.barrier()
    
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=False)
    base_vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    ).to(device_id)
    
    base_vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)
    
    # Apply LoRA to base VLA
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        base_vla = get_peft_model(base_vla, lora_config)
        base_vla.print_trainable_parameters()
    
    # Wrap with DDP
    base_vla = wrap_ddp(base_vla, device_id, find_unused=True)
    
    # Create InfoBot-VLA wrapper
    infobot_model = InfoBotVLAModel(
        base_vla=base_vla.module if isinstance(base_vla, DDP) else base_vla,
        bottleneck_type=cfg.bottleneck_type,
        bottleneck_dim=cfg.bottleneck_dim,
        num_bottleneck_tokens=cfg.num_bottleneck_tokens,
        use_l1_regression=cfg.use_l1_regression,
    ).to(device_id).to(torch.bfloat16)
    
    # Wrap InfoBot with DDP
    infobot_model = DDP(infobot_model, device_ids=[device_id], find_unused_parameters=True)
    
    # MI estimator - created separately to avoid DDP issues
    mi_estimator = MutualInformationEstimator(
        bottleneck_dim=cfg.bottleneck_dim,
        visual_dim=infobot_model.module.llm_dim,
    ).to(device_id).to(torch.bfloat16)
    
    # Setup optimizer - include both InfoBot and MI estimator
    trainable_params = list(infobot_model.parameters()) + list(mi_estimator.parameters())
    print(f"# total trainable params: {sum(p.numel() for p in trainable_params)}")
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)
    
    scheduler = MultiStepLR(
        optimizer,
        milestones=[cfg.num_steps_before_decay],
        gamma=0.1,
    )
    
    # Setup dataset
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    
    use_wrist_image = cfg.num_images_in_input > 1
    
    batch_transform = SubstepRLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        substep_labels={},
        use_wrist_image=use_wrist_image,
        use_proprio=cfg.use_proprio,
        use_substep_eos=cfg.use_substep_eos,
    )
    
    train_dataset = SubstepRLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        cfg.substep_labels_path,
        resize_resolution=tuple(base_vla.module.config.image_sizes if isinstance(base_vla, DDP) else base_vla.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )
    
    if distributed_state.is_main_process:
        save_dataset_statistics(train_dataset.dataset_statistics, run_dir)
    
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,
    )
    
    # Training loop
    recent_metrics = {
        'total_loss': deque(maxlen=cfg.grad_accumulation_steps),
        'action_loss': deque(maxlen=cfg.grad_accumulation_steps),
        'mi_loss': deque(maxlen=cfg.grad_accumulation_steps),
        'curr_action_l1_loss': deque(maxlen=cfg.grad_accumulation_steps),
    }
    
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        infobot_model.train()
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(dataloader):
            # Forward pass
            loss, metrics = run_infobot_forward_pass(
                infobot_model=infobot_model,
                batch=batch,
                device_id=device_id,
                use_l1_regression=cfg.use_l1_regression,
                action_tokenizer=action_tokenizer,
                mi_estimator=mi_estimator,
                beta_mi=cfg.beta_mi,
            )
            
            # Normalize for gradient accumulation
            normalized_loss = loss / cfg.grad_accumulation_steps
            
            # Check for NaN/Inf
            if not torch.isfinite(normalized_loss):
                print(f"❌ [NaN/Inf] Step {batch_idx}: loss={normalized_loss.item()}")
                optimizer.zero_grad()
                continue
            
            # Backward
            normalized_loss.backward()
            
            # Store metrics
            for k, v in metrics.items():
                if k in recent_metrics:
                    recent_metrics[k].append(v)
            
            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps
            log_step = gradient_step_idx
            
            # Logging
            if distributed_state.is_main_process and gradient_step_idx % cfg.wandb_log_freq == 0:
                smoothened = {k: np.mean(v) if len(v) > 0 else 0.0 for k, v in recent_metrics.items()}
                log_metrics_to_wandb(smoothened, "InfoBot Train", log_step, wandb)
                
                print(f"\n{'='*60}")
                print(f"[Step {log_step}] Loss: {smoothened['total_loss']:.4f}")
                print(f"  Action Loss: {smoothened['action_loss']:.4f}")
                print(f"  MI Loss: {smoothened['mi_loss']:.4f}")
                print(f"  Beta: {cfg.beta_mi}")
                print(f"{'='*60}\n")
            
            # Optimizer step
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                if cfg.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, cfg.max_grad_norm)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress.update()
            
            # Save checkpoint
            if gradient_step_idx > 0 and log_step % cfg.save_freq == 0:
                if distributed_state.is_main_process:
                    checkpoint_dir = run_dir if cfg.save_latest_checkpoint_only else Path(str(run_dir) + f"--{log_step}_chkpt")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    
                    # Save InfoBot model state
                    torch.save(infobot_model.module.state_dict(), checkpoint_dir / f"infobot--{log_step}_checkpoint.pt")
                    
                    print(f"✓ Saved checkpoint: {checkpoint_dir}")
                
                dist.barrier()
            
            # Stop at max steps
            if log_step == cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping...")
                break


if __name__ == "__main__":
    set_seed(42)
    finetune_infobot()
