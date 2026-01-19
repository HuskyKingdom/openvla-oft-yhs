"""Implementations of various action heads, which serve as alternatives to VLM sequential token prediction."""

import math

import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sine- and cosine-based positional encoding that produces embeddings of a batch of timesteps.

    For example, at train time, the input might be a batch of 32 randomly sampled diffusion timesteps -> shape (32,)
    Then the output would be a batch of 32 timestep embeddings -> shape (32, D)

    Adapted from: https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/positional_embedding.py
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim  # dimensionality of the positional encoding

    def forward(self, x):
        # x: (batch_size,)
        device = x.device
        assert self.dim % 2 == 0, f"# dimensions must be even but got {self.dim}"
        half_dim = self.dim // 2
        exponent = torch.arange(half_dim, device=device) * -math.log(10000) / (half_dim - 1)  # shape: (D/2,)
        emb = torch.exp(exponent)  # shape: (D/2,)
        emb = x[:, None] * emb[None, :]  # shape: (batch_size, 1) * (1, D/2) -> (batch_size, D/2)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # shape: (batch_size, D)
        return emb


class MLPResNetBlock(nn.Module):
    """One MLP ResNet block with a residual connection."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.ffn = nn.Sequential(  # feedforward network, similar to the ones in Transformers
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: (batch_size, hidden_dim)
        # We follow the module ordering of "Pre-Layer Normalization" feedforward networks in Transformers as
        # described here: https://arxiv.org/pdf/2002.04745.pdf
        identity = x
        x = self.ffn(x)
        x = x + identity
        return x


class MLPResNet(nn.Module):
    """MLP with residual connection blocks."""
    def __init__(self, num_blocks, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.mlp_resnet_blocks.append(MLPResNetBlock(dim=hidden_dim))
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch_size, input_dim)
        x = self.layer_norm1(x)  # shape: (batch_size, input_dim)
        x = self.fc1(x)  # shape: (batch_size, hidden_dim)
        x = self.relu(x)  # shape: (batch_size, hidden_dim)
        for block in self.mlp_resnet_blocks:
            x = block(x)  # shape: (batch_size, hidden_dim)
        x = self.layer_norm2(x)  # shape: (batch_size, hidden_dim)
        x = self.fc2(x)  # shape: (batch_size, output_dim)
        return x


class L1RegressionActionHead(nn.Module):
    """Simple MLP-based action head that generates continuous actions via L1 regression."""
    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=None,  # Will use ACTION_DIM from constants if not specified
    ):
        super().__init__()
        # Use ACTION_DIM from constants if action_dim not specified
        if action_dim is None:
            action_dim = ACTION_DIM
        self.action_dim = action_dim
        
        self.model = MLPResNet(
            num_blocks=2, 
            input_dim=input_dim*action_dim,
            hidden_dim=hidden_dim, 
            output_dim=action_dim
        )

    def predict_action(self, actions_hidden_states):
        # actions_hidden_states: last hidden states of Transformer corresponding to action tokens in sequence
        # - shape: (batch_size, NUM_ACTIONS_CHUNK * action_dim, hidden_dim)
        # Output:
        # - shape: (batch_size, NUM_ACTIONS_CHUNK, action_dim)
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device
        # Reshape: (B, NUM_ACTIONS_CHUNK * action_dim, hidden_dim) 
        #       -> (B, NUM_ACTIONS_CHUNK, action_dim * hidden_dim)
        rearranged_actions_hidden_states = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)
        action = self.model(rearranged_actions_hidden_states)
        
        return action


class NoisePredictionModel(nn.Module):
    """
    Diffusion noise prediction model that takes an observation embedding (which fuses the
    noisy action, diffusion timestep, and image-language observation embeddings) and
    outputs a noise prediction.
    """

    def __init__(
        self,
        transformer_hidden_dim,  # Transformer hidden embedding size
        hidden_dim,  # MLP hidden size
        action_dim=7,  # action dimensionality
    ):
        super().__init__()
        self.mlp_resnet = MLPResNet(
            num_blocks=2,
            input_dim=transformer_hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
        )

    def forward(
        self,
        obs,
    ):
        # obs: observation embeddings to condition the generation on
        # - shape: (batch_size, chunk_len, rearranged_hidden_dim=action_dim*hidden_dim)
        #
        # output: predicted noise
        # - shape: (batch_size, action_dim)
        output = self.mlp_resnet(obs)
        return output


class DiffusionActionHead(nn.Module):
    """
    Simple MLP-based action head that generates continuous actions via conditional denoising diffusion process.

    Loosely inspired by: https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/transformer_for_diffusion.py
    """

    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=None,  # Will use ACTION_DIM from constants if not specified
        num_diffusion_steps_train=50,
    ):
        super().__init__()
        # Use ACTION_DIM from constants if action_dim not specified
        if action_dim is None:
            from prismatic.vla.constants import ACTION_DIM
            action_dim = ACTION_DIM
        self.action_dim = action_dim
        
        self.noise_predictor = NoisePredictionModel(
            transformer_hidden_dim=hidden_dim*action_dim,
            hidden_dim=hidden_dim, 
            action_dim=action_dim
        )
        self.num_diffusion_steps_train = num_diffusion_steps_train
        self.noise_scheduler = DDIMScheduler(num_train_timesteps=num_diffusion_steps_train, beta_schedule="squaredcos_cap_v2")
        self.time_encoder = SinusoidalPositionalEncoding(dim=hidden_dim)

    def sample_noisy_actions(self, ground_truth_actions):
        """
        Samples noise and applies noise to ground-truth actions to produce noisy actions, which are
        used as input in the noise prediction network. Returns noise, noisy actions, and the
        corresponding diffusion timestep embeddings.
        """
        # ground_truth_actions: ground-truth actions
        # - shape: (batch_size, chunk_len, action_dim)
        batch_size = ground_truth_actions.shape[0]
        device = ground_truth_actions.device
        # Sample random noise with shape equal to actions, used for closed-form forward diffusion.
        noise = torch.randn(size=(batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM), device=device, dtype=ground_truth_actions.dtype)  # (B, chunk_len, action_dim)
        # Sample random diffusion timesteps (one for each action in batch).
        timesteps = torch.randint(
            low=0, high=self.noise_scheduler.config.num_train_timesteps, size=(batch_size,), device=device
        )
        # Add noise to clean actions according to the magnitude at each diffusion timestep via
        # closed-form forward diffusion.
        noisy_actions = self.noise_scheduler.add_noise(ground_truth_actions, noise, timesteps)  # (B, chunk_len, action_dim)

        # Get diffusion timestep embeddings as well
        diffusion_timestep_embeddings = self.time_encoder(timesteps).to(noisy_actions.dtype).to(noisy_actions.device)  # (B, llm_dim)
        diffusion_timestep_embeddings = diffusion_timestep_embeddings.unsqueeze(1)  # (B, 1, llm_dim)

        return_dict = dict(
            noise=noise,
            noisy_actions=noisy_actions,
            diffusion_timestep_embeddings=diffusion_timestep_embeddings,
        )

        return return_dict

    def predict_noise(self, actions_hidden_states):
        """
        Given a batch of last hidden Transformer layer embeddings (which fuse the vision-language observation embeddings,
        noisy action embeddings, and diffusion timestep embedding), predicts the noise applied to the actions.
        """
        # actions_hidden_states: last hidden states of Transformer corresponding to action tokens in sequence
        # - shape: (batch_size, chunk_len * action_dim, hidden_dim)
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device
        rearranged_actions_hidden_states = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)  # (batch_size, chunk_len, action_dim * hidden_dim)
        # Get diffusion model's noise prediction.
        noise_pred = self.noise_predictor(rearranged_actions_hidden_states)
        return noise_pred


class EOSClassificationHead(nn.Module):
    """
    独立的 EOS 分类头，用于预测 substep 结束位置。
    
    设计要点：
    - 输入：action hidden states (B, NUM_ACTIONS_CHUNK * ACTION_DIM, D)
    - 输出：EOS logits (B, NUM_ACTIONS_CHUNK, 1) - 每个action位置一个logit
    - 结构：2层 MLP + BatchNorm + Dropout (防止过拟合)
    - 激活：最后不加sigmoid（留给BCEWithLogitsLoss）
    
    与 L1RegressionActionHead 的区别：
    - 用途不同：分类 vs 回归
    - 输出维度：1 (logit) vs ACTION_DIM (7)
    - 损失函数：BCE vs L1
    """
    def __init__(
        self,
        input_dim=4096,      # LLM hidden dim
        hidden_dim=1024,     # MLP hidden dim (可以比action head小)
        dropout=0.1,         # Dropout率
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 计算输入特征维度：input_dim * ACTION_DIM
        # 因为每个action的hidden states是 ACTION_DIM 个token的拼接
        from prismatic.vla.constants import ACTION_DIM
        feature_dim = input_dim * ACTION_DIM
        
        # 2层MLP结构
        self.model = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),  # 输出单个logit
        )
    
    def forward(self, actions_hidden_states):
        """
        预测每个action位置的EOS logit
        
        Args:
            actions_hidden_states: (B, NUM_ACTIONS_CHUNK * ACTION_DIM, hidden_dim)
                                   每个action由ACTION_DIM个token组成
        
        Returns:
            eos_logits: (B, NUM_ACTIONS_CHUNK, 1)
                       每个action位置一个logit值
        """
        batch_size = actions_hidden_states.shape[0]
        # Reshape: (B, NUM_ACTIONS_CHUNK * ACTION_DIM, D) 
        #       -> (B, NUM_ACTIONS_CHUNK, ACTION_DIM * D)
        rearranged = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)
        # MLP: (B, NUM_ACTIONS_CHUNK, ACTION_DIM * D) -> (B, NUM_ACTIONS_CHUNK, 1)
        eos_logits = self.model(rearranged)
        return eos_logits


class EOSBufferManager:
    """
    EOS 样本缓冲区管理器，用于批次累积平衡策略。
    
    核心思想：
    - 累积正负样本直到达到目标比例 (如 20:35)
    - 返回平衡的batch用于loss计算
    - 只有凑够样本才计算loss，否则返回0
    
    优势：
    - 精确控制正负样本比例
    - 不需要额外的样本权重
    - 训练更稳定
    """
    def __init__(
        self,
        target_positive=20,    # 目标正样本数
        target_negative=35,    # 目标负样本数
        device='cuda'
    ):
        self.target_positive = target_positive
        self.target_negative = target_negative
        self.device = device
        
        # 缓冲区：存储 (logits, labels)
        self.positive_buffer = {
            'logits': [],
            'labels': []
        }
        self.negative_buffer = {
            'logits': [],
            'labels': []
        }
        
        # 统计信息
        self.total_positive_seen = 0
        self.total_negative_seen = 0
        self.total_updates = 0
    
    def add_samples(self, eos_logits, eos_labels):
        """
        添加新样本到缓冲区
        
        Args:
            eos_logits: (N,) tensor - EOS预测logits (未经sigmoid)
            eos_labels: (N,) tensor - EOS真实标签 (0 or 1)
        """
        # 确保在正确的设备上
        eos_logits = eos_logits.detach().to(self.device)
        eos_labels = eos_labels.detach().to(self.device)
        
        # 分离正负样本
        positive_mask = eos_labels > 0.5
        negative_mask = eos_labels <= 0.5
        
        # 添加到对应缓冲区
        if positive_mask.any():
            pos_logits = eos_logits[positive_mask]
            pos_labels = eos_labels[positive_mask]
            self.positive_buffer['logits'].append(pos_logits)
            self.positive_buffer['labels'].append(pos_labels)
            self.total_positive_seen += len(pos_logits)
        
        if negative_mask.any():
            neg_logits = eos_logits[negative_mask]
            neg_labels = eos_labels[negative_mask]
            self.negative_buffer['logits'].append(neg_logits)
            self.negative_buffer['labels'].append(neg_labels)
            self.total_negative_seen += len(neg_logits)
    
    def can_compute_loss(self):
        """检查是否累积了足够的样本"""
        num_positive = sum(len(x) for x in self.positive_buffer['logits'])
        num_negative = sum(len(x) for x in self.negative_buffer['logits'])
        
        return (num_positive >= self.target_positive and 
                num_negative >= self.target_negative)
    
    def get_balanced_batch(self):
        """
        获取平衡的batch并清空缓冲区
        
        Returns:
            balanced_logits: (target_positive + target_negative,) tensor
            balanced_labels: (target_positive + target_negative,) tensor
        """
        # 合并缓冲区中的所有样本
        all_pos_logits = torch.cat(self.positive_buffer['logits'])
        all_pos_labels = torch.cat(self.positive_buffer['labels'])
        all_neg_logits = torch.cat(self.negative_buffer['logits'])
        all_neg_labels = torch.cat(self.negative_buffer['labels'])
        
        # 随机采样目标数量
        pos_indices = torch.randperm(len(all_pos_logits), device=self.device)[:self.target_positive]
        neg_indices = torch.randperm(len(all_neg_logits), device=self.device)[:self.target_negative]
        
        sampled_pos_logits = all_pos_logits[pos_indices]
        sampled_pos_labels = all_pos_labels[pos_indices]
        sampled_neg_logits = all_neg_logits[neg_indices]
        sampled_neg_labels = all_neg_labels[neg_indices]
        
        # 合并正负样本
        balanced_logits = torch.cat([sampled_pos_logits, sampled_neg_logits])
        balanced_labels = torch.cat([sampled_pos_labels, sampled_neg_labels])
        
        # 随机打乱
        shuffle_indices = torch.randperm(len(balanced_logits), device=self.device)
        balanced_logits = balanced_logits[shuffle_indices]
        balanced_labels = balanced_labels[shuffle_indices]
        
        # 清空缓冲区（保留多余的样本）
        if len(all_pos_logits) > self.target_positive:
            # 找到未被选中的索引
            all_indices = torch.arange(len(all_pos_logits), device=self.device)
            mask = torch.ones(len(all_pos_logits), dtype=torch.bool, device=self.device)
            mask[pos_indices] = False
            remaining_pos = all_pos_logits[mask]
            remaining_pos_labels = all_pos_labels[mask]
            self.positive_buffer['logits'] = [remaining_pos]
            self.positive_buffer['labels'] = [remaining_pos_labels]
        else:
            self.positive_buffer['logits'] = []
            self.positive_buffer['labels'] = []
        
        if len(all_neg_logits) > self.target_negative:
            all_indices = torch.arange(len(all_neg_logits), device=self.device)
            mask = torch.ones(len(all_neg_logits), dtype=torch.bool, device=self.device)
            mask[neg_indices] = False
            remaining_neg = all_neg_logits[mask]
            remaining_neg_labels = all_neg_labels[mask]
            self.negative_buffer['logits'] = [remaining_neg]
            self.negative_buffer['labels'] = [remaining_neg_labels]
        else:
            self.negative_buffer['logits'] = []
            self.negative_buffer['labels'] = []
        
        self.total_updates += 1
        
        return balanced_logits, balanced_labels
    
    def get_stats(self):
        """获取统计信息"""
        num_positive = sum(len(x) for x in self.positive_buffer['logits'])
        num_negative = sum(len(x) for x in self.negative_buffer['logits'])
        
        return {
            'buffer_positive': num_positive,
            'buffer_negative': num_negative,
            'total_positive_seen': self.total_positive_seen,
            'total_negative_seen': self.total_negative_seen,
            'total_updates': self.total_updates,
            'positive_utilization': self.total_updates * self.target_positive / max(1, self.total_positive_seen),
            'negative_utilization': self.total_updates * self.target_negative / max(1, self.total_negative_seen),
        }
    
    def reset(self):
        """重置缓冲区"""
        self.positive_buffer = {'logits': [], 'labels': []}
        self.negative_buffer = {'logits': [], 'labels': []}
        self.total_positive_seen = 0
        self.total_negative_seen = 0
        self.total_updates = 0


def compute_classification_metrics(pred, gt):
    """
    计算二分类指标
    
    Args:
        pred: (N,) bool tensor - 预测结果 (True/False)
        gt: (N,) bool tensor - 真实标签 (True/False)
    
    Returns:
        dict: 包含 accuracy, precision, recall, F1 和混淆矩阵的指标字典
    """
    import torch
    
    # 确保是 bool tensor
    pred = pred.bool()
    gt = gt.bool()
    
    # 计算混淆矩阵元素
    # True Positives, False Positives, True Negatives, False Negatives
    tp = (pred & gt).sum().float()
    fp = (pred & ~gt).sum().float()
    tn = (~pred & ~gt).sum().float()
    fn = (~pred & gt).sum().float()
    
    # 计算指标
    accuracy = (tp + tn) / (tp + fp + tn + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {
        "eos_accuracy": accuracy.item(),
        "eos_precision": precision.item(),
        "eos_recall": recall.item(),
        "eos_f1": f1.item(),
        "eos_tp": int(tp.item()),
        "eos_fp": int(fp.item()),
        "eos_tn": int(tn.item()),
        "eos_fn": int(fn.item()),
    }
