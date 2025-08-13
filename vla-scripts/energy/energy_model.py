from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F



class MLPResNetBlock(nn.Module):
    """One MLP ResNet block with a residual connection."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.ffn = nn.Sequential(  # feedforward network, similar to the ones in Transformers
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
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
        self.act = nn.SiLU()
        self.mlp_resnet_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.mlp_resnet_blocks.append(MLPResNetBlock(dim=hidden_dim))
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch_size, input_dim)
        x = self.layer_norm1(x)  # shape: (batch_size, input_dim)
        x = self.fc1(x)  # shape: (batch_size, hidden_dim)
        x = self.act(x)  # shape: (batch_size, hidden_dim)
        for block in self.mlp_resnet_blocks:
            x = block(x)  # shape: (batch_size, hidden_dim)
        x = self.layer_norm2(x)  # shape: (batch_size, hidden_dim)
        x = self.fc2(x)  # shape: (batch_size, output_dim)
        return x



class EnergyModel(nn.Module):
    """
    E_phi(s, a):
    input: hN(s) [B, D_h], a [B, D_a] 
    output: energy [B, 1]
    """
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        hidden: int = 256,
        n_layers: int = 2,
    ):
        super().__init__()
        in_dim = state_dim + hidden
        self.action_proj = nn.Linear(act_dim, hidden)
        self.action_proj_act = nn.SiLU()
        self.model = MLPResNet(
            num_blocks=n_layers, input_dim=in_dim, hidden_dim=hidden, output_dim=1
        )

    def forward(self, hN: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        hN: [B, D_h], a: [B, D_a]
        return: energy [B, 1]
        """
        a = self.action_proj_act(self.action_proj(a))
        x = torch.cat([hN, a], dim=-1)
        E = self.model(x)
        return E
    





@torch.no_grad()
def one_step_energy_correction(
    energy_head: EnergyModel,
    hN: torch.Tensor,
    a_bc: torch.Tensor,
    alpha: float = 0.1,
    clip_frac: float = 0.2,
    act_range: Optional[torch.Tensor] = None,
    correct_dims: Optional[List[int]] = None,
) -> torch.Tensor:
    """
      a_ref = a_bc - alpha * ∇_a E(s, a_bc)
    in:
      hN:        [B, D_h]
      a_bc:      [B, D_a]（
      alpha:     step magnitude
      clip_frac:  clip
      act_range: [D_a] 每维动作量程（上界-下界），用于裁剪；若为 None 则按 a_bc 的范数做全局裁剪
      correct_dims: 只校正这些维度（如末端位姿维度），None 表示全维校正
    返回:
      a_ref: [B, D_a]
    """
    # 需要对 a 求梯度，因此以下不加 no_grad
    a = a_bc.detach().requires_grad_(True)          # [B, D_a]
    E = energy_head(hN, a).sum()                    # 标量化以便 grad
    grad_a = torch.autograd.grad(E, a, create_graph=False, retain_graph=False)[0]  # [B, D_a]

    step = alpha * grad_a

    if correct_dims is not None:
        mask = torch.zeros_like(step)
        mask[..., correct_dims] = 1.0
        step = step * mask

    if act_range is not None:
        max_step = clip_frac * act_range.to(step.device)
        step = torch.clamp(step, -max_step, max_step)
    else:
        max_norm = (a_bc.detach().norm(dim=-1, keepdim=True) * clip_frac) + 1e-6
        step_norm = step.norm(dim=-1, keepdim=True) + 1e-6
        step = step * torch.minimum(torch.ones_like(step_norm), max_norm / step_norm)

    a_ref = a - step
    return a_ref.detach()