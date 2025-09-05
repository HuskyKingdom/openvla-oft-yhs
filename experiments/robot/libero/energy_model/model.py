from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SeqPool(nn.Module):
    def __init__(self, mode="mean"):
        super().__init__()
        assert mode in ["cls", "mean"]
        self.mode = mode

    def forward(self, h):  # h: [B,S,Dh]
        if self.mode == "cls":
            return h[:, 0, :]                       
        else:
            return h.mean(dim=1)
        


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
    input: hN(s) [B, seq, D_h], a [B, chunk, D_a] 
    output: energy [B, 1]
    """
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        hidden: int = 512,
        n_layers: int = 4,
        NUM_ACTIONS_CHUNK = 8,
    ):
        super().__init__()
        in_dim = hidden * 3
        self.action_proj = nn.Linear(act_dim, hidden)
        self.action_proj_act = nn.SiLU()
        
        self.state_dim = state_dim
        self.pool = SeqPool(mode="mean")
        self.proj_hidden  = nn.Linear(state_dim, hidden)

        self.model = MLPResNet(
            num_blocks=n_layers, input_dim=in_dim, hidden_dim=hidden, output_dim=1
        )


        # pos emb
        self.pos_emb = nn.Embedding(NUM_ACTIONS_CHUNK, hidden)


    def forward(self, hN: torch.Tensor, a: torch.Tensor, reduce="sum", gamma=None) -> torch.Tensor:
        """
        hN: [B, S, D_h], a: [B, H,  D_a]
        return: energy [B, 1]
        """

        B, H, Da = a.shape
        c = self.pool(hN) # [B, 1]
        c = self.proj_hidden(c) # [B, Hd]
        c = c.unsqueeze(1).expand(B, H, c.shape[-1])  # [B,H,Hd]

        a = self.action_proj_act(self.action_proj(a)) # [B,H,Hd]

        # pos emb
        feats = [c, a]
        pos_ids = torch.arange(H, device=a.device).unsqueeze(0).expand(B, H)  # [B,H]
        p = self.pos_emb(pos_ids)                                             # [B,H,Hid]
        feats.append(p)
        x = torch.cat(feats, dim=-1)         

        E_steps = self.model(x)           # [B,H,1]
        # E_steps = F.softplus(E_steps) # reg
        E_steps = 0.5 * (E_steps ** 2) + 1e-6

        if reduce == "sum":
            if gamma is None:
                E = E_steps.sum(dim=1)        # [B,1]
            else:
                w = torch.pow(gamma, torch.arange(H, device=a.device)).view(1,H,1)  # discount
                E = (E_steps * w).sum(dim=1)
        elif reduce == "mean":
            E = E_steps.mean(dim=1)
        else:
            raise ValueError("reduce must be 'sum' or 'mean'")


        return E, E_steps
    

