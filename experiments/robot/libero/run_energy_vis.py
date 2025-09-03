from energy_model.model import EnergyModel
import torch
from typing import Dict, Sequence


def denorm_actions_torch(
    actions: torch.Tensor,   # [..., 7]  归一化后的动作
    stats: Dict[str, Sequence[float]],
    clamp_to_range: bool = True,
    discretize_gripper: bool = True,
) -> torch.Tensor:
    """
    依据 OpenVLA 保存的 norm_stats['action'] 做反归一化。
    - 对 mask[i]==True 的维做 x = z*std + mean
    - 对 mask[i]==False（通常是 gripper）保持原值；可选阈值二值化到 {0,1}
    - 最后可选按 per-dim min/max 限幅

    actions: 形状 [..., 7]（如 [B, T, 7]），float/bfloat16 都行
    stats:   dict，含 keys: 'mean','std','min','max','mask'
    """
    assert actions.shape[-1] == 7, "Expect last dim = 7 (x,y,z,roll,pitch,yaw,gripper)"
    device = actions.device if torch.is_tensor(actions) else "cpu"
    dtype  = torch.float32

    mean = torch.tensor(stats["mean"], dtype=dtype, device=device)  # [7]
    std  = torch.tensor(stats["std"],  dtype=dtype, device=device)  # [7]
    vmin = torch.tensor(stats["min"],  dtype=dtype, device=device)  # [7]
    vmax = torch.tensor(stats["max"],  dtype=dtype, device=device)  # [7]
    mask = torch.tensor(stats["mask"], dtype=torch.bool, device=device)  # [7]

    # broadcast 到 actions 的形状
    view_shape = [1] * (actions.ndim - 1) + [7]
    mean = mean.view(view_shape)
    std  = std.view(view_shape)
    vmin = vmin.view(view_shape)
    vmax = vmax.view(view_shape)
    mask = mask.view(view_shape)

    # 反归一化（仅在被标准化的维度上做 z*std+mean；其它维度原样保留）
    z = actions.to(dtype)
    x = torch.where(mask, z * std + mean, z)

    # gripper: 可选二值化（>=0.5为闭合）
    if discretize_gripper:
        g = x[..., 6]
        x[..., 6] = (g >= 0.5).to(dtype)

    # 可选按数据统计的 min/max 限幅（更安全，避免数值越界）
    if clamp_to_range:
        x = torch.minimum(torch.maximum(x, vmin), vmax)

    return x

# params
device = "cuda"
norm_stats_action = {
    "mean": [0.01820324920117855, 0.05858374014496803, -0.05592384561896324,
             0.004626928828656673, 0.00289608770981431, -0.007673131301999092, 0.5457824468612671],
    "std":  [0.2825464606285095, 0.35904666781425476, 0.3673802614212036,
             0.03770702704787254, 0.05429719388484955, 0.08725254982709885, 0.49815231561660767],
    "max":  [0.9375, 0.9375, 0.9375, 0.30000001192092896, 0.29357144236564636, 0.375, 1.0],
    "min":  [-0.9375, -0.9375, -0.9375, -0.23642857372760773, -0.3053571283817291, -0.3675000071525574, 0.0],
    "q01":  [-0.6348214149475098, -0.7741071581840515, -0.7633928656578064,
             -0.09749999642372131, -0.14819999992847435, -0.2742857038974762, 0.0],
    "q99":  [0.7714285850524902, 0.8464285731315613, 0.9375,
             0.13928571343421936, 0.15964286029338837, 0.3246428668498993, 1.0],
    "mask": [True, True, True, True, True, True, False],
}

normalized = torch.tensor(
  [[[-0.3984, -0.6055, -0.4375, -0.1484, -0.6758, -0.2832, 1.0000],
    [-0.4238, -0.5742, -0.4688, -0.1484, -0.6289, -0.2412, 1.0000],
    [-0.4219, -0.4922, -0.4766, -0.1484, -0.6094, -0.1865, 1.0000],
    [-0.4238, -0.5039, -0.4805, -0.1484, -0.5938, -0.1826, 1.0000],
    [-0.4336, -0.4980, -0.5000, -0.1484, -0.5820, -0.1738, 1.0000],
    [-0.4277, -0.4883, -0.5156, -0.1484, -0.5781, -0.1660, 1.0000],
    [-0.4102, -0.5000, -0.5859, -0.1484, -0.5156, -0.1406, 1.0000],
    [-0.4102, -0.4766, -0.6523, -0.1484, -0.5391, -0.1201, 1.0000]]]
).to(device).to(torch.bfloat16)


def sample_rand_like_train(stats, B=1, H=8, device="cuda"):
    # 0-5 维：标准化域内的截断高斯；第6维(gripper)：{0,1}
    import torch
    z = torch.randn(B, H, 6, device=device)
    z = torch.clamp(z, -3.0, 3.0)  # 约等于在训练“可见域”内
    g = torch.randint(0, 2, (B, H, 1), device=device).float()  # 离散抓取
    a = torch.cat([z, g], dim=-1)  # 已是“标准化空间”
    # （如你训练时还额外 clip 到 [-0.9375, 0.9375]，也同步 clip 一下）
    a[..., :6] = torch.clamp(a[..., :6], -0.94, 0.94)
    return a

x = sample_rand_like_train()

denorm = denorm_actions_torch(normalized, norm_stats_action,
                              clamp_to_range=True, discretize_gripper=True)


CKPT_DIR = "/work1/aiginternal/yuhang/openvla-oft-yhs/ckpoints/openvla-7b+libero_4_task_suites_no_noops+b3+lr-0.0005+lora-r32+dropout-0.0--image_aug--energy_finetuned--200000_chkpt"
energy_model = EnergyModel(4096,7,512,2,8).to(device).to(torch.bfloat16)
energy_model.load_state_dict(torch.load(CKPT_DIR + "/energy_model--200000_checkpoint.pt"))
energy_model.eval()

# loading variables
CONTEXT_PATH = "energy_vis/context_hidden_ts1.pt"
context_hidden = torch.load(CONTEXT_PATH, map_location=device).to(torch.bfloat16)



energy_turth, energy_turth_step = energy_model(context_hidden, normalized)
energy_neg, energy_neg_step = energy_model(context_hidden, x)


print(energy_turth_step,energy_neg_step)
print(f"expert : {energy_turth[0][0]:.8f} | rand : {energy_neg[0][0]:.8f}")
