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

denorm = denorm_actions_torch(normalized, norm_stats_action,
                              clamp_to_range=True, discretize_gripper=True)


CKPT_DIR = "/work1/aiginternal/yuhang/openvla-oft-yhs/ckpoints/openvla-7b+libero_4_task_suites_no_noops+b3+lr-0.0005+lora-r32+dropout-0.0--image_aug--energy_finetuned--200000_chkpt"
energy_model = EnergyModel(4096,7,512,2,8).to(device).to(torch.bfloat16)
energy_model.load_state_dict(torch.load(CKPT_DIR + "/energy_model--200000_checkpoint.pt"))

# loading variables
CONTEXT_PATH = "energy_vis/context_hidden_ts1.pt"
context_hidden = torch.load(CONTEXT_PATH, map_location=device).to(torch.bfloat16)



energy_turth, _ = energy_model(context_hidden, normalized)

print(context_hidden.shape, energy_turth)



import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# -----------------------------
# 1) 选定 batch 与时间步 t
# -----------------------------
device = context_hidden.device
dtype  = context_hidden.dtype  # 你目前是 bfloat16

b, seq_len, hdim = context_hidden.shape
_, _, adim = normalized.shape
assert adim == 7, f"Expect adim=7, got {adim}"

t = 0  # 可改：你想看的时间步
h_t = context_hidden[0:1, t:t+1, :]              # (1,1,hdim)
a_star = normalized[0:1, t:t+1, :]               # (1,1,adim)
# 通常夹爪是离散/二值的，默认把它固定（最后一维 index=6）
freeze_dims = [6]  # 如不想固定夹爪可设为 []

# -----------------------------
# 2) 基础工具
# -----------------------------
def _make_basis_axis_slice(adim, i, j):
    """标准坐标轴切片：返回 e_i, e_j 两个单位向量（adim维）。"""
    e_i = torch.zeros(adim, device=device, dtype=torch.float32)
    e_j = torch.zeros_like(e_i)
    e_i[i] = 1.0
    e_j[j] = 1.0
    return e_i, e_j

@torch.no_grad()
def _energy_of_actions(h_batch, a_batch):
    """
    h_batch: (N,1,hdim)
    a_batch: (N,1,adim)
    返回: (N,) energies
    """
    # 转成与模型一致的 dtype（你的模型是 bfloat16）
    h_in = h_batch.to(device=device, dtype=dtype)
    a_in = a_batch.to(device=device, dtype=dtype)
    E, *_ = energy_model(h_in, a_in)   # 假设 forward 返回 (E, other)
    # 形状可能是 (N,1) 或 (N,)，这里统一 squeeze
    return E.squeeze(-1).squeeze(-1).float()

def _apply_freeze_dims(a, a_star_flat, freeze_idx):
    """把 a 的 freeze_idx 维度强行设置为专家动作的值。"""
    if not freeze_idx:
        return a
    a = a.clone()
    for k in freeze_idx:
        a[..., k] = a_star_flat[..., k]
    return a

def _clamp_normalized(a):
    """
    依据你的规范化范围（通常关节在 [-1,1]），做一下夹紧；
    如你有更准确的范围，可在这里替换规则。
    """
    return a.clamp(-1.0, 1.0)

# -----------------------------
# 3) 在二维平面上评估能量并画图
# -----------------------------
@torch.no_grad()
def plot_energy_landscape_axis_slice(
    h_t, a_star, dims=(0,1), steps=101, span=2.5, out_prefix="axis",
):
    """
    在 axis-slice 平面上 (x沿 dim i, y沿 dim j) 评估能量:
    a(x,y) = a* + x * e_i + y * e_j (其他维度固定为 a*)
    """
    i, j = dims
    e_i, e_j = _make_basis_axis_slice(adim, i, j)  # float32

    # 网格（用 float32 计算，再转 dtype）
    xs = torch.linspace(-span, span, steps, device=device, dtype=torch.float32)
    ys = torch.linspace(-span, span, steps, device=device, dtype=torch.float32)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")  # (steps, steps)

    a_star_flat = a_star.view(1, adim).to(torch.float32)  # (1,adim)
    A = a_star_flat + X.unsqueeze(-1)*e_i + Y.unsqueeze(-1)*e_j  # (steps,steps,adim)

    # 可选：冻结某些维度（如夹爪）
    A = _apply_freeze_dims(A, a_star_flat, freeze_dims)

    # 夹紧到合法范围（若你的 normalized 空间是别的范围请相应调整）
    A = _clamp_normalized(A)

    # 打平成批
    A_flat = A.reshape(-1, adim).to(device=device)
    H_flat = h_t.expand(A_flat.shape[0], -1, -1)  # (N,1,hdim)
    A_flat = A_flat.unsqueeze(1)                  # (N,1,adim)

    # 评估能量
    E_flat = _energy_of_actions(H_flat, A_flat)   # (N,)
    E = E_flat.view(steps, steps).cpu().numpy()

    # 专家点能量
    E_star = _energy_of_actions(h_t, a_star).item()

    # 画 3D 曲面
    Xn = X.cpu().numpy(); Yn = Y.cpu().numpy()
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Xn, Yn, E, rstride=1, cstride=1, linewidth=0, antialiased=True, alpha=0.9)
    ax.scatter([0.0], [0.0], [E_star], s=60, marker='o')  # 专家动作为(0,0)
    ax.set_xlabel(f"x (dim {i})")
    ax.set_ylabel(f"y (dim {j})")
    ax.set_zlabel("Energy")
    ax.set_title(f"Energy landscape @ t={t} (axis slice dims {i},{j})")
    plt.tight_layout()
    png_path = f"{out_prefix}_t{t}_dims{i}{j}.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    # 额外：2D 等高线（可选）
    fig2 = plt.figure(figsize=(6,5))
    ax2 = fig2.add_subplot(111)
    cs = ax2.contourf(Xn, Yn, E, levels=40)
    ax2.scatter([0.0], [0.0], s=40, marker='x')
    ax2.set_xlabel(f"x (dim {i})")
    ax2.set_ylabel(f"y (dim {j})")
    ax2.set_title(f"Energy contour @ t={t} (dims {i},{j})")
    plt.colorbar(cs, ax=ax2)
    plt.tight_layout()
    png2_path = f"{out_prefix}_t{t}_dims{i}{j}_contour.png"
    fig2.savefig(png2_path, dpi=150)
    plt.close(fig2)

    print(f"[Saved] {png_path} and {png2_path}. Expert E* = {E_star:.4f}")

# -----------------------------
# 4) Gradient 平面（局部最陡方向）
# -----------------------------
def plot_energy_landscape_grad_plane(
    h_t, a_star, steps=101, span=1.5, out_prefix="grad"
):
    """
    以专家动作处的梯度方向为第一轴 v1，随机正交向量为第二轴 v2：
    a(x,y) = a* + x*v1 + y*v2
    """
    # 需要梯度
    a_star_var = a_star.detach().clone().to(device=device, dtype=torch.float32).requires_grad_(True)
    E_star, *_ = energy_model(h_t.to(dtype=dtype), a_star_var.to(dtype=dtype))
    # 统一到标量（有时会是 (1,1)）
    E_star = E_star.squeeze()
    g, = torch.autograd.grad(E_star, a_star_var, retain_graph=False, create_graph=False)
    g = g.detach().view(-1)  # (adim,)

    # 若梯度几乎为0，退化到 axis-slice
    if torch.norm(g) < 1e-8:
        print("[Warn] grad is ~0; fallback to axis-slice dims (0,1).")
        return plot_energy_landscape_axis_slice(h_t, a_star, dims=(0,1), steps=steps, span=span, out_prefix=out_prefix)

    v1 = g / (torch.norm(g) + 1e-12)
    # 随机向量并正交化得到 v2
    rnd = torch.randn_like(v1)
    v2 = rnd - torch.dot(rnd, v1) * v1
    v2 = v2 / (torch.norm(v2) + 1e-12)

    # 可选：不在被冻结的维度上扰动
    if freeze_dims:
        mask = torch.ones_like(v1)
        for k in freeze_dims:
            mask[k] = 0.0
        # 重新正交化并归一
        v1 = v1 * mask
        v2 = v2 * mask
        if torch.norm(v1) > 0:
            v1 = v1 / (torch.norm(v1) + 1e-12)
        if torch.norm(v2) > 0:
            v2 = v2 / (torch.norm(v2) + 1e-12)

    xs = torch.linspace(-span, span, steps, device=device, dtype=torch.float32)
    ys = torch.linspace(-span, span, steps, device=device, dtype=torch.float32)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")

    a0 = a_star.view(1, adim).to(torch.float32)
    A = a0 + X.unsqueeze(-1)*v1 + Y.unsqueeze(-1)*v2  # (steps,steps,adim)
    A = _apply_freeze_dims(A, a0, freeze_dims)
    A = _clamp_normalized(A)

    A_flat = A.reshape(-1, adim).to(device=device)
    H_flat = h_t.expand(A_flat.shape[0], -1, -1)
    E_flat = _energy_of_actions(H_flat, A_flat.unsqueeze(1))
    E = E_flat.view(steps, steps).cpu().numpy()

    # 重新算专家点能量（用 no_grad）
    with torch.no_grad():
        E_star_val = _energy_of_actions(h_t, a_star).item()

    Xn = X.cpu().numpy(); Yn = Y.cpu().numpy()
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Xn, Yn, E, rstride=1, cstride=1, linewidth=0, antialiased=True, alpha=0.9)
    ax.scatter([0.0], [0.0], [E_star_val], s=60, marker='o')
    ax.set_xlabel("x (grad dir)")
    ax.set_ylabel("y (orth dir)")
    ax.set_zlabel("Energy")
    ax.set_title(f"Energy landscape @ t={t} (grad-plane)")
    plt.tight_layout()
    png_path = f"{out_prefix}_t{t}_gradplane.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    # 2D 等高线
    fig2 = plt.figure(figsize=(6,5))
    ax2 = fig2.add_subplot(111)
    cs = ax2.contourf(Xn, Yn, E, levels=40)
    ax2.scatter([0.0], [0.0], s=40, marker='x')
    ax2.set_xlabel("x (grad dir)")
    ax2.set_ylabel("y (orth dir)")
    ax2.set_title(f"Energy contour @ t={t} (grad-plane)")
    plt.colorbar(cs, ax=ax2)
    plt.tight_layout()
    png2_path = f"{out_prefix}_t{t}_gradplane_contour.png"
    fig2.savefig(png2_path, dpi=150)
    plt.close(fig2)

    print(f"[Saved] {png_path} and {png2_path}. Expert E* = {E_star_val:.4f}")

# -----------------------------
# 5) 实际调用示例
# -----------------------------
# A) 轴向切片：比如维度0与1（两关节）
plot_energy_landscape_axis_slice(h_t, a_star, dims=(0,1), steps=121, span=2.0, out_prefix="axis")

# B) 梯度平面（观察局部最陡方向）
plot_energy_landscape_grad_plane(h_t, a_star, steps=121, span=1.0, out_prefix="grad")