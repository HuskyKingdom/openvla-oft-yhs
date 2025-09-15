import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

def gaussian_perturbation_energy_landscape():
    """高斯扰动参数的energy landscape可视化"""
    
    # 加载数据
    device = "cuda"
    CKPT_DIR = "/work1/aiginternal/yuhang/openvla-oft-yhs/ckpoints/openvla-7b-oft-finetuned-libero-spatial-object-goal-10+libero_4_task_suites_no_noops+b24+lr-0.0005+lora-r32+dropout-0.0--image_aug--energy_freeze--100000_chkpt"
    
    energy_model = EnergyModel(4096, 7).to(device).to(torch.bfloat16)
    energy_model.load_state_dict(torch.load(CKPT_DIR + "/energy_model--100000_checkpoint.pt"))
    energy_model.eval()
    
    context_hidden = torch.load("energy_vis/context_hidden_ts1.pt", map_location=device).to(torch.bfloat16)
    expert_action = torch.tensor(
        [[[-0.0069, -0.7031,  0.6914, -0.3750, -0.6055, -1.0312, -0.6172],
          [ 0.5781, -0.5664,  0.5078,  0.1982,  0.0024, -1.3047, -0.1680],
          # ... 其他expert actions
        ]]
    ).to(device).to(torch.bfloat16)
    
    # 扰动参数网格
    noise_means = np.linspace(-0.5, 0.5, 20)      # x轴: 噪声均值
    noise_stds = np.linspace(0.01, 1.0, 20)       # y轴: 噪声标准差
    
    energy_surface = np.zeros((len(noise_means), len(noise_stds)))
    
    print("Computing energy landscape...")
    for i, noise_mean in enumerate(noise_means):
        for j, noise_std in enumerate(noise_stds):
            # 生成扰动动作
            noise = torch.randn_like(expert_action) * noise_std + noise_mean
            perturbed_action = expert_action + noise
            
            # 计算energy（多次采样取平均，减少随机性）
            energies = []
            for _ in range(5):
                noise_sample = torch.randn_like(expert_action) * noise_std + noise_mean
                action_sample = expert_action + noise_sample
                with torch.no_grad():
                    energy = energy_model(context_hidden, action_sample)
                    energies.append(energy.cpu().item())
            
            energy_surface[i, j] = np.mean(energies)
    
    # 3D可视化
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    X, Y = np.meshgrid(noise_means, noise_stds)
    surf = ax.plot_surface(X, Y, energy_surface.T, cmap='viridis', alpha=0.8)
    
    # 标记expert action点
    expert_energy = energy_model(context_hidden, expert_action).cpu().item()
    ax.scatter([0], [0], [expert_energy], color='red', s=100, label='Expert Action')
    
    ax.set_xlabel('Noise Mean')
    ax.set_ylabel('Noise Std')
    ax.set_zlabel('Energy')
    ax.set_title('Energy Landscape: Gaussian Perturbation Parameters')
    plt.colorbar(surf)
    plt.legend()
    plt.savefig('energy_landscape_gaussian_params.png', dpi=300)