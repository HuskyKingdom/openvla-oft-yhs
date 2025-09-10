# 方案1: 改进in-batch方法的负样本质量
# 在您的代码第536行之前添加：

def select_better_negatives_for_inbatch(layer_actions, ground_truth_actions, context_hidden, energy_model, energy_mask):
    """为in-batch方法选择更好的负样本，替换batch内的一些样本"""
    if len(layer_actions) < 2:
        return ground_truth_actions  # 如果没有足够的layer actions，返回原始
    
    B, H, Da = ground_truth_actions.shape
    with torch.no_grad():
        energy_model.eval()
        
        # 计算不同层动作的能量，选择中等质量的作为负样本
        layer_energies = []
        for i, layer_action in enumerate(layer_actions[:-1]):  # 排除最后一层
            try:
                energy = energy_model(context_hidden, layer_action, energy_mask)
                layer_energies.append((i, layer_action, energy.mean().item()))
            except:
                continue
        
        if not layer_energies:
            energy_model.train()
            return ground_truth_actions
            
        # 选择能量中等的作为负样本（不是最好也不是最差）
        layer_energies.sort(key=lambda x: x[2])
        mid_idx = len(layer_energies) // 2
        selected_negative = layer_energies[mid_idx][1]
        
        # 用选中的负样本替换batch中的一些样本
        improved_actions = ground_truth_actions.clone()
        replace_indices = torch.randperm(B)[:B//3]  # 替换1/3的样本
        improved_actions[replace_indices] = selected_negative[replace_indices]
        
        energy_model.train()
        return improved_actions

# 使用方法：在第536行替换为：
# improved_actions = select_better_negatives_for_inbatch(layer_actions, ground_truth_actions, context_hidden, energy_model, energy_mask)
# swap_loss, E_pos_mean, E_neg_mean = energy_inbatch_swap_infonce(energy_model, context_hidden, improved_actions, energy_mask)


# 方案2: 切换到标准InfoNCE方法 (需要更多改动)
def improved_negative_sampling(layer_actions, ground_truth_actions, context_hidden, energy_model, energy_mask):
    """生成高质量负样本用于标准InfoNCE"""
    B, H, Da = ground_truth_actions.shape
    negatives = []
    
    with torch.no_grad():
        energy_model.eval()
        
        # 策略1: 从不同层选择能量适中的动作
        if len(layer_actions) > 1:
            layer_energies = []
            for i, layer_action in enumerate(layer_actions[:-1]):
                try:
                    energy = energy_model(context_hidden, layer_action, energy_mask)
                    layer_energies.append((layer_action, energy.mean().item()))
                except:
                    continue
            
            if layer_energies:
                # 按能量排序，选择中等的
                layer_energies.sort(key=lambda x: x[1])
                n_select = min(2, len(layer_energies))
                for i in range(n_select):
                    idx = len(layer_energies) * (i + 1) // (n_select + 1)
                    negatives.append(layer_energies[idx][0])
        
        energy_model.train()
    
    # 策略2: 添加不同强度的噪声
    for sigma in [0.1, 0.3]:
        noise_action = ground_truth_actions + torch.randn_like(ground_truth_actions) * sigma
        negatives.append(noise_action)
    
    # 策略3: 时序shuffle
    shuffle_idx = torch.randperm(H)
    shuffle_action = ground_truth_actions[:, shuffle_idx, :]
    negatives.append(shuffle_action)
    
    # 确保至少有一个负样本
    if not negatives:
        negatives = [ground_truth_actions + torch.randn_like(ground_truth_actions) * 0.2]
    
    return torch.stack(negatives, dim=1)  # [B, M, H, Da]

# 使用方法：替换第536-537行为：
# A_negatives = improved_negative_sampling(layer_actions, ground_truth_actions, context_hidden, energy_model, energy_mask)
# from energy.energy_model import energy_infonce_loss
# energy_loss, E_pos_mean, E_neg_mean = energy_infonce_loss(energy_model, context_hidden, ground_truth_actions, A_negatives, energy_mask, tau=0.3)


# 方案3: 混合策略 (最稳健)
def hybrid_energy_loss(energy_model, context_hidden, ground_truth_actions, layer_actions, energy_mask, step):
    """混合使用两种方法"""
    
    # 早期训练使用in-batch (更稳定)
    if step < 10000:
        improved_actions = select_better_negatives_for_inbatch(
            layer_actions, ground_truth_actions, context_hidden, energy_model, energy_mask
        )
        loss, E_pos, E_neg = energy_inbatch_swap_infonce(
            energy_model, context_hidden, improved_actions, energy_mask, tau=0.3
        )
        
    # 后期训练使用标准InfoNCE (更精确)
    else:
        A_negatives = improved_negative_sampling(
            layer_actions, ground_truth_actions, context_hidden, energy_model, energy_mask
        )
        from energy.energy_model import energy_infonce_loss
        loss, E_pos, E_neg = energy_infonce_loss(
            energy_model, context_hidden, ground_truth_actions, A_negatives, energy_mask, tau=0.3
        )
    
    return loss, E_pos, E_neg

# 使用方法：替换第536-537行为：
# energy_loss, E_pos_mean, E_neg_mean = hybrid_energy_loss(
#     energy_model, context_hidden, ground_truth_actions, layer_actions, energy_mask, batch_idx
# )
