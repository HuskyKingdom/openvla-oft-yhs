"""
Debug script for EOS detection
检查 EOS head 是否被正确训练和加载
"""
import torch
from pathlib import Path

def check_eos_checkpoint(checkpoint_path: str):
    """检查 EOS checkpoint 的状态"""
    checkpoint_path = Path(checkpoint_path)
    
    # 查找 EOS head checkpoint
    eos_checkpoint_candidates = [
        checkpoint_path / "eos_head--latest_checkpoint.pt",
        *sorted(checkpoint_path.glob("eos_head--*_checkpoint.pt"), reverse=True),
    ]
    
    eos_checkpoint_file = None
    for candidate in eos_checkpoint_candidates:
        if candidate.exists():
            eos_checkpoint_file = candidate
            break
    
    if eos_checkpoint_file is None:
        print(f"❌ No EOS checkpoint found in {checkpoint_path}")
        return False
    
    print(f"✓ Found EOS checkpoint: {eos_checkpoint_file}")
    
    # 加载 checkpoint
    state_dict = torch.load(eos_checkpoint_file, map_location='cpu', weights_only=True)
    
    # 去除 DDP 前缀
    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            clean_state_dict[k[7:]] = v
        else:
            clean_state_dict[k] = v
    
    print(f"\n{'='*80}")
    print(f"EOS Head Checkpoint Analysis")
    print(f"{'='*80}\n")
    
    # 检查每层权重
    print("Weight Statistics:")
    for key, param in clean_state_dict.items():
        mean = param.mean().item()
        std = param.std().item()
        min_val = param.min().item()
        max_val = param.max().item()
        
        # 判断权重是否像是训练过的
        is_trained = abs(mean) > 0.01 or std > 0.1
        status = "✓ Trained" if is_trained else "⚠️  Random?"
        
        print(f"  {key:30s} {status}")
        print(f"    shape={list(param.shape):20s} mean={mean:8.4f} std={std:8.4f} range=[{min_val:.4f}, {max_val:.4f}]")
    
    print(f"\n{'='*80}\n")
    
    # 检查最后一层 bias（通常能看出是否训练）
    final_bias_key = None
    for key in clean_state_dict.keys():
        if 'bias' in key:
            final_bias_key = key
    
    if final_bias_key:
        final_bias = clean_state_dict[final_bias_key]
        print(f"Final layer bias: {final_bias.item():.4f}")
        print(f"  If close to 0: model predicts ~50% probability (random)")
        print(f"  If very negative (< -5): model predicts low probability (trained to predict mostly 0)")
        print(f"  If very positive (> 5): model predicts high probability (trained to predict mostly 1)")
    
    return True


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python debug_eos_detection.py <checkpoint_path>")
        print("Example: python debug_eos_detection.py runs/openvla-7b+libero_object_no_noops--substep")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    check_eos_checkpoint(checkpoint_path)

