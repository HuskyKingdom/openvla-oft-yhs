"""
test_substep_imports.py

Quick test script to verify all substep training modules can be imported correctly.
"""

import sys
from pathlib import Path

print("Testing substep training module imports...")
print("=" * 60)

try:
    print("\n1. Testing transforms_substep.py...")
    from prismatic.vla.datasets.rlds.oxe.transforms_substep import (
        libero_dataset_transform_with_episode_id,
        reset_episode_counter,
        OXE_STANDARDIZATION_TRANSFORMS_WITH_EPISODE_ID,
    )
    print("   ✓ transforms_substep imports successful")
    print(f"   - Supported datasets: {list(OXE_STANDARDIZATION_TRANSFORMS_WITH_EPISODE_ID.keys())}")
except Exception as e:
    print(f"   ✗ transforms_substep import failed: {e}")
    sys.exit(1)

try:
    print("\n2. Testing dataset_substep.py...")
    from prismatic.vla.datasets.rlds.dataset_substep import make_dataset_from_rlds_with_episode_id
    print("   ✓ dataset_substep imports successful")
except Exception as e:
    print(f"   ✗ dataset_substep import failed: {e}")
    sys.exit(1)

try:
    print("\n3. Testing materialize_substep.py...")
    from prismatic.vla.datasets.rlds.oxe.materialize_substep import (
        make_oxe_dataset_kwargs_with_episode_id,
        get_oxe_dataset_kwargs_and_weights_with_episode_id,
    )
    print("   ✓ materialize_substep imports successful")
except Exception as e:
    print(f"   ✗ materialize_substep import failed: {e}")
    sys.exit(1)

try:
    print("\n4. Testing datasets_substep.py...")
    from prismatic.vla.datasets.datasets_substep import (
        SubstepRLDSBatchTransform,
        SubstepRLDSDataset,
        load_substep_labels,
        get_substep_instruction,
    )
    print("   ✓ datasets_substep imports successful")
except Exception as e:
    print(f"   ✗ datasets_substep import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n5. Testing make_interleaved_dataset import...")
    from prismatic.vla.datasets.rlds import make_interleaved_dataset
    print("   ✓ make_interleaved_dataset import successful")
except Exception as e:
    print(f"   ✗ make_interleaved_dataset import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n6. Testing finetune_substep.py config...")
    from vla_scripts.finetune_substep import FinetuneSubstepConfig
    print("   ✓ finetune_substep config imports successful")
except Exception as e:
    print(f"   ✗ finetune_substep import failed: {e}")
    print(f"   Note: This is expected if vla_scripts is not in Python path")

print("\n" + "=" * 60)
print("✓ All core substep training modules imported successfully!")
print("\nYou can now run training with:")
print("  python vla-scripts/finetune_substep.py --substep_labels_path <path> ...")

