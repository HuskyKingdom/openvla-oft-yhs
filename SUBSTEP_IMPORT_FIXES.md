# Substep Training Import Fixes

This document tracks all import issues that were identified and fixed during the implementation of the substep training pipeline.

## Fixed Import Issues

### 1. ImageTransform Import Error

**Error:**
```
ImportError: cannot import name 'ImageTransform' from 'prismatic.util.data_utils'
```

**Root Cause:**
`ImageTransform` was being imported from the wrong module.

**Fix:**
In `prismatic/vla/datasets/datasets_substep.py`:
```python
# ❌ WRONG
from prismatic.util.data_utils import ImageTransform

# ✅ CORRECT
from prismatic.models.backbones.vision import ImageTransform
```

**File Modified:** `prismatic/vla/datasets/datasets_substep.py` (line 21)

---

### 2. make_interleaved_dataset Import Error

**Error:**
```
ImportError: cannot import name 'make_interleaved_dataset' from 'prismatic.vla.datasets.rlds.oxe'
```

**Root Cause:**
`make_interleaved_dataset` is located in the parent `rlds` module, not the `oxe` submodule.

**Fix:**
In `prismatic/vla/datasets/datasets_substep.py`:
```python
# ❌ WRONG
from prismatic.vla.datasets.rlds.oxe import make_interleaved_dataset

# ✅ CORRECT
from prismatic.vla.datasets.rlds import make_interleaved_dataset
```

**File Modified:** `prismatic/vla/datasets/datasets_substep.py` (line 397)

---

## Correct Import Patterns

For reference, here are the correct import patterns used throughout the substep training pipeline:

### Core VLA Imports
```python
from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.overwatch import initialize_overwatch
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import (
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    NUM_ACTIONS_CHUNK,
    IGNORE_INDEX,
)
```

### Dataset Imports
```python
from prismatic.vla.datasets.datasets import rephrase
from prismatic.vla.datasets.rlds import make_interleaved_dataset
from prismatic.vla.datasets.rlds.oxe.mixtures import OXE_NAMED_MIXTURES
```

### Substep-Specific Imports
```python
from prismatic.vla.datasets.datasets_substep import (
    SubstepRLDSBatchTransform,
    SubstepRLDSDataset,
    load_substep_labels,
    get_substep_instruction,
)

from prismatic.vla.datasets.rlds.oxe.transforms_substep import (
    libero_dataset_transform_with_episode_id,
    reset_episode_counter,
    OXE_STANDARDIZATION_TRANSFORMS_WITH_EPISODE_ID,
)

from prismatic.vla.datasets.rlds.oxe.materialize_substep import (
    make_oxe_dataset_kwargs_with_episode_id,
    get_oxe_dataset_kwargs_and_weights_with_episode_id,
)

from prismatic.vla.datasets.rlds.dataset_substep import (
    make_dataset_from_rlds_with_episode_id,
)
```

---

## Verification

To verify all imports are working correctly, run:

```bash
python test_substep_imports.py
```

Expected output:
```
Testing substep training module imports...
============================================================

1. Testing transforms_substep.py...
   ✓ transforms_substep imports successful
   - Supported datasets: ['libero_spatial_no_noops', ...]

2. Testing dataset_substep.py...
   ✓ dataset_substep imports successful

3. Testing materialize_substep.py...
   ✓ materialize_substep imports successful

4. Testing datasets_substep.py...
   ✓ datasets_substep imports successful

5. Testing make_interleaved_dataset import...
   ✓ make_interleaved_dataset import successful

6. Testing finetune_substep.py config...
   ✓ finetune_substep config imports successful

============================================================
✓ All core substep training modules imported successfully!
```

---

## Common Import Debugging Steps

If you encounter import errors:

1. **Run the test script first:**
   ```bash
   python test_substep_imports.py
   ```
   This will identify which specific import is failing.

2. **Check the error message carefully:**
   - Note the exact module path in the error
   - Note what it's trying to import

3. **Compare with original code:**
   - Check how the same import is done in `prismatic/vla/datasets/datasets.py`
   - Check the module structure in `prismatic/vla/datasets/rlds/`

4. **Verify file exists:**
   ```bash
   ls -la prismatic/vla/datasets/rlds/oxe/
   ```

5. **Check `__init__.py` files:**
   Sometimes imports fail if `__init__.py` doesn't properly expose the function.

---

## Module Structure Reference

```
prismatic/
├── models/
│   └── backbones/
│       ├── llm/
│       │   └── prompting.py  (PromptBuilder)
│       └── vision/
│           └── __init__.py  (ImageTransform)
├── overwatch.py  (initialize_overwatch)
└── vla/
    ├── action_tokenizer.py  (ActionTokenizer)
    ├── constants.py  (ACTION_PROPRIO_NORMALIZATION_TYPE, etc.)
    └── datasets/
        ├── datasets.py  (rephrase, RLDSBatchTransform, RLDSDataset)
        ├── datasets_substep.py  (SubstepRLDSBatchTransform, SubstepRLDSDataset)
        └── rlds/
            ├── __init__.py  (make_interleaved_dataset)
            ├── dataset.py  (make_dataset_from_rlds)
            ├── dataset_substep.py  (make_dataset_from_rlds_with_episode_id)
            └── oxe/
                ├── __init__.py
                ├── mixtures.py  (OXE_NAMED_MIXTURES)
                ├── materialize.py  (make_oxe_dataset_kwargs)
                ├── materialize_substep.py  (make_oxe_dataset_kwargs_with_episode_id)
                ├── transforms.py  (OXE_STANDARDIZATION_TRANSFORMS)
                └── transforms_substep.py  (OXE_STANDARDIZATION_TRANSFORMS_WITH_EPISODE_ID)
```

---

## Status

✅ All known import issues have been resolved as of the latest update.

If you encounter any new import errors, please:
1. Run `python test_substep_imports.py` to diagnose
2. Check this document for similar issues
3. Verify you're using the latest versions of all files

