# How label_substeps.py Handles Timestep Substep Labeling

## Overview

`label_substeps.py` automatically labels each timestep in RLDS episodes with substep information by analyzing robot telemetry data (gripper states, end-effector positions) and mapping detected actions to APD plan steps.

---

## Complete Pipeline

```
RLDS Episode → Gripper Analysis → Action Detection → Temporal Expansion → APD Mapping → Output
     ↓               ↓                    ↓                   ↓                ↓
  Telemetry    Find transitions    Label segments     Expand ranges    Assign substeps
```

---

## Step-by-Step Process

### 1. **Gripper Transition Detection**

**Function**: `detect_gripper_transitions()`

**Input**: Gripper states array `(T, 2)` where each timestep has gripper position

**Method**: Relative change detection (not absolute threshold)

```python
# Uses windowed comparison
window = 8
for t in range(window, T - window):
    before_window = gripper_smooth[t-window:t]
    after_window = gripper_smooth[t+1:t+window+1]

    # Detect closing: before > after (PICK)
    if before_avg > after_avg + threshold:
        pick_moments.append(t)

    # Detect opening: after > before (PLACE)
    elif after_avg > before_avg + threshold * 0.4:
        place_moments.append(t)
```

**Output**:
- `pick_moments`: `[45, 178]` (timesteps where gripper closes)
- `place_moments`: `[112, 245]` (timesteps where gripper opens)

**Key Insight**: Uses relative change instead of absolute threshold to handle different object sizes!

---

### 2. **Temporal Range Expansion**

**Functions**: `expand_pick_range()`, `expand_place_range()`

**Purpose**: A single gripper transition moment doesn't capture the full action. We need to expand temporally to include:
- **Pick**: Approach descent + grasp + lift
- **Place**: Descent + release + retract

**Expansion Strategy**:

```python
# Pick expansion
pick_start = pick_t - pick_expand_backward   # Default: -30 steps
pick_end = pick_t + pick_expand_forward      # Default: +20 steps

# Place expansion
place_start = place_t - place_expand_backward  # Default: -100 steps (larger!)
place_end = place_t + place_expand_forward     # Default: +80 steps
```

**Why larger for place?**
- Place includes the entire "move to target" phase before releasing
- Carrying the object from pick location to target location

**Example**:
```
Pick moment at t=50:
  → Pick range: [20, 70]  (30 before, 20 after)

Place moment at t=120:
  → Place range: [20, 200]  (100 before, 80 after)
```

---

### 3. **Action Labeling**

**Function**: `label_actions()`

**Process**:
1. Initialize all timesteps as `"move"`
2. Detect all pick moments → Expand ranges → Label as `"pick"`
3. Detect all place moments → Expand ranges → Label as `"place"`
4. Remaining timesteps stay as `"move"`

**Important**: Later labels overwrite earlier ones (handle overlaps)

**Output**: Array of action types for each timestep
```python
action_labels = [
    "move", "move", ...,          # t=0-19
    "pick", "pick", ...,          # t=20-69
    "place", "place", ...,        # t=20-199  (overwrites pick!)
    "move", "move", ...           # t=200+
]
```

---

### 4. **Block-Based Temporal Assignment**

**Function**: `map_timesteps_to_apd_steps()`

**Key Innovation**: Merges Move segments INTO Pick/Place blocks

**Strategy**:
- Move before Pick → Part of Pick preparation
- Move after Pick (carrying object) → Part of Place preparation
- **Only output Pick and Place timesteps** (Move is merged!)

**Block Creation**:

```python
for cycle_idx in range(num_cycles):
    # Pick block
    if cycle_idx < len(pick_moments):
        pick_start = blocks[-1]['end'] if len(blocks) > 0 else 0
        pick_end = min(T, pick_t + pick_expand_forward)

        blocks.append({
            'start': pick_start,
            'end': pick_end,
            'type': 'pick',
            'cycle': cycle_idx,
            'core_moment': pick_t
        })

    # Place block
    if cycle_idx < len(place_moments):
        place_start = blocks[-1]['end']
        place_end = min(T, place_t + place_expand_forward)

        blocks.append({
            'start': place_start,
            'end': place_end,
            'type': 'place',
            'cycle': cycle_idx,
            'core_moment': place_t
        })
```

**Example with dual pick-place**:

```
Episode: T=300 timesteps
Pick moments: [50, 180]
Place moments: [120, 250]

Blocks created:
  Block 0: Pick  [0, 70]     cycle=0  (includes initial move)
  Block 1: Place [70, 200]   cycle=0  (includes move to target)
  Block 2: Pick  [200, 200]  cycle=1  (minimal, starts immediately)
  Block 3: Place [200, 270]  cycle=1

Timesteps labeled:
  t=0-69:    Pick (cycle 0)
  t=70-199:  Place (cycle 0)
  t=200-269: Place (cycle 1)  (Pick cycle 1 has no timesteps because blocks don't overlap!)
```

**Non-overlap guarantee**: Each block starts where previous ended!

---

### 5. **APD Step Assignment**

**Function**: `map_timesteps_to_apd_steps()` (continued)

**Matching strategy**:

```python
# Extract pick/place steps from APD plan
apd_pick_steps = [step for step in apd_plan
                  if 'pick' in step['subgoal'].lower()]
apd_place_steps = [step for step in apd_plan
                   if 'place' in step['subgoal'].lower()]

# Assign to blocks by cycle index
for block in blocks:
    if block['type'] == 'pick':
        cycle = block['cycle']
        if cycle < len(apd_pick_steps):
            block['apd_step'] = apd_pick_steps[cycle]['subgoal']

    elif block['type'] == 'place':
        cycle = block['cycle']
        if cycle < len(apd_place_steps):
            block['apd_step'] = apd_place_steps[cycle]['subgoal']
```

**Example**:

APD Plan:
```json
[
  {"subgoal": "Pick up the cream cheese"},
  {"subgoal": "Place cream cheese in the bowl"}
]
```

Blocks:
```
Block 0 (pick, cycle=0)  → "Pick up the cream cheese"
Block 1 (place, cycle=0) → "Place cream cheese in the bowl"
```

---

### 6. **Timestep Label Generation**

**Final output format**:

```python
for block in blocks:
    for t in range(block['start'], block['end']):
        timestep_labels.append({
            "timestep": t,
            "action": block['type'],
            "APD_step": block['apd_step'],
            "cycle": block['cycle'],
            "is_substep_end": (t == block['end'] - 1)  # EOS marker!
        })
```

**Example output**:

```json
[
  {
    "timestep": 0,
    "action": "pick",
    "APD_step": "Pick up the cream cheese",
    "cycle": 0,
    "is_substep_end": false
  },
  {
    "timestep": 1,
    "action": "pick",
    "APD_step": "Pick up the cream cheese",
    "cycle": 0,
    "is_substep_end": false
  },
  ...
  {
    "timestep": 69,
    "action": "pick",
    "APD_step": "Pick up the cream cheese",
    "cycle": 0,
    "is_substep_end": true     ← EOS marker for substep boundary!
  },
  {
    "timestep": 70,
    "action": "place",
    "APD_step": "Place cream cheese in the bowl",
    "cycle": 0,
    "is_substep_end": false
  }
]
```

**Key field**: `is_substep_end` marks the last timestep of each substep!

---

## Key Design Decisions

### 1. **Relative Change Detection (Not Absolute Threshold)**

**Why?**
- Different objects have different sizes → different gripper widths
- Absolute threshold (e.g., "gripper > 0.04 = open") fails for small objects

**Solution**: Detect significant changes in gripper value
```python
# Good: Detects 0.01→0.02 change (small object)
# Good: Detects 0.02→0.04 change (large object)
```

---

### 2. **Merge Move Into Pick/Place**

**Why?**
- Move alone has no clear boundaries
- "Move to object" is part of Pick preparation
- "Move to target" is part of Place preparation

**Solution**: Only output Pick and Place labels, Move is absorbed

**Benefit**: Cleaner substep boundaries, easier EOS detection

---

### 3. **Block-Based Non-Overlapping Assignment**

**Why?**
- Avoid confusion from overlapping labels
- Ensure each timestep has exactly ONE substep

**Solution**: Each block starts where previous ended

```python
pick_start = blocks[-1]['end'] if len(blocks) > 0 else 0
```

**Guarantees**:
- No gaps (all timesteps covered)
- No overlaps (mutually exclusive)

---

### 4. **Large Place Backward Expansion**

**Why?**
- Place includes the entire "carrying" phase
- From pick location → to target location → descend → release

**Default**: `place_expand_backward = 100` (vs `pick_expand_backward = 30`)

**Example**:
```
Pick at t=50:  [20, 70]
Place at t=120: [20, 200]  ← Starts from same place as pick!
                             (Merged into continuous block)
```

---

## Comparison: label_substeps.py vs label_substeps_so101.py

| Aspect | LIBERO (label_substeps.py) | SO101 (label_substeps_so101.py) |
|--------|---------------------------|--------------------------------|
| **Detection Method** | Relative change (windowed) | Relative change (windowed) | ← Identical |
| **label_actions() step** | ✅ Intermediate labeling | ✅ Intermediate labeling | ← Identical |
| **Move Handling** | Merged into Pick/Place | Merged into Pick/Place | ← Identical |
| **Block Strategy** | Non-overlapping continuous | Non-overlapping continuous | ← Identical |
| **Block end clip** | `next_pick_t - pick_expand_backward` | `next_pick_t - pick_expand_backward` | ← Identical |
| **Dual Cycle** | ✅ Automatic | ✅ Automatic | ← Identical |
| **APD Matching** | Keyword-based (`subgoal` field) | `action_type` + `cycle` field | SO101 APD schema |
| **Move prep context** | `apd_prep_step` (optional) | `APD_prep_step` in output | SO101 has explicit move steps |
| **EOS Marker** | ✅ `is_substep_end` | ✅ `is_substep_end` | ← Identical |

---

## Example Walkthrough

### Input Episode

```
T = 200 timesteps
Gripper states:
  t=0-40:   [0.03, 0.03, ...]  (open)
  t=41-45:  [0.03, 0.025, 0.02, 0.015, 0.01]  (closing)
  t=46-100: [0.01, 0.01, ...]  (closed, carrying)
  t=101-105: [0.01, 0.015, 0.02, 0.025, 0.03]  (opening)
  t=106-200: [0.03, 0.03, ...]  (open)
```

### Step 1: Detect Transitions

```
Pick moment: t=45 (closing detected at window comparison)
Place moment: t=105 (opening detected)
```

### Step 2: Expand Ranges

```
Pick:  [45-30, 45+20] = [15, 65]
Place: [105-100, 105+80] = [5, 185]
```

### Step 3: Create Blocks (Non-overlapping)

```
Block 0 (Pick):  [0, 65]    (starts from episode start)
Block 1 (Place): [65, 185]  (starts where Block 0 ended)
```

### Step 4: Assign APD Steps

```
APD Plan: [
  {"subgoal": "Pick up the cream cheese"},
  {"subgoal": "Place cream cheese in the bowl"}
]

Block 0 → "Pick up the cream cheese"
Block 1 → "Place cream cheese in the bowl"
```

### Step 5: Generate Labels

```json
[
  {"timestep": 0, "action": "pick", "APD_step": "Pick up the cream cheese", "is_substep_end": false},
  {"timestep": 1, "action": "pick", "APD_step": "Pick up the cream cheese", "is_substep_end": false},
  ...
  {"timestep": 64, "action": "pick", "APD_step": "Pick up the cream cheese", "is_substep_end": true},
  {"timestep": 65, "action": "place", "APD_step": "Place cream cheese in the bowl", "is_substep_end": false},
  ...
  {"timestep": 184, "action": "place", "APD_step": "Place cream cheese in the bowl", "is_substep_end": true}
]
```

**Result**: 185 labeled timesteps (t=0-184), 15 unlabeled (t=185-199)

---

## Why This Matters for Training

### During Training (SubstepRLDSBatchTransform)

For each training sample at timestep `t`:

```python
# Query substep label
episode_id = 5
timestep = 42
substep_instruction, is_eos = get_substep_instruction(
    substep_labels, dataset_name, task, episode_id, timestep
)

# Result:
# substep_instruction = "Pick up the cream cheese"
# is_eos = False
```

**Model sees**:
- **Image**: Current observation at t=42
- **Instruction**: "Pick up the cream cheese" (NOT full task instruction!)
- **Action**: Ground truth action at t=42
- **EOS**: False (not end of substep)

At t=64:
- **Instruction**: "Pick up the cream cheese"
- **EOS**: True ← Signals substep boundary!

**Benefit**: Model learns fine-grained language conditioning!

---

## Summary

`label_substeps.py` performs **automatic temporal segmentation** of robot episodes:

1. **Detect** gripper transitions (pick/place moments)
2. **Expand** temporal ranges around transitions
3. **Create** non-overlapping blocks (merge move into pick/place)
4. **Map** blocks to APD plan substeps
5. **Generate** per-timestep labels with EOS markers

**Key innovation**: Block-based assignment ensures clean boundaries and enables EOS detection for substep-aware training!
