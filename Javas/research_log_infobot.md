# InfoBot-VLA Research Log

## Project: Information Bottleneck Constrained VLA for ICML

### Core Innovation
Address $H(L|V) \approx 0$ problem by forcing VLA to predict actions through a language bottleneck,
preventing direct visual overfitting.

### Key Insight
Current VLA: Visual → Action (ignores language)
InfoBot-VLA: Visual → Language Bottleneck → Action (must use language)

### Implementation Status

#### Day 1-2 (Completed)
- [x] Created InfoBot bottleneck architecture modules
  - `InfoBottleneckLayer`: Cross-attention based compression
  - `LanguageConditionedBottleneck`: Language-parameterized bottleneck
  - `MutualInformationEstimator`: InfoNCE-based MI estimation
  - `InfoBotActionHead`: Action prediction from bottleneck features
- [x] Created standalone training script `finetune_infobot.py`
- [x] Committed changes and pushed to GitHub

#### Day 2-3 (Current - Training In Progress)
- [x] **Training job submitted to AMD cluster**
  - Job ID: 6032 (final working version)
  - Partition: mi3508xl (8x MI350X)
  - Runtime: 12 hours
  - Status: **TRAINING SUCCESSFULLY** ✅
  - Current Step: 370+
  - Loss: ~0.30-0.40 (stable)
  
**Issues Encountered & Fixed:**
1. Wrong partition (mi3008xl → mi3508xl)
2. DDP issues with MI estimator (moved outside DDP wrapper)
3. NaN losses from MI estimator (disabled for now)
4. SLURM script syntax errors

**Current Configuration:**
- Bottleneck: cross_attn, dim=256, tokens=8
- MI regularization: DISABLED (caused NaN)
- Training with L1 action loss only
- Beta = 0.0

### Implementation Plan
1. [x] Create InfoBotVLA architecture with bottleneck layer
2. [x] Implement mutual information regularization loss
3. [x] Integrate with existing APD dataset
4. [x] Train on AMD cluster (Job 6032 running successfully)
5. [ ] Evaluate on LIBERO-PRO benchmark (pending training completion)

### Technical Details

#### Loss Function
$$\mathcal{L}_{total} = \mathcal{L}_{action} + \beta \cdot I(Z_v; V | L)$$

Where:
- $\mathcal{L}_{action}$: L1 action prediction loss
- $I(Z_v; V | L)$: Conditional mutual information (minimized) - **DISABLED**
- $\beta = 0.0$ (MI loss disabled due to NaN)

#### Architecture Components
1. **Bottleneck Layer**: Compresses visual features from (B, N_v, D) → (B, K, D_b)
   - K=8 bottleneck tokens (configurable)
   - D_b=256 compressed dimension
   - Cross-attention with language conditioning

2. **MI Estimator**: InfoNCE-based contrastive estimation (**DISABLED**)
   - Projects bottleneck and visual features to 128-dim
   - Temperature-scaled similarity matrix
   - Symmetric InfoNCE loss
   - Issue: Causes NaN losses even with numerical stability fixes

3. **Action Head**: Predicts actions from bottleneck + language
   - Cross-attention: action queries → context
   - MLP prediction per action dimension

### Expected Outcomes
- Higher attention scores on instruction tokens
- Better robustness on Position/Semantic perturbations
- Lower success rate on empty prompt (desired: should fail)

### Timeline
- Day 1: Architecture implementation ✅
- Day 2: Training script + SLURM job ✅
- Day 3: Training on AMD cluster (in progress - stable at step 370+) 🔄
- Day 4-5: Continue training to 200K steps
- Day 6: Evaluation on LIBERO-PRO
- Day 7: Analysis and paper writing
