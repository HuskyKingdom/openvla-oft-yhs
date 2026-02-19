# InfoBot-VLA Research Log

## Project: Information Bottleneck Constrained VLA for ICML

### Core Innovation
Address $H(L|V) \approx 0$ problem by forcing VLA to predict actions through a language bottleneck,
preventing direct visual overfitting.

### Key Insight
Current VLA: Visual → Action (ignores language)
InfoBot-VLA: Visual → Language Bottleneck → Action (must use language)

### Implementation Status

#### Day 1 (Current)
- [x] Created InfoBot bottleneck architecture modules
  - `InfoBottleneckLayer`: Cross-attention based compression
  - `LanguageConditionedBottleneck`: Language-parameterized bottleneck
  - `MutualInformationEstimator`: InfoNCE-based MI estimation
  - `InfoBotActionHead`: Action prediction from bottleneck features
- [x] Created standalone training script `finetune_infobot.py`
- [x] Committed changes and pushed to GitHub
- [x] **Training job submitted to AMD cluster**
  - Job ID: 6029 (updated to mi3508xl partition)
  - Partition: mi3508xl (8x MI350X)
  - Runtime: 12 hours
  - Status: **SUBMITTED** ✅

### Implementation Plan
1. [x] Create InfoBotVLA architecture with bottleneck layer
2. [x] Implement mutual information regularization loss
3. [x] Integrate with existing APD dataset
4. [x] Train on AMD cluster (Job 6026 running)
5. [ ] Evaluate on LIBERO-PRO benchmark

### Technical Details

#### Loss Function
$$\mathcal{L}_{total} = \mathcal{L}_{action} + \beta \cdot I(Z_v; V | L)$$

Where:
- $\mathcal{L}_{action}$: L1 action prediction loss
- $I(Z_v; V | L)$: Conditional mutual information (minimized)
- $\beta = 0.1$ (tunable hyperparameter)

#### Architecture Components
1. **Bottleneck Layer**: Compresses visual features from (B, N_v, D) → (B, K, D_b)
   - K=8 bottleneck tokens (configurable)
   - D_b=256 compressed dimension
   - Cross-attention with language conditioning

2. **MI Estimator**: InfoNCE-based contrastive estimation
   - Projects bottleneck and visual features to 128-dim
   - Temperature-scaled similarity matrix
   - Symmetric InfoNCE loss

3. **Action Head**: Predicts actions from bottleneck + language
   - Cross-attention: action queries → context
   - MLP prediction per action dimension

### Expected Outcomes
- Higher attention scores on instruction tokens
- Better robustness on Position/Semantic perturbations
- Lower success rate on empty prompt (desired: should fail)

### Timeline
- Day 1: Architecture implementation ✅
- Day 2: Training script + SLURM job
- Day 3-4: Training on AMD cluster
- Day 5-6: Evaluation on LIBERO-PRO
- Day 7: Analysis and paper writing
