# Stiffness Policy Learning: Comprehensive Analysis

**Date**: 2025-11-17  
**Analysis**: Three approaches to diagnose poor model performance (R² < 0)

---

## 1. Leave-One-Trial-Out Cross-Validation (LOTO-CV)

### Setup
- **Model**: Behavior Cloning (BC, MLP 256-dim, 3 layers)
- **Mode**: Unified (20D observations → 9D stiffness)
- **Training**: 50 epochs per fold
- **Folds**: 15 (one per demo)

### Results

| Metric | Mean ± Std | Min | Max |
|--------|------------|-----|-----|
| **RMSE** | 101.47 ± 29.85 | 53.98 | 150.50 |
| **MAE** | 77.00 ± 21.96 | 43.71 | 121.31 |
| **R²** | **-23.89 ± 24.82** | **-89.45** | **-1.84** |

### Key Findings
- **All folds have negative R²**: Model performs worse than predicting the mean
- **High variance across folds**: Some trials are easier (R²=-1.8) vs. very hard (R²=-89.4)
- **Train/test distribution mismatch**: Each demo has unique stiffness patterns that don't generalize
- **Random split (75/25) was NOT the issue**: LOTO-CV confirms the problem persists

---

## 2. Temporal Model Analysis

### Setup
Tested sequential models to capture temporal dependencies:
1. **LSTM-GMM**: LSTM encoder + GMM decoder (sequence_window=10, 100 epochs)
2. **Diffusion**: Conditional denoising (100 epochs)

### Results

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| **BC** (baseline) | 76.20 | 59.45 | -1.49 |
| **LSTM-GMM** | 96.74 | 70.54 | **-2.55** |
| **Diffusion (DDPM)** | 71.90 | 57.31 | -1.25 |
| **Diffusion (DDIM)** | 69.46 | 55.29 | **-1.10** |

### Key Findings
- **Temporal context does NOT help**: LSTM-GMM performs worse than BC
- **Diffusion slightly better** but still R² < 0
- **Stiffness is not primarily a temporal sequence problem**: Current observation lacks sufficient information

---

## 3. Feature Correlation Analysis

### Observation-Stiffness Correlations

**Top 20 Strongest Correlations:**
```
 1. s1_fy           <-> if_k1   : +0.4728
 2. ee_if_px        <-> if_k1   : -0.4026
 3. s1_fy           <-> mf_k2   : +0.3556
 4. s3_fy           <-> th_k3   : -0.3309
 5. s1_fx           <-> mf_k3   : +0.3223
... (rest < 0.32)
```

**Summary Statistics:**
- **Maximum |correlation|**: 0.47 (s1_fy ↔ if_k1)
- **Mean |correlation|**: **0.14**
- **Median |correlation|**: **0.14**

**Per-Action Max Correlations:**
```
th_k1: -0.16 (with s2_fz)    ← VERY WEAK
th_k2: -0.21 (with ee_if_px)
th_k3: -0.33 (with s3_fy)
if_k1: +0.47 (with s1_fy)    ← STRONGEST
if_k2: -0.23 (with s3_fy)
if_k3: +0.31 (with s1_fy)
mf_k1: +0.25 (with deform_ecc)
mf_k2: +0.36 (with s1_fy)
mf_k3: +0.32 (with s1_fx)
```

### Variance Analysis

**Observations:**
- Forces (s1/s2/s3_fx/fy/fz): mean=-2 to +1, std=0.6 to 2.0 N
- Deform: circ=0.85±0.03, ecc=0.38±0.12
- EE positions: std=0.01-0.02 m (very stable)

**Actions (Stiffness):**
- All k1/k2/k3: mean=173-260, std=65-115
- Range: 50-680 (K_MIN clipping active)

### Key Findings
- **Correlations are VERY WEAK** (mean 0.14): Current observations explain only ~2% of stiffness variance (0.14²)
- **No single feature has strong predictive power**: Even the best (s1_fy → if_k1, r=0.47) explains only 22%
- **TH finger particularly weak**: th_k1 max correlation is only -0.16
- **Missing information**: Stiffness likely depends on:
  - **EMG activation history** (not available in observation)
  - **Task intent** (grasp type, force target)
  - **Proprioceptive feedback** (muscle fatigue, comfort)

---

## Architectural Comparison: Unified vs. Per-Finger

### Unified Model (20D → 9D)
**50 epochs**: RMSE=76.20, MAE=59.45, R²=-1.49  
**200 epochs**: RMSE=76.30, MAE=58.96, R²=-1.55

### Per-Finger Models (3 × 8D → 3D)
**50 epochs**:
- TH: RMSE=89.15, MAE=69.36, R²=-4.50
- IF: RMSE=108.21, MAE=70.27, R²=-6.34
- MF: RMSE=119.88, MAE=95.94, R²=-3.88

**200 epochs** (overfitting):
- TH: RMSE=95.23, MAE=72.82, R²=-5.32
- IF: RMSE=121.20, MAE=72.85, R²=-7.56
- MF: RMSE=117.18, MAE=91.06, R²=-3.64

### Conclusion
**Unified model is better** despite sensor-finger independence, because:
1. Deform_circ/ecc are shared features providing global context
2. Per-finger models lose cross-finger coordination signals
3. More parameters help even with weak correlations

---

## Root Cause Diagnosis

### Why All Models Fail (R² < 0)?

1. **Insufficient Observability**: Current sensors (force, deform, EE position) have **mean correlation = 0.14** with target stiffness
   - Missing: EMG activation, task intent, user preference

2. **Stiffness Generation Process**: 
   ```
   EMG → T_F (projection) → Force → Stiffness (K = K_INIT + T_K @ emg_proj)
   ```
   - **Inverse problem is ill-posed**: Multiple EMG patterns → same force
   - **K_MIN=50 clipping** introduces discontinuities
   - **No direct force-stiffness mapping** in biological control

3. **Data Distribution Issues**:
   - High inter-trial variance (LOTO R² std = 24.8)
   - Each demo has unique stiffness strategy
   - Train/test distribution mismatch even with LOTO-CV

4. **Model Capacity vs. Signal Strength**:
   - BC (256-dim MLP) is sufficient for the task complexity
   - Problem is **signal-to-noise ratio**, not model architecture
   - More parameters or complex models (LSTM, Diffusion) don't help

---

## Recommendations

### ❌ What DOESN'T Work
1. ❌ **Per-finger models**: Worse than unified (lose shared context)
2. ❌ **More epochs**: Leads to overfitting (R² worsens)
3. ❌ **Temporal models**: LSTM-GMM R²=-2.5 < BC R²=-1.5
4. ❌ **Better architectures**: Diffusion only marginally better

### ✅ What MIGHT Work

1. **Add EMG Features** (if available):
   - Direct EMG amplitudes per channel
   - EMG-derived muscle synergies
   - Expected correlation: 0.7-0.9 (strong)

2. **Task/Context Labels**:
   - Grasp type (power/precision/pinch)
   - Target force magnitude
   - Object properties (size, compliance)

3. **Reformulate Problem**:
   - **Option A**: Predict force → stiffness (if force target known)
   - **Option B**: Predict stiffness change Δk instead of absolute k
   - **Option C**: Classify stiffness regime (low/medium/high) instead of regression

4. **Data Augmentation**:
   - Collect more diverse trials (different objects, forces)
   - Explicit task labels during data collection
   - Controlled force ramps to build force-stiffness maps

5. **Physics-Informed Priors**:
   - Encode impedance control theory: k ∝ √(force_error)
   - Constrain predictions to physiologically plausible ranges
   - Multi-task learning: predict force AND stiffness jointly

---

## Conclusion

**Current stiffness policy learning is fundamentally limited by observation insufficiency.**

The **correlation analysis (mean r=0.14)** reveals that force, deformation, and EE position alone **cannot predict EMG-derived stiffness** with any model architecture. The problem is not:
- ❌ Train/test split (LOTO-CV confirmed)
- ❌ Model architecture (BC = LSTM = Diffusion)
- ❌ Training procedure (50-200 epochs tested)

The problem IS:
- ✅ **Missing EMG information** (stiffness comes from muscle activation)
- ✅ **Ill-posed inverse problem** (force ← stiffness, many-to-one mapping)
- ✅ **No causal pathway** (sensors → stiffness is indirect)

**Next steps**: Either (1) add EMG observations, (2) reformulate as classification/change-prediction, or (3) accept stiffness as unobservable and use force-only control.
