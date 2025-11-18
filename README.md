
```
cat /sys/bus/usb-serial/devices/ttyUSB0/latency_timer 
sudo vi /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
   # change to 16 -> 1
   
```


## Stiffness Policy Learning Pipeline

### 1. Generate Stiffness Profiles
Extract stiffness from EMG and force data:
```bash
python3 scripts/generate_stiffness_profiles.py
```

### 2. Data Augmentation
Augment demonstrations with physics-aware noise:
```bash
cd /home/songwoo/ros2_ws/icra2025/src/hri_falcon_robot_bridge

# 기본 설정 (5x augmentation)
python3 scripts/augment_demonstration_data.py --num-augment 5

# 또는 더 aggressive (noise 증가)
python3 scripts/augment_demonstration_data.py \
  --num-augment 5 \
  --noise-force 0.03 \
  --noise-stiffness 0.08 \
  --noise-deform 0.02
```
- Input: 15 original demonstrations
- Output: 90 total demonstrations (15 original + 75 augmented)
- Applies Gaussian noise to forces, stiffness, deformation, end-effector positions
- Preserves physical constraints (K ≥ 1 N/m)


### 3. Train Stiffness Prediction Models
Train all models (BC, GMM, GMR, LSTM-GMM, Diffusion, IBC, GP, MDN):

```bash
cd /home/songwoo/ros2_ws/icra2025/src/hri_falcon_robot_bridge
python3 scripts/3_model_learning/run_stiffness_policy_benchmarks.py \
  --stiffness-dir outputs/analysis/stiffness_profiles_global_tk \
  --mode per-finger \
  --models all \
  --bc-epochs 200 \
  --diffusion-epochs 200 \
  --augment \
  --augment-num 1 \
  --augment-noise-force 0.015 \
  --augment-noise-stiffness 0.04 \
  --augment-temporal-jitter 1
```

**Available Models:**
- **GMM/GMR**: Gaussian Mixture Model/Regression (multimodal, uncertainty)
- **BC**: Behavior Cloning (MLP regression, baseline)
- **LSTM-GMM**: Sequence model with mixture outputs
- **Diffusion**: Diffusion Policy (C/T variants)
- **IBC**: Implicit Behavior Cloning (energy-based)
- **GP**: Gaussian Process Regression (small data + uncertainty)
- **MDN**: Mixture Density Network (GMM + deep learning)

**Tip:** Line continuation uses backslash (\) as the very last character — no trailing spaces. To use global-T_K, change `--stiffness-dir` to `outputs/analysis/stiffness_profiles_global_tk`.

### GP 대용량 처리 메모 (중요)
- Exact GP는 메모리 O(N^2), 시간 O(N^3) 복잡도로, 수십만 샘플에서 메모리 에러가 납니다.
- 본 레포 GPBaseline은 자동으로 학습용 샘플을 최대 4k로 랜덤 서브샘플링하고, 예측은 배치로 분할해 메모리 폭주를 막습니다.
- 설정은 `scripts/3_model_learning/configs/stiffness_policy/gp.yaml`에서 조정 가능합니다:
  - `max_train_points`: 학습 최대 샘플 수 (기본 4000)
  - `batch_predict_size`: 배치 예측 크기 (기본 2048)
  - `subsample_strategy`: random (추후 kmeans 등 추가 가능)
  
대용량에서 GP가 너무 느리면, `--stride`를 늘려 입력 데이터 자체를 다운샘플링하거나, GP를 제외하고 다른 모델군(MDN/BC/Diffusion 등)을 활용하세요.
```
오프라인 증강

장점
재현성 높음: 한 번 생성하면 모든 실험에서 동일 데이터 사용.
학습 속도 안정: 학습 중 증강 계산 오버헤드가 없음.
복잡/무거운 물리 연산을 미리 계산해 둘 수 있음.
디버깅 쉬움: 실제 CSV가 있으니 케이스 재현이 쉬움.
단점
디스크 용량 증가.
데이터 split 주의 필요: 원본과 증강본이 서로 다른 split으로 섞이면 데이터 누수 위험.
다양성이 고정됨: 만든 만큼만 다양함.
온더플라이 증강

장점
디스크 추가 사용 없음.
실행마다(또는 매 epoch) 조금씩 다른 샘플 생성 → 다양성↑, 과적합 완화에 도움.
데이터 누수 위험 낮음: 보통 train split 결정 후에 증강을 적용하므로 테스트로 새어 나가기 어려움.
단점
매 학습 스텝마다 증강을 계산 → CPU/GPU 오버헤드 증가 가능.
재현성 낮아짐: seed로 고정 가능하지만, 완전 동일한 “셋”을 반복하기는 어려움.
너무 강한/복잡한 물리 제약·후처리를 매번 하면 병목이 될 수 있음.
```
| unified <-> per-finger

**Observation (20D):**
- Force sensors: 9D (3 sensors × 3 axes)
- End-effector positions: 9D (3 fingers × 3 axes)
- Deformation: 2D (circumferential, eccentricity)

**Action (9D):**
- Stiffness: 9D (3 fingers × 3 DOF: K1, K2, K3)


### 4. Monitor Training with TensorBoard
View real-time training metrics:
```bash
# Start TensorBoard
tensorboard --logdir outputs/models/stiffness_policies/tensorboard --port 6006

# Access in browser at http://localhost:6006
```

### 5. Evaluate Trained Models
Evaluate all models on held-out test demonstrations:
```bash
# 각 학습 결과에 대해 평가
python3 scripts/3_model_learning/evaluate_stiffness_policy.py \
  --artifact-dir outputs/models/policy_learning/artifacts/<timestamp> \
  --models all

# Global T_K 버전 평가
python3 scripts/3_model_learning/evaluate_stiffness_policy.py \
  --artifact-dir outputs/models/policy_learning_global_tk/artifacts/<timestamp> \
  --models all
```

Options:
- `--diffusion-sampler ddpm` : DDPM or DDIM sampling

### 6. Visualize Results
Generate comprehensive comparison plots:
```bash
# 모든 모델 비교 시각화
python3 scripts/3_model_learning/visualize_all_models.py
```

Outputs:
- `all_models_comparison.png` : All models vs ground truth (3×3 grid)
- `top3_models_comparison.png` : Top 3 models by R²
- Per-model performance metrics table

### Directory Structure
```
outputs/
├── logs/success/                           # Raw demonstration logs (*.csv)
├── analysis/
│   └── stiffness_profiles_global_tk/      # Generated stiffness profiles (*_paper_profile.csv)
└── models/stiffness_policies/
    ├── artifacts/                          # Trained model checkpoints
    │   └── <timestamp>/
    │       ├── bc.pt                       # Behavior cloning model
    │       ├── gmm.pkl, gmr.pkl           # Gaussian mixture models
    │       ├── lstm_gmm.pt                 # LSTM-GMM model
    │       ├── diffusion_*.pt              # Diffusion models (C/T, DDPM/DDIM)
    │       ├── ibc.pt                      # Implicit BC model
    │       ├── scalers.pkl                 # Data normalization scalers
    │       └── manifest.json               # Model metadata
    ├── tensorboard/                        # TensorBoard logs
    │   └── <model_name>/                   # Per-model event files
    ├── plots/                              # Evaluation visualizations
    └── benchmark_summary_<timestamp>.json  # Performance summary
```

### Common Commands

**Check Training Status:**
```bash
# View training log
tail -f outputs/unified_all_models.log

# Check if training is running
ps aux | grep "run_stiffness_policy_benchmarks.py"
```

**View Benchmark Results:**
```bash
# List all benchmarks (sorted by time)
ls -lt outputs/models/stiffness_policies/benchmark_summary_*.json | head -5

# Pretty-print latest benchmark
ls -t outputs/models/stiffness_policies/benchmark_summary_*.json | head -1 | xargs cat | python3 -m json.tool

# Extract model performance
cat benchmark_summary_<timestamp>.json | python3 -c "
import json, sys
data = json.load(sys.stdin)
for name, metrics in data['models'].items():
    print(f\"{name:15s} R²={metrics['r2']:7.4f} RMSE={metrics['rmse']:7.2f}\")
"
```

**Quick Data Stats:**
```bash
# Count demonstrations
ls outputs/logs/success/*.csv | wc -l

# Count augmented demos
ls outputs/logs/success/*_aug*.csv | wc -l

# Check stiffness profiles
ls outputs/analysis/stiffness_profiles_global_tk/*_paper_profile.csv | wc -l
```
