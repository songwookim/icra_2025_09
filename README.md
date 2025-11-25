
```
cat /sys/bus/usb-serial/devices/ttyUSB0/latency_timer 
sudo vi /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
   # change to 16 -> 1
   

   # 특정 토픽의 메시지 실시간 출력 (force sensor 예시)
ros2 topic echo /force_sensor/s1/wrench

# EE pose 토픽 확인
ros2 topic echo /ee_pose_if
ros2 topic echo /ee_pose_mf
ros2 topic echo /ee_pose_th

# JointState 확인
ros2 topic echo /joint_states

# 실행 중인 ROS2 노드 확인
ros2 node list

# force_sensor 관련 프로세스 확인
ps aux | grep force_sensor

# 프로세스 강제 종료
pkill -9 -f force_sensor_node

# 모든 활성 토픽 나열
ros2 topic list -t

ros2 topic info /force_sensor/s1/wrench -v

pkill -9 -f "ros2|python3.*hri_falcon"

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

**4가지 구성 중 선택하여 실행:**

```bash
cd /home/songwoo/ros2_ws/icra2025/src/hri_falcon_robot_bridge

# [구성 1] Unified + Original
python3 scripts/3_model_learning/run_stiffness_policy_benchmarks.py \
  --mode unified \
  --stiffness-dir outputs/analysis/stiffness_profiles \
  --models all \
  --bc-epochs 200 \
  --diffusion-epochs 200 \
  --augment --augment-num 1 \
  --augment-noise-force 0.015 \
  --augment-noise-stiffness 0.04 \
  --augment-temporal-jitter 1

# [구성 2] Unified + Global T_K (추천: 성능 가장 좋음)
python3 scripts/3_model_learning/run_stiffness_policy_benchmarks.py \
  --mode unified \
  --stiffness-dir outputs/analysis/stiffness_profiles_global_tk \
  --models all \
  --bc-epochs 200 \
  --diffusion-epochs 200 \
  --augment --augment-num 1 \
  --augment-noise-force 0.015 \
  --augment-noise-stiffness 0.04 \
  --augment-temporal-jitter 1

# [구성 3] Per-Finger + Original
python3 scripts/3_model_learning/run_stiffness_policy_benchmarks.py \
  --mode per-finger \
  --stiffness-dir outputs/analysis/stiffness_profiles \
  --models all \
  --bc-epochs 200 \
  --augment --augment-num 1 \
  --augment-noise-force 0.015 \
  --augment-noise-stiffness 0.04 \
  --augment-temporal-jitter 1

# [구성 4] Per-Finger + Global T_K
python3 scripts/3_model_learning/run_stiffness_policy_benchmarks.py \
  --mode per-finger \
  --stiffness-dir outputs/analysis/stiffness_profiles_global_tk \
  --models all \
  --bc-epochs 200 \
  --augment --augment-num 1 \
  --augment-noise-force 0.015 \
  --augment-noise-stiffness 0.04 \
  --augment-temporal-jitter 1

# 4가지 모두 자동 실행 (배치 스크립트)
bash run_all_with_tb.sh
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
tensorboard --logdir src/hri_falcon_robot_bridge/outputs/models/policy_learning/tensorboard --port 6006

# Access in browser at http://localhost:6006
```

### 5. Evaluate Trained Models

**먼저 최신 timestamp 확인:**

```bash
# 모든 artifact 디렉토리 확인 (최신 순)
ls -1dt outputs/models/policy_learning*/artifacts/*

# 또는 특정 구성만
ls -1t outputs/models/stiffness_policies/policy_learning_unified/artifacts/*
ls -1t outputs/models/stiffness_policies/policy_learning_global_tk_unified/artifacts/*
ls -1t outputs/models/stiffness_policies/policy_learning_per_finger/artifacts/*
ls -1t outputs/models/stiffness_policies/policy_learning_global_tk_per_finger/artifacts/*
```

**평가 실행 (아래 `YYYYMMDD_HHMMSS`를 위에서 확인한 실제 timestamp로 교체):**

```bash
# Unified + Original 평가
python3 scripts/3_model_learning/evaluate_stiffness_policy.py \
  --artifact-dir outputs/models/stiffness_policies/policy_learning_unified/artifacts/YYYYMMDD_HHMMSS \
  --models all

# Unified + Global T_K 평가
python3 scripts/3_model_learning/evaluate_stiffness_policy.py \
  --artifact-dir outputs/models/stiffness_policies/policy_learning_global_tk_unified/artifacts/YYYYMMDD_HHMMSS \
  --models all

# Per-Finger + Original 평가
python3 scripts/3_model_learning/evaluate_stiffness_policy.py \
  --artifact-dir outputs/models/stiffness_policies/policy_learning_per_finger/artifacts/YYYYMMDD_HHMMSS \
  --models bc

# Per-Finger + Global T_K 평가
python3 scripts/3_model_learning/evaluate_stiffness_policy.py \
  --artifact-dir outputs/models/stiffness_policies/policy_learning_global_tk_per_finger/artifacts/YYYYMMDD_HHMMSS \
  --models all
```

**자동으로 최신 timestamp 사용 (고급):**

```bash
# Unified + Global T_K 최신 결과 자동 평가
LATEST=$(ls -1dt outputs/models/stiffness_policies/policy_learning_global_tk_unified/artifacts/* | head -1)
python3 scripts/3_model_learning/evaluate_stiffness_policy.py \
  --artifact-dir "$LATEST" \
  --models all
```

**평가 옵션:**
- `--diffusion-sampler ddpm` : DDPM or DDIM sampling
- `--models all` : 모든 모델 평가 (또는 bc, gmr, diffusion_c 등 개별 지정)

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
  ├── policy_learning_unified/            # Unified (original stiffness) models
  │   ├── artifacts/<timestamp>/
  │   │   ├── bc.pt
  │   │   ├── diffusion_c.pt / diffusion_t.pt
  │   │   ├── gmm.pkl
  │   │   ├── ibc.pt
  │   │   ├── lstm_gmm.pt
  │   │   ├── scalers.pkl
  │   │   └── manifest.json
  │   └── tensorboard/<timestamp>/        # TensorBoard events (bc, ibc, diffusion, lstm_gmm, mdn)
  ├── policy_learning_global_tk_unified/  # Unified (global T_K) models
  │   ├── artifacts/<timestamp>/ ... (same layout)
  │   └── tensorboard/<timestamp>/
  ├── policy_learning_per_finger/         # Per-finger (original stiffness) models
  │   ├── artifacts/<timestamp>/
  │   │   ├── th/bc.pt, scalers.pkl, manifest.json
  │   │   ├── if/bc.pt, scalers.pkl, manifest.json
  │   │   ├── mf/bc.pt, scalers.pkl, manifest.json
  │   │   └── manifest.json (aggregate)
  │   └── tensorboard/<timestamp>/        # (Reserved for future per-finger TB runs)
  ├── policy_learning_global_tk_per_finger/ # Per-finger (global T_K) models
  │   ├── artifacts/<timestamp>/ (same per-finger layout)
  │   └── tensorboard/<timestamp>/
  ├── artifacts/                          # Legacy (symlink to unified) – backward compatibility
  │   └── <timestamp>/                    # (Old path retained for existing scripts)
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

### 4가지 구성별 실행 & 평가 (Unified / Per-Finger × Original / Global T_K)

| 구성 | 설명 | 학습 명령 예시 | 평가 경로 예시 |
|------|------|----------------|----------------|
| Unified + Original | 단일 모델 (20D→9D) 원본 강성 | `--mode unified --stiffness-dir outputs/analysis/stiffness_profiles` | `outputs/models/stiffness_policies/policy_learning_unified/artifacts/<ts>` |
| Unified + Global T_K | 단일 모델 (20D→9D) 전역 T_K | `--mode unified --stiffness-dir outputs/analysis/stiffness_profiles_global_tk` | `outputs/models/stiffness_policies/policy_learning_global_tk_unified/artifacts/<ts>` |
| Per-Finger + Original | 손가락별 3개 모델 (8D→3D) 원본 강성 | `--mode per-finger --stiffness-dir outputs/analysis/stiffness_profiles` | `outputs/models/stiffness_policies/policy_learning_per_finger/artifacts/<ts>` |
| Per-Finger + Global T_K | 손가락별 3개 모델 (8D→3D) 전역 T_K | `--mode per-finger --stiffness-dir outputs/analysis/stiffness_profiles_global_tk` | `outputs/models/stiffness_policies/policy_learning_global_tk_per_finger/artifacts/<ts>` |

학습 예시 (4-way 모두 실행):

```bash
bash run_all_with_tb.sh
```

개별 실행 예시 (Unified + Global T_K):

```bash
python3 scripts/3_model_learning/run_stiffness_policy_benchmarks.py \
  --mode unified \
  --stiffness-dir outputs/analysis/stiffness_profiles_global_tk \
  --models all \
  --augment --augment-num 1 \
  --augment-noise-force 0.015 \
  --augment-noise-stiffness 0.04 \
  --augment-temporal-jitter 1
```

평가 (Unified + Global T_K):

```bash
python3 scripts/3_model_learning/evaluate_stiffness_policy.py \
  --artifact-dir outputs/models/stiffness_policies/policy_learning_global_tk_unified/artifacts/<timestamp> \
  --models all
```

평가 (Per-Finger + Original):

```bash
python3 scripts/3_model_learning/evaluate_stiffness_policy.py \
  --artifact-dir outputs/models/stiffness_policies/policy_learning_per_finger/artifacts/<timestamp> \
  --models bc
```

주의: 기존 경로(`policy_learning`, `policy_learning_global_tk`)는 새 구조의 unified 폴더로 심볼릭 링크가 생성되어 기존 스크립트 호환이 유지됩니다.

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
