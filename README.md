# HRI Falcon Robot Bridge

**Human-Robot Interaction Stiffness Policy Learning for Dexterous Manipulation**

ROS 2 Humble 기반의 햅틱 로봇 제어 및 강성(Stiffness) 정책 학습 프레임워크입니다. Force 센서, Falcon 햅틱 장치, Dynamixel 모터를 연동하여 인간 시연 데이터로부터 물체별 강성 정책을 학습하고 배포합니다.

---

## 📋 Table of Contents

1. [Features](#-features)
2. [System Architecture](#-system-architecture)
3. [Experiment Progress](#-experiment-progress)
4. [Installation](#-installation)
5. [Package Structure](#-package-structure)
6. [Pipeline Overview](#-pipeline-overview)
7. [Quick Start](#-quick-start)
8. [Detailed Usage](#-detailed-usage)
9. [ROS 2 Nodes](#-ros-2-nodes)
10. [Troubleshooting](#-troubleshooting)

---

## ✨ Features

- **Multi-finger Stiffness Estimation**: 3-finger (Thumb, Index, Middle) × 3-axis (X, Y, Z) 강성 프로파일 생성
- **Multiple Policy Learning Models**: BC, Diffusion Policy, LSTM-GMM, IBC, GMR 지원
- **Multi-Object Experiments**: 물체별(풍선, 사과, 귤, 토마토) 시연 데이터 수집 및 정책 학습
- **Real-time Deployment**: 학습된 모델을 ROS 2 노드로 실시간 배포
- **Comprehensive Evaluation**: Pearson 상관계수, R², RMSE 기반 비교 분석
- **Data Augmentation**: Physics-aware 데이터 증강으로 일반화 성능 향상

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Hardware Layer                                     │
├─────────────────┬─────────────────┬─────────────────┬───────────────────────┤
│   Force Sensors │  Falcon Haptic  │   Dynamixel     │      SenseGlove       │
│   (ATI Mini45)  │    Devices      │    Motors       │   (Hand Tracking)     │
└────────┬────────┴────────┬────────┴────────┬────────┴───────────┬───────────┘
         │                 │                 │                     │
         ▼                 ▼                 ▼                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ROS 2 Node Layer                                   │
├─────────────────┬─────────────────┬─────────────────┬───────────────────────┤
│ force_sensor_   │   falcon_node   │ dynamixel_      │  sense_glove_node     │
│ node            │                 │ control         │                       │
└────────┬────────┴────────┬────────┴────────┬────────┴───────────┬───────────┘
         │                 │                 │                     │
         ▼                 ▼                 ▼                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Processing Layer                                      │
├─────────────────┬─────────────────┬─────────────────────────────────────────┤
│  data_logger    │  deformity_     │  robot_controller_node                  │
│  _node          │  tracker_node   │  (Impedance Control)                    │
└────────┬────────┴────────┬────────┴────────────────────┬────────────────────┘
         │                 │                              │
         ▼                 ▼                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Policy Learning Pipeline                                │
├───────────────┬───────────────┬───────────────┬───────────────┬─────────────┤
│ 1. Stiffness  │ 2. Data       │ 3. Model      │ 4. Policy     │ 5. Result   │
│   Profiling   │   Augment     │   Learning    │   Deploy      │   Analysis  │
└───────────────┴───────────────┴───────────────┴───────────────┴─────────────┘
```

---

## 🧪 Experiment Progress

### 실험 대상 물체

| 물체 | 상태 | 시연 수 | 비고 |
|------|------|---------|------|
| 🎈 **풍선 (Balloon)** | ✅ 완료 | 10회 (+ 증강 데이터) | 기본 실험 완료, 모델 학습 및 평가 완료 |
| 🍎 **사과 (Apple)** | ⬜ 예정 | - | - |
| 🍊 **귤 (Tangerine)** | ⬜ 예정 | - | - |
| 🍅 **토마토 (Tomato)** | ⬜ 예정 | - | - |

### 풍선 실험 결과 요약

- **시연 데이터**: 10회 성공 시연 (2025.11.22 수집)
- **강성 프로파일**: Sign-aligned Global T_K 기반 생성 완료
- **학습 모델**: BC, Diffusion Policy (seq16_h2), LSTM-GMM, IBC, GMR
- **평가 지표**: Pearson 상관계수, R², RMSE
- **결과 시각화**: 6종 논문용 Figure 생성 완료

---

## 📦 Installation

### Prerequisites

- Ubuntu 22.04
- ROS 2 Humble
- Python 3.10+
- CUDA 11.8+ (GPU 학습 시)

### Build

```bash
# Clone repository
cd ~/ros2_ws/src
git clone https://github.com/your-repo/hri_falcon_robot_bridge.git

# Install dependencies
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y

# Build
colcon build --packages-select hri_falcon_robot_bridge
source install/setup.bash
```

### Python Dependencies

```bash
pip install torch torchvision numpy pandas scipy scikit-learn matplotlib seaborn
pip install gpytorch hydra-core omegaconf tensorboard
```

---

## 📁 Package Structure

```
hri_falcon_robot_bridge/
├── hri_falcon_robot_bridge/          # Python ROS 2 노드들
│   ├── data_logger_node.py           # 시연 데이터 로깅
│   ├── deformity_tracker_node.py     # 변형도 추적 (eccentricity)
│   ├── force_sensor_node.py          # Force 센서 인터페이스
│   ├── robot_controller_node.py      # 로봇 제어 (Impedance)
│   ├── run_policy_node.py            # 학습된 정책 실행
│   └── ...
│
├── src/                              # C++ 노드들
│   ├── falcon_node.cpp               # Falcon 햅틱 장치 드라이버
│   └── sense_glove_node.cpp          # SenseGlove 인터페이스
│
├── scripts/                          # 파이프라인 스크립트
│   ├── 1_stiffness_profiling/        # 강성 프로파일 생성
│   ├── 2_data_augmentation/          # 데이터 증강
│   ├── 3_model_learning/             # 모델 학습 & 평가
│   ├── 4_policy_depolyer/            # 정책 배포
│   ├── 5_plot_result/                # 결과 시각화
│   ├── analysis/                     # 추가 분석 스크립트
│   └── legacy/                       # 레거시 코드 (DMP 등)
│
├── launch/                           # ROS 2 Launch 파일
│   └── full_stiffness_pipeline.launch.py
│
├── outputs/                          # 출력 데이터 (물체별 정리)
│   ├── logs/                         # 시연 로그
│   │   ├── balloon/                  # 풍선 시연 (✅ 완료)
│   │   │   ├── success/              #   성공 시연 10회 (20251122)
│   │   │   ├── success_251117/       #   이전 성공 시연 + 증강
│   │   │   └── 20251122/             #   날짜별 원본
│   │   ├── apple/success/            # 사과 시연 (⬜ 예정)
│   │   ├── tangerine/success/        # 귤 시연 (⬜ 예정)
│   │   └── tomato/success/           # 토마토 시연 (⬜ 예정)
│   ├── stiffness_profiles/           # 강성 프로파일
│   │   └── balloon/                  #   풍선 프로파일 (✅)
│   ├── stiffness_profiles_signaligned/ # Sign-aligned 프로파일
│   │   └── balloon/                  #   풍선 (✅)
│   ├── models/                       # 학습 모델
│   │   └── balloon/                  #   풍선 모델 (✅)
│   ├── analysis/                     # 분석 결과
│   │   └── balloon/plots/            #   풍선 intent mapping (✅)
│   └── plots/                        # 시각화
│       └── balloon/full_profiles/    #   풍선 프로파일 플롯 (✅)
│
└── docs/                             # 문서
```

---

## 🔄 Pipeline Overview

### 전체 파이프라인 (5단계)

| 단계 | 스크립트 위치 | 설명 |
|------|--------------|------|
| **1. Stiffness Profiling** | `scripts/1_stiffness_profiling/` | Force/EMG 데이터에서 강성 프로파일 추출 |
| **2. Data Augmentation** | `scripts/2_data_augmentation/` | Physics-aware 노이즈로 데이터 증강 |
| **3. Model Learning** | `scripts/3_model_learning/` | 다양한 모델 학습 (BC, Diffusion 등) |
| **4. Policy Deploy** | `scripts/4_policy_depolyer/` | 학습된 모델 실시간 배포 |
| **5. Result Analysis** | `scripts/5_plot_result/` | 결과 비교 및 시각화 |

### 지원 모델

| Model | Type | 특징 |
|-------|------|------|
| **BC** | Behavior Cloning | MLP 기반 회귀, 기본 baseline |
| **Diffusion Policy** | Generative | DDPM/DDIM, seq16_h2 최적 구성 |
| **LSTM-GMM** | Sequence | 시계열 + GMM 출력 |
| **IBC** | Energy-based | Implicit Behavior Cloning |
| **GMR** | Probabilistic | Gaussian Mixture Regression |

---

## 🚀 Quick Start

### 1. 시연 데이터 수집

```bash
# ROS 2 파이프라인 실행 후 시연 녹화
ros2 launch hri_falcon_robot_bridge full_stiffness_pipeline.launch.py
# 성공 시연 CSV → outputs/logs/<object>/success/ 에 저장
```

### 2. 강성 프로파일 생성

```bash
cd ~/ros2_ws/src/hri_falcon_robot_bridge

# Sign-aligned Global T_K 방식으로 강성 프로파일 생성
python3 scripts/1_stiffness_profiling/generate_stiffness_profiles_global_tk_sign_aligned.py
```

### 3. 모델 학습

```bash
# 모든 모델 학습 (Unified + Global T_K) - 추천
python3 scripts/3_model_learning/run_stiffness_policy_benchmarks.py \
  --mode unified \
  --models all \
  --bc-epochs 200 \
  --diffusion-epochs 200 \
  --augment --augment-num 1
```

### 4. 결과 시각화

```bash
python3 scripts/5_plot_result/compare_stiffness_sessions.py
```

---

## 📖 Detailed Usage

### 학습 구성 옵션

| 구성 | 설명 | 명령 옵션 |
|------|------|----------|
| **Unified + Sign-aligned** | 단일 모델 (20D→9D), Sign-aligned Global T_K ⭐ | `--mode unified` |
| **Per-Finger** | 손가락별 모델 (8D→3D) | `--mode per-finger` |

### 데이터 차원

**Observation (20D):**
- Force 센서: 9D (3 센서 × 3 축)
- End-effector 위치: 9D (3 손가락 × 3 축)
- 변형도: 2D (circumferential, eccentricity)

**Action (9D):**
- Stiffness: 9D (3 손가락 × 3 DOF: K1, K2, K3)

### 배치 학습 실행

```bash
# Unified 모드 전체 실행
bash run_all_with_tb.sh
```

### 데이터 증강 옵션

```bash
python3 scripts/3_model_learning/run_stiffness_policy_benchmarks.py \
  --augment \
  --augment-num 1 \
  --augment-noise-force 0.015 \
  --augment-noise-stiffness 0.04 \
  --augment-temporal-jitter 1
```

---

## 🤖 ROS 2 Nodes

### Core Nodes

| Node | Topic (Pub/Sub) | 설명 |
|------|-----------------|------|
| `force_sensor_node` | `/force_sensor/s{1,2,3}/wrench` | ATI 센서 데이터 발행 |
| `falcon_node` | `/falcon/{position,force}` | Falcon 위치/힘 피드백 |
| `robot_controller_node` | `/joint_commands` | Dynamixel 제어 |
| `data_logger_node` | 다수 Subscribe | 시연 데이터 CSV 저장 |
| `deformity_tracker_node` | `/deformity` | 변형도 계산 |
| `run_policy_node` | `/stiffness_command` | 학습된 정책 실행 |

### Launch

```bash
# 전체 파이프라인 실행
ros2 launch hri_falcon_robot_bridge full_stiffness_pipeline.launch.py
```

### 유용한 명령어

```bash
# 토픽 확인
ros2 topic list -t
ros2 topic echo /force_sensor/s1/wrench

# 노드 확인
ros2 node list

# USB 레이턴시 최적화 (Force 센서용)
sudo sh -c 'echo 1 > /sys/bus/usb-serial/devices/ttyUSB0/latency_timer'
```

---

## 🔧 Troubleshooting

### 프로세스 강제 종료

```bash
pkill -9 -f "ros2|python3.*hri_falcon"
```

### USB 레이턴시 최적화 (Force 센서)

```bash
sudo sh -c 'echo 1 > /sys/bus/usb-serial/devices/ttyUSB0/latency_timer'
```

### TensorBoard 포트 충돌

```bash
tensorboard --logdir outputs/models/balloon/policy_learning_unified/tensorboard --port 6007
```

---

## 📊 Output Files

### 데이터 디렉토리 구조 (물체별 분류)

```
outputs/
├── logs/<object>/success/            # 시연 CSV 데이터
├── stiffness_profiles/<object>/      # 강성 프로파일 CSV & 플롯
├── stiffness_profiles_signaligned/<object>/  # Sign-aligned 프로파일
├── models/<object>/                  # 학습된 모델 체크포인트
│   ├── benchmark_sweep/              #   하이퍼파라미터 탐색 결과
│   └── policy_learning_unified/      #   Unified 학습 결과
│       ├── artifacts/<timestamp>/    #     모델 파일 (.pt, .pkl)
│       └── tensorboard/              #     TensorBoard 로그
├── analysis/<object>/plots/          # Intent mapping 플롯
└── plots/<object>/full_profiles/     # 전체 프로파일 시각화
```

### 논문용 Figure (풍선 실험)

```
outputs/stiffness_comparison/paper_figures/
├── fig_gt_diffusion_force_comparison.png          # GT vs Diffusion 힘 비교
├── fig_stiffness_per_model_all_axes.png            # 모델별 전 축 강성 비교
├── fig_stiffness_per_model_all_axes_unified_y.png  # 통일 Y축 버전
├── fig_stiffness_per_model_thumb_sorted.png        # Thumb 정렬 비교
├── fig_stiffness_per_model_optimal_aligned_raw.png # Optimal alignment 비교
└── fig_stiffness_per_model_optimal_aligned_raw_unified_y.png  # 통일 Y축 버전
```

---

## � Data & Model Storage

실험 데이터와 학습된 모델은 **GitHub에 포함되지 않으며**, Google Drive에 비공개로 별도 보관합니다.

| 데이터 | 경로 | 저장 위치 |
|--------|------|-----------|
| 학습된 모델 (`.pt`, `.pkl`) | `outputs/models/` | Google Drive |
| 실험 로그 (CSV) | `outputs/stiffness_logs/` | Google Drive |
| 분석 결과 | `outputs/analysis/` | Google Drive |
| EMG 데이터 | `outputs/emg/` | Google Drive |
| DMP 모델 | `dmp_models/` | Google Drive |

### 데이터 복원 방법

```bash
# Google Drive에서 outputs/ 폴더를 다운로드한 후:
cp -r ~/Downloads/outputs ./outputs/
```

> `.gitignore`에 의해 `outputs/`, `*.pt`, `*.pkl`, `*.log`, `__pycache__/` 등은 자동으로 Git 추적에서 제외됩니다.

---

## �📚 References

- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)
- [Implicit Behavior Cloning](https://implicitbc.github.io/)
- [ROS 2 Humble Documentation](https://docs.ros.org/en/humble/)

---

## 📝 License

MIT License

## 👥 Contact

- Maintainer: Songwoo Kim
