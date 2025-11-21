# Stiffness Policy Deployer - 사용 가이드

## 개요
학습된 stiffness policy 모델을 실시간으로 로봇에 배포하는 ROS2 노드입니다.

## 지원 모델
- **BC** (Behavior Cloning): 가장 빠른 추론 속도, 부드러운 예측
- **Diffusion**: 높은 정확도, 약간 느린 속도 (DDPM/DDIM 샘플러)
- **LSTM-GMM**: 시계열 정보 활용, 단일 스텝 모드 지원
- **GMM/GMR**: 가우시안 혼합 모델 기반

## 입력 (Subscriptions)
센서 토픽에서 실시간으로 데이터를 받아옵니다:

```
/force_sensor/s1/wrench          # Force sensor 1 (WrenchStamped)
/force_sensor/s2/wrench          # Force sensor 2 (WrenchStamped)
/force_sensor/s3/wrench          # Force sensor 3 (WrenchStamped)
/deformity_tracker/eccentricity  # Object eccentricity (Float32)
/ee_pose_if                       # Index finger EE pose (PoseStamped)
/ee_pose_mf                       # Middle finger EE pose (PoseStamped)
/ee_pose_th                       # Thumb EE pose (PoseStamped)
```

**관찰 벡터 (19차원):**
- Force sensors: s1_fx, s1_fy, s1_fz, s2_fx, s2_fy, s2_fz, s3_fx, s3_fy, s3_fz (9D)
- Deformity: deform_ecc (1D)
- EE positions: ee_if_px, ee_if_py, ee_if_pz, ee_mf_px, ee_mf_py, ee_mf_pz, ee_th_px, ee_th_py, ee_th_pz (9D)

## 출력 (Publications)
```
/stiffness_policy/predicted      # 예측된 stiffness (Float32MultiArray, 9D)
/stiffness_policy/status         # 상태 메시지 (String)
```

**예측 벡터 (9차원):**
- Stiffness: th_k1, th_k2, th_k3, if_k1, if_k2, if_k3, mf_k1, mf_k2, mf_k3
- 단위: N/mm (물리적 제약으로 최소값 1.0 이상)

## 사용 방법

### 1. 기본 실행 (직접 노드 실행)
```bash
cd /home/songwoo/ros2_ws/icra2025/src/hri_falcon_robot_bridge/scripts/4_policy_depolyer

# Python 실행 권한 부여
chmod +x stiffness_policy_deployer_node.py

# 직접 실행
ros2 run hri_falcon_robot_bridge stiffness_policy_deployer_node.py \
  --ros-args \
  -p model_type:=diffusion \
  -p artifact_dir:=/home/songwoo/ros2_ws/icra2025/outputs/policy_learning_global_tk_unified/artifacts/20251119_192314 \
  -p rate_hz:=100.0
```

### 2. 런치 파일 사용 (권장)

#### Diffusion 모델 (DDPM 샘플러, 100Hz)
```bash
ros2 launch hri_falcon_robot_bridge deploy_stiffness_policy.launch.py \
  model_type:=diffusion \
  artifact_dir:=/home/songwoo/ros2_ws/icra2025/outputs/policy_learning_global_tk_unified/artifacts/20251119_192314 \
  rate_hz:=100.0 \
  diffusion_sampler:=ddpm \
  diffusion_n_samples:=1
```

#### Diffusion 모델 (DDIM 샘플러, 더 빠른 추론)
```bash
ros2 launch hri_falcon_robot_bridge deploy_stiffness_policy.launch.py \
  model_type:=diffusion \
  artifact_dir:=/home/songwoo/ros2_ws/icra2025/outputs/policy_learning_global_tk_unified/artifacts/20251119_192314 \
  rate_hz:=100.0 \
  diffusion_sampler:=ddim \
  diffusion_n_samples:=1
```

#### BC 모델 (가장 빠른 추론)
```bash
ros2 launch hri_falcon_robot_bridge deploy_stiffness_policy.launch.py \
  model_type:=bc \
  artifact_dir:=/home/songwoo/ros2_ws/icra2025/outputs/policy_learning_global_tk_unified/artifacts/20251119_192314 \
  rate_hz:=200.0
```

#### LSTM-GMM 모델
```bash
ros2 launch hri_falcon_robot_bridge deploy_stiffness_policy.launch.py \
  model_type:=lstm_gmm \
  artifact_dir:=/home/songwoo/ros2_ws/icra2025/outputs/policy_learning_global_tk_unified/artifacts/20251119_192314 \
  rate_hz:=50.0
```

### 3. 예측 결과 확인
```bash
# 실시간 stiffness 예측 확인
ros2 topic echo /stiffness_policy/predicted

# 상태 메시지 확인
ros2 topic echo /stiffness_policy/status

# 토픽 Hz 확인
ros2 topic hz /stiffness_policy/predicted
```

### 4. 센서 데이터 확인
```bash
# Force sensor 확인
ros2 topic echo /force_sensor/s1/wrench

# EE pose 확인
ros2 topic echo /ee_pose_if

# Deformity 확인
ros2 topic echo /deformity_tracker/eccentricity
```

## 파라미터 설정

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|-------|------|
| `model_type` | string | `'bc'` | 모델 종류: bc, diffusion, lstm_gmm, gmm, gmr |
| `artifact_dir` | string | `''` | 모델 artifacts 디렉토리 (필수) |
| `rate_hz` | float | `100.0` | 예측 주파수 (Hz) |
| `diffusion_sampler` | string | `'ddpm'` | Diffusion 샘플러: ddpm 또는 ddim |
| `diffusion_n_samples` | int | `1` | Diffusion 샘플 개수 (1=평균값) |

## Artifact 디렉토리 찾기
학습 완료 후 artifact 디렉토리 경로를 확인하세요:

```bash
# Unified mode (Global T_K)
ls -lt /home/songwoo/ros2_ws/icra2025/outputs/policy_learning_global_tk_unified/artifacts/

# Per-Finger mode (Global T_K)
ls -lt /home/songwoo/ros2_ws/icra2025/outputs/policy_learning_global_tk_per_finger/artifacts/
```

가장 최근 디렉토리 (예: `20251119_192314`)를 `artifact_dir`에 사용하세요.

## 모델별 추론 속도 비교 (참고)

| 모델 | 추론 속도 | 추천 rate_hz | 특징 |
|-----|----------|-------------|------|
| BC | 매우 빠름 (~1ms) | 200Hz | 가장 부드러운 예측, 실시간 제어에 최적 |
| Diffusion (DDPM) | 느림 (~50ms) | 20-50Hz | 높은 정확도, 노이즈 많음 |
| Diffusion (DDIM) | 보통 (~10ms) | 50-100Hz | DDPM보다 빠름, 정확도 유사 |
| LSTM-GMM | 보통 (~5ms) | 100Hz | 시계열 정보 활용, 단일 스텝 모드 |
| GMM/GMR | 빠름 (~2ms) | 150Hz | 가볍고 빠름, 정확도 낮음 |

**권장 설정:**
- **실시간 제어**: BC (200Hz) 또는 Diffusion DDIM (100Hz)
- **높은 정확도**: Diffusion DDPM (50Hz) 또는 LSTM-GMM (100Hz)
- **균형**: Diffusion DDIM (100Hz) ← **추천**

## 문제 해결

### 1. "artifact_dir parameter is required" 에러
```bash
# artifact_dir 파라미터를 반드시 지정해야 합니다
ros2 launch ... artifact_dir:=/full/path/to/artifacts/YYYYMMDD_HHMMSS
```

### 2. "Waiting for all sensor data..." 경고
- 모든 센서 토픽이 publish되고 있는지 확인하세요
- 센서 노드가 실행 중인지 확인하세요

```bash
ros2 topic list | grep -E "(force_sensor|ee_pose|deformity)"
```

### 3. "PyTorch is required" 에러
```bash
# PyTorch 설치
pip install torch
```

### 4. Diffusion 모델이 느릴 때
- `diffusion_sampler:=ddim`으로 변경 (DDPM보다 5-10배 빠름)
- `rate_hz`를 낮추기 (100 → 50 → 20)
- BC 모델로 변경 (가장 빠름)

### 5. "Prediction error" 로그
- 모델 파일이 올바르게 로드되었는지 확인
- artifact_dir 경로가 올바른지 확인
- manifest.json 파일이 존재하는지 확인

## 통합 실행 예제

```bash
# Terminal 1: 센서 노드 실행 (data_logger_node 기반)
ros2 run hri_falcon_robot_bridge data_logger_node.py

# Terminal 2: Stiffness policy deployer (Diffusion DDIM, 100Hz)
ros2 launch hri_falcon_robot_bridge deploy_stiffness_policy.launch.py \
  model_type:=diffusion \
  artifact_dir:=/home/songwoo/ros2_ws/icra2025/outputs/policy_learning_global_tk_unified/artifacts/20251119_192314 \
  rate_hz:=100.0 \
  diffusion_sampler:=ddim

# Terminal 3: 예측 결과 모니터링
ros2 topic echo /stiffness_policy/predicted

# Terminal 4: 제어기 (예측된 stiffness를 받아서 로봇 제어)
# ros2 run your_package stiffness_controller_node
```

## 다음 단계
1. 예측된 stiffness를 받아서 실제 로봇을 제어하는 controller node 구현
2. 제어기와 deployer를 하나의 launch file로 통합
3. 성능 벤치마크 (지연시간, 정확도) 평가
4. Safety layer 추가 (stiffness 범위 제한, 이상치 감지)

## 참고
- 모델 학습: `src/hri_falcon_robot_bridge/scripts/3_model_learning/run_stiffness_policy_benchmarks.py`
- 오프라인 평가: `src/hri_falcon_robot_bridge/scripts/3_model_learning/evaluate_stiffness_policy.py`
- 데이터 로거: `src/hri_falcon_robot_bridge/hri_falcon_robot_bridge/data_logger_node.py`
