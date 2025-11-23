# Run Policy Node - 사용 가이드

실시간으로 센서 데이터를 받아 학습된 정책으로 stiffness를 예측하고 impedance control 명령을 발행하는 ROS2 노드입니다.

## 기능

- **실시간 센서 구독**: Force, deformity, EE pose 데이터 수집
- **모델 추론**: BC, Diffusion, GMM/GMR 모델 지원
- **자동 모델 탐지**: artifact 디렉토리 자동 검색
- **스무딩**: Moving average로 예측값 안정화
- **안전 제한**: Min/max 클램핑으로 안전한 stiffness 범위 유지

## 토픽

### 구독 (Subscriptions)
- `/force_sensor/s1/wrench` (geometry_msgs/WrenchStamped) - Thumb force
- `/force_sensor/s2/wrench` (geometry_msgs/WrenchStamped) - Index force  
- `/force_sensor/s3/wrench` (geometry_msgs/WrenchStamped) - Middle force
- `/deformity_tracker/eccentricity` (std_msgs/Float32) - Deformity metric
- `/ee_pose_if` (geometry_msgs/PoseStamped) - Index finger EE pose
- `/ee_pose_mf` (geometry_msgs/PoseStamped) - Middle finger EE pose
- `/ee_pose_th` (geometry_msgs/PoseStamped) - Thumb EE pose

### 발행 (Publications)
- `/impedance_control/target_stiffness` (std_msgs/Float32MultiArray) - 9D stiffness vector
  - `[th_k1, th_k2, th_k3, if_k1, if_k2, if_k3, mf_k1, mf_k2, mf_k3]`

## 파라미터

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `model_type` | string | `bc` | 모델 타입: bc, diffusion_c, diffusion_t, gmm, gmr |
| `mode` | string | `unified` | 모드: unified 또는 per-finger |
| `artifact_dir` | string | `` | 모델 artifact 경로 (비어있으면 자동 탐지) |
| `rate_hz` | float | `50.0` | 제어 루프 주파수 (Hz) |
| `stiffness_scale` | float | `1.0` | Stiffness 스케일 팩터 |
| `stiffness_min` | float | `0.0` | 최소 stiffness 클램핑 값 |
| `stiffness_max` | float | `1000.0` | 최대 stiffness 클램핑 값 |
| `smooth_window` | int | `5` | Moving average 윈도우 크기 |

## 사용 방법

### 1. 기본 실행 (BC 모델)

```bash
# 빌드 및 환경 설정
cd ~/ros2_ws/icra2025
source install/setup.bash

# BC 모델로 실행
ros2 run hri_falcon_robot_bridge run_policy_node
```

### 2. 파라미터 설정하여 실행

```bash
# Diffusion conditional 모델, 100Hz로 실행
ros2 run hri_falcon_robot_bridge run_policy_node \
  --ros-args \
  -p model_type:=diffusion_c \
  -p rate_hz:=100.0 \
  -p stiffness_scale:=1.5

# GMR 모델, per-finger 모드
ros2 run hri_falcon_robot_bridge run_policy_node \
  --ros-args \
  -p model_type:=gmr \
  -p mode:=per-finger \
  -p smooth_window:=10
```

### 3. 특정 artifact 디렉토리 지정

```bash
ros2 run hri_falcon_robot_bridge run_policy_node \
  --ros-args \
  -p model_type:=bc \
  -p artifact_dir:=/path/to/your/artifact/dir
```

### 4. 로그 레벨 조정

```bash
# INFO 레벨 (기본)
ros2 run hri_falcon_robot_bridge run_policy_node

# DEBUG 레벨 (자세한 로그)
ros2 run hri_falcon_robot_bridge run_policy_node \
  --ros-args --log-level debug

# WARN 레벨 (경고만)
ros2 run hri_falcon_robot_bridge run_policy_node \
  --ros-args --log-level warn
```

## 모델 타입별 특징

### BC (Behavior Cloning)
- 가장 단순하고 빠른 모델
- 단일 forward pass로 즉시 예측
- 추천: 실시간 제어가 중요한 경우

### Diffusion (Conditional/Temporal)
- 더 정확하지만 계산량 많음
- DDPM/DDIM 샘플러 사용
- 추천: 정확도가 중요하고 계산 자원이 충분한 경우

### GMM (Gaussian Mixture Model)
- 확률적 샘플링
- 다양한 stiffness 전략 탐색
- 추천: 탐색적 제어가 필요한 경우

### GMR (Gaussian Mixture Regression)
- GMM 기반 조건부 기댓값
- 안정적인 예측
- 추천: 안정성이 중요한 경우

## 출력 예시

```
[INFO] [run_policy_node]: RunPolicy node started: model=bc, mode=unified, rate=50.0Hz, artifact=/home/.../artifacts/20251122_181241
[INFO] [run_policy_node]: Loading model from: /home/.../artifacts/20251122_181241
[INFO] [run_policy_node]: Loaded observation and action scalers
[INFO] [run_policy_node]: BC model: obs_dim=19, act_dim=9, hidden=256, depth=3
[INFO] [run_policy_node]: Model loaded successfully: bc
[INFO] [run_policy_node]: RunPolicy node running. Press Ctrl+C to exit.
[WARN] [run_policy_node]: Waiting for sensor data: forces, ee_poses
[INFO] [run_policy_node]: Stiffness: TH=[45.2,38.1,52.3], IF=[67.4,71.2,69.8], MF=[54.1,49.7,51.2]
```

## 트러블슈팅

### 센서 데이터가 안 들어옴
```bash
# 토픽 리스트 확인
ros2 topic list

# 특정 토픽 확인
ros2 topic echo /force_sensor/s1/wrench
ros2 topic echo /ee_pose_if
```

### 모델 로딩 실패
```bash
# Artifact 디렉토리 확인
ls -la ~/ros2_ws/icra2025/src/hri_falcon_robot_bridge/outputs/models/policy_learning_unified/artifacts/

# 로그에서 자세한 에러 확인
ros2 run hri_falcon_robot_bridge run_policy_node --ros-args --log-level debug
```

### Stiffness 값이 이상함
- `stiffness_scale` 파라미터로 스케일 조정
- `stiffness_min`, `stiffness_max`로 범위 제한
- `smooth_window` 늘려서 더 부드럽게

## 성능 최적화

### 높은 주파수 제어 (100Hz+)
```bash
ros2 run hri_falcon_robot_bridge run_policy_node \
  --ros-args \
  -p model_type:=bc \
  -p rate_hz:=100.0 \
  -p smooth_window:=3
```

### 안정적인 제어 (낮은 주파수)
```bash
ros2 run hri_falcon_robot_bridge run_policy_node \
  --ros-args \
  -p rate_hz:=20.0 \
  -p smooth_window:=10
```

## 다음 단계

1. **Impedance Controller 연결**: 실제 로봇 컨트롤러와 연동
2. **시각화**: rqt_plot으로 stiffness 실시간 모니터링
3. **데이터 로깅**: rosbag으로 제어 성능 기록 및 분석
