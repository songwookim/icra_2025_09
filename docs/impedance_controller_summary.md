# Impedance Controller Node - 완성

## 개요

학습된 stiffness를 사용해 Cartesian impedance control을 수행하는 ROS2 노드가 완성되었습니다.

## 생성된 파일

- **impedance_controller_node.py** - Impedance controller 노드
- **launch.json** - 디버그 설정 추가 (8. run_policy_node, 9. impedance_controller_node)

## 전체 시스템 구조

```
센서 데이터 → run_policy_node → stiffness (K)
                                       ↓
                          impedance_controller_node ← desired EE pos
                                       ↓
                                  joint units
                                       ↓
                              robot_controller_node
                                       ↓
                                  실제 로봇
```

## 실행 방법

### 1. 전체 시스템 실행 (터미널 4개 필요)

```bash
# Terminal 1: MuJoCo 시뮬레이션 (센서 데이터 생성)
ros2 run hri_falcon_robot_bridge sense_glove_mj_node

# Terminal 2: Policy 실행 (stiffness 예측)
ros2 run hri_falcon_robot_bridge run_policy_node \
  --ros-args -p model_type:=bc -p rate_hz:=50.0

# Terminal 3: Impedance 컨트롤러 (stiffness로 제어)
ros2 run hri_falcon_robot_bridge impedance_controller_node \
  --ros-args -p rate_hz:=50.0 -p use_mujoco:=true

# Terminal 4: 로봇 컨트롤러 (실제 모터 제어)
ros2 run hri_falcon_robot_bridge robot_controller_node \
  --ros-args -p safe_mode:=false
```

### 2. VSCode 디버거로 실행

1. **F5** 누르고 **"8. run_policy_node"** 선택하여 실행
2. **F5** 누르고 **"9. impedance_controller_node"** 선택하여 실행
3. 다른 필요한 노드들도 동일하게 실행

### 3. Desired EE position 설정 (테스트용)

```bash
# Index finger 목표 위치
ros2 topic pub /ee_pose_desired_if geometry_msgs/PoseStamped "{
  pose: {position: {x: 0.1, y: 0.0, z: 0.15}}
}" -1

# Middle finger 목표 위치
ros2 topic pub /ee_pose_desired_mf geometry_msgs/PoseStamped "{
  pose: {position: {x: 0.1, y: 0.03, z: 0.15}}
}" -1

# Thumb 목표 위치
ros2 topic pub /ee_pose_desired_th geometry_msgs/PoseStamped "{
  pose: {position: {x: 0.08, y: -0.03, z: 0.15}}
}" -1
```

## 주요 기능

### run_policy_node
- **입력**: Force, deformity, EE pose 센서 데이터
- **출력**: 9D stiffness [th_k1..k3, if_k1..k3, mf_k1..k3]
- **모델**: BC, Diffusion, GMM/GMR 지원
- **주파수**: 기본 50Hz

### impedance_controller_node
- **입력**: 
  - Target stiffness (from policy)
  - Current qpos (from robot/mujoco)
  - Desired EE positions
- **출력**: Joint position commands (units)
- **제어 법칙**: F = K * (x_desired - x_current), τ = J^T * F
- **주파수**: 기본 50Hz

## 토픽 구조

### run_policy_node
**구독**:
- `/force_sensor/s{1..3}/wrench` - 힘 센서
- `/deformity_tracker/eccentricity` - 변형 메트릭
- `/ee_pose_{if|mf|th}` - 현재 EE 위치

**발행**:
- `/impedance_control/target_stiffness` - 9D stiffness

### impedance_controller_node
**구독**:
- `/impedance_control/target_stiffness` - 목표 stiffness
- `/hand_tracker/qpos` - 현재 관절 위치
- `/ee_pose_desired_{if|mf|th}` - 목표 EE 위치

**발행**:
- `/hand_tracker/targets_units` - 로봇 명령

## 파라미터

### run_policy_node
```bash
model_type: bc          # bc, diffusion_c, diffusion_t, gmm, gmr
mode: unified           # unified or per-finger
rate_hz: 50.0
stiffness_scale: 1.0
smooth_window: 5
```

### impedance_controller_node
```bash
rate_hz: 50.0
use_mujoco: true        # MuJoCo FK 사용 여부
position_gain: 1.0      # P gain
damping_ratio: 0.7
smooth_alpha: 0.3       # 명령 스무딩
max_step_units: 50.0    # 최대 단계 변화
```

## 모니터링

```bash
# Stiffness 확인
ros2 topic echo /impedance_control/target_stiffness

# Joint commands 확인
ros2 topic echo /hand_tracker/targets_units

# 토픽 리스트
ros2 topic list | grep -E "impedance|policy|ee_pose"
```

## 디버깅

```bash
# 로그 레벨 조정
ros2 run hri_falcon_robot_bridge run_policy_node --ros-args --log-level debug
ros2 run hri_falcon_robot_bridge impedance_controller_node --ros-args --log-level debug

# rqt_graph로 노드 연결 확인
rqt_graph

# rqt_plot으로 실시간 모니터링
rqt_plot /impedance_control/target_stiffness/data[0]:data[3]:data[6]
```

## 성능 최적화

### 고속 제어 (100Hz)
```bash
ros2 run hri_falcon_robot_bridge run_policy_node \
  --ros-args -p model_type:=bc -p rate_hz:=100.0 -p smooth_window:=3

ros2 run hri_falcon_robot_bridge impedance_controller_node \
  --ros-args -p rate_hz:=100.0 -p smooth_alpha:=0.2
```

### 안정적 제어 (낮은 gain)
```bash
ros2 run hri_falcon_robot_bridge impedance_controller_node \
  --ros-args -p position_gain:=0.5 -p smooth_alpha:=0.5
```

## 다음 단계

1. **실제 로봇 테스트**: safe_mode:=false로 실제 모터 제어
2. **Desired position 자동화**: 별도 trajectory planner 노드 추가
3. **성능 평가**: rosbag으로 제어 성능 기록 및 분석
4. **안전 기능**: Force/torque 제한, 충돌 감지 추가
