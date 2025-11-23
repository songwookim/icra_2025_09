# Run Policy Node - 완성

## 개요

실시간으로 센서 데이터를 받아 학습된 stiffness policy로 impedance control을 수행하는 ROS2 노드가 완성되었습니다.

## 파일 위치

- **노드**: `/hri_falcon_robot_bridge/hri_falcon_robot_bridge/run_policy_node.py`
- **문서**: `/hri_falcon_robot_bridge/docs/run_policy_node_usage.md`
- **테스트**: `/hri_falcon_robot_bridge/scripts/test_run_policy_node.py`

## 테스트 결과

```bash
$ python3 src/hri_falcon_robot_bridge/scripts/test_run_policy_node.py

[✓] Model loaded successfully
    Model type: bc
    Artifact dir: .../artifacts/20251122_181241
    Rate: 50.0 Hz
[✓] Observation scaler loaded
[✓] Action scaler loaded

[TEST] Node initialization complete!
```

## 주요 기능

1. **센서 데이터 구독**
   - Force sensors (s1/s2/s3): 3개 손가락 힘 측정
   - Deformity tracker: 변형 메트릭 (eccentricity)
   - EE pose: 3개 손가락 end-effector 위치

2. **모델 추론**
   - BC (Behavior Cloning): 가장 빠름
   - Diffusion (Conditional/Temporal): 가장 정확
   - GMM/GMR: 확률적 샘플링

3. **Stiffness 출력**
   - 9D vector: [th_k1, th_k2, th_k3, if_k1, if_k2, if_k3, mf_k1, mf_k2, mf_k3]
   - Moving average 스무딩으로 안정화
   - Min/max 클램핑으로 안전 범위 유지

## 빠른 시작

### 1. 빌드
```bash
cd ~/ros2_ws/icra2025
colcon build --packages-select hri_falcon_robot_bridge
source install/setup.bash
```

### 2. 실행
```bash
# BC 모델로 기본 실행
ros2 run hri_falcon_robot_bridge run_policy_node

# Diffusion 모델, 100Hz
ros2 run hri_falcon_robot_bridge run_policy_node \
  --ros-args \
  -p model_type:=diffusion_c \
  -p rate_hz:=100.0
```

### 3. 토픽 모니터링
```bash
# Stiffness 명령 확인
ros2 topic echo /impedance_control/target_stiffness

# 센서 데이터 확인
ros2 topic list | grep -E "force_sensor|ee_pose|deformity"
```

## 다음 단계

1. **Impedance Controller 통합**
   - `/impedance_control/target_stiffness` 토픽을 구독하는 실제 컨트롤러 구현
   - Cartesian impedance control: `tau = J^T * (Kp*(goal-current) - Kd*vel)`

2. **실시간 모니터링**
   ```bash
   # rqt_plot으로 시각화
   rqt_plot /impedance_control/target_stiffness/data[0]:data[3]:data[6]
   ```

3. **성능 평가**
   - rosbag으로 데이터 기록
   - 예측 stiffness와 실제 필요 stiffness 비교

## 참고사항

- **관측 벡터 (19D)**: s1/s2/s3 force (9D) + deform_ecc (1D) + ee positions (9D)
- **행동 벡터 (9D)**: th/if/mf 각 3D stiffness (k1/k2/k3)
- **자동 artifact 탐지**: `outputs/models/policy_learning_unified/artifacts/` 최신 디렉토리
- **Type hints 경고**: Pylance 경고는 있지만 실행에 문제없음
