# DMP Multi-Demo Training with Trajectory Alignment

## 📌 개요

사람이 여러 번 반복한 데모 궤적들을 평균내어 하나의 부드러운 DMP 모델을 학습하는 기능이 추가되었습니다.

### 왜 필요한가?

사람이 10번 동작하면 **속도와 타이밍**이 매번 다릅니다.
- 그냥 평균: 중요한 특징(꺾이는 부분)이 뭉개져 버림 → "물타기 현상"
- **시간 정규화 후 평균**: 모든 궤적을 0%→100% 진행률로 맞춘 뒤 평균 → 특징 유지 ✅

---

## 🚀 사용법

### 1. 단일 CSV 모드 (기존 방식)

```bash
python compare_dmp_kf_global.py \
  --csv outputs/demo_001.csv \
  --finger if \
  --n_bfs 50
```

### 2. 다중 CSV 모드 (새로운 기능) ⭐

```bash
python compare_dmp_kf_global.py \
  --csv_pattern "outputs/demo_*.csv" \
  --finger if \
  --n_bfs 50 \
  --target_len 200
```

**옵션 설명:**
- `--csv_pattern`: glob 패턴 (e.g., `"outputs/*.csv"`)
- `--target_len`: 정규화할 표준 길이 (기본값: 200)
- `--visualize_alignment`: 정렬 품질 확인 플롯 출력

**출력:**
- `dmp_if_multi_10demos.pkl` (10개 데모 평균 학습 결과)

---

## 🔍 정렬 품질 확인 (DTW 필요성 판단)

```bash
python compare_dmp_kf_global.py \
  --csv_pattern "outputs/demo_*.csv" \
  --finger if \
  --visualize_alignment
```

**판단 기준:**
- ✅ **피크(꺾이는 점)들이 비슷한 x축 위치에 모여 있다** → 선형 정규화만으로 충분 (현재 구현)
- ❌ **피크들이 중구난방으로 퍼져 있다** → DTW 구현 필요

---

## 📊 시각화 결과 해석

### Multi-Demo Training 결과 그래프

```
Axis X/Y/Z (N demos)
├─ 회색 반투명 선들: 원본 데모들 (x_attr 역산 결과)
├─ 검은 점들: 시간 정규화 후 평균 궤적
└─ 빨간 굵은선: DMP 학습 결과 (부드러운 재생성)
```

**기대 결과:**
- 회색 선들이 비슷한 형태로 겹쳐져 있음
- DMP 출력(빨강)이 평균(검정)을 부드럽게 따라감

---

## 🛠️ 구현 세부사항

### 1단계: 선형 시간 정규화 (현재 구현됨)

```python
def get_mean_trajectory_simple(demo_list, target_len=200):
    """
    모든 데모를 0%→100% 진행률로 치환 후 같은 길이(target_len)로 맞춤
    사람이 '비슷한 속도'로 움직였다면 이것만으로 충분함
    """
    # 각 궤적을 target_len 길이로 선형 보간
    # 평균 계산
```

**장점:**
- 구현 간단, 계산 빠름
- 사람이 비교적 일정한 속도로 움직였다면 80~90점 성능

**단점:**
- 속도 변화가 심한 구간에서는 특징 손실 가능

### 2단계: DTW (필요시 추가 구현)

```python
# 향후 필요시 구현 예정
# fastdtw 라이브러리 활용
# Reference 궤적 선택 → 나머지 Warping
```

---

## 📝 실전 워크플로우

### Step 1: 데이터 수집
```bash
# 10개 데모 녹화 완료
ls outputs/
# demo_001.csv, demo_002.csv, ..., demo_010.csv
```

### Step 2: 정렬 품질 확인
```bash
python compare_dmp_kf_global.py \
  --csv_pattern "outputs/demo_*.csv" \
  --finger if \
  --visualize_alignment
```

→ 그래프 보고 피크 정렬 상태 확인

### Step 3: DMP 학습
```bash
python compare_dmp_kf_global.py \
  --csv_pattern "outputs/demo_*.csv" \
  --finger if \
  --n_bfs 50 \
  --target_len 200
```

→ `dmp_if_multi_10demos.pkl` 생성

### Step 4: ROS 노드 실행
```bash
ros2 run hri_falcon_robot_bridge dmp_desired_node \
  --model-path dmp_models/dmp_if_multi_10demos.pkl
```

---

## 🔧 트러블슈팅

### Q1. "aug 파일도 같이 읽혀요"
A: 자동으로 제외됩니다 (`'aug' not in Path(f).name`)

### Q2. "데모 길이가 너무 다릅니다 (50 vs 500)"
A: `--target_len 200`으로 자동 정규화됩니다

### Q3. "DMP 결과가 지그재그입니다"
A: 
1. `--visualize_alignment`로 정렬 상태 확인
2. 피크들이 중구난방이면 DTW 필요
3. `--n_bfs` 값 조정 (기본 50 → 30~100 시도)

### Q4. "메모리 부족 에러"
A: CSV 파일 수가 너무 많습니다
- 대표적인 10~20개만 선택
- `--target_len` 값 줄이기 (200 → 100)

---

## 📚 참고 자료

**DTW 개념:**
- Dynamic Time Warping: 시간 축을 고무줄처럼 늘려서 두 시계열의 형상을 맞추는 기술
- 노래방 비유: 빠르게 부른 사람 목소리를 빨리 감기 해서 박자 맞추기

**구현 라이브러리 (향후 필요시):**
```bash
pip install fastdtw
```

```python
from fastdtw import fastdtw
# 사용 예시는 DTW 구현 시 추가 예정
```
