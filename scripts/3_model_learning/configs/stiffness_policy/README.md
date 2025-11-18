# Stiffness Policy Hyperparameter Cheat Sheet

이 디렉터리는 `scripts/3_model_learning/run_stiffness_policy_benchmarks.py` 실행 시
조정하기 좋은 주요 하이퍼파라미터를 모델별 YAML로 정리한 것입니다. 스크립트는
현재 YAML을 자동으로 읽지는 않으며, 여기 정리된 이름을 그대로 CLI 인수에
전달해 수동으로 조정하면 됩니다.

- `training_defaults.yaml`: 전 모델에 공통으로 영향을 주는 전역 제어.
- `gmm.yaml`: GMM / GMR이 참고하는 혼합 모델 구성.
- `behavior_cloning.yaml`: MLP 기반 BC 모델 구조 및 학습률.
- `diffusion_conditional.yaml`: 조건부(비시퀀스) 확산 정책.
- `diffusion_temporal.yaml`: 시퀀스를 사용하는 확산 정책.
- `ibc.yaml`: Implicit BC (score-based) 학습 및 샘플링 파라미터.
- `lstm_gmm.yaml`: LSTM 인코더와 GMM 디코더로 구성된 순차 모델.

필요에 따라 값을 바꾸고, 대응되는 CLI 옵션(예: `--diffusion-epochs`)을 실행할 때
함께 넘겨 주세요. 향후 자동 로드를 원한다면 `--config` 옵션을 추가하는 방식으로
확장할 수 있습니다.
