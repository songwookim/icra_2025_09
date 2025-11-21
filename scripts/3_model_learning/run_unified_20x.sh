#!/bin/bash
# Run Unified mode training with 20x augmentation

cd /home/songwoo/ros2_ws/icra2025/src/hri_falcon_robot_bridge

echo "=== Starting Unified Training with 20x Augmentation ==="
echo "Expected: 90 demos → 1,890 demos (21x total)"
echo "Expected: ~480K samples → ~6.2M samples"
echo ""

python3 scripts/3_model_learning/run_stiffness_policy_benchmarks.py \
  --mode unified \
  --log-dir outputs/logs/success \
  --stiffness-dir outputs/analysis/stiffness_profiles_global_tk \
  --models all \
  --augment \
  --augment-num 20 \
  --bc-epochs 200 \
  --lstm-gmm-epochs 200 \
  --diffusion-epochs 200 \
  --seed 42
