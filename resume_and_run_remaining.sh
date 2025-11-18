#!/bin/bash
set -e
cd /home/songwoo/ros2_ws/icra2025/src/hri_falcon_robot_bridge

echo "[1/4] GP+MDN (unified, original)"
python3 scripts/3_model_learning/run_stiffness_policy_benchmarks.py \
  --stiffness-dir outputs/analysis/stiffness_profiles \
  --mode unified --models gp,mdn \
  --augment --augment-num 1 \
  --augment-noise-force 0.015 \
  --augment-noise-stiffness 0.04 \
  --augment-temporal-jitter 1

echo "[2/4] All models (unified, global)"
python3 scripts/3_model_learning/run_stiffness_policy_benchmarks.py \
  --stiffness-dir outputs/analysis/stiffness_profiles_global_tk \
  --mode unified --models all \
  --augment --augment-num 1 \
  --augment-noise-force 0.015 \
  --augment-noise-stiffness 0.04 \
  --augment-temporal-jitter 1

echo "[3/4] All models (per-finger, original)"
python3 scripts/3_model_learning/run_stiffness_policy_benchmarks.py \
  --stiffness-dir outputs/analysis/stiffness_profiles \
  --mode per-finger --models all \
  --augment --augment-num 1 \
  --augment-noise-force 0.015 \
  --augment-noise-stiffness 0.04 \
  --augment-temporal-jitter 1

echo "[4/4] All models (per-finger, global)"
python3 scripts/3_model_learning/run_stiffness_policy_benchmarks.py \
  --stiffness-dir outputs/analysis/stiffness_profiles_global_tk \
  --mode per-finger --models all \
  --augment --augment-num 1 \
  --augment-noise-force 0.015 \
  --augment-noise-stiffness 0.04 \
  --augment-temporal-jitter 1

echo "✅ All runs completed."
