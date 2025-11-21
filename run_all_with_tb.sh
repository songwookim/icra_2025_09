#!/bin/bash
set -e
cd /home/songwoo/ros2_ws/icra2025/src/hri_falcon_robot_bridge

BASE_ARGS="--augment --augment-num 1 \
  --augment-noise-force 0.015 \
  --augment-noise-stiffness 0.04 \
  --augment-temporal-jitter 1 \
  --gmm-components 5 \
  --bc-hidden 512 --bc-depth 4 \
  --ibc-langevin-steps 10 --ibc-step-size 0.003 \
  --models all --save-predictions --tensorboard"

echo "[1/4] unified + original"
python3 scripts/3_model_learning/run_stiffness_policy_benchmarks.py \
  --stiffness-dir outputs/analysis/stiffness_profiles \
  --mode unified $BASE_ARGS

echo "[2/4] unified + global"
python3 scripts/3_model_learning/run_stiffness_policy_benchmarks.py \
  --stiffness-dir outputs/analysis/stiffness_profiles_global_tk \
  --mode unified $BASE_ARGS

echo "[3/4] per-finger + original"
python3 scripts/3_model_learning/run_stiffness_policy_benchmarks.py \
  --stiffness-dir outputs/analysis/stiffness_profiles \
  --mode per-finger $BASE_ARGS

echo "[4/4] per-finger + global"
python3 scripts/3_model_learning/run_stiffness_policy_benchmarks.py \
  --stiffness-dir outputs/analysis/stiffness_profiles_global_tk \
  --mode per-finger $BASE_ARGS

echo "✅ Completed all four configurations."
