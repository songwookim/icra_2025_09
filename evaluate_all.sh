#!/bin/bash
set -e
cd /home/songwoo/ros2_ws/icra2025/src/hri_falcon_robot_bridge

echo "=== Evaluating Unified + Original ==="
python3 scripts/3_model_learning/evaluate_stiffness_policy.py \
  --artifact-dir outputs/models/policy_learning/artifacts/20251119_100949 \
  --models all

echo "=== Evaluating Unified + Global T_K ==="
python3 scripts/3_model_learning/evaluate_stiffness_policy.py \
  --artifact-dir outputs/models/policy_learning_global_tk/artifacts/20251119_112601 \
  --models all

echo "✅ Evaluation completed"
