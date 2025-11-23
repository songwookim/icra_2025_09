#!/usr/bin/env python3
"""
Quick test script for run_policy_node - checks if it can load a model.
"""
import sys
from pathlib import Path

# Add package to path
pkg_root = Path(__file__).parents[1]
sys.path.insert(0, str(pkg_root))

from hri_falcon_robot_bridge.run_policy_node import RunPolicyNode

import rclpy


def test_model_loading():
    """Test if the node can initialize and load a BC model."""
    print("[TEST] Initializing ROS2...")
    rclpy.init()

    try:
        print("[TEST] Creating RunPolicyNode with BC model...")
        node = RunPolicyNode()

        # Check if model loaded
        if node.model is not None:
            print("[✓] Model loaded successfully")
            print(f"    Model type: {node.model_type}")
            print(f"    Artifact dir: {node.artifact_dir}")
            print(f"    Rate: {node.rate_hz} Hz")
        else:
            print("[✗] Model is None - loading failed")
            return False

        # Check scalers
        if node.obs_scaler is not None:
            print("[✓] Observation scaler loaded")
        else:
            print("[!] No observation scaler (may use raw values)")

        if node.act_scaler is not None:
            print("[✓] Action scaler loaded")
        else:
            print("[!] No action scaler (may use raw values)")

        print("\n[TEST] Node initialization complete!")
        return True

    except Exception as e:
        print(f"[✗] Error during test: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    success = test_model_loading()
    sys.exit(0 if success else 1)
