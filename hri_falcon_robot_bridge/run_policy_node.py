#!/usr/bin/env python3
"""
Real-time stiffness policy execution node for impedance control.

Subscribes to sensor observations (force, deformity, EE poses),
predicts stiffness using trained BC/Diffusion/GMM model,
and publishes impedance control commands.

Topics subscribed:
- `/force_sensor/s{1..3}/wrench` (geometry_msgs/WrenchStamped)
- `/deformity_tracker/eccentricity` (std_msgs/Float32)
- `/ee_pose_{if|mf|th}` (geometry_msgs/PoseStamped)

Topics published:
- `/impedance_control/target_stiffness` (std_msgs/Float32MultiArray) - 9D stiffness [th_k1..k3, if_k1..k3, mf_k1..k3]

Parameters:
- `model_type` (str): bc, diffusion_c, diffusion_t, gmm, gmr
- `mode` (str): unified or per-finger
- `artifact_dir` (str): path to model artifacts (auto-detect if empty)
- `rate_hz` (float): control loop rate (default: 50.0)
- `stiffness_scale` (float): scale factor for predictions (default: 1.0)
- `stiffness_min` (float): minimum stiffness clamp (default: 0.0)
- `stiffness_max` (float): maximum stiffness clamp (default: 1000.0)
- `smooth_window` (int): moving average window size (default: 5)
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rclpy
from scipy.signal import butter, lfilter  # For low-pass filter
import threading
import select
import termios
import tty
import time
from geometry_msgs.msg import PoseStamped, WrenchStamped
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Float32, Float32MultiArray, String

# Try importing torch
try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Package root for finding models
_THIS_FILE = Path(__file__).resolve()
_PKG_ROOT = _THIS_FILE.parents[1]  # hri_falcon_robot_bridge package root
_MODELS_ROOT = _PKG_ROOT / "outputs" / "models"

# Source scripts path (for importing benchmark classes like IBCBaseline, DiffusionPolicyBaseline)
# Try multiple possible locations (installed vs source)
# Path structure varies depending on how the node is invoked:
#   - As executable: /install/hri_falcon_robot_bridge/lib/hri_falcon_robot_bridge/run_policy_node -> parents[4] = workspace root
#   - As Python module: /install/hri_falcon_robot_bridge/lib/python3.10/site-packages/hri_falcon_robot_bridge/run_policy_node.py -> parents[6] = workspace root
_SCRIPTS_PATH_CANDIDATES = [
    _PKG_ROOT / "scripts" / "3_model_learning",  # From installed package (unlikely)
    _THIS_FILE.parents[4] / "src" / "hri_falcon_robot_bridge" / "scripts" / "3_model_learning",  # Executable in lib/hri_falcon_robot_bridge/
    _THIS_FILE.parents[6] / "src" / "hri_falcon_robot_bridge" / "scripts" / "3_model_learning",  # Python module in lib/python3.10/site-packages/
    Path("/home/songwoo/ros2_ws/icra2025/src/hri_falcon_robot_bridge/scripts/3_model_learning"),  # Fallback absolute
]
_SCRIPTS_PATH = None
for candidate in _SCRIPTS_PATH_CANDIDATES:
    if candidate.exists():
        _SCRIPTS_PATH = candidate
        break


class BehaviorCloningModel(nn.Module):
    """Simple BC model matching the training script structure."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256, depth: int = 3):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for _ in range(max(1, depth)):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        return self.net(obs)


class RunPolicyNode(Node):
    """ROS2 node for real-time stiffness policy execution."""

    # Observation feature names (19D for unified mode)
    OBS_COLUMNS = [
        "s1_fx", "s1_fy", "s1_fz",
        "s2_fx", "s2_fy", "s2_fz",
        "s3_fx", "s3_fy", "s3_fz",
        "deform_ecc",
        "ee_if_px", "ee_if_py", "ee_if_pz",
        "ee_mf_px", "ee_mf_py", "ee_mf_pz",
        "ee_th_px", "ee_th_py", "ee_th_pz",
    ]

    # Action feature names (default 9D stiffness for unified mode)
    ACTION_COLUMNS = [
        "th_k1", "th_k2", "th_k3",
        "if_k1", "if_k2", "if_k3",
        "mf_k1", "mf_k2", "mf_k3",
    ]

    def __init__(self):
        super().__init__("run_policy_node")
        
        # Create reentrant callback group to allow parallel execution of callbacks
        self.callback_group = ReentrantCallbackGroup()

        # Declare all parameters
        self.declare_parameter("model_type", "lstm_gmm")  # Default: lstm_gmm (best R²=0.9961)
        self.declare_parameter("mode", "unified")
        self.declare_parameter("artifact_dir", "")
        self.declare_parameter("rate_hz", 30.0)  # Match camera FPS (30Hz) for sync
        self.declare_parameter("stiffness_scale", 1.0)
        self.declare_parameter("stiffness_min", 0.0)
        self.declare_parameter("stiffness_max", 1000.0)
        self.declare_parameter("smooth_window", 5)
        # Low-pass filter parameters (Butterworth) - 2Hz cutoff for faster response while still smooth
        self.declare_parameter("lowpass_enabled", False)  # Disable LP filter for faster response
        self.declare_parameter("lowpass_cutoff_hz", 5.0)  # Cutoff frequency in Hz (higher = faster response)
        self.declare_parameter("lowpass_order", 2)  # Filter order (2 = good balance)
        # Time-based stiffness scaling parameters
        self.declare_parameter("time_ramp_duration", 8.0)  # seconds to ramp up (increased for gentler start)
        self.declare_parameter("initial_stiffness_scale", 0.1)  # initial 10% (reduced from 30%)
        self.declare_parameter("final_stiffness_scale", 1.0)  # final 100%
        # (NEW) Sensor topic parameters for flexibility
        self.declare_parameter("force_topics", [
            "/force_sensor/s1/wrench",
            "/force_sensor/s2/wrench",
            "/force_sensor/s3/wrench"
        ])
        self.declare_parameter("deform_topic", "/deformity_tracker/eccentricity")
        self.declare_parameter("ee_pose_if_topic", "/ee_pose_if")
        self.declare_parameter("ee_pose_mf_topic", "/ee_pose_mf")
        self.declare_parameter("ee_pose_th_topic", "/ee_pose_th")
        self.declare_parameter("debug_inputs", True)
        self.declare_parameter("debug_topic_scan", True)
        # Force influence: how much actual force affects observation (vs training mean)
        # 1.0 = full actual force, 0.0 = training mean only (no force variation)
        self.declare_parameter("force_influence", 1.0)

        # Get parameters (add explicit typing + None fallbacks for static analysis clarity)
        def _p(name: str):
            val = self.get_parameter(name).value
            return val

        self.model_type: str = str(_p("model_type") or "bc")
        self.mode: str = str(_p("mode") or "unified")
        self.artifact_dir: str = str(_p("artifact_dir") or "")
        # Env override (workaround for launch argument propagation issue)
        env_artifact = os.environ.get("POLICY_ARTIFACT_DIR", "").strip()
        if env_artifact:
            self.get_logger().info(f"Env POLICY_ARTIFACT_DIR override detected -> {env_artifact}")
            self.artifact_dir = env_artifact
        self.rate_hz: float = float(_p("rate_hz") or 50.0)
        self.stiffness_scale: float = float(_p("stiffness_scale") or 1.0)
        self.stiffness_min: float = float(_p("stiffness_min") or 0.0)
        self.stiffness_max: float = float(_p("stiffness_max") or 1000.0)
        self.smooth_window: int = max(1, int(_p("smooth_window") or 5))
        
        # Low-pass filter parameters
        self.lowpass_enabled: bool = bool(_p("lowpass_enabled") if _p("lowpass_enabled") is not None else True)
        self.lowpass_cutoff_hz: float = float(_p("lowpass_cutoff_hz") or 1.0)
        self.lowpass_order: int = int(_p("lowpass_order") or 2)
        
        ft = _p("force_topics") or []
        self.force_topics: List[str] = list(ft)
        self.deform_topic: str = str(_p("deform_topic") or "/deformity_tracker/eccentricity")
        self.ee_pose_if_topic: str = str(_p("ee_pose_if_topic") or "/ee_pose_if")
        self.ee_pose_mf_topic: str = str(_p("ee_pose_mf_topic") or "/ee_pose_mf")
        self.ee_pose_th_topic: str = str(_p("ee_pose_th_topic") or "/ee_pose_th")
        self.debug_inputs: bool = bool(_p("debug_inputs") if _p("debug_inputs") is not None else True)
        
        # Time-based stiffness scaling
        self.time_ramp_duration: float = float(_p("time_ramp_duration") or 5.0)
        self.initial_scale: float = float(_p("initial_stiffness_scale") or 0.3)
        self.final_scale: float = float(_p("final_stiffness_scale") or 1.0)
        self.policy_start_time: Optional[float] = None  # Set when first prediction starts
        
        # Mock mode removed: always wait for real sensor data
        self.allow_mock_missing: bool = False
        self.mock_start_timeout_sec: float = 0.0

        # Log resolved sensor topics & core numeric params once
        self.get_logger().info(
            "Resolved params: rate_hz=%.2f scale=%.2f clamp=[%.1f, %.1f] smooth_window=%d" % (
                self.rate_hz, self.stiffness_scale, self.stiffness_min, self.stiffness_max, self.smooth_window
            )
        )
        self.get_logger().info(
            f"Sensor topics: forces={self.force_topics}, deform={self.deform_topic}, EE(if|mf|th)=({self.ee_pose_if_topic}|{self.ee_pose_mf_topic}|{self.ee_pose_th_topic})"
        )

        # State holders for sensor data
        self.forces = [None, None, None]  # s1, s2, s3
        self.deform_ecc_raw = None  # RAW eccentricity for observation (policy uses this!)
        self.deform_ecc_smoothed = None  # Smoothed eccentricity for plotting only
        self.ee_positions = {"if": None, "mf": None, "th": None}
        
        # === FORCE BASELINE CALIBRATION (matches training data preprocessing) ===
        # Training data uses: Fc = F - F[:10%].mean() (baseline removal)
        # We collect first N samples at startup to compute baseline, then subtract it.
        self.force_calibration_samples = 50  # ~0.5s at 100Hz (first 10% equivalent)
        self.force_baseline_buffer: List[List[Dict]] = [[], [], []]  # Per-sensor buffer
        self.force_baselines: List[Optional[Dict]] = [None, None, None]  # Computed baselines
        self.force_calibrated = False  # Flag: baseline computed?
        
        # === FORCE INFLUENCE CONTROL ===
        # Force has 6x more influence than ecc on stiffness prediction.
        # force_influence=1.0: use actual force (full influence)
        # force_influence=0.0: use training mean (no influence, ecc-only behavior)
        # force_influence=0.5: blend 50% actual + 50% training mean
        self.force_influence = float(_p("force_influence") if _p("force_influence") is not None else 1.0)
        
        # === FORCE AMPLIFICATION ===
        # Live force range is often smaller than training data (~20-50% of training std).
        # force_amplify > 1.0: amplify force values to match training distribution
        # force_amplify=2.0: double the force values (compensate for ~50% live/train ratio)
        self.force_amplify = 2.  # Default 2x amplification
        
        # Training data force means (after baseline removal, these should be ~0)
        # From scaler: baseline-removed force should center around 0
        self.force_training_mean = {
            's1': {'fx': 0.0, 'fy': 0.0, 'fz': 0.0},  # After baseline removal
            's2': {'fx': 0.0, 'fy': 0.0, 'fz': 0.0},
            's3': {'fx': 0.0, 'fy': 0.0, 'fz': 0.0},
        }
        
        # === Z-SCORE AMPLIFICATION ===
        # Model outputs z-scores in narrow range (~0.7σ: -1.84 to -1.13), but training data has ~4σ range.
        # Amplify z-scores before inverse_transform to get fuller stiffness range.
        # z_amp=1.0: no change, z_amp=2.0: amplify z-scores by 2.0x
        # With z_amp=2.0: z-scores -1.84~-1.13 become -3.68~-2.26 → stiffness range 48~48 (still low!)
        # Need to shift the center, not just amplify. Try z_amp=1.5 first.
        self.stiffness_z_amp = 1.5  # Moderate amplification
        
        # Deformity smoothing buffer (moving average for obs input)
        self.deform_buffer: List[float] = []
        self.deform_buffer_size = 5  # Enable smoothing (5-sample moving average)

        # Stiffness prediction buffer for smoothing
        self.stiffness_buffer: List[np.ndarray] = []
        
        # Low-pass filter state (Butterworth IIR filter)
        self._lp_b: Optional[np.ndarray] = None
        self._lp_a: Optional[np.ndarray] = None
        self._lp_zi: Optional[np.ndarray] = None  # Filter state for each dimension
        self._lp_initialized = False
        if self.lowpass_enabled:
            self._init_lowpass_filter()

        # === DEBUG LOGGING: Save observations & predictions for analysis ===
        self.debug_log_data: List[Dict] = []  # Store all data for CSV export
        self.debug_log_enabled = True  # Enable/disable logging
        self.debug_log_max_samples = 10000  # Max samples to keep in memory

        # [NOTE] Model architecture: Seq2One (16 obs → 1 action)
        # - Input: sequence of 16 past observations (temporal context via GRU encoder)
        # - Output: single action for current timestep (NOT action chunking!)
        # The temporal ensembling code below is ONLY for backward compatibility with
        # action-chunking models (action_horizon > 1). Current diffusion_t outputs (1, 9).
        self.action_horizon = 1  # Seq2One: single action output (NOT 16!)
        self.action_buffer: List[np.ndarray] = []  # Not used for Seq2One
        self.exec_horizon = 1  # how many steps to execute per prediction
        self.action_step_counter = 0  # tracks current step
        self.max_action_buffer = 1  # Not used for Seq2One
        
        # [CRITICAL] Observation history buffer for temporal models (diffusion_t)
        # diffusion_t uses GRU encoder that needs sequence of past observations
        self.sequence_window = 4  # Default value, updated from manifest in _load_model()
        self.obs_history: List[np.ndarray] = []  # Ring buffer of past scaled observations
        
        # [DEBUG] Sensor callback counters
        self._force_callback_count = [0, 0, 0]
        self._deform_callback_count = 0
        self._ee_callback_count = {"if": 0, "mf": 0, "th": 0}

        # Model and scalers
        self.model: Optional[Any] = None
        self.obs_scaler: Optional[Any] = None
        self.act_scaler: Optional[Any] = None
        self.manifest: Optional[Dict[str, Any]] = None

        # Load model
        self._load_model()

        # Setup ROS subscribers
        self._setup_subscribers()

        # Publisher for stiffness commands
        self.stiffness_pub = self.create_publisher(
            Float32MultiArray, "/impedance_control/target_stiffness", 10
        )
        
        # Publisher for raw eccentricity (re-publish for torque controller logging)
        self.raw_ecc_pub = self.create_publisher(
            Float32, "/deformity_tracker/eccentricity", 10
        )
        
        # Publisher for smoothed eccentricity (for comparison in plots)
        self.smoothed_ecc_pub = self.create_publisher(
            Float32, "/deformity_tracker/eccentricity_smoothed", 10
        )
        
        # Publisher for model name (for logging/plotting)
        self.model_name_pub = self.create_publisher(
            String, "/policy/model_name", 10
        )
        # Publish model name immediately and periodically
        self._publish_model_name()

        # Control timer (using reentrant callback group)
        period = 1.0 / self.rate_hz
        self.timer = self.create_timer(period, self._control_callback, callback_group=self.callback_group)

        # Topic scan timer (diagnostics)
        self.debug_topic_scan: bool = bool(_p("debug_topic_scan") if _p("debug_topic_scan") is not None else True)
        self._scan_counter = 0
        if self.debug_topic_scan:
            self._scan_timer = self.create_timer(2.0, self._scan_topics, callback_group=self.callback_group)
        else:
            self._scan_timer = None

        # Logging counter
        self._log_counter = 0
        # Ready announcement flag
        self._ready_announced = False
        # (NEW) one-shot debug flags
        self._force_logged = [False, False, False]
        self._deform_logged = False
        self._ee_logged = {"if": False, "mf": False, "th": False}
        self._sensor_waiting_logged = False  # One-time sensor waiting warning

        self.get_logger().info(
            f"RunPolicy node started: model={self.model_type}, mode={self.mode}, "
            f"rate={self.rate_hz}Hz, artifact={self.artifact_dir}"
        )
        # Mock fallback state
        self._mock_activated = False
        self._node_start_time = time.time()
        
        # Register shutdown callback to save debug log
        import atexit
        atexit.register(self._save_debug_log)

    def _setup_subscribers(self):
        """Setup ROS topic subscribers for sensor data."""
        # Force sensors (3 sensors) using parameterized topics
        for i in range(3):
            topic = self.force_topics[i] if i < len(self.force_topics) else f"/force_sensor/s{i+1}/wrench"
            self.create_subscription(
                WrenchStamped, topic, lambda msg, idx=i: self._on_force(idx, msg), 10,
                callback_group=self.callback_group
            )
            if self.debug_inputs:
                self.get_logger().info(f"Subscribed force[{i}] -> {topic}")

        # Deformity eccentricity
        self.create_subscription(
            Float32, self.deform_topic, self._on_deform_ecc, 10,
            callback_group=self.callback_group
        )
        if self.debug_inputs:
            self.get_logger().info(f"Subscribed deform_ecc -> {self.deform_topic}")

        # End-effector positions (3 fingers)
        self.create_subscription(
            PoseStamped, self.ee_pose_if_topic, self._on_ee_pose_if, 10,
            callback_group=self.callback_group
        )
        self.create_subscription(
            PoseStamped, self.ee_pose_mf_topic, self._on_ee_pose_mf, 10,
            callback_group=self.callback_group
        )
        self.create_subscription(
            PoseStamped, self.ee_pose_th_topic, self._on_ee_pose_th, 10,
            callback_group=self.callback_group
        )
        if self.debug_inputs:
            self.get_logger().info(
                f"Subscribed EE poses -> if:{self.ee_pose_if_topic}, mf:{self.ee_pose_mf_topic}, th:{self.ee_pose_th_topic}"
            )

    def _on_force(self, idx: int, msg: WrenchStamped):
        """Callback for force sensor data.
        
        [BASELINE CALIBRATION] Collects first N samples to compute force baseline,
        matching training data preprocessing: Fc = F - F[:10%].mean()
        """
        try:
            w = msg.wrench
            force_dict = {
                "fx": w.force.x,
                "fy": w.force.y,
                "fz": w.force.z,
                "tx": w.torque.x,
                "ty": w.torque.y,
                "tz": w.torque.z,
            }
            
            # Store raw force values
            self.forces[idx] = force_dict
            
            # === BASELINE CALIBRATION: collect samples for baseline computation ===
            if not self.force_calibrated:
                self.force_baseline_buffer[idx].append(force_dict.copy())
                
                # Check if all sensors have enough samples
                all_ready = all(
                    len(buf) >= self.force_calibration_samples 
                    for buf in self.force_baseline_buffer
                )
                if all_ready:
                    self._compute_force_baselines()
            
            self._force_callback_count[idx] += 1
            if self.debug_inputs and not self._force_logged[idx]:
                self.get_logger().info(
                    f"Force s{idx+1} first msg fx={w.force.x:.2f} fy={w.force.y:.2f} fz={w.force.z:.2f}"
                )
                self._force_logged[idx] = True
        except Exception as e:
            self.get_logger().warning(f"Force callback error (s{idx+1}): {e}")
    
    def _compute_force_baselines(self):
        """Compute force baseline from collected samples (first N samples mean).
        
        This matches training data preprocessing:
        Fc = F - F[:rest].mean() where rest = int(len(F) * 0.1)
        """
        for idx in range(3):
            buf = self.force_baseline_buffer[idx]
            if len(buf) == 0:
                continue
            
            # Compute mean of each force component
            baseline = {}
            for key in ["fx", "fy", "fz", "tx", "ty", "tz"]:
                values = [sample[key] for sample in buf]
                baseline[key] = sum(values) / len(values)
            
            self.force_baselines[idx] = baseline
            self.get_logger().info(
                f"[FORCE CALIBRATION] s{idx+1} baseline: "
                f"fx={baseline['fx']:.3f} fy={baseline['fy']:.3f} fz={baseline['fz']:.3f}"
            )
        
        self.force_calibrated = True
        self.get_logger().info(
            f"[FORCE CALIBRATION] Complete! Baseline computed from {self.force_calibration_samples} samples per sensor."
        )

    def _publish_model_name(self):
        """Publish the current model name for logging/plotting purposes."""
        msg = String()
        msg.data = self.model_type
        self.model_name_pub.publish(msg)

    def _on_deform_ecc(self, msg: Float32):
        """Callback for deformity eccentricity. Smoothed value goes to observation."""
        raw_value = msg.data
        
        # Store RAW value for logging/comparison
        self.deform_ecc_raw = raw_value
        
        # Publish RAW eccentricity for torque controller logging
        self.raw_ecc_pub.publish(Float32(data=raw_value))
        
        # Add to buffer for smoothing
        self.deform_buffer.append(raw_value)
        if len(self.deform_buffer) > self.deform_buffer_size:
            self.deform_buffer.pop(0)
        
        # Compute smoothed value - THIS IS USED IN OBSERVATION!
        self.deform_ecc_smoothed = sum(self.deform_buffer) / len(self.deform_buffer)
        
        # Publish smoothed eccentricity for comparison in plots
        self.smoothed_ecc_pub.publish(Float32(data=self.deform_ecc_smoothed))
        
        self._deform_callback_count += 1
        if self.debug_inputs and not self._deform_logged:
            self.get_logger().info(f"Deform eccentricity first msg raw={raw_value:.3f}, smoothed={self.deform_ecc_smoothed:.3f}")
            self._deform_logged = True

    def _on_ee_pose_if(self, msg: PoseStamped):
        """Callback for index finger end-effector pose."""
        try:
            pos = msg.pose.position
            self.ee_positions["if"] = np.array([pos.x, pos.y, pos.z], dtype=np.float32)
            self._ee_callback_count["if"] += 1
            if self.debug_inputs and not self._ee_logged["if"]:
                self.get_logger().info(f"EE if first msg pos=({pos.x:.3f},{pos.y:.3f},{pos.z:.3f})")
                self._ee_logged["if"] = True
        except Exception as e:
            self.get_logger().warning(f"EE pose callback error (if): {e}")

    def _on_ee_pose_mf(self, msg: PoseStamped):
        """Callback for middle finger end-effector pose."""
        try:
            pos = msg.pose.position
            self.ee_positions["mf"] = np.array([pos.x, pos.y, pos.z], dtype=np.float32)
            self._ee_callback_count["mf"] += 1
            if self.debug_inputs and not self._ee_logged["mf"]:
                self.get_logger().info(f"EE mf first msg pos=({pos.x:.3f},{pos.y:.3f},{pos.z:.3f})")
                self._ee_logged["mf"] = True
        except Exception as e:
            self.get_logger().warning(f"EE pose callback error (mf): {e}")

    def _on_ee_pose_th(self, msg: PoseStamped):
        """Callback for thumb end-effector pose."""
        try:
            pos = msg.pose.position
            self.ee_positions["th"] = np.array([pos.x, pos.y, pos.z], dtype=np.float32)
            self._ee_callback_count["th"] += 1
            if self.debug_inputs and not self._ee_logged["th"]:
                self.get_logger().info(f"EE th first msg pos=({pos.x:.3f},{pos.y:.3f},{pos.z:.3f})")
                self._ee_logged["th"] = True
        except Exception as e:
            self.get_logger().warning(f"EE pose callback error (th): {e}")

    def _find_latest_artifact(self) -> Optional[str]:
        """Auto-detect latest artifact directory for the specified mode.
        
        Supports model_type formats:
        - Simple: 'lstm_gmm', 'bc', 'diffusion_t'
        - Benchmark sweep: 'lstm_gmm_seq4', 'diffusion_t_seq4_h1', etc.
        """
        # First, check if model_type matches a benchmark_sweep subfolder
        benchmark_sweep_dir = _MODELS_ROOT / "benchmark_sweep"
        if benchmark_sweep_dir.exists():
            # Check for exact match (e.g., 'diffusion_t_seq4_h1')
            sweep_model_dir = benchmark_sweep_dir / self.model_type
            if sweep_model_dir.exists():
                # Find artifacts inside
                artifacts_dir = sweep_model_dir / "policy_learning_unified" / "artifacts"
                if artifacts_dir.exists():
                    dirs = sorted([d for d in artifacts_dir.iterdir() if d.is_dir()])
                    if dirs:
                        artifact_path = str(dirs[-1])  # Latest
                        self.get_logger().info(f"[AUTO-DETECT] Found benchmark_sweep model: {artifact_path}")
                        return artifact_path
            
            # Check for partial match (e.g., 'lstm_gmm' -> 'lstm_gmm_seq4')
            for subdir in benchmark_sweep_dir.iterdir():
                if subdir.is_dir() and subdir.name.startswith(self.model_type):
                    artifacts_dir = subdir / "policy_learning_unified" / "artifacts"
                    if artifacts_dir.exists():
                        dirs = sorted([d for d in artifacts_dir.iterdir() if d.is_dir()])
                        if dirs:
                            artifact_path = str(dirs[-1])
                            self.get_logger().info(f"[AUTO-DETECT] Found matching model: {subdir.name} -> {artifact_path}")
                            return artifact_path
        
        # Fallback: check standard paths
        search_paths = [
            _MODELS_ROOT / f"policy_learning_{self.mode}" / "artifacts",
            _PKG_ROOT.parents[2] / "outputs" / "models" / f"policy_learning_{self.mode}" / "artifacts",
        ]

        for search_dir in search_paths:
            if not search_dir.exists():
                continue

            # Find latest timestamped directory
            dirs = sorted([d for d in search_dir.iterdir() if d.is_dir()])
            if not dirs:
                continue

            # Check from newest to oldest
            for artifact_dir in reversed(dirs):
                # Check if model file exists
                model_files = {
                    "bc": "bc.pt",
                    "diffusion_c": "diffusion_c.pt",
                    "diffusion_t": "diffusion_t.pt",
                    "ibc": "ibc.pt",
                    "gmm": "gmm.pkl",
                    "gmr": "gmm.pkl",
                }

                model_file = artifact_dir / model_files.get(self.model_type, "")
                if model_file.exists():
                    return str(artifact_dir)

        return None

    def _load_model(self):
        """Load trained model and scalers from artifact directory."""
        # Parse model_type to determine base type and parameters
        # e.g., 'diffusion_t_seq4_h1' -> base='diffusion_t', seq=4, horizon=1
        #       'lstm_gmm_seq4' -> base='lstm_gmm', seq=4
        self._parse_model_type()
        
        # Auto-detect if not specified
        if not self.artifact_dir:
            self.artifact_dir = self._find_latest_artifact()

        if not self.artifact_dir:
            raise RuntimeError(
                f"Could not find model artifacts for {self.model_type} in {self.mode} mode"
            )

        artifact_path = Path(self.artifact_dir)
        self.get_logger().info(f"Loading model from: {artifact_path}")

        # Load manifest (optional)
        manifest_path = artifact_path / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, "r") as f:
                self.manifest = json.load(f)
                self.get_logger().info(f"Loaded manifest: {self.manifest}")
                
                # Extract action horizon and sequence window from manifest if available
                model_config = self.manifest.get("models", {}).get(self._base_model_type, {})
                if model_config.get("temporal", False):
                    seq_len = model_config.get("seq_len", 16)
                    self.sequence_window = seq_len  # Update from manifest!
                    # [FIX] action_horizon is separate from seq_len!
                    # seq_len = input observation sequence length (e.g., 4 or 16)
                    # action_horizon = output action sequence length (Seq2One = 1)
                    self.action_horizon = model_config.get("action_horizon", 1)  # Default 1 for Seq2One
                    self.get_logger().info(f"[TEMPORAL] sequence_window={self.sequence_window}, action_horizon={self.action_horizon}")

        # Load scalers
        scaler_path = artifact_path / "scalers.pkl"
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                scalers = pickle.load(f)
                # Handle different key names
                self.obs_scaler = scalers.get("obs_scaler") or scalers.get("obs")
                self.act_scaler = scalers.get("act_scaler") or scalers.get("act")
            self.get_logger().info("Loaded observation and action scalers")
            
            # === FIX: Clip scaler scale_ to prevent extreme z-scores ===
            # Some features (e.g., ee_if_px) have near-zero std in training data,
            # causing extreme z-scores (>10) for minor deviations in live data.
            # Clip scale_ to minimum value to prevent OOD scaled values.
            if self.obs_scaler is not None and hasattr(self.obs_scaler, 'scale_'):
                min_scale = 0.01  # Minimum scale (1cm for position, 0.01N for force)
                original_scales = self.obs_scaler.scale_.copy()
                self.obs_scaler.scale_ = np.maximum(self.obs_scaler.scale_, min_scale)
                clipped = np.where(original_scales < min_scale)[0]
                if len(clipped) > 0:
                    obs_names = [
                        's1_fx', 's1_fy', 's1_fz', 's2_fx', 's2_fy', 's2_fz',
                        's3_fx', 's3_fy', 's3_fz', 'deform_ecc',
                        'ee_if_px', 'ee_if_py', 'ee_if_pz',
                        'ee_mf_px', 'ee_mf_py', 'ee_mf_pz',
                        'ee_th_px', 'ee_th_py', 'ee_th_pz',
                    ]
                    clipped_names = [obs_names[i] if i < len(obs_names) else f"idx{i}" for i in clipped]
                    self.get_logger().warning(
                        f"[SCALER FIX] Clipped scale_ for features with near-zero std: {clipped_names}"
                    )
                    for idx in clipped:
                        name = obs_names[idx] if idx < len(obs_names) else f"idx{idx}"
                        self.get_logger().info(
                            f"  {name}: scale {original_scales[idx]:.6f} -> {min_scale:.4f}"
                        )
        else:
            self.get_logger().warning("No scalers found - using raw values (may degrade performance)")

        # Load model based on BASE type (not full model_type)
        base = self._base_model_type
        if base in ["bc", "diffusion_c", "diffusion_t", "diffusion_t_ddim", "ibc"]:
            if not TORCH_AVAILABLE:
                raise RuntimeError(f"PyTorch required for {self.model_type} model but not available")
            self._load_torch_model(artifact_path)
        elif base.startswith("lstm_gmm"):
            if not TORCH_AVAILABLE:
                raise RuntimeError(f"PyTorch required for {self.model_type} model but not available")
            self._load_lstm_gmm_model(artifact_path)
        elif base in ["gmm", "gmr"]:
            self._load_gmm_model(artifact_path)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type} (base: {base})")

        self.get_logger().info(f"Model loaded successfully: {self.model_type} (base: {base})")

    def _parse_model_type(self):
        """Parse model_type to extract base type and parameters.
        
        Examples:
        - 'diffusion_t_seq4_h1' -> base='diffusion_t', seq=4, horizon=1
        - 'lstm_gmm_seq4' -> base='lstm_gmm', seq=4
        - 'bc' -> base='bc'
        """
        import re
        
        self._base_model_type = self.model_type
        self._parsed_seq_len = None
        self._parsed_horizon = None
        
        # Pattern for diffusion_t_seq{N}_h{M}
        diffusion_match = re.match(r'^(diffusion_[ct])_seq(\d+)_h(\d+)$', self.model_type)
        if diffusion_match:
            self._base_model_type = diffusion_match.group(1)
            self._parsed_seq_len = int(diffusion_match.group(2))
            self._parsed_horizon = int(diffusion_match.group(3))
            self.sequence_window = self._parsed_seq_len
            self.get_logger().info(
                f"[PARSE] {self.model_type} -> base={self._base_model_type}, "
                f"seq={self._parsed_seq_len}, horizon={self._parsed_horizon}"
            )
            return
        
        # Pattern for lstm_gmm_seq{N}
        lstm_match = re.match(r'^(lstm_gmm)_seq(\d+)$', self.model_type)
        if lstm_match:
            self._base_model_type = lstm_match.group(1)
            self._parsed_seq_len = int(lstm_match.group(2))
            self.sequence_window = self._parsed_seq_len
            self.get_logger().info(
                f"[PARSE] {self.model_type} -> base={self._base_model_type}, seq={self._parsed_seq_len}"
            )
            return
        
        # No parsing needed for simple types
        self.get_logger().info(f"[PARSE] {self.model_type} -> base={self._base_model_type} (no params)")


    def _load_torch_model(self, artifact_path: Path):
        """Load PyTorch-based model (BC, Diffusion, or IBC)."""
        model_map = {
            "bc": "bc.pt",
            "diffusion_c": "diffusion_c.pt",
            "diffusion_t": "diffusion_t.pt",
            "diffusion_t_ddim": "diffusion_t.pt",
            "ibc": "ibc.pt",
        }
        # Use base model type (parsed from model_type like diffusion_t_seq4_h1 -> diffusion_t)
        base_type = self._base_model_type
        if base_type not in model_map:
            raise KeyError(f"Unknown base model type: {base_type} (from {self.model_type})")
        model_path = artifact_path / model_map[base_type]
        
        # [PERFORMANCE] Use GPU if available for faster inference
        self.inference_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"[DEVICE] Using {self.inference_device.upper()} for model inference")
        
        checkpoint = torch.load(model_path, map_location=self.inference_device)

        if base_type == "bc":
            # Reconstruct BC model from config
            config = checkpoint.get("config", {})
            obs_dim = config.get("obs_dim", 19)
            act_dim = config.get("act_dim", 9)
            hidden_dim = config.get("hidden_dim", 256)
            depth = config.get("depth", 3)

            self.model = BehaviorCloningModel(obs_dim, act_dim, hidden_dim, depth)
            state_dict_key = "state_dict" if "state_dict" in checkpoint else "model_state_dict"
            self.model.load_state_dict(checkpoint[state_dict_key])
            self.model.eval()

            self.get_logger().info(
                f"BC model: obs_dim={obs_dim}, act_dim={act_dim}, hidden={hidden_dim}, depth={depth}"
            )

        elif "diffusion" in base_type:
            # Diffusion model requires DiffusionPolicyBaseline class
            # For simplicity, we'll attempt to import from the benchmark script
            if _SCRIPTS_PATH and str(_SCRIPTS_PATH) not in sys.path:
                sys.path.insert(0, str(_SCRIPTS_PATH))
            try:
                from run_stiffness_policy_benchmarks import DiffusionPolicyBaseline

                # Reconstruct diffusion model
                config = checkpoint.get("config", {})
                # Determine if temporal based on base_type suffix ('c' = False, 't' = True)
                is_temporal = base_type.split("_")[1] == "t" if "_" in base_type else False
                
                # [FIX] Get action_horizon from config (for action chunking models)
                action_horizon = config.get("action_horizon", 1)
                self.action_horizon = action_horizon  # Store for inference
                
                self.model = DiffusionPolicyBaseline(
                    obs_dim=config.get("obs_dim", 19),
                    act_dim=config.get("act_dim", 9),
                    hidden_dim=config.get("hidden_dim", 256),
                    time_dim=config.get("time_dim", 16),
                    timesteps=config.get("timesteps", 100),
                    temporal=is_temporal,
                    action_horizon=action_horizon,  # [FIX] Pass action_horizon!
                    device=self.inference_device,  # [PERFORMANCE] Use GPU
                )
                # Try different possible keys for state dict
                state_dict_key = "model_state_dict" if "model_state_dict" in checkpoint else "state_dict"
                self.model.model.load_state_dict(checkpoint[state_dict_key])
                self.model.model.eval()

                self.get_logger().info(
                    f"Diffusion model loaded: {self.model_type}, timesteps={config.get('timesteps')}, temporal={is_temporal}"
                )

            except ImportError as e:
                self.get_logger().error(f"Failed to import DiffusionPolicyBaseline: {e}")
                raise RuntimeError(
                    "Diffusion model requires DiffusionPolicyBaseline from run_stiffness_policy_benchmarks.py"
                )

        elif base_type == "ibc":
            # IBC (Implicit Behavior Cloning) - energy-based model with Langevin sampling
            if _SCRIPTS_PATH and str(_SCRIPTS_PATH) not in sys.path:
                sys.path.insert(0, str(_SCRIPTS_PATH))
            try:
                from run_stiffness_policy_benchmarks import IBCBaseline

                config = checkpoint.get("config", {})
                self.model = IBCBaseline(
                    obs_dim=config.get("obs_dim", 19),
                    act_dim=config.get("act_dim", 9),
                    hidden_dim=config.get("hidden_dim", 256),
                    depth=config.get("depth", 3),
                    noise_std=config.get("noise_std", 0.5),
                    langevin_steps=config.get("langevin_steps", 30),
                    step_size=config.get("step_size", 1e-2),
                    device=self.inference_device,
                )
                state_dict_key = "model_state_dict" if "model_state_dict" in checkpoint else "state_dict"
                self.model.model.load_state_dict(checkpoint[state_dict_key])
                self.model.model.eval()

                self.get_logger().info(
                    f"IBC model loaded: obs_dim={config.get('obs_dim', 19)}, "
                    f"langevin_steps={config.get('langevin_steps', 30)}"
                )

            except ImportError as e:
                self.get_logger().error(f"Failed to import IBCBaseline: {e}")
                raise RuntimeError(
                    "IBC model requires IBCBaseline from run_stiffness_policy_benchmarks.py"
                )

    def _load_lstm_gmm_model(self, artifact_path: Path):
        """Load LSTM-GMM model."""
        model_path = artifact_path / "lstm_gmm.pt"
        if not model_path.exists():
            # Try alternative naming
            for alt in ["lstm_gmm_seq4.pt", "lstm_gmm_seq8.pt", "lstm_gmm_seq16.pt"]:
                alt_path = artifact_path / alt
                if alt_path.exists():
                    model_path = alt_path
                    break
        
        if not model_path.exists():
            raise FileNotFoundError(f"LSTM-GMM model not found in {artifact_path}")
        
        # Import LSTMGMMHead from benchmark script
        sys.path.insert(0, str(_PKG_ROOT / "scripts" / "3_model_learning"))
        from run_stiffness_policy_benchmarks import LSTMGMMHead
        
        self.inference_device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(model_path, map_location=self.inference_device)
        
        config = checkpoint.get("config", {})
        obs_dim = config.get("obs_dim", 19)
        act_dim = config.get("act_dim", 9)
        hidden_dim = config.get("hidden_dim", 256)
        n_layers = config.get("n_layers", 1)
        n_components = config.get("n_components", 5)
        self.sequence_window = config.get("seq_len", 4)  # Update sequence window from config
        
        self.model = LSTMGMMHead(obs_dim, act_dim, hidden_dim, n_layers, n_components)
        state_dict_key = "state_dict" if "state_dict" in checkpoint else "model_state_dict"
        self.model.load_state_dict(checkpoint[state_dict_key])
        self.model.to(self.inference_device)
        self.model.eval()
        
        self.get_logger().info(
            f"LSTM-GMM model loaded: obs_dim={obs_dim}, act_dim={act_dim}, "
            f"hidden={hidden_dim}, layers={n_layers}, components={n_components}, seq_len={self.sequence_window}"
        )

    def _load_gmm_model(self, artifact_path: Path):
        """Load GMM/GMR model (GMMConditional with built-in predict method)."""
        model_path = artifact_path / "gmm.pkl"
        
        # GMMConditional class must be available for unpickling
        if _SCRIPTS_PATH and str(_SCRIPTS_PATH) not in sys.path:
            sys.path.insert(0, str(_SCRIPTS_PATH))
        
        try:
            from run_stiffness_policy_benchmarks import GMMConditional
            
            # CRITICAL: Register GMMConditional in globals for pickle to find it
            # pickle looks for classes in the module where the pickle was loaded,
            # so we need to make it available in the current namespace
            import __main__
            __main__.GMMConditional = GMMConditional
            
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)  # This is a GMMConditional object
            
            self.get_logger().info(
                f"GMM/GMR model loaded from {model_path}: "
                f"obs_dim={getattr(self.model, 'obs_dim', 'N/A')}, "
                f"act_dim={getattr(self.model, 'act_dim', 'N/A')}"
            )
        except ImportError as e:
            self.get_logger().error(f"Failed to import GMMConditional: {e}")
            raise RuntimeError(
                "GMM/GMR model requires GMMConditional from run_stiffness_policy_benchmarks.py"
            )

    def _get_observation(self) -> Optional[np.ndarray]:
        """Construct 19D observation vector from current sensor data.
        
        Uses RAW eccentricity (matches training data, no smoothing applied).
        [BASELINE REMOVAL] Force values have baseline subtracted (matches training preprocessing).
        """
        # ALL sensors are REQUIRED: forces, ee_positions, deform_ecc_raw
        forces_ready = all(f is not None for f in self.forces)
        ee_ready = all((p is not None and isinstance(p, np.ndarray) and p.size == 3 and np.all(np.isfinite(p)))
                       for p in self.ee_positions.values())
        deform_ready = self.deform_ecc_raw is not None and isinstance(self.deform_ecc_raw, (int, float))
        
        # Also require force calibration to be complete
        calibration_ready = self.force_calibrated
        
        if not (forces_ready and ee_ready and deform_ready and calibration_ready):
            return None

        obs = []

        # Force features (9D: s1/s2/s3 fx/fy/fz) - WITH BASELINE REMOVAL + INFLUENCE CONTROL
        sensor_keys = ['s1', 's2', 's3']
        for i in range(3):
            f = self.forces[i]
            baseline = self.force_baselines[i]
            if f is not None and baseline is not None:
                # Subtract baseline: Fc = F - baseline (matches training preprocessing)
                fx_raw = f["fx"] - baseline["fx"]
                fy_raw = f["fy"] - baseline["fy"]
                fz_raw = f["fz"] - baseline["fz"]
                
                # Apply force amplification: scale up to match training data distribution
                # Live force is typically ~20-50% of training std, so amplify to compensate
                fx_amp = fx_raw * self.force_amplify
                fy_amp = fy_raw * self.force_amplify
                fz_amp = fz_raw * self.force_amplify
                
                # Apply force influence: blend actual force with training mean
                # force_influence=1.0: full actual force
                # force_influence=0.0: training mean (effectively removes force influence)
                train_mean = self.force_training_mean[sensor_keys[i]]
                alpha = self.force_influence
                fx = alpha * fx_amp + (1 - alpha) * train_mean['fx']
                fy = alpha * fy_amp + (1 - alpha) * train_mean['fy']
                fz = alpha * fz_amp + (1 - alpha) * train_mean['fz']
                
                obs.extend([fx, fy, fz])
            else:
                # Should not happen since we gate on forces_ready and calibration_ready
                obs.extend([0.0, 0.0, 0.0])

        # Eccentricity (1D) - USE RAW VALUE (matches training data, no smoothing)
        obs.append(self.deform_ecc_raw if self.deform_ecc_raw is not None else 0.0)

        # End-effector positions (9D: if/mf/th px/py/pz)
        for finger in ["if", "mf", "th"]:
            pos = self.ee_positions[finger]
            if pos is not None:
                obs.extend(pos.tolist())
            else:
                obs.extend([0.0, 0.0, 0.0])

        return np.array(obs, dtype=np.float32)

    def _predict_stiffness(self, obs: np.ndarray) -> np.ndarray:
        """Predict stiffness from observation using loaded model (9D)."""
        import time as time_module
        t_start = time_module.time()
        
        # [DEBUG] Log raw observation (every 2 seconds)
        if self._log_counter % int(self.rate_hz * 2) == 0:
            self.get_logger().info(
                f"[POLICY] Raw obs (first 6): {obs[:6]}, "
                f"force_cb=[{self._force_callback_count[0]},{self._force_callback_count[1]},{self._force_callback_count[2]}], "
                f"deform_cb={self._deform_callback_count}, "
                f"ee_cb=[{self._ee_callback_count['if']},{self._ee_callback_count['mf']},{self._ee_callback_count['th']}]"
            )
        
        # Scale observation
        if self.obs_scaler:
            obs_scaled = self.obs_scaler.transform(obs.reshape(1, -1))
        else:
            obs_scaled = obs.reshape(1, -1)

        # [DEBUG] Log scaled observation (every 2 seconds)
        if self._log_counter % int(self.rate_hz * 2) == 0:
            self.get_logger().info(f"[POLICY] Scaled obs (first 6): {obs_scaled[0, :6]}")

        # Predict based on model type
        if self.model_type == "bc":
            with torch.no_grad():
                obs_t = torch.from_numpy(obs_scaled.astype(np.float32))
                act_scaled = self.model(obs_t).numpy()

        elif "diffusion" in self.model_type:
            # Diffusion policy with temporal ensembling
            # [PERFORMANCE FIX] Force DDIM sampler for faster inference (10x speedup)
            sampler = "ddim"  # Force DDIM instead of DDPM
            if self.manifest:
                model_config = self.manifest.get("models", {}).get(self.model_type, {})
                manifest_sampler = model_config.get("sampler", "ddpm")
                # Only use manifest sampler if it's already DDIM
                if manifest_sampler == "ddim":
                    sampler = manifest_sampler
                else:
                    # self.get_logger().warn(
                    #     f"[PERFORMANCE] Overriding sampler '{manifest_sampler}' → 'ddim' for real-time performance"
                    # )
                    pass
            
            # [CRITICAL] Use fewer inference steps for real-time performance
            n_inference_steps = 10  # 75 -> 10 steps = ~7x speedup
            
            # Check if model is temporal (diffusion_t) or non-temporal (diffusion_c)
            is_temporal = getattr(self.model, 'temporal', False) or self.model_type.startswith("diffusion_t")
            
            if is_temporal:
                # [TEMPORAL MODEL: diffusion_t] Build observation sequence
                self.obs_history.append(obs_scaled[0].copy())  # (obs_dim,)
                if len(self.obs_history) > self.sequence_window:
                    self.obs_history.pop(0)  # Keep only last sequence_window observations
                
                # Create sequence input: (1, seq_len, obs_dim)
                if len(self.obs_history) < self.sequence_window:
                    # Pad with first observation if not enough history
                    pad_count = self.sequence_window - len(self.obs_history)
                    padded_hist = [self.obs_history[0]] * pad_count + list(self.obs_history)
                    obs_input = np.array(padded_hist)[np.newaxis, :, :]  # (1, seq_len, obs_dim)
                else:
                    obs_input = np.array(list(self.obs_history))[np.newaxis, :, :]  # (1, seq_len, obs_dim)
                
                if self._log_counter % int(self.rate_hz * 0.5) == 0:
                    self.get_logger().info(
                        f"[POLICY] diffusion_t obs_seq shape: {obs_input.shape}, history_len: {len(self.obs_history)}"
                    )
            else:
                # [NON-TEMPORAL MODEL: diffusion_c] Use 2D input directly
                obs_input = obs_scaled  # (1, obs_dim)
                
                if self._log_counter % int(self.rate_hz * 0.5) == 0:
                    self.get_logger().info(
                        f"[POLICY] diffusion_c obs shape: {obs_input.shape}"
                    )
            
            # Predict action
            action_seq = self.model.predict(
                obs_input, 
                n_samples=1, 
                sampler=sampler, 
                eta=0.0,
                n_inference_steps=n_inference_steps  # [NEW] Fast inference
            )
            
            # [Seq2One] Model outputs single action: (1, 9) or (1, 1, 9)
            # No temporal ensembling needed - just use the output directly
            if len(action_seq.shape) == 3:
                # Squeeze if model returns (1, 1, 9)
                act_scaled = action_seq[0, 0:1, :]  # (1, 9)
            else:
                # Already (1, 9)
                act_scaled = action_seq.reshape(1, -1)
            
            self.action_step_counter += 1
            
            # [DEBUG] Log action output (every 0.5 seconds)
            if self._log_counter % int(self.rate_hz * 0.5) == 0:
                self.get_logger().info(
                    f"[POLICY] Seq2One output shape: {action_seq.shape}, act_scaled[:3]={act_scaled[0, :3]}"
                )

        elif self.model_type.startswith("lstm_gmm"):
            # LSTM-GMM: requires sequence of observations
            self.obs_history.append(obs_scaled[0].copy())
            if len(self.obs_history) > self.sequence_window:
                self.obs_history.pop(0)
            
            # Pad if needed
            if len(self.obs_history) < self.sequence_window:
                pad_count = self.sequence_window - len(self.obs_history)
                padded_hist = [self.obs_history[0]] * pad_count + list(self.obs_history)
                obs_seq = np.array(padded_hist, dtype=np.float32)[np.newaxis, :, :]
            else:
                obs_seq = np.array(list(self.obs_history), dtype=np.float32)[np.newaxis, :, :]
            
            with torch.no_grad():
                seq_tensor = torch.from_numpy(obs_seq).to(self.inference_device)
                mean, logvar, logits = self.model(seq_tensor)
                # Weighted mean prediction (most stable)
                weights = torch.softmax(logits, dim=-1)
                pred = torch.sum(weights.unsqueeze(-1) * mean, dim=1)
                act_scaled = pred.cpu().numpy()
            
            if self._log_counter % int(self.rate_hz * 0.5) == 0:
                self.get_logger().info(
                    f"[LSTM-GMM] seq shape: {obs_seq.shape}, pred: {act_scaled[0, :3]}"
                )

        elif self.model_type == "gmm":
            # GMM: Conditional sampling from mixture (sample from predicted distribution)
            # GMMConditional.predict with mode="sample" samples from conditional distribution
            act_scaled = self.model.predict(obs_scaled, mode="sample", n_samples=1)
            
            if self._log_counter % int(self.rate_hz * 0.5) == 0:
                self.get_logger().info(
                    f"[GMM] obs shape: {obs_scaled.shape}, pred: {act_scaled[0, :3]}"
                )

        elif self.model_type == "gmr":
            # GMR: Gaussian Mixture Regression - conditional expectation (mean)
            # GMMConditional.predict with mode="mean" computes weighted mean of conditional
            act_scaled = self.model.predict(obs_scaled, mode="mean")
            
            if self._log_counter % int(self.rate_hz * 0.5) == 0:
                self.get_logger().info(
                    f"[GMR] obs shape: {obs_scaled.shape}, pred: {act_scaled[0, :3]}"
                )

        elif self.model_type == "ibc":
            # IBC (Implicit Behavior Cloning) - energy-based model with Langevin sampling
            # Uses 2D input: (B, obs_dim) -> (B, act_dim)
            act_scaled = self.model.predict(obs_scaled, n_samples=1)
            
            if self._log_counter % int(self.rate_hz * 0.5) == 0:
                self.get_logger().info(
                    f"[IBC] obs shape: {obs_scaled.shape}, pred: {act_scaled[0, :3]}"
                )

        else:
            act_scaled = np.zeros((1, 9))

        # [DEBUG] Log scaled action (every 2 seconds)
        if self._log_counter % int(self.rate_hz * 2) == 0:
            self.get_logger().info(f"[POLICY] Scaled action (act_scaled): {act_scaled[0, :3]}")

        # === Z-SCORE AMPLIFICATION ===
        # Model outputs narrow z-score range. Amplify to get fuller stiffness range.
        # act_scaled is already z-score (StandardScaler normalized), so we can directly amplify.
        if self.stiffness_z_amp != 1.0:
            act_scaled = act_scaled * self.stiffness_z_amp
            if self._log_counter % int(self.rate_hz * 2) == 0:
                self.get_logger().info(f"[POLICY] Amplified z-score (x{self.stiffness_z_amp}): {act_scaled[0, :3]}")

        # Inverse scale
        if self.act_scaler:
            stiffness = self.act_scaler.inverse_transform(act_scaled)
        else:
            stiffness = act_scaled

        # [DEBUG] Log inverse-transformed stiffness (before smoothing)
        if self._log_counter % int(self.rate_hz * 2) == 0:
            t_elapsed = (time_module.time() - t_start) * 1000  # ms
            self.get_logger().info(f"[POLICY] Raw stiffness (before smooth): {stiffness[0, :3]}, prediction_time={t_elapsed:.1f}ms")

        return stiffness.flatten()

    # _predict_action removed (emg_bc no longer supported)

    def _init_lowpass_filter(self) -> None:
        """Initialize Butterworth low-pass filter coefficients."""
        try:
            nyquist = self.rate_hz / 2.0
            normalized_cutoff = self.lowpass_cutoff_hz / nyquist
            normalized_cutoff = min(normalized_cutoff, 0.99)  # Ensure valid range
            
            self._lp_b, self._lp_a = butter(self.lowpass_order, normalized_cutoff, btype='low')
            # [FIX] Initialize filter state to None - will be set on first sample
            self._lp_zi = None  # Will be initialized with first sample value
            self._lp_initialized = True
            self._lp_first_sample = True  # Flag to initialize with first sample
            
            self.get_logger().info(
                f"Low-pass filter initialized: cutoff={self.lowpass_cutoff_hz}Hz, "
                f"order={self.lowpass_order}, sample_rate={self.rate_hz}Hz"
            )
        except Exception as e:
            self.get_logger().error(f"Failed to initialize low-pass filter: {e}")
            self.lowpass_enabled = False

    def _apply_lowpass_filter(self, stiffness: np.ndarray) -> np.ndarray:
        """Apply real-time low-pass filter to stiffness prediction."""
        if not self.lowpass_enabled or not self._lp_initialized:
            return stiffness
        
        try:
            # [FIX] Initialize filter state on first sample to avoid transient
            if self._lp_first_sample:
                from scipy.signal import lfilter_zi
                zi = lfilter_zi(self._lp_b, self._lp_a)
                # Initialize state so that first output equals first input (no transient)
                self._lp_zi = np.zeros((len(zi), len(stiffness)))
                for dim in range(len(stiffness)):
                    self._lp_zi[:, dim] = zi * stiffness[dim]
                self._lp_first_sample = False
                return stiffness  # Return first sample unchanged
            
            # Apply IIR filter with state preservation for real-time processing
            filtered = np.zeros_like(stiffness)
            for dim in range(len(stiffness)):
                # lfilter returns (filtered_value, new_zi)
                filtered_val, self._lp_zi[:, dim] = lfilter(
                    self._lp_b, self._lp_a, 
                    [stiffness[dim]], 
                    zi=self._lp_zi[:, dim]
                )
                filtered[dim] = filtered_val[0]
            return filtered
        except Exception as e:
            # Fallback to raw value on error
            return stiffness

    def _smooth_stiffness(self, stiffness: np.ndarray) -> np.ndarray:
        """Apply low-pass filter OR moving average smoothing (not both to avoid over-smoothing)."""
        # [FIX] Use ONLY LP filter if enabled, skip moving average (avoid double smoothing)
        if self.lowpass_enabled:
            stiffness = self._apply_lowpass_filter(stiffness)
            # Skip moving average when LP is enabled to avoid over-smoothing
            smoothed = np.clip(stiffness, self.stiffness_min, self.stiffness_max)
            return smoothed
        
        # Fallback: use moving average smoothing if LP disabled
        self.stiffness_buffer.append(stiffness)

        if len(self.stiffness_buffer) > self.smooth_window:
            self.stiffness_buffer.pop(0)

        # Compute mean
        smoothed = np.mean(self.stiffness_buffer, axis=0)

        # Clamp stiffness (NOTE: stiffness_scale removed - use launch param or model scaling instead)
        smoothed = np.clip(smoothed, self.stiffness_min, self.stiffness_max)

        return smoothed

    def _scan_topics(self):
        try:
            # Ready check without ambiguous NumPy truth-values
            forces_ready = all(f is not None for f in self.forces)
            ee_ready = all(v is not None for v in self.ee_positions.values())
            if forces_ready and self.deform_ecc_smoothed is not None and ee_ready:
                # All ready; stop scanning
                if self._scan_timer is not None:
                    self._scan_timer.cancel()
                return
            # Check which topics are absent from the ROS graph
            names_types = dict(self.get_topic_names_and_types())
            absent_force = [t for t in self.force_topics if t not in names_types]
            absent_ee = [t for t in [self.ee_pose_if_topic, self.ee_pose_mf_topic, self.ee_pose_th_topic] if t not in names_types]
            if absent_force:
                self.get_logger().warning(f"Force topics absent in graph: {absent_force}")
            if absent_ee:
                self.get_logger().warning(f"EE pose topics absent in graph: {absent_ee}")
        except Exception as e:
            self.get_logger().warning(f"Topic scan error: {e}")

    def _control_callback(self):
        """Main control loop callback - runs at rate_hz frequency."""
        # Get current observation
        obs = self._get_observation()

        # [DEBUG] Log callback entry
        # if self._log_counter % int(self.rate_hz * 2) == 0:
        #     self.get_logger().info(f"[DEBUG] _control_callback entered, obs={'OK' if obs is not None else 'NONE'}")

        if obs is None:
            # Not enough data yet - skip this iteration
            self._log_counter += 1  # Increment first so the condition below works
            if not self._sensor_waiting_logged:
                missing = []
                if not all(f is not None for f in self.forces):
                    missing_idx = [i for i in range(3) if self.forces[i] is None]
                    missing.append(f"forces({[self.force_topics[i] for i in missing_idx]})")
                if self.deform_ecc_smoothed is None:
                    missing.append(f"deform_ecc({self.deform_topic})")
                if not all(v is not None for v in self.ee_positions.values()):
                    missing_fingers = [k for k,v in self.ee_positions.items() if v is None]
                    missing.append(f"ee_poses({missing_fingers})")
                self.get_logger().warning(
                    f"[INIT] Waiting for sensor data: {', '.join(missing)}"
                )
                self._sensor_waiting_logged = True
            return

        try:
            # Start timing when first observation is ready
            if self.policy_start_time is None:
                self.policy_start_time = time.time()
                self.get_logger().info(
                    f"[TIME_SCALE] Policy started! Ramping {self.initial_scale:.0%} → {self.final_scale:.0%} over {self.time_ramp_duration:.1f}s"
                )
            
            if not self._ready_announced:
                self.get_logger().info("All required sensor inputs received -> starting policy predictions")
                self._ready_announced = True

            # Compute time-based scale factor (ramp up over time)
            elapsed = time.time() - self.policy_start_time
            ramp_progress = min(elapsed / self.time_ramp_duration, 1.0)
            time_scale = self.initial_scale + (self.final_scale - self.initial_scale) * ramp_progress

            # Predict stiffness (9D)
            stiffness = self._predict_stiffness(obs)

            # [CHANGED] time_scale is now applied in torque_impedance_controller, not here
            # This ensures stiffness.csv logs the raw policy output for analysis
            stiffness_before_scale = stiffness.copy()
            # stiffness = stiffness * time_scale  # REMOVED - applied in controller
            
            # [DEBUG] Save pre-LP filter value for comparison
            stiffness_before_lp = stiffness.copy()

            # Smooth and scale (LP filter applied here)
            stiffness = self._smooth_stiffness(stiffness)

            # === DEBUG: Collect data for analysis ===
            if self.debug_log_enabled and len(self.debug_log_data) < self.debug_log_max_samples:
                f1, f2, f3 = self.forces[0], self.forces[1], self.forces[2]
                # Scale observation for comparison
                obs_scaled = self.obs_scaler.transform(obs.reshape(1, -1))[0] if self.obs_scaler else obs
                
                log_entry = {
                    'time_s': time.time() - self.policy_start_time,
                    # Raw observation (19D)
                    's1_fx': f1['fx'], 's1_fy': f1['fy'], 's1_fz': f1['fz'],
                    's2_fx': f2['fx'], 's2_fy': f2['fy'], 's2_fz': f2['fz'],
                    's3_fx': f3['fx'], 's3_fy': f3['fy'], 's3_fz': f3['fz'],
                    'deform_ecc': self.deform_ecc_smoothed,
                    'ee_if_px': self.ee_positions['if'][0], 'ee_if_py': self.ee_positions['if'][1], 'ee_if_pz': self.ee_positions['if'][2],
                    'ee_mf_px': self.ee_positions['mf'][0], 'ee_mf_py': self.ee_positions['mf'][1], 'ee_mf_pz': self.ee_positions['mf'][2],
                    'ee_th_px': self.ee_positions['th'][0], 'ee_th_py': self.ee_positions['th'][1], 'ee_th_pz': self.ee_positions['th'][2],
                    # Scaled observation (for model input comparison)
                    's1_fz_scaled': obs_scaled[2], 's2_fz_scaled': obs_scaled[5], 's3_fz_scaled': obs_scaled[8],
                    'deform_ecc_scaled': obs_scaled[9],
                    # Policy output stages: raw -> time_scaled -> LP_filtered
                    'th_k1_raw': stiffness_before_scale[0], 'th_k2_raw': stiffness_before_scale[1], 'th_k3_raw': stiffness_before_scale[2],
                    'mf_k1_raw': stiffness_before_scale[6], 'mf_k2_raw': stiffness_before_scale[7], 'mf_k3_raw': stiffness_before_scale[8],
                    # Before LP filter (after time_scale)
                    'th_k1_pre_lp': stiffness_before_lp[0], 'th_k2_pre_lp': stiffness_before_lp[1], 'th_k3_pre_lp': stiffness_before_lp[2],
                    'mf_k1_pre_lp': stiffness_before_lp[6], 'mf_k2_pre_lp': stiffness_before_lp[7], 'mf_k3_pre_lp': stiffness_before_lp[8],
                    # After LP filter (final published value)
                    'th_k1': stiffness[0], 'th_k2': stiffness[1], 'th_k3': stiffness[2],
                    'if_k1': stiffness[3], 'if_k2': stiffness[4], 'if_k3': stiffness[5],
                    'mf_k1': stiffness[6], 'mf_k2': stiffness[7], 'mf_k3': stiffness[8],
                    'time_scale': time_scale,
                    'lp_enabled': self.lowpass_enabled,  # Track if LP was actually enabled
                }
                self.debug_log_data.append(log_entry)

            # [DEBUG] After smoothing (every 2 seconds)
            if self._log_counter % int(self.rate_hz * 2) == 0:
                # Log observation values for debugging
                f1, f2, f3 = self.forces[0], self.forces[1], self.forces[2]
                self.get_logger().info(
                    f"[OBS] Force: s1=({f1['fx']:.2f},{f1['fy']:.2f},{f1['fz']:.2f}), "
                    f"s2=({f2['fx']:.2f},{f2['fy']:.2f},{f2['fz']:.2f}), "
                    f"s3=({f3['fx']:.2f},{f3['fy']:.2f},{f3['fz']:.2f})"
                )
                self.get_logger().info(
                    f"[OBS] ecc={self.deform_ecc_smoothed:.3f}, "
                    f"EE_z: if={self.ee_positions['if'][2]:.4f}, mf={self.ee_positions['mf'][2]:.4f}, th={self.ee_positions['th'][2]:.4f}"
                )
                self.get_logger().info(
                    f"[POLICY] time_scale={time_scale:.0%}, stiffness={stiffness[:3]} (th), {stiffness[3:6]} (if), {stiffness[6:9]} (mf)"
                )
                # Training data reference
                self.get_logger().info(
                    f"[REF] Training force range: s1_fz=[-3.2,0.1], s2_fz=[-4.8,0.1], s3_fz=[-5.6,0.1]"
                )
                self.get_logger().info(
                    f"[DEBUG] Logged {len(self.debug_log_data)} samples so far"
                )

            # Publish stiffness command
            msg = Float32MultiArray()
            msg.data = stiffness.tolist()
            self.stiffness_pub.publish(msg)

            # Increment counter after successful prediction
            self._log_counter += 1

        except Exception as e:
            self.get_logger().error(f"Control callback error: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())

    def _save_debug_log(self):
        """Save collected debug data to CSV for analysis."""
        if not self.debug_log_data:
            return
        
        try:
            import pandas as pd
            from datetime import datetime
            
            # Create output directory
            output_dir = Path(_PKG_ROOT) / "outputs" / "policy_debug_logs"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"policy_debug_{timestamp}.csv"
            
            # Save to CSV
            df = pd.DataFrame(self.debug_log_data)
            df.to_csv(output_path, index=False)
            
            print(f"\\n{'='*60}")
            print(f"[DEBUG LOG SAVED] {output_path}")
            print(f"  Samples: {len(self.debug_log_data)}")
            print(f"  Duration: {df['time_s'].max():.1f}s")
            print(f"\\n  Force ranges:")
            print(f"    s1_fz: [{df['s1_fz'].min():.3f}, {df['s1_fz'].max():.3f}]  (training: [-3.2, 0.1])")
            print(f"    s2_fz: [{df['s2_fz'].min():.3f}, {df['s2_fz'].max():.3f}]  (training: [-4.8, 0.1])")
            print(f"    s3_fz: [{df['s3_fz'].min():.3f}, {df['s3_fz'].max():.3f}]  (training: [-5.6, 0.1])")
            print(f"\\n  Deformity range:")
            print(f"    ecc: [{df['deform_ecc'].min():.3f}, {df['deform_ecc'].max():.3f}]  (training: [0.02, 0.54])")
            print(f"\\n  Stiffness (mf_k3) range:")
            print(f"    raw: [{df['mf_k3_raw'].min():.1f}, {df['mf_k3_raw'].max():.1f}]")
            print(f"    final: [{df['mf_k3'].min():.1f}, {df['mf_k3'].max():.1f}]  (training: [91, 655])")
            print(f"{'='*60}\\n")
            
        except Exception as e:
            print(f"[ERROR] Failed to save debug log: {e}")


def main(args=None):
    rclpy.init(args=args)

    try:
        node = RunPolicyNode()
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)

        node.get_logger().info("RunPolicy node running. Press Ctrl+C to exit.")
        executor.spin()

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback

        traceback.print_exc()
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
