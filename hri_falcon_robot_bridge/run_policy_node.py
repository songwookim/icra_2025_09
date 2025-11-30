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
from std_msgs.msg import Float32, Float32MultiArray

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
        self.declare_parameter("model_type", "diffusion_t")  # Default: diffusion_t (temporal, best performance)
        self.declare_parameter("mode", "unified")
        self.declare_parameter("artifact_dir", "")
        self.declare_parameter("rate_hz", 100.0)
        self.declare_parameter("stiffness_scale", 1.0)
        self.declare_parameter("stiffness_min", 0.0)
        self.declare_parameter("stiffness_max", 1000.0)
        self.declare_parameter("smooth_window", 5)
        # Low-pass filter parameters (Butterworth) - recommended: 3Hz cutoff for smooth stiffness
        self.declare_parameter("lowpass_enabled", True)  # Enable LP filter by default
        self.declare_parameter("lowpass_cutoff_hz", 3.0)  # Cutoff frequency in Hz (best: 3.0Hz)
        self.declare_parameter("lowpass_order", 2)  # Filter order
        # Time-based stiffness scaling parameters
        self.declare_parameter("time_ramp_duration", 5.0)  # seconds to ramp up
        self.declare_parameter("initial_stiffness_scale", 0.3)  # initial 30%
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
        self.lowpass_cutoff_hz: float = float(_p("lowpass_cutoff_hz") or 3.0)
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
        
        # Deformity smoothing buffer (for plotting only, NOT for observation!)
        self.deform_buffer: List[float] = []
        self.deform_buffer_size = 10

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

        # Temporal ensembling for diffusion policy (action chunking)
        self.action_horizon = 16  # prediction horizon from manifest
        self.action_buffer: List[np.ndarray] = []  # stores (horizon, 9) predictions
        self.exec_horizon = 1  # how many steps to execute per prediction
        self.action_step_counter = 0  # tracks current step in action sequence
        
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
        
        # Publisher for smoothed eccentricity (for comparison in plots)
        self.smoothed_ecc_pub = self.create_publisher(
            Float32, "/deformity_tracker/eccentricity_smoothed", 10
        )

        # Attractor publisher (removed emg_bc support)

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
        """Callback for force sensor data."""
        try:
            w = msg.wrench
            self.forces[idx] = {
                "fx": w.force.x,
                "fy": w.force.y,
                "fz": w.force.z,
                "tx": w.torque.x,
                "ty": w.torque.y,
                "tz": w.torque.z,
            }
            self._force_callback_count[idx] += 1
            if self.debug_inputs and not self._force_logged[idx]:
                self.get_logger().info(
                    f"Force s{idx+1} first msg fx={w.force.x:.2f} fy={w.force.y:.2f} fz={w.force.z:.2f}"
                )
                self._force_logged[idx] = True
        except Exception as e:
            self.get_logger().warning(f"Force callback error (s{idx+1}): {e}")

    def _on_deform_ecc(self, msg: Float32):
        """Callback for deformity eccentricity. Smoothed value goes to observation."""
        raw_value = msg.data
        
        # Store RAW value for logging/comparison
        self.deform_ecc_raw = raw_value
        
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
        """Auto-detect latest artifact directory for the specified mode."""
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
                    "gmm": "gmm_model.pkl",
                    "gmr": "gmm_model.pkl",
                }

                model_file = artifact_dir / model_files.get(self.model_type, "")
                if model_file.exists():
                    return str(artifact_dir)

        return None

    def _load_model(self):
        """Load trained model and scalers from artifact directory."""
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
                
                # Extract action horizon from manifest if available
                model_config = self.manifest.get("models", {}).get(self.model_type, {})
                if model_config.get("temporal", False):
                    self.action_horizon = model_config.get("seq_len", 16)
                    self.get_logger().info(f"[TEMPORAL] Action horizon set to {self.action_horizon}")

        # Load scalers
        scaler_path = artifact_path / "scalers.pkl"
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                scalers = pickle.load(f)
                # Handle different key names
                self.obs_scaler = scalers.get("obs_scaler") or scalers.get("obs")
                self.act_scaler = scalers.get("act_scaler") or scalers.get("act")
            self.get_logger().info("Loaded observation and action scalers")
        else:
            self.get_logger().warning("No scalers found - using raw values (may degrade performance)")

        # Load model based on type
        if self.model_type in ["bc", "diffusion_c", "diffusion_t", "diffusion_t_ddim"]:
            if not TORCH_AVAILABLE:
                raise RuntimeError(f"PyTorch required for {self.model_type} model but not available")
            self._load_torch_model(artifact_path)
        elif self.model_type in ["gmm", "gmr"]:
            self._load_gmm_model(artifact_path)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        self.get_logger().info(f"Model loaded successfully: {self.model_type}")

    def _load_torch_model(self, artifact_path: Path):
        """Load PyTorch-based model (BC or Diffusion)."""
        model_map = {
            "bc": "bc.pt",
            "diffusion_c": "diffusion_c.pt",
            "diffusion_t": "diffusion_t.pt",
            "diffusion_t_ddim": "diffusion_t.pt",
        }
        model_path = artifact_path / model_map[self.model_type]
        checkpoint = torch.load(model_path, map_location="cpu")

        if self.model_type == "bc":
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

        elif "diffusion" in self.model_type:
            # Diffusion model requires DiffusionPolicyBaseline class
            # For simplicity, we'll attempt to import from the benchmark script
            sys.path.insert(0, str(_PKG_ROOT / "scripts" / "3_model_learning"))
            try:
                from run_stiffness_policy_benchmarks import DiffusionPolicyBaseline

                # Reconstruct diffusion model
                config = checkpoint.get("config", {})
                # Determine if temporal based on model_type suffix ('c' = False, 't' = True)
                is_temporal = self.model_type.split("_")[1] == "t" if "_" in self.model_type else False
                
                self.model = DiffusionPolicyBaseline(
                    obs_dim=config.get("obs_dim", 19),
                    act_dim=config.get("act_dim", 9),
                    hidden_dim=config.get("hidden_dim", 256),
                    time_dim=config.get("time_dim", 16),
                    timesteps=config.get("timesteps", 100),
                    temporal=is_temporal,
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

    def _load_gmm_model(self, artifact_path: Path):
        """Load GMM/GMR model."""
        model_path = artifact_path / "gmm_model.pkl"

        with open(model_path, "rb") as f:
            gmm_data = pickle.load(f)

        if self.model_type == "gmm":
            self.model = gmm_data.get("gmm")
        else:  # gmr
            self.model = gmm_data  # GMR uses the full dict with regression helpers

        self.get_logger().info(f"GMM/GMR model loaded from {model_path}")

    def _get_observation(self) -> Optional[np.ndarray]:
        """Construct 19D observation vector from current sensor data.
        
        Uses SMOOTHED eccentricity for stable policy predictions.
        """
        # ALL sensors are REQUIRED: forces, ee_positions, deform_ecc_smoothed
        forces_ready = all(f is not None for f in self.forces)
        ee_ready = all((p is not None and isinstance(p, np.ndarray) and p.size == 3 and np.all(np.isfinite(p)))
                       for p in self.ee_positions.values())
        deform_ready = self.deform_ecc_smoothed is not None and isinstance(self.deform_ecc_smoothed, (int, float))
        
        if not (forces_ready and ee_ready and deform_ready):
            return None

        obs = []

        # Force features (9D: s1/s2/s3 fx/fy/fz)
        for i in range(3):
            f = self.forces[i]
            if f is not None:
                obs.extend([f["fx"], f["fy"], f["fz"]])
            else:
                # Should not happen since we gate on forces_ready
                obs.extend([0.0, 0.0, 0.0])

        # Eccentricity (1D) - USE SMOOTHED VALUE for stable policy!
        obs.append(self.deform_ecc_smoothed if self.deform_ecc_smoothed is not None else 0.0)

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
                    self.get_logger().warn(
                        f"[PERFORMANCE] Overriding sampler '{manifest_sampler}' → 'ddim' for real-time performance"
                    )
            
            # Predict action sequence (shape: (1, horizon, 9) or (1, 9))
            action_seq = self.model.predict(obs_scaled, n_samples=1, sampler=sampler, eta=0.0)
            
            # Check if temporal model (returns sequence)
            if len(action_seq.shape) == 3 and action_seq.shape[1] == self.action_horizon:
                # Temporal ensembling: store prediction and average overlapping actions
                self.action_buffer.append(action_seq[0])  # (horizon, 9)
                
                # Keep only recent predictions (sliding window)
                max_buffer = self.action_horizon
                if len(self.action_buffer) > max_buffer:
                    self.action_buffer.pop(0)
                
                # Temporal ensembling: average all predictions for current timestep
                current_step = self.action_step_counter
                valid_actions = []
                for i, pred_seq in enumerate(self.action_buffer):
                    # pred_seq[j] corresponds to timestep (buffer_start + i + j)
                    # We want actions for current_step
                    offset = current_step - (len(self.action_buffer) - 1 - i)
                    if 0 <= offset < len(pred_seq):
                        valid_actions.append(pred_seq[offset])
                
                if valid_actions:
                    act_scaled = np.mean(valid_actions, axis=0, keepdims=True)  # (1, 9)
                else:
                    act_scaled = action_seq[0:1, 0, :]  # fallback to first action
                
                self.action_step_counter += 1
            else:
                # Non-temporal or single action
                act_scaled = action_seq.reshape(1, -1)
                self.action_step_counter += 1

        elif self.model_type == "gmm":
            # GMM sampling
            act_scaled = self.model.sample(1)[0].reshape(1, -1)

        elif self.model_type == "gmr":
            # GMR regression (requires conditional expectation computation)
            # Simplified: assuming the dict has a 'predict' method
            if hasattr(self.model, "predict"):
                act_scaled = self.model.predict(obs_scaled)
            else:
                # Fallback to GMM sampling if no predict method
                gmm = self.model.get("gmm")
                if gmm:
                    act_scaled = gmm.sample(1)[0].reshape(1, -1)
                else:
                    act_scaled = np.zeros((1, 9))

        else:
            act_scaled = np.zeros((1, 9))

        # [DEBUG] Log scaled action (every 2 seconds)
        if self._log_counter % int(self.rate_hz * 2) == 0:
            self.get_logger().info(f"[POLICY] Scaled action (act_scaled): {act_scaled[0, :3]}")

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
            # Initialize filter state for 9D stiffness
            from scipy.signal import lfilter_zi
            zi = lfilter_zi(self._lp_b, self._lp_a)
            self._lp_zi = np.tile(zi[:, np.newaxis], (1, 9))  # (order, 9)
            self._lp_initialized = True
            
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
        """Apply low-pass filter and moving average smoothing to stiffness predictions."""
        # First apply low-pass filter (if enabled)
        if self.lowpass_enabled:
            stiffness = self._apply_lowpass_filter(stiffness)
        
        # Then apply moving average smoothing
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
            if forces_ready and self.deform_ecc is not None and ee_ready:
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

            # Apply time-based scaling BEFORE smoothing
            stiffness_before_scale = stiffness.copy()
            stiffness = stiffness * time_scale

            # Smooth and scale
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
                    # Policy output (before time scale, after time scale, after smoothing)
                    'th_k1_raw': stiffness_before_scale[0], 'th_k2_raw': stiffness_before_scale[1], 'th_k3_raw': stiffness_before_scale[2],
                    'mf_k1_raw': stiffness_before_scale[6], 'mf_k2_raw': stiffness_before_scale[7], 'mf_k3_raw': stiffness_before_scale[8],
                    'th_k1': stiffness[0], 'th_k2': stiffness[1], 'th_k3': stiffness[2],
                    'if_k1': stiffness[3], 'if_k2': stiffness[4], 'if_k3': stiffness[5],
                    'mf_k1': stiffness[6], 'mf_k2': stiffness[7], 'mf_k3': stiffness[8],
                    'time_scale': time_scale,
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
            # self.get_logger().error(f"Control callback error: {e}")
            pass

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
