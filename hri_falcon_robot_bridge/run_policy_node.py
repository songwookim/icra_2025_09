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
- `/impedance_control/target_attractor` (std_msgs/Float32MultiArray) - 9D attractor EE positions [if_xyz, mf_xyz, th_xyz] (emg_bc 전용)

Parameters:
- `model_type` (str): bc, diffusion_c, diffusion_t, gmm, gmr, emg_bc
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
import threading
import select
import termios
import tty
import time
from geometry_msgs.msg import PoseStamped, WrenchStamped
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

        # Declare parameters
        self.declare_parameter("model_type", "bc")
        self.declare_parameter("mode", "unified")
        self.declare_parameter("artifact_dir", "")
        self.declare_parameter("rate_hz", 50.0)
        self.declare_parameter("stiffness_scale", 1.0)
        self.declare_parameter("stiffness_min", 0.0)
        self.declare_parameter("stiffness_max", 1000.0)
        self.declare_parameter("smooth_window", 5)
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
        # (NEW) Manual start via keyboard
        self.declare_parameter("manual_start", True)
        self.declare_parameter("start_key", "p")

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
        ft = _p("force_topics") or []
        self.force_topics: List[str] = list(ft)
        self.deform_topic: str = str(_p("deform_topic") or "/deformity_tracker/eccentricity")
        self.ee_pose_if_topic: str = str(_p("ee_pose_if_topic") or "/ee_pose_if")
        self.ee_pose_mf_topic: str = str(_p("ee_pose_mf_topic") or "/ee_pose_mf")
        self.ee_pose_th_topic: str = str(_p("ee_pose_th_topic") or "/ee_pose_th")
        self.debug_inputs: bool = bool(_p("debug_inputs") if _p("debug_inputs") is not None else True)
        self.manual_start: bool = bool(_p("manual_start") if _p("manual_start") is not None else True)
        self.start_key: str = str(_p("start_key") or "p")

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
        self.deform_ecc = None
        self.ee_positions = {"if": None, "mf": None, "th": None}

        # Stiffness prediction buffer for smoothing
        self.stiffness_buffer: List[np.ndarray] = []

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

        # (NEW) Publisher for attractor EE positions (emg_bc)
        self.attractor_pub = self.create_publisher(
            Float32MultiArray, "/impedance_control/target_attractor", 10
        )

        # Control timer
        period = 1.0 / self.rate_hz
        self.timer = self.create_timer(period, self._control_callback)

        # Topic scan timer (diagnostics)
        self.debug_topic_scan: bool = bool(_p("debug_topic_scan") if _p("debug_topic_scan") is not None else True)
        self._scan_counter = 0
        if self.debug_topic_scan:
            self._scan_timer = self.create_timer(2.0, self._scan_topics)
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

        # (NEW) policy start gate
        self._started = not self.manual_start
        self._start_prompted = False
        if self.manual_start:
            self._start_thread = threading.Thread(target=self._keyboard_waiter, daemon=True)
            self._start_thread.start()

        self.get_logger().info(
            f"RunPolicy node started: model={self.model_type}, mode={self.mode}, "
            f"rate={self.rate_hz}Hz, artifact={self.artifact_dir}"
        )

    def _keyboard_waiter(self):
        """Wait for a single key press (start_key) to start policy."""
        try:
            if not sys.stdin.isatty():
                self.get_logger().warn("stdin is not a TTY; auto-starting policy (manual_start ignored)")
                self._started = True
                return

            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setcbreak(fd)
                self.get_logger().info(f"Press '{self.start_key}' to START policy predictions …")
                while rclpy.ok() and not self._started:
                    # Use select for non-blocking check
                    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
                    if rlist:
                        ch = sys.stdin.read(1)
                        if ch.lower() == self.start_key.lower():
                            self._started = True
                            self.get_logger().info("Policy STARTED by keyboard input")
                            break
                    time.sleep(0.05)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except Exception as e:
            self.get_logger().warn(f"Keyboard waiter error: {e}; auto-starting policy")
            self._started = True

    def _setup_subscribers(self):
        """Setup ROS topic subscribers for sensor data."""
        # Force sensors (3 sensors) using parameterized topics
        for i in range(3):
            topic = self.force_topics[i] if i < len(self.force_topics) else f"/force_sensor/s{i+1}/wrench"
            self.create_subscription(WrenchStamped, topic, lambda msg, idx=i: self._on_force(idx, msg), 10)
            if self.debug_inputs:
                self.get_logger().info(f"Subscribed force[{i}] -> {topic}")

        # Deformity eccentricity
        self.create_subscription(Float32, self.deform_topic, self._on_deform_ecc, 10)
        if self.debug_inputs:
            self.get_logger().info(f"Subscribed deform_ecc -> {self.deform_topic}")

        # End-effector positions (3 fingers)
        self.create_subscription(PoseStamped, self.ee_pose_if_topic, self._on_ee_pose_if, 10)
        self.create_subscription(PoseStamped, self.ee_pose_mf_topic, self._on_ee_pose_mf, 10)
        self.create_subscription(PoseStamped, self.ee_pose_th_topic, self._on_ee_pose_th, 10)
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
            if self.debug_inputs and not self._force_logged[idx]:
                self.get_logger().info(
                    f"Force s{idx+1} first msg fx={w.force.x:.2f} fy={w.force.y:.2f} fz={w.force.z:.2f}"
                )
                self._force_logged[idx] = True
        except Exception as e:
            self.get_logger().warn(f"Force callback error (s{idx+1}): {e}", throttle_duration_sec=5.0)

    def _on_deform_ecc(self, msg: Float32):
        """Callback for deformity eccentricity."""
        self.deform_ecc = msg.data
        if self.debug_inputs and not self._deform_logged:
            self.get_logger().info(f"Deform eccentricity first msg ecc={self.deform_ecc:.3f}")
            self._deform_logged = True

    def _on_ee_pose_if(self, msg: PoseStamped):
        """Callback for index finger end-effector pose."""
        try:
            pos = msg.pose.position
            self.ee_positions["if"] = np.array([pos.x, pos.y, pos.z], dtype=np.float32)
            if self.debug_inputs and not self._ee_logged["if"]:
                self.get_logger().info(f"EE if first msg pos=({pos.x:.3f},{pos.y:.3f},{pos.z:.3f})")
                self._ee_logged["if"] = True
        except Exception as e:
            self.get_logger().warn(f"EE pose callback error (if): {e}", throttle_duration_sec=5.0)

    def _on_ee_pose_mf(self, msg: PoseStamped):
        """Callback for middle finger end-effector pose."""
        try:
            pos = msg.pose.position
            self.ee_positions["mf"] = np.array([pos.x, pos.y, pos.z], dtype=np.float32)
            if self.debug_inputs and not self._ee_logged["mf"]:
                self.get_logger().info(f"EE mf first msg pos=({pos.x:.3f},{pos.y:.3f},{pos.z:.3f})")
                self._ee_logged["mf"] = True
        except Exception as e:
            self.get_logger().warn(f"EE pose callback error (mf): {e}", throttle_duration_sec=5.0)

    def _on_ee_pose_th(self, msg: PoseStamped):
        """Callback for thumb end-effector pose."""
        try:
            pos = msg.pose.position
            self.ee_positions["th"] = np.array([pos.x, pos.y, pos.z], dtype=np.float32)
            if self.debug_inputs and not self._ee_logged["th"]:
                self.get_logger().info(f"EE th first msg pos=({pos.x:.3f},{pos.y:.3f},{pos.z:.3f})")
                self._ee_logged["th"] = True
        except Exception as e:
            self.get_logger().warn(f"EE pose callback error (th): {e}", throttle_duration_sec=5.0)

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
            self.get_logger().warn("No scalers found - using raw values (may degrade performance)")

        # Load model based on type
        if self.model_type in ["bc", "diffusion_c", "diffusion_t", "emg_bc"]:
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
            "emg_bc": "emg_bc.pt",
        }

        model_path = artifact_path / model_map[self.model_type]
        checkpoint = torch.load(model_path, map_location="cpu")

        if self.model_type in ("bc", "emg_bc"):
            # Reconstruct BC model from config
            config = checkpoint.get("config", {})
            obs_dim = config.get("obs_dim", 19)
            # act_dim: 9 for bc, 18 for emg_bc (x_attr 9 + K_emg 9)
            default_act_dim = 18 if self.model_type == "emg_bc" else 9
            act_dim = config.get("act_dim", default_act_dim)
            hidden_dim = config.get("hidden_dim", 256)
            depth = config.get("depth", 3)

            self.model = BehaviorCloningModel(obs_dim, act_dim, hidden_dim, depth)
            # Load state_dict (key name might be 'state_dict' or 'model_state_dict')
            state_dict_key = "state_dict" if "state_dict" in checkpoint else "model_state_dict"
            self.model.load_state_dict(checkpoint[state_dict_key])
            self.model.eval()

            self.get_logger().info(
                f"{self.model_type.upper()} model: obs_dim={obs_dim}, act_dim={act_dim}, hidden={hidden_dim}, depth={depth}"
            )

        elif "diffusion" in self.model_type:
            # Diffusion model requires DiffusionPolicyBaseline class
            # For simplicity, we'll attempt to import from the benchmark script
            sys.path.insert(0, str(_PKG_ROOT / "scripts" / "3_model_learning"))
            try:
                from run_stiffness_policy_benchmarks import DiffusionPolicyBaseline

                # Reconstruct diffusion model
                config = checkpoint.get("config", {})
                self.model = DiffusionPolicyBaseline(
                    obs_dim=config.get("obs_dim", 19),
                    act_dim=config.get("act_dim", 9),
                    hidden_dim=config.get("hidden_dim", 256),
                    time_dim=config.get("time_dim", 16),
                    timesteps=config.get("timesteps", 100),
                    model_type=self.model_type.split("_")[1],  # 'c' or 't'
                )
                self.model.model.load_state_dict(checkpoint["model_state_dict"])
                self.model.model.eval()

                self.get_logger().info(
                    f"Diffusion model loaded: {self.model_type}, timesteps={config.get('timesteps')}"
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
        """Construct 19D observation vector from current sensor data."""
        # Check if we have minimum required data
        if not all(self.forces):
            return None
        if self.deform_ecc is None:
            return None
        if not all(self.ee_positions.values()):
            return None

        obs = []

        # Force features (9D: s1/s2/s3 fx/fy/fz)
        for i in range(3):
            if self.forces[i]:
                obs.extend([self.forces[i]["fx"], self.forces[i]["fy"], self.forces[i]["fz"]])
            else:
                obs.extend([0.0, 0.0, 0.0])

        # Eccentricity (1D)
        obs.append(self.deform_ecc)

        # End-effector positions (9D: if/mf/th px/py/pz)
        for finger in ["if", "mf", "th"]:
            if self.ee_positions[finger] is not None:
                obs.extend(self.ee_positions[finger].tolist())
            else:
                obs.extend([0.0, 0.0, 0.0])

        return np.array(obs, dtype=np.float32)

    def _predict_stiffness(self, obs: np.ndarray) -> np.ndarray:
        """Predict stiffness from observation using loaded model (9D)."""
        # Scale observation
        if self.obs_scaler:
            obs_scaled = self.obs_scaler.transform(obs.reshape(1, -1))
        else:
            obs_scaled = obs.reshape(1, -1)

        # Predict based on model type
        if self.model_type == "bc":
            with torch.no_grad():
                obs_t = torch.from_numpy(obs_scaled.astype(np.float32))
                act_scaled = self.model(obs_t).numpy()

        elif "diffusion" in self.model_type:
            # Diffusion policy prediction
            act_scaled = self.model.predict(obs_scaled, n_samples=1, sampler="ddpm", eta=0.0)

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

        # Inverse scale
        if self.act_scaler:
            stiffness = self.act_scaler.inverse_transform(act_scaled)
        else:
            stiffness = act_scaled

        return stiffness.flatten()

    def _predict_action(self, obs: np.ndarray) -> np.ndarray:
        """General action prediction for variable act_dim (used for emg_bc)."""
        if self.obs_scaler:
            obs_scaled = self.obs_scaler.transform(obs.reshape(1, -1))
        else:
            obs_scaled = obs.reshape(1, -1)

        if self.model_type in ("bc", "emg_bc"):
            with torch.no_grad():
                obs_t = torch.from_numpy(obs_scaled.astype(np.float32))
                act_scaled = self.model(obs_t).numpy()
        elif "diffusion" in self.model_type:
            act_scaled = self.model.predict(obs_scaled, n_samples=1, sampler="ddpm", eta=0.0)
        else:
            # Fallback
            act_scaled = np.zeros((1, 9), dtype=np.float32)

        if self.act_scaler:
            act = self.act_scaler.inverse_transform(act_scaled)
        else:
            act = act_scaled
        return act.flatten()

    def _smooth_stiffness(self, stiffness: np.ndarray) -> np.ndarray:
        """Apply moving average smoothing to stiffness predictions."""
        self.stiffness_buffer.append(stiffness)

        if len(self.stiffness_buffer) > self.smooth_window:
            self.stiffness_buffer.pop(0)

        # Compute mean
        smoothed = np.mean(self.stiffness_buffer, axis=0)

        # Apply scale and clamp
        smoothed = smoothed * self.stiffness_scale
        smoothed = np.clip(smoothed, self.stiffness_min, self.stiffness_max)

        return smoothed

    def _scan_topics(self):
        try:
            if all(self.forces) and self.deform_ecc is not None and all(self.ee_positions.values()):
                # All ready; stop scanning
                if self._scan_timer is not None:
                    self._scan_timer.cancel()
                return
            names_types = dict(self.get_topic_names_and_types())
            self._scan_counter += 1
            missing_force_indices = [i for i in range(3) if self.forces[i] is None]
            missing_ee = [k for k,v in self.ee_positions.items() if v is None]
            force_topics_present = [t for t in self.force_topics if t in names_types]
            ee_topics_present = [t for t in [self.ee_pose_if_topic, self.ee_pose_mf_topic, self.ee_pose_th_topic] if t in names_types]
            self.get_logger().info(
                f"[scan#{self._scan_counter}] present(force)={force_topics_present} present(ee)={ee_topics_present} missing_force_idx={missing_force_indices} missing_ee={missing_ee}"
            )
            # Extra hint if topics absent entirely
            absent_force = [t for t in self.force_topics if t not in names_types]
            absent_ee = [t for t in [self.ee_pose_if_topic, self.ee_pose_mf_topic, self.ee_pose_th_topic] if t not in names_types]
            if absent_force:
                self.get_logger().warn(f"Force topics absent in graph: {absent_force}")
            if absent_ee:
                self.get_logger().warn(f"EE pose topics absent in graph: {absent_ee}")
        except Exception as e:
            self.get_logger().warn(f"Topic scan error: {e}")

    def _control_callback(self):
        """Main control loop callback - runs at rate_hz frequency."""
        # Get current observation
        obs = self._get_observation()

        if obs is None:
            # Not enough data yet - skip this iteration
            if self._log_counter % int(self.rate_hz * 2) == 0:  # Log every 2 seconds
                missing = []
                if not all(self.forces):
                    missing.append("forces")
                if self.deform_ecc is None:
                    missing.append("deform_ecc")
                if not all(self.ee_positions.values()):
                    missing.append("ee_poses")
                self.get_logger().warn(
                    f"Waiting for sensor data: {', '.join(missing)}", throttle_duration_sec=2.0
                )
            self._log_counter += 1
            return

        try:
            if not self._ready_announced:
                self.get_logger().info("All required sensor inputs received -> starting policy predictions")
                self._ready_announced = True

            # Manual start gate
            if not self._started:
                if not self._start_prompted:
                    self.get_logger().info(f"Waiting for key '{self.start_key}' to start policy …")
                    self._start_prompted = True
                # Throttle reminder
                self._log_counter += 1
                if self._log_counter % int(self.rate_hz * 2) == 0:
                    self.get_logger().info(f"Still waiting for '{self.start_key}' …")
                return

            if self.model_type == "emg_bc":
                act = self._predict_action(obs)
                # Split: x_attr(9) + K_emg(9)
                if act.shape[0] < 18:
                    raise RuntimeError(f"emg_bc expected 18-dim action, got {act.shape[0]}")
                x_attr = act[:9]
                stiffness = act[9:18]

                # Smooth/clamp stiffness as before
                stiffness = self._smooth_stiffness(stiffness)

                # Publish attractor
                attr_msg = Float32MultiArray()
                attr_msg.data = x_attr.tolist()
                self.attractor_pub.publish(attr_msg)

                # Publish stiffness
                st_msg = Float32MultiArray()
                st_msg.data = stiffness.tolist()
                self.stiffness_pub.publish(st_msg)

                # Log occasionally
                self._log_counter += 1
                if self._log_counter % int(self.rate_hz) == 0:
                    self.get_logger().info(
                        f"Attr(IF|MF|TH)=[{x_attr[0]:.3f},{x_attr[1]:.3f},{x_attr[2]:.3f}]|"
                        f"[{x_attr[3]:.3f},{x_attr[4]:.3f},{x_attr[5]:.3f}]|"
                        f"[{x_attr[6]:.3f},{x_attr[7]:.3f},{x_attr[8]:.3f}] | "
                        f"K: TH=[{stiffness[0]:.1f},{stiffness[1]:.1f},{stiffness[2]:.1f}], "
                        f"IF=[{stiffness[3]:.1f},{stiffness[4]:.1f},{stiffness[5]:.1f}], "
                        f"MF=[{stiffness[6]:.1f},{stiffness[7]:.1f},{stiffness[8]:.1f}]"
                    )
            else:
                # Predict stiffness (9D)
                stiffness = self._predict_stiffness(obs)

                # Smooth and scale
                stiffness = self._smooth_stiffness(stiffness)

                # Publish stiffness command
                msg = Float32MultiArray()
                msg.data = stiffness.tolist()
                self.stiffness_pub.publish(msg)

                # Log occasionally
                self._log_counter += 1
                if self._log_counter % int(self.rate_hz) == 0:  # Log every second
                    self.get_logger().info(
                        f"Stiffness: TH=[{stiffness[0]:.1f},{stiffness[1]:.1f},{stiffness[2]:.1f}], "
                        f"IF=[{stiffness[3]:.1f},{stiffness[4]:.1f},{stiffness[5]:.1f}], "
                        f"MF=[{stiffness[6]:.1f},{stiffness[7]:.1f},{stiffness[8]:.1f}]"
                    )

        except Exception as e:
            self.get_logger().error(f"Control callback error: {e}", throttle_duration_sec=1.0)


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
