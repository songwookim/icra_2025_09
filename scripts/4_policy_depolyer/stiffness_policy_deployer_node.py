#!/usr/bin/env python3
"""
Real-time Stiffness Policy Deployer Node

실시간으로 센서 데이터를 받아서 학습된 stiffness policy로 예측하고 publish하는 ROS2 노드

Subscriptions:
  - /force_sensor/s{1,2,3}/wrench (geometry_msgs/WrenchStamped): Force sensor data
  - /deformity_tracker/circularity (std_msgs/Float32): Object circularity
  - /deformity_tracker/eccentricity (std_msgs/Float32): Object eccentricity
  - /ee_pose_{if,mf,th} (geometry_msgs/PoseStamped): End-effector positions

Publications:
  - /stiffness_policy/predicted (std_msgs/Float32MultiArray): 
      Predicted stiffness [th_k1, th_k2, th_k3, if_k1, if_k2, if_k3, mf_k1, mf_k2, mf_k3]
  - /stiffness_policy/status (std_msgs/String): Policy status messages

Parameters:
  - model_type (string): 'bc', 'diffusion', 'lstm_gmm', etc.
  - artifact_dir (string): Path to model artifacts directory
  - rate_hz (float, default 100.0): Prediction frequency
  - diffusion_sampler (string, default 'ddpm'): 'ddpm' or 'ddim' for diffusion models
  - diffusion_n_samples (int, default 1): Number of samples for diffusion prediction
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional, List, Tuple, Any
import json

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import WrenchStamped, PoseStamped
from std_msgs.msg import Float32, Float32MultiArray, String

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    TORCH_AVAILABLE = False


# Observation columns (should match training)
OBS_COLUMNS = [
    "s1_fx", "s1_fy", "s1_fz",
    "s2_fx", "s2_fy", "s2_fz",
    "s3_fx", "s3_fy", "s3_fz",
    "deform_ecc",
    "ee_if_px", "ee_if_py", "ee_if_pz",
    "ee_mf_px", "ee_mf_py", "ee_mf_pz",
    "ee_th_px", "ee_th_py", "ee_th_pz",
]  # 19D (deform_circ removed)

# Action columns
ACTION_COLUMNS = [
    'th_k1', 'th_k2', 'th_k3',
    'if_k1', 'if_k2', 'if_k3',
    'mf_k1', 'mf_k2', 'mf_k3'
]  # 9D


class BehaviorCloningModel(nn.Module):
    """BC model architecture (must match training)"""
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


class StiffnessPolicyDeployer(Node):
    def __init__(self):
        super().__init__('stiffness_policy_deployer')
        
        # Parameters
        self.declare_parameter('model_type', 'bc')
        self.declare_parameter('artifact_dir', '')
        self.declare_parameter('rate_hz', 100.0)
        self.declare_parameter('diffusion_sampler', 'ddpm')
        self.declare_parameter('diffusion_n_samples', 1)
        
        self.model_type = self.get_parameter('model_type').get_parameter_value().string_value
        artifact_dir_str = self.get_parameter('artifact_dir').get_parameter_value().string_value
        self.rate_hz = self.get_parameter('rate_hz').get_parameter_value().double_value
        self.diffusion_sampler = self.get_parameter('diffusion_sampler').get_parameter_value().string_value
        self.diffusion_n_samples = self.get_parameter('diffusion_n_samples').get_parameter_value().integer_value
        
        if not artifact_dir_str:
            self.get_logger().error("artifact_dir parameter is required!")
            raise RuntimeError("artifact_dir parameter is required")
        
        self.artifact_dir = Path(artifact_dir_str)
        if not self.artifact_dir.exists():
            self.get_logger().error(f"Artifact directory not found: {self.artifact_dir}")
            raise RuntimeError(f"Artifact directory not found: {self.artifact_dir}")
        
        # Load model and scalers
        self.get_logger().info(f"Loading {self.model_type} model from {self.artifact_dir}")
        self._load_model()
        
        # State: latest sensor readings
        self._force_s1: Optional[Tuple[float, float, float]] = None  # fx, fy, fz
        self._force_s2: Optional[Tuple[float, float, float]] = None
        self._force_s3: Optional[Tuple[float, float, float]] = None
        self._deform_ecc: Optional[float] = None
        self._ee_if: Optional[Tuple[float, float, float]] = None  # px, py, pz
        self._ee_mf: Optional[Tuple[float, float, float]] = None
        self._ee_th: Optional[Tuple[float, float, float]] = None
        
        # Subscribers
        self.create_subscription(WrenchStamped, '/force_sensor/s1/wrench', 
                                lambda msg: self._on_force(1, msg), 10)
        self.create_subscription(WrenchStamped, '/force_sensor/s2/wrench', 
                                lambda msg: self._on_force(2, msg), 10)
        self.create_subscription(WrenchStamped, '/force_sensor/s3/wrench', 
                                lambda msg: self._on_force(3, msg), 10)
        self.create_subscription(Float32, '/deformity_tracker/eccentricity', 
                                self._on_eccentricity, 10)
        self.create_subscription(PoseStamped, '/ee_pose_if', self._on_ee_if, 10)
        self.create_subscription(PoseStamped, '/ee_pose_mf', self._on_ee_mf, 10)
        self.create_subscription(PoseStamped, '/ee_pose_th', self._on_ee_th, 10)
        
        # Publishers
        self._pred_pub = self.create_publisher(Float32MultiArray, '/stiffness_policy/predicted', 10)
        self._status_pub = self.create_publisher(String, '/stiffness_policy/status', 10)
        
        # Timer for prediction
        period = max(0.001, 1.0 / max(1e-6, self.rate_hz))
        self.timer = self.create_timer(period, self._prediction_callback)
        
        self._ready_count = 0
        self.get_logger().info(f'Stiffness Policy Deployer started ({self.model_type}, {self.rate_hz:.1f}Hz)')
        self._publish_status(f'Policy deployer ready: {self.model_type}')
    
    def _load_model(self):
        """Load model artifacts (manifest, scalers, model weights)"""
        # Load manifest
        manifest_path = self.artifact_dir / "manifest.json"
        if not manifest_path.exists():
            raise RuntimeError(f"Manifest not found: {manifest_path}")
        
        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)
        
        # Load scalers
        scalers_path = self.artifact_dir / self.manifest["scalers"]
        with open(scalers_path, 'rb') as f:
            scalers = pickle.load(f)
        self.obs_scaler = scalers['obs_scaler']
        self.act_scaler = scalers['act_scaler']
        
        self.get_logger().info(f"Loaded scalers: obs_dim={self.obs_scaler.n_features_in_}, "
                              f"act_dim={self.act_scaler.n_features_in_}")
        
        # Load model based on type
        if self.model_type == 'bc':
            self._load_bc_model()
        elif self.model_type.startswith('diffusion'):
            self._load_diffusion_model()
        elif self.model_type == 'lstm_gmm':
            self._load_lstm_gmm_model()
        elif self.model_type in ['gmm', 'gmr']:
            self._load_gmm_model()
        else:
            raise RuntimeError(f"Unsupported model type: {self.model_type}")
    
    def _load_bc_model(self):
        """Load Behavior Cloning model"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for BC model")
        
        model_entry = self.manifest['models'].get('bc')
        if not model_entry:
            raise RuntimeError("BC model not found in manifest")
        
        model_path = self.artifact_dir / model_entry['path']
        checkpoint = torch.load(model_path, map_location='cpu')
        
        config = checkpoint.get('config', {})
        self.model = BehaviorCloningModel(
            obs_dim=config.get('obs_dim', len(OBS_COLUMNS)),
            act_dim=config.get('act_dim', len(ACTION_COLUMNS)),
            hidden_dim=config.get('hidden_dim', 256),
            depth=config.get('depth', 3)
        )
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        
        self.get_logger().info("BC model loaded successfully")
    
    def _load_diffusion_model(self):
        """Load Diffusion model"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for Diffusion model")
        
        # Import diffusion baseline from benchmarks
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / '3_model_learning'))
        from run_stiffness_policy_benchmarks import DiffusionPolicyBaseline
        
        # Find diffusion model in manifest
        model_key = None
        for key in ['diffusion_c', 'diffusion_c_ddpm', 'diffusion_c_ddim']:
            if key in self.manifest['models']:
                model_key = key
                break
        
        if not model_key:
            raise RuntimeError("Diffusion model not found in manifest")
        
        model_entry = self.manifest['models'][model_key]
        model_path = self.artifact_dir / model_entry['path']
        checkpoint = torch.load(model_path, map_location='cpu')
        
        config = checkpoint.get('config', {})
        self.model = DiffusionPolicyBaseline(
            obs_dim=config.get('obs_dim', len(OBS_COLUMNS)),
            act_dim=config.get('act_dim', len(ACTION_COLUMNS)),
            timesteps=config.get('timesteps', 50),
            hidden_dim=config.get('hidden_dim', 256),
            time_dim=config.get('time_dim', 64),
            lr=config.get('lr', 1e-3),
            batch_size=config.get('batch_size', 256),
            epochs=config.get('epochs', 0),
            seed=config.get('seed', 0),
            temporal=config.get('temporal', False),
            device='cpu'
        )
        self.model.model.load_state_dict(checkpoint['state_dict'])
        self.model.model.eval()
        
        self.get_logger().info(f"Diffusion model loaded (sampler={self.diffusion_sampler}, "
                              f"n_samples={self.diffusion_n_samples})")
    
    def _load_lstm_gmm_model(self):
        """Load LSTM-GMM model"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for LSTM-GMM model")
        
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / '3_model_learning'))
        from run_stiffness_policy_benchmarks import LSTMGMMBaseline
        
        model_entry = self.manifest['models'].get('lstm_gmm')
        if not model_entry:
            raise RuntimeError("LSTM-GMM model not found in manifest")
        
        model_path = self.artifact_dir / model_entry['path']
        checkpoint = torch.load(model_path, map_location='cpu')
        
        config = checkpoint.get('config', {})
        self.model = LSTMGMMBaseline(
            obs_dim=config.get('obs_dim', len(OBS_COLUMNS)),
            act_dim=config.get('act_dim', len(ACTION_COLUMNS)),
            seq_len=config.get('seq_len', 10),
            n_components=config.get('n_components', 5),
            hidden_dim=config.get('hidden_dim', 256),
            n_layers=config.get('n_layers', 1),
            lr=config.get('lr', 1e-3),
            batch_size=config.get('batch_size', 256),
            epochs=config.get('epochs', 0),
            seed=config.get('seed', 0),
            device='cpu'
        )
        self.model.model.load_state_dict(checkpoint['state_dict'])
        self.model.model.eval()
        
        self.get_logger().warn("LSTM-GMM requires sequence data - using single-step prediction mode")
        self.get_logger().info("LSTM-GMM model loaded")
    
    def _load_gmm_model(self):
        """Load GMM/GMR model"""
        model_key = 'gmr' if self.model_type == 'gmr' else 'gmm'
        model_entry = self.manifest['models'].get(model_key)
        if not model_entry:
            raise RuntimeError(f"{model_key.upper()} model not found in manifest")
        
        model_path = self.artifact_dir / model_entry['path']
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        self.gmm_mode = 'mean' if self.model_type == 'gmr' else 'sample'
        self.get_logger().info(f"{model_key.upper()} model loaded (mode={self.gmm_mode})")
    
    def _on_force(self, sensor_id: int, msg: WrenchStamped):
        """Force sensor callback"""
        force = (msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z)
        if sensor_id == 1:
            self._force_s1 = force
        elif sensor_id == 2:
            self._force_s2 = force
        elif sensor_id == 3:
            self._force_s3 = force
    
    def _on_eccentricity(self, msg: Float32):
        """Eccentricity callback"""
        self._deform_ecc = float(msg.data)
    
    def _on_ee_if(self, msg: PoseStamped):
        """Index finger EE pose callback"""
        self._ee_if = (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)
    
    def _on_ee_mf(self, msg: PoseStamped):
        """Middle finger EE pose callback"""
        self._ee_mf = (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)
    
    def _on_ee_th(self, msg: PoseStamped):
        """Thumb EE pose callback"""
        self._ee_th = (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)
    
    def _build_observation(self) -> Optional[np.ndarray]:
        """Build observation vector from current sensor readings"""
        # Check if all required data is available
        if (self._force_s1 is None or self._force_s2 is None or self._force_s3 is None or
            self._deform_ecc is None or
            self._ee_if is None or self._ee_mf is None or self._ee_th is None):
            return None
        
        # Build observation in correct order matching OBS_COLUMNS
        obs = np.array([
            # Force sensors (3x3 = 9)
            self._force_s1[0], self._force_s1[1], self._force_s1[2],
            self._force_s2[0], self._force_s2[1], self._force_s2[2],
            self._force_s3[0], self._force_s3[1], self._force_s3[2],
            # Deformity (1)
            self._deform_ecc,
            # End-effector positions (3x3 = 9)
            self._ee_if[0], self._ee_if[1], self._ee_if[2],
            self._ee_mf[0], self._ee_mf[1], self._ee_mf[2],
            self._ee_th[0], self._ee_th[1], self._ee_th[2],
        ], dtype=np.float32)
        
        return obs.reshape(1, -1)  # (1, 19)
    
    def _predict(self, obs: np.ndarray) -> Optional[np.ndarray]:
        """Predict stiffness from observation"""
        try:
            # Scale observation
            obs_scaled = self.obs_scaler.transform(obs)
            
            # Predict based on model type
            if self.model_type == 'bc':
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs_scaled)
                    pred_scaled = self.model(obs_tensor).numpy()
            
            elif self.model_type.startswith('diffusion'):
                pred_scaled = self.model.predict(
                    obs_scaled,
                    n_samples=self.diffusion_n_samples,
                    sampler=self.diffusion_sampler,
                    eta=0.0 if self.diffusion_sampler == 'ddpm' else 1.0
                )
            
            elif self.model_type == 'lstm_gmm':
                # For single-step, repeat observation to create sequence
                seq_len = self.model.seq_len
                obs_seq = np.repeat(obs_scaled[np.newaxis, :, :], seq_len, axis=1)
                pred_scaled = self.model.predict(obs_seq, mode='mean', n_samples=1)
            
            elif self.model_type in ['gmm', 'gmr']:
                pred_scaled = self.model.predict(obs_scaled, mode=self.gmm_mode, n_samples=1)
            
            else:
                return None
            
            # Inverse transform to get actual stiffness values
            pred = self.act_scaler.inverse_transform(pred_scaled)
            
            # Ensure positive stiffness values (physical constraint)
            pred = np.maximum(pred, 1.0)
            
            return pred.flatten()
        
        except Exception as e:
            self.get_logger().error(f"Prediction error: {e}")
            return None
    
    def _prediction_callback(self):
        """Main prediction loop - called at rate_hz"""
        # Build observation from current sensor data
        obs = self._build_observation()
        
        if obs is None:
            # Not all sensors ready yet
            if self._ready_count < 10:  # Warn only first few times
                self.get_logger().warn("Waiting for all sensor data...")
                self._ready_count += 1
            return
        
        if self._ready_count == 10:
            self.get_logger().info("All sensors ready! Starting predictions...")
            self._publish_status("All sensors ready - predictions active")
            self._ready_count += 1
        
        # Predict stiffness
        predicted_stiffness = self._predict(obs)
        
        if predicted_stiffness is None:
            return
        
        # Publish prediction
        msg = Float32MultiArray()
        msg.data = predicted_stiffness.tolist()
        self._pred_pub.publish(msg)
    
    def _publish_status(self, message: str):
        """Publish status message"""
        msg = String()
        msg.data = message
        self._status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = StiffnessPolicyDeployer()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
