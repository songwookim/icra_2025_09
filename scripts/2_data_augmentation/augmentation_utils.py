#!/usr/bin/env python3
"""Data augmentation utilities for robotic manipulation demonstrations.

Provides physics-aware augmentation techniques for force-torque sensor data,
stiffness profiles, and end-effector positions while preserving physical constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class Trajectory:
    """Single demonstration trajectory."""
    name: str
    observations: np.ndarray  # shape (T, obs_dim)
    actions: np.ndarray       # shape (T, act_dim)


class DataAugmentor:
    """Physics-aware data augmentation for manipulation demonstrations."""
    
    def __init__(self, seed: int = 0):
        """Initialize augmentor with random seed."""
        self.rng = np.random.RandomState(seed)
    
    def augment_trajectory(
        self,
        traj: Trajectory,
        obs_columns: List[str],
        num_augmentations: int = 3,
        noise_std_force: float = 0.02,
        noise_std_deform: float = 0.01,
        noise_std_ee: float = 0.001,
        noise_std_stiffness: float = 0.05,
        scale_range_force: Tuple[float, float] = (0.95, 1.05),
        scale_range_stiffness: Tuple[float, float] = (0.90, 1.10),
        jitter_timesteps: int = 3,
        enable_temporal_shift: bool = True,
        enable_noise: bool = True,
        enable_scaling: bool = True,
        enable_temporal_jitter: bool = True,
    ) -> List[Trajectory]:
        """
        Generate augmented versions of a trajectory.
        
        Augmentation techniques:
        1. Gaussian noise injection (sensor noise simulation)
        2. Magnitude scaling (different force/stiffness regimes)
        3. Temporal jittering (sub-sampling variation)
        4. Temporal shift (phase variation)
        
        Args:
            traj: Original trajectory
            obs_columns: Column names to identify obs types
            num_augmentations: Number of augmented copies to generate
            noise_std_force: Noise std for force sensors (relative to signal)
            noise_std_deform: Noise std for deformation descriptors
            noise_std_ee: Noise std for end-effector positions (m)
            noise_std_stiffness: Noise std for stiffness actions (relative)
            scale_range_force: (min, max) scaling factors for forces
            scale_range_stiffness: (min, max) scaling factors for stiffness
            jitter_timesteps: Max temporal jitter (timesteps)
            enable_temporal_shift: Enable temporal shifting
            enable_noise: Enable Gaussian noise
            enable_scaling: Enable magnitude scaling
            enable_temporal_jitter: Enable temporal jittering
        
        Returns:
            List of augmented trajectories (including original)
        """
        augmented = [traj]  # Include original
        
        # Parse observation structure
        force_indices = self._find_column_indices(obs_columns, ['fx', 'fy', 'fz'])
        deform_indices = self._find_column_indices(obs_columns, ['deform_circ', 'deform_ecc'])
        ee_indices = self._find_column_indices(obs_columns, ['ee_', 'px', 'py', 'pz'])
        emg_indices = self._find_column_indices(obs_columns, ['emg_'])
        
        for i in range(num_augmentations):
            aug_obs = traj.observations.copy()
            aug_act = traj.actions.copy()
            
            # 1. Temporal shift (phase variation)
            if enable_temporal_shift and len(aug_obs) > 20:
                shift = self.rng.randint(-jitter_timesteps, jitter_timesteps + 1)
                if shift != 0:
                    aug_obs, aug_act = self._temporal_shift(aug_obs, aug_act, shift)
            
            # 2. Gaussian noise injection
            if enable_noise:
                # Force noise (sensor noise)
                if force_indices:
                    force_signal = aug_obs[:, force_indices]
                    noise_level = noise_std_force * np.abs(force_signal).mean()
                    aug_obs[:, force_indices] += self.rng.normal(0, noise_level, force_signal.shape)
                
                # Deformation noise
                if deform_indices:
                    deform_signal = aug_obs[:, deform_indices]
                    noise_level = noise_std_deform * np.abs(deform_signal).mean()
                    aug_obs[:, deform_indices] += self.rng.normal(0, noise_level, deform_signal.shape)
                
                # End-effector position noise (very small - in mm range)
                if ee_indices:
                    aug_obs[:, ee_indices] += self.rng.normal(0, noise_std_ee, 
                                                              (len(aug_obs), len(ee_indices)))
                
                # EMG noise (if present)
                if emg_indices:
                    emg_signal = aug_obs[:, emg_indices]
                    noise_level = 0.01 * np.abs(emg_signal).mean()  # 1% noise
                    aug_obs[:, emg_indices] += self.rng.normal(0, noise_level, emg_signal.shape)
                
                # Stiffness noise
                stiffness_signal = aug_act
                noise_level = noise_std_stiffness * np.abs(stiffness_signal).mean()
                aug_act += self.rng.normal(0, noise_level, aug_act.shape)
            
            # 3. Magnitude scaling (different force/stiffness regimes)
            if enable_scaling:
                # Force scaling
                if force_indices:
                    force_scale = self.rng.uniform(*scale_range_force)
                    aug_obs[:, force_indices] *= force_scale
                
                # Stiffness scaling (correlated with force)
                stiffness_scale = self.rng.uniform(*scale_range_stiffness)
                aug_act *= stiffness_scale
            
            # 4. Temporal jittering (random sub-sampling)
            if enable_temporal_jitter and len(aug_obs) > 20:
                jitter = self.rng.randint(1, jitter_timesteps + 1)
                if jitter > 1:
                    # Random offset for sub-sampling
                    offset = self.rng.randint(0, jitter)
                    aug_obs = aug_obs[offset::jitter]
                    aug_act = aug_act[offset::jitter]
            
            # Ensure stiffness remains positive (physical constraint)
            aug_act = np.maximum(aug_act, 1.0)  # Min stiffness = 1 N/m
            
            # Ensure finite values
            if not (np.isfinite(aug_obs).all() and np.isfinite(aug_act).all()):
                continue  # Skip invalid augmentation
            
            augmented.append(Trajectory(
                name=f"{traj.name}_aug{i+1}",
                observations=aug_obs,
                actions=aug_act,
            ))
        
        return augmented
    
    def _find_column_indices(self, column_names: List[str], keywords: List[str]) -> List[int]:
        """Find column indices containing any of the keywords."""
        indices = []
        for i, col in enumerate(column_names):
            if any(kw in col for kw in keywords):
                indices.append(i)
        return indices
    
    def _temporal_shift(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        shift: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Shift trajectory in time (phase variation)."""
        if shift == 0:
            return obs, act
        
        if shift > 0:
            # Shift forward (truncate beginning)
            return obs[shift:], act[shift:]
        else:
            # Shift backward (truncate end)
            return obs[:shift], act[:shift]
    
    def augment_dataset(
        self,
        trajectories: List[Trajectory],
        obs_columns: List[str],
        num_augmentations_per_traj: int = 3,
        **kwargs,
    ) -> List[Trajectory]:
        """
        Augment entire dataset.
        
        Args:
            trajectories: List of original trajectories
            obs_columns: Column names for observation parsing
            num_augmentations_per_traj: Number of augmented copies per trajectory
            **kwargs: Additional arguments passed to augment_trajectory
        
        Returns:
            List of all trajectories (original + augmented)
        """
        all_trajectories = []
        
        for traj in trajectories:
            augmented = self.augment_trajectory(
                traj,
                obs_columns,
                num_augmentations=num_augmentations_per_traj,
                **kwargs,
            )
            all_trajectories.extend(augmented)
        
        return all_trajectories


def smart_augmentation(
    trajectories: List[Trajectory],
    obs_columns: List[str],
    target_samples: int = 200000,
    min_aug_per_demo: int = 1,
    max_aug_per_demo: int = 10,
    seed: int = 0,
    **kwargs,
) -> List[Trajectory]:
    """
    Smart augmentation that adapts to dataset size.
    
    Automatically determines number of augmentations needed to reach
    target sample count while respecting min/max constraints.
    
    Args:
        trajectories: Original demonstrations
        obs_columns: Column names for parsing
        target_samples: Target total sample count
        min_aug_per_demo: Minimum augmentations per demo
        max_aug_per_demo: Maximum augmentations per demo
        seed: Random seed
        **kwargs: Additional augmentation parameters
    
    Returns:
        Augmented dataset
    """
    # Count current samples
    current_samples = sum(len(t.observations) for t in trajectories)
    
    print(f"Current dataset: {len(trajectories)} demos, {current_samples:,} samples")
    
    # Calculate required augmentations
    if current_samples >= target_samples:
        print(f"Dataset already has {current_samples:,} >= {target_samples:,} samples")
        aug_per_demo = min_aug_per_demo
    else:
        samples_needed = target_samples - current_samples
        avg_traj_length = current_samples / len(trajectories)
        aug_per_demo = int(np.ceil(samples_needed / (len(trajectories) * avg_traj_length)))
        aug_per_demo = np.clip(aug_per_demo, min_aug_per_demo, max_aug_per_demo)
    
    print(f"Generating {aug_per_demo} augmentations per demo...")
    
    augmentor = DataAugmentor(seed=seed)
    augmented = augmentor.augment_dataset(
        trajectories,
        obs_columns,
        num_augmentations_per_traj=aug_per_demo,
        **kwargs,
    )
    
    total_samples = sum(len(t.observations) for t in augmented)
    print(f"Augmented dataset: {len(augmented)} demos, {total_samples:,} samples")
    print(f"Augmentation factor: {len(augmented) / len(trajectories):.1f}x")
    
    return augmented


if __name__ == "__main__":
    # Test augmentation
    print("Testing data augmentation...")
    
    # Create dummy trajectory
    T = 100
    obs_dim = 20
    act_dim = 9
    
    dummy_traj = Trajectory(
        name="test_demo",
        observations=np.random.randn(T, obs_dim),
        actions=np.random.rand(T, act_dim) * 200 + 50,  # Stiffness 50-250
    )
    
    obs_columns = [
        's1_fx', 's1_fy', 's1_fz',
        's2_fx', 's2_fy', 's2_fz',
        's3_fx', 's3_fy', 's3_fz',
        'deform_circ', 'deform_ecc',
        'ee_if_px', 'ee_if_py', 'ee_if_pz',
        'ee_mf_px', 'ee_mf_py', 'ee_mf_pz',
        'ee_th_px', 'ee_th_py', 'ee_th_pz',
    ]
    
    augmentor = DataAugmentor(seed=42)
    augmented = augmentor.augment_trajectory(
        dummy_traj,
        obs_columns,
        num_augmentations=5,
    )
    
    print(f"\n✅ Generated {len(augmented)} trajectories (1 original + 5 augmented)")
    for traj in augmented:
        print(f"  - {traj.name}: {len(traj.observations)} samples")
        print(f"    Obs range: [{traj.observations.min():.2f}, {traj.observations.max():.2f}]")
        print(f"    Act range: [{traj.actions.min():.2f}, {traj.actions.max():.2f}]")
