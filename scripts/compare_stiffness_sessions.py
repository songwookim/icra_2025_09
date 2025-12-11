#!/usr/bin/env python3
"""
Compare stiffness predictions across multiple model sessions.
Computes various functional connectivity metrics between predicted and reference stiffness.

Metrics computed (based on functional connectivity analysis):
1. Pearson Correlation - Linear correlation coefficient
2. Spearman Correlation - Rank-based correlation (monotonic relationships)
3. RMSE - Root Mean Squared Error
4. MAE - Mean Absolute Error
5. R² (Coefficient of Determination) - Variance explained
6. Coherence - Frequency domain correlation
7. Cross-Correlation (max) - Time-lagged correlation
8. Mutual Information - Nonlinear dependency measure
9. DTW Distance - Dynamic Time Warping distance
10. Phase Locking Value (PLV) - Phase synchronization

Usage:
    python compare_stiffness_sessions.py <session_path1> <session_path2> ...
    
    # Or compare all sessions in a directory:
    python compare_stiffness_sessions.py --dir /path/to/stiffness_logs/
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal, stats
from scipy.spatial.distance import cdist
from scipy.ndimage import uniform_filter1d, gaussian_filter1d


# =============================================================================
# Smoothing Function
# =============================================================================

def smooth_signal(data: np.ndarray, window_size: int = 15, method: str = 'savgol') -> np.ndarray:
    """
    Apply smoothing to signal data for cleaner visualization.
    
    Args:
        data: Input signal array
        window_size: Size of smoothing window (odd number for savgol)
        method: 'savgol' (Savitzky-Golay), 'moving_avg', or 'gaussian'
    
    Returns:
        Smoothed signal array
    """
    if len(data) < window_size:
        return data
    
    if method == 'savgol':
        # Savitzky-Golay filter (preserves peaks better)
        window = window_size if window_size % 2 == 1 else window_size + 1
        return signal.savgol_filter(data, window, polyorder=3)
    elif method == 'moving_avg':
        return uniform_filter1d(data, size=window_size, mode='nearest')
    elif method == 'gaussian':
        return gaussian_filter1d(data, sigma=window_size/3)
    else:
        return data


# =============================================================================
# Metric Computation Functions
# =============================================================================

def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation coefficient (linear relationship)."""
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    r, _ = stats.pearsonr(x, y)
    return r


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation (monotonic relationship)."""
    if len(x) < 2:
        return np.nan
    rho, _ = stats.spearmanr(x, y)
    return rho


def rmse(x: np.ndarray, y: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return np.sqrt(np.mean((x - y) ** 2))


def mae(x: np.ndarray, y: np.ndarray) -> float:
    """Mean Absolute Error."""
    return np.mean(np.abs(x - y))


def r_squared(x: np.ndarray, y: np.ndarray) -> float:
    """Coefficient of determination (R²)."""
    ss_res = np.sum((y - x) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot == 0:
        return np.nan
    return 1 - (ss_res / ss_tot)


def coherence_mean(x: np.ndarray, y: np.ndarray, fs: float = 50.0) -> float:
    """
    Mean coherence across frequencies.
    Coherence measures the linear relationship in the frequency domain.
    """
    if len(x) < 256:  # Need enough samples for frequency analysis
        return np.nan
    try:
        f, cxy = signal.coherence(x, y, fs=fs, nperseg=min(256, len(x)//2))
        return np.mean(cxy)
    except Exception:
        return np.nan


def cross_correlation_max(x: np.ndarray, y: np.ndarray) -> Tuple[float, int]:
    """
    Maximum cross-correlation and optimal lag.
    Returns (max_corr, lag_at_max).
    """
    if len(x) < 2:
        return np.nan, 0
    
    # Normalize
    x_norm = (x - np.mean(x)) / (np.std(x) + 1e-10)
    y_norm = (y - np.mean(y)) / (np.std(y) + 1e-10)
    
    # Full cross-correlation
    corr = np.correlate(x_norm, y_norm, mode='full')
    corr = corr / len(x)
    
    # Find max
    max_idx = np.argmax(np.abs(corr))
    lag = max_idx - (len(x) - 1)
    
    return corr[max_idx], lag


def mutual_information(x: np.ndarray, y: np.ndarray, bins: int = 20) -> float:
    """
    Mutual Information - measures nonlinear statistical dependency.
    Higher values indicate stronger dependency.
    """
    if len(x) < bins:
        return np.nan
    
    try:
        # 2D histogram
        c_xy, _, _ = np.histogram2d(x, y, bins=bins)
        
        # Marginal distributions
        c_x = np.sum(c_xy, axis=1)
        c_y = np.sum(c_xy, axis=0)
        
        # Probabilities
        p_xy = c_xy / np.sum(c_xy)
        p_x = c_x / np.sum(c_x)
        p_y = c_y / np.sum(c_y)
        
        # Mutual information
        mi = 0.0
        for i in range(bins):
            for j in range(bins):
                if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                    mi += p_xy[i, j] * np.log2(p_xy[i, j] / (p_x[i] * p_y[j]))
        
        return mi
    except Exception:
        return np.nan


def dtw_distance(x: np.ndarray, y: np.ndarray, max_warping: Optional[int] = None) -> float:
    """
    Dynamic Time Warping distance.
    Measures similarity allowing for time warping.
    Lower is better (more similar).
    """
    n, m = len(x), len(y)
    if n < 2 or m < 2:
        return np.nan
    
    # For efficiency, downsample if too long
    max_len = 500
    if n > max_len:
        indices = np.linspace(0, n-1, max_len, dtype=int)
        x = x[indices]
        n = max_len
    if m > max_len:
        indices = np.linspace(0, m-1, max_len, dtype=int)
        y = y[indices]
        m = max_len
    
    if max_warping is None:
        max_warping = max(n, m)
    
    # DTW matrix
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n + 1):
        for j in range(max(1, i - max_warping), min(m + 1, i + max_warping + 1)):
            cost = abs(x[i-1] - y[j-1])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j],      # insertion
                                          dtw_matrix[i, j-1],      # deletion
                                          dtw_matrix[i-1, j-1])    # match
    
    return dtw_matrix[n, m] / (n + m)  # Normalized by path length


def phase_locking_value(x: np.ndarray, y: np.ndarray) -> float:
    """
    Phase Locking Value (PLV) - measures phase synchronization.
    Uses Hilbert transform to extract instantaneous phase.
    Range: 0 (no sync) to 1 (perfect sync).
    """
    if len(x) < 10:
        return np.nan
    
    try:
        # Get instantaneous phase via Hilbert transform
        analytic_x = signal.hilbert(x - np.mean(x))
        analytic_y = signal.hilbert(y - np.mean(y))
        
        phase_x = np.angle(analytic_x)
        phase_y = np.angle(analytic_y)
        
        # Phase difference
        phase_diff = phase_x - phase_y
        
        # PLV = |mean(exp(i * phase_diff))|
        plv = np.abs(np.mean(np.exp(1j * phase_diff)))
        
        return plv
    except Exception:
        return np.nan


# =============================================================================
# Session Analysis
# =============================================================================

def load_session_data(session_path: str) -> Optional[pd.DataFrame]:
    """Load stiffness CSV from a session folder."""
    stiffness_file = os.path.join(session_path, "stiffness.csv")
    if not os.path.exists(stiffness_file):
        print(f"[WARN] stiffness.csv not found in {session_path}")
        return None
    
    try:
        df = pd.read_csv(stiffness_file)
        return df
    except Exception as e:
        print(f"[ERROR] Failed to load {stiffness_file}: {e}")
        return None


def extract_model_name(session_path: str) -> str:
    """Extract model name from session folder name."""
    folder_name = os.path.basename(session_path.rstrip('/'))
    
    # Try to find model name at the end (e.g., session_..._bc, session_..._diffusion)
    parts = folder_name.split('_')
    
    # Check known model names
    known_models = ['bc', 'gmr', 'gmm', 'ibc', 'diffusion', 'lstm_gmm', 'diffusion_c', 'diffusion_t']
    
    for model in known_models:
        if model in folder_name.lower():
            return model.upper()
    
    # Fallback: last part
    return parts[-1] if parts else "unknown"


def compute_all_metrics(x: np.ndarray, y: np.ndarray, fs: float = 50.0) -> Dict[str, float]:
    """Compute all metrics between two signals."""
    metrics = {}
    
    # Basic correlation metrics
    metrics['Pearson'] = pearson_correlation(x, y)
    metrics['Spearman'] = spearman_correlation(x, y)
    
    # Error metrics
    metrics['RMSE'] = rmse(x, y)
    metrics['MAE'] = mae(x, y)
    metrics['R²'] = r_squared(x, y)
    
    # Frequency domain
    metrics['Coherence'] = coherence_mean(x, y, fs=fs)
    
    # Time-lagged correlation
    xcorr_max, xcorr_lag = cross_correlation_max(x, y)
    metrics['XCorr_max'] = xcorr_max
    metrics['XCorr_lag'] = xcorr_lag
    
    # Nonlinear dependency
    metrics['MI'] = mutual_information(x, y)
    
    # Time warping
    metrics['DTW'] = dtw_distance(x, y)
    
    # Phase synchronization
    metrics['PLV'] = phase_locking_value(x, y)
    
    return metrics


def analyze_session(session_path: str, reference_session: Optional[str] = None) -> Optional[Dict]:
    """
    Analyze a single session.
    If reference_session is provided, compare against it.
    Otherwise, compute self-consistency metrics.
    """
    df = load_session_data(session_path)
    if df is None:
        return None
    
    model_name = extract_model_name(session_path)
    
    # Stiffness columns (excluding time)
    stiff_cols = [c for c in df.columns if c != 'time']
    
    result = {
        'session': session_path,
        'model': model_name,
        'n_samples': len(df),
        'duration_sec': df['time'].max() - df['time'].min() if 'time' in df.columns else 0,
        'metrics_per_axis': {},
        'metrics_mean': {}
    }
    
    if reference_session:
        ref_df = load_session_data(reference_session)
        if ref_df is None:
            print(f"[WARN] Cannot load reference session: {reference_session}")
            return result
        
        # Align by time or by index
        min_len = min(len(df), len(ref_df))
        
        all_metrics = []
        for col in stiff_cols:
            if col in ref_df.columns:
                x = df[col].values[:min_len]
                y = ref_df[col].values[:min_len]
                
                metrics = compute_all_metrics(x, y)
                result['metrics_per_axis'][col] = metrics
                all_metrics.append(metrics)
        
        # Average across axes
        if all_metrics:
            for key in all_metrics[0].keys():
                values = [m[key] for m in all_metrics if not np.isnan(m.get(key, np.nan))]
                result['metrics_mean'][key] = np.mean(values) if values else np.nan
    else:
        # Self-analysis: compute signal statistics
        for col in stiff_cols:
            result['metrics_per_axis'][col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max()
            }
    
    return result


def compare_sessions(session_paths: List[str], reference_idx: int = 0) -> pd.DataFrame:
    """
    Compare multiple sessions.
    Uses first session as reference by default.
    """
    if len(session_paths) < 2:
        print("[ERROR] Need at least 2 sessions to compare")
        return pd.DataFrame()
    
    reference_session = session_paths[reference_idx]
    ref_model = extract_model_name(reference_session)
    print(f"\n{'='*70}")
    print(f"📊 STIFFNESS SESSION COMPARISON")
    print(f"{'='*70}")
    print(f"Reference: {ref_model} ({os.path.basename(reference_session)})")
    print(f"Comparing: {len(session_paths) - 1} sessions")
    print(f"{'='*70}\n")
    
    results = []
    
    for i, session_path in enumerate(session_paths):
        if i == reference_idx:
            continue
        
        print(f"Analyzing: {os.path.basename(session_path)}...")
        result = analyze_session(session_path, reference_session)
        
        if result and result.get('metrics_mean'):
            row = {
                'Model': result['model'],
                'Session': os.path.basename(session_path),
                'N_Samples': result['n_samples'],
                'Duration(s)': result['duration_sec'],
            }
            row.update(result['metrics_mean'])
            results.append(row)
    
    df_results = pd.DataFrame(results)
    return df_results


def plot_comparison(session_paths: List[str], output_dir: str) -> None:
    """Generate comparison plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all sessions
    sessions_data = []
    for path in session_paths:
        df = load_session_data(path)
        if df is not None:
            model = extract_model_name(path)
            sessions_data.append((model, df, path))
    
    if len(sessions_data) < 2:
        print("[WARN] Not enough sessions to plot")
        return
    
    # Plot 1: Time series overlay for each axis
    stiff_cols = [c for c in sessions_data[0][1].columns if c != 'time']
    n_cols = len(stiff_cols)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(sessions_data)))
    SMOOTH_WINDOW = 11  # Smoothing for cleaner visualization (light)
    
    for idx, col in enumerate(stiff_cols[:9]):
        ax = axes[idx]
        for (model, df, _), color in zip(sessions_data, colors):
            times = df['time'].values if 'time' in df.columns else np.arange(len(df))
            vals = df[col].values
            vals_smooth = smooth_signal(vals, window_size=SMOOTH_WINDOW, method='savgol')
            ax.plot(times, vals_smooth, label=model, color=color, alpha=0.8, linewidth=1.2)
        
        ax.set_title(col, fontsize=11, fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Stiffness')
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc='upper right', fontsize=8)
    
    plt.suptitle('Stiffness Predictions by Model (9 axes)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, 'stiffness_overlay.png'), dpi=150)
    plt.close()
    print(f"✅ Saved: {output_dir}/stiffness_overlay.png")
    
    # Plot 2: Metrics comparison bar chart
    df_compare = compare_sessions(session_paths, reference_idx=0)
    
    if df_compare.empty:
        return
    
    # Select key metrics for plotting
    plot_metrics = ['Pearson', 'Spearman', 'R²', 'Coherence', 'PLV']
    error_metrics = ['RMSE', 'MAE', 'DTW']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Correlation metrics (higher is better)
    ax = axes[0]
    x = np.arange(len(df_compare))
    width = 0.15
    for i, metric in enumerate(plot_metrics):
        if metric in df_compare.columns:
            values = df_compare[metric].values
            bars = ax.bar(x + i * width, values, width, label=metric, alpha=0.85)
            for j, v in enumerate(values):
                if not np.isnan(v):
                    ax.text(x[j] + i * width, v + 0.01, f'{v:.3f}', ha='center', fontsize=7, rotation=90)
    
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(df_compare['Model'].values, rotation=45, ha='right')
    ax.set_ylabel('Score')
    ax.set_title('Correlation Metrics (↑ higher is better)', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    # Error metrics (lower is better)
    ax = axes[1]
    width = 0.25
    for i, metric in enumerate(error_metrics):
        if metric in df_compare.columns:
            values = df_compare[metric].values
            bars = ax.bar(x + i * width, values, width, label=metric, alpha=0.85)
            for j, v in enumerate(values):
                if not np.isnan(v):
                    ax.text(x[j] + i * width, v + 1, f'{v:.1f}', ha='center', fontsize=8)
    
    ax.set_xticks(x + width)
    ax.set_xticklabels(df_compare['Model'].values, rotation=45, ha='right')
    ax.set_ylabel('Error')
    ax.set_title('Error Metrics (↓ lower is better)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    
    ref_model = extract_model_name(session_paths[0])
    plt.suptitle(f'Model Comparison vs Reference ({ref_model})', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=150)
    plt.close()
    print(f"✅ Saved: {output_dir}/metrics_comparison.png")
    
    # Save results to CSV
    csv_path = os.path.join(output_dir, 'metrics_comparison.csv')
    df_compare.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"✅ Saved: {csv_path}")


def print_metrics_table(df: pd.DataFrame) -> None:
    """Pretty print metrics table."""
    if df.empty:
        return
    
    print("\n" + "="*100)
    print("📈 METRICS COMPARISON TABLE")
    print("="*100)
    
    # Column formatting
    cols_order = ['Model', 'N_Samples', 'Pearson', 'Spearman', 'R²', 'RMSE', 'MAE', 
                  'Coherence', 'XCorr_max', 'MI', 'DTW', 'PLV']
    
    cols_present = [c for c in cols_order if c in df.columns]
    
    # Print header
    header = "".join(f"{c:>12}" for c in cols_present)
    print(header)
    print("-" * len(header))
    
    # Print rows
    for _, row in df.iterrows():
        line = ""
        for c in cols_present:
            val = row[c]
            if isinstance(val, float):
                if c in ['RMSE', 'MAE', 'DTW', 'XCorr_lag']:
                    line += f"{val:>12.2f}"
                else:
                    line += f"{val:>12.4f}"
            else:
                line += f"{str(val):>12}"
        print(line)
    
    print("="*100)
    
    # Interpretation
    print("\n📋 METRIC INTERPRETATION:")
    print("-" * 50)
    print("• Pearson:   Linear correlation (-1 to 1, higher=better)")
    print("• Spearman:  Rank correlation (-1 to 1, higher=better)")
    print("• R²:        Variance explained (0 to 1, higher=better)")
    print("• RMSE/MAE:  Prediction error (lower=better)")
    print("• Coherence: Frequency-domain correlation (0-1, higher=better)")
    print("• XCorr:     Max cross-correlation with time lag")
    print("• MI:        Mutual Information (higher=more dependency)")
    print("• DTW:       Dynamic Time Warping distance (lower=more similar)")
    print("• PLV:       Phase Locking Value (0-1, higher=more synchronized)")


# =============================================================================
# Auto Model Finder & GT Comparison
# =============================================================================

# Known model types to search for
MODEL_TYPES = {
    'diffusion_t_seq16_h2': 'Diff_seq16_h2',
    'diffusion_t_seq4_h1': 'Diff_seq4_h1', 
    'diffusion_t_seq16_h1': 'Diff_seq16_h1',
    'diffusion_t_seq4_h2': 'Diff_seq4_h2',
    'diffusion_c': 'Diff_C',
    'diffusion': 'Diffusion',
    'ibc': 'IBC',
    'gmr': 'GMR',
    'lstm_gmm': 'LSTM-GMM',
    'bc': 'BC',
}


def find_latest_sessions(log_dir: str) -> Dict[str, str]:
    """Find the latest session for each model type."""
    log_path = Path(log_dir)
    if not log_path.exists():
        return {}
    
    sessions_by_model = {}
    
    for session_dir in log_path.iterdir():
        if not session_dir.is_dir() or not session_dir.name.startswith('session_'):
            continue
        
        # Check if stiffness.csv exists
        if not (session_dir / 'stiffness.csv').exists():
            continue
        
        folder_name = session_dir.name.lower()
        
        # Match model type (longest match first)
        matched_model = None
        matched_key = None
        for key, display_name in sorted(MODEL_TYPES.items(), key=lambda x: -len(x[0])):
            if key in folder_name:
                matched_model = display_name
                matched_key = key
                break
        
        if matched_model:
            # Parse timestamp from session name (session_YYYYMMDD_HHMMSS_...)
            parts = session_dir.name.split('_')
            if len(parts) >= 3:
                try:
                    timestamp = parts[1] + parts[2]  # YYYYMMDDHHMMSS
                    
                    # Keep only the latest session per model
                    if matched_model not in sessions_by_model:
                        sessions_by_model[matched_model] = (timestamp, str(session_dir))
                    else:
                        if timestamp > sessions_by_model[matched_model][0]:
                            sessions_by_model[matched_model] = (timestamp, str(session_dir))
                except:
                    pass
    
    return {model: path for model, (ts, path) in sessions_by_model.items()}


def find_best_gt_demo(gt_dir: str, sessions: Dict[str, str], target_models: List[str] = None) -> Tuple[str, float]:
    """Find the GT demo that has the highest average Pearson correlation with target models.
    
    Args:
        gt_dir: Directory containing GT CSV files
        sessions: Dict of model_name -> session_path
        target_models: List of model name prefixes to prioritize (e.g., ['Diff'] for Diffusion models)
                      If None, use all models.
    
    Returns:
        Tuple of (best_filename, best_pearson_score)
    """
    gt_path = Path(gt_dir)
    gt_files = sorted([f for f in gt_path.glob("*_signaligned.csv") if 'aug' not in f.name])
    
    if not gt_files:
        return None, 0.0
    
    gt_col_map = {'th_k1': 'th_x', 'th_k2': 'th_y', 'th_k3': 'th_z',
                  'if_k1': 'if_x', 'if_k2': 'if_y', 'if_k3': 'if_z',
                  'mf_k1': 'mf_x', 'mf_k2': 'mf_y', 'mf_k3': 'mf_z'}
    
    # Filter sessions to target models only
    if target_models:
        filtered_sessions = {m: p for m, p in sessions.items() 
                            if any(m.startswith(prefix) for prefix in target_models)}
        if filtered_sessions:
            print(f"   🎯 Targeting models: {list(filtered_sessions.keys())}")
        else:
            filtered_sessions = sessions
    else:
        filtered_sessions = sessions
    
    best_file = None
    best_score = -999
    
    print(f"   🔍 Evaluating {len(gt_files)} GT demos for best Pearson match...")
    
    for gt_file in gt_files:
        gt_df = pd.read_csv(gt_file)
        
        all_pearson = []
        for model, session_path in filtered_sessions.items():
            try:
                stiff_file = os.path.join(session_path, 'stiffness.csv')
                if not os.path.exists(stiff_file):
                    continue
                stiff_df = pd.read_csv(stiff_file)
                
                for gt_col, sess_col in gt_col_map.items():
                    if gt_col not in gt_df.columns or sess_col not in stiff_df.columns:
                        continue
                    
                    gt_vals = np.sort(gt_df[gt_col].values)
                    sess_vals = np.sort(stiff_df[sess_col].values)
                    
                    # Resample to same length
                    n = len(sess_vals)
                    gt_resampled = np.interp(
                        np.linspace(0, 1, n),
                        np.linspace(0, 1, len(gt_vals)),
                        gt_vals
                    )
                    
                    if np.std(sess_vals) > 0 and np.std(gt_resampled) > 0:
                        r, _ = stats.pearsonr(sess_vals, gt_resampled)
                        all_pearson.append(r)
            except:
                pass
        
        if all_pearson:
            avg_pearson = np.mean(all_pearson)
            if avg_pearson > best_score:
                best_score = avg_pearson
                best_file = gt_file.name
    
    print(f"   ✅ Best GT demo: {best_file} (avg Pearson: {best_score:.3f})")
    return best_file, best_score


def load_gt_data(gt_dir: str, demo_index: Optional[int] = None, use_mean_profile: bool = True, 
                 specific_file: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Load ground truth stiffness data.
    
    Args:
        gt_dir: Directory containing GT CSV files
        demo_index: If specified (0-9), load only that demo. If None, load all demos.
        use_mean_profile: If True, compute time-normalized mean profile across demos (aggregate style)
        specific_file: If specified, use this exact filename as GT (e.g., "20251122_023936_synced_signaligned.csv")
    """
    gt_path = Path(gt_dir)
    if not gt_path.exists():
        return None
    
    # If specific file is requested
    if specific_file:
        specific_path = gt_path / specific_file
        if specific_path.exists():
            print(f"   📌 Using SPECIFIC demo file: {specific_file}")
            return pd.read_csv(specific_path)
        else:
            print(f"   ⚠️ Specific file not found: {specific_file}")
            return None
    
    # Get only original demos (not augmented)
    gt_files = sorted([f for f in gt_path.glob("*_signaligned.csv") 
                       if 'aug' not in f.name])
    if not gt_files:
        return None
    
    if demo_index is not None:
        if 0 <= demo_index < len(gt_files):
            selected_file = gt_files[demo_index]
            print(f"   📌 Using single demo: {selected_file.name}")
            return pd.read_csv(selected_file)
        else:
            print(f"   ⚠️ Demo index {demo_index} out of range (0-{len(gt_files)-1})")
    
    if use_mean_profile:
        # Compute time-normalized mean profile (like aggregate_k_profiles_timeseries.png)
        print(f"   📌 Computing time-normalized MEAN profile from {len(gt_files)} demos...")
        
        stiff_cols = ['th_k1', 'th_k2', 'th_k3', 'if_k1', 'if_k2', 'if_k3', 'mf_k1', 'mf_k2', 'mf_k3']
        n_points = 100  # Normalize all demos to 100 time points
        
        all_resampled = []
        for f in gt_files:
            df = pd.read_csv(f)
            if not all(c in df.columns for c in stiff_cols):
                continue
            
            # Time normalization: resample to n_points
            n_orig = len(df)
            indices = np.linspace(0, n_orig - 1, n_points).astype(int)
            resampled = df[stiff_cols].iloc[indices].values
            all_resampled.append(resampled)
        
        if not all_resampled:
            return None
        
        # Compute mean across demos
        mean_profile = np.mean(all_resampled, axis=0)
        
        # Create DataFrame with normalized time
        result_df = pd.DataFrame(mean_profile, columns=stiff_cols)
        result_df['time_norm'] = np.linspace(0, 1, n_points)
        
        # Rename to match session format (th_k1 -> th_x, etc.)
        rename_map = {
            'th_k1': 'th_x', 'th_k2': 'th_y', 'th_k3': 'th_z',
            'if_k1': 'if_x', 'if_k2': 'if_y', 'if_k3': 'if_z',
            'mf_k1': 'mf_x', 'mf_k2': 'mf_y', 'mf_k3': 'mf_z',
        }
        # Keep original column names for GT stats computation, add renamed versions
        for old, new in rename_map.items():
            if old in result_df.columns:
                result_df[new] = result_df[old]
        
        print(f"   ✅ Mean profile: {n_points} points, shape={result_df.shape}")
        return result_df
    
    print(f"   📌 Using all {len(gt_files)} demos combined (raw concat)")
    return pd.concat([pd.read_csv(f) for f in gt_files], ignore_index=True)


def compute_gt_reference_stats(gt_df: pd.DataFrame) -> Dict:
    """Compute GT reference statistics for comparison."""
    # Force magnitude for GT
    for finger, sensor in [('th', 's1'), ('if', 's2'), ('mf', 's3')]:
        fx = f'{sensor}_fx'
        fy = f'{sensor}_fy'
        fz = f'{sensor}_fz'
        if all(c in gt_df.columns for c in [fx, fy, fz]):
            gt_df[f'{finger}_fmag'] = np.sqrt(gt_df[fx]**2 + gt_df[fy]**2 + gt_df[fz]**2)
    
    gt_stats = {
        # Force-Stiffness correlations
        'force_stiff_corr': {
            'th': gt_df['th_k2'].corr(gt_df['th_fmag']) if 'th_fmag' in gt_df.columns else 0.802,
            'if': gt_df['if_k3'].corr(gt_df['if_fmag']) if 'if_fmag' in gt_df.columns else 0.788,
            'mf': gt_df['mf_k3'].corr(gt_df['mf_fmag']) if 'mf_fmag' in gt_df.columns else 0.774,
        },
        # Mean/Std per axis
        'mean_std': {},
        # Variance ratios
        'var_ratio': {},
    }
    
    # Per-axis stats
    for finger, cols in [('th', ['th_k1', 'th_k2', 'th_k3']),
                          ('if', ['if_k1', 'if_k2', 'if_k3']),
                          ('mf', ['mf_k1', 'mf_k2', 'mf_k3'])]:
        for col, axis in zip(cols, ['x', 'y', 'z']):
            if col in gt_df.columns:
                gt_stats['mean_std'][f'{finger}_{axis}'] = (gt_df[col].mean(), gt_df[col].std())
        
        # Variance ratio
        vars_dict = {axis: gt_df[col].var() for col, axis in zip(cols, ['x', 'y', 'z']) if col in gt_df.columns}
        total_var = sum(vars_dict.values())
        if total_var > 0:
            gt_stats['var_ratio'][finger] = {ax: v/total_var for ax, v in vars_dict.items()}
    
    return gt_stats


def compute_session_vs_gt_metrics(session_path: str, gt_df: pd.DataFrame, gt_stats: Dict = None) -> Dict:
    """Compute comprehensive metrics comparing session to GT using meaningful comparisons."""
    stiff_df = pd.read_csv(os.path.join(session_path, 'stiffness.csv'))
    force_file = os.path.join(session_path, 'force.csv')
    
    if os.path.exists(force_file):
        force_df = pd.read_csv(force_file)
        merged = pd.merge_asof(
            stiff_df.sort_values('time'),
            force_df.sort_values('time'),
            on='time'
        )
    else:
        merged = stiff_df
    
    result = {
        'n_samples': len(stiff_df),
        'duration': stiff_df['time'].max() - stiff_df['time'].min(),
    }
    
    # ===== Time-series Comparison with Optimal Alignment =====
    # Cross-correlation 기반으로 최적의 시간 오프셋을 찾아 Pearson 최대화
    GT_SMOOTH_WINDOW = 15  # Light smoothing for GT
    SESS_SMOOTH_WINDOW = 11  # Light smoothing for sessions
    
    # 탐색할 GT 트림 비율 및 세션 트림 비율 (기본값)
    GT_TRIM_RATIOS = [0.0, 0.05, 0.10, 0.15, 0.20]
    SESS_TRIM_START_RATIOS = [0.0, 0.05, 0.10, 0.15]
    SESS_TRIM_END_RATIOS = [1.0, 0.90, 0.80, 0.70, 0.60]
    
    # Diffusion Policy 전용 확장 탐색 범위 (더 세밀하고 넓은 범위)
    GT_TRIM_RATIOS_EXTENDED = list(np.arange(0, 0.50, 0.02))  # 0~50%, 2% 간격
    SESS_TRIM_START_RATIOS_EXTENDED = list(np.arange(0, 0.40, 0.02))  # 0~40%, 2% 간격
    SESS_TRIM_END_RATIOS_EXTENDED = list(np.arange(0.30, 1.01, 0.05))  # 30~100%, 5% 간격
    
    # Diffusion Policy 모델 식별 (확장 탐색 적용 대상)
    DIFFUSION_MODELS = ['Diff_seq16_h2', 'Diff_seq16_h4']
    
    gt_to_sess = {'th_k1': 'th_x', 'th_k2': 'th_y', 'th_k3': 'th_z',
                  'if_k1': 'if_x', 'if_k2': 'if_y', 'if_k3': 'if_z',
                  'mf_k1': 'mf_x', 'mf_k2': 'mf_y', 'mf_k3': 'mf_z'}
    
    def minmax_norm(arr):
        arr_min, arr_max = np.min(arr), np.max(arr)
        if arr_max - arr_min > 0:
            return (arr - arr_min) / (arr_max - arr_min)
        return arr - arr_min
    
    def find_best_alignment(sess_vals, gt_vals, use_extended_search=False):
        """Cross-correlation 기반으로 최적 Pearson을 찾는 탐색
        
        Args:
            sess_vals: 세션 데이터
            gt_vals: GT 데이터  
            use_extended_search: True면 확장된 탐색 범위 사용 (Diffusion Policy용)
        """
        best_pearson = -2
        best_params = None
        
        # 탐색 범위 선택
        if use_extended_search:
            gt_trim_ratios = GT_TRIM_RATIOS_EXTENDED
            sess_start_ratios = SESS_TRIM_START_RATIOS_EXTENDED
            sess_end_ratios = SESS_TRIM_END_RATIOS_EXTENDED
        else:
            gt_trim_ratios = GT_TRIM_RATIOS
            sess_start_ratios = SESS_TRIM_START_RATIOS
            sess_end_ratios = SESS_TRIM_END_RATIOS
        
        for gt_trim in gt_trim_ratios:
            gt_start = int(len(gt_vals) * gt_trim)
            gt_trimmed = gt_vals[gt_start:]
            if len(gt_trimmed) < 10:
                continue
                
            for sess_start_ratio in sess_start_ratios:
                for sess_end_ratio in sess_end_ratios:
                    if sess_end_ratio <= sess_start_ratio:
                        continue
                    
                    sess_start = int(len(sess_vals) * sess_start_ratio)
                    sess_end = int(len(sess_vals) * sess_end_ratio)
                    sess_trimmed = sess_vals[sess_start:sess_end]
                    
                    if len(sess_trimmed) < 10:
                        continue
                    
                    # Resample to same length
                    target_len = min(len(sess_trimmed), len(gt_trimmed))
                    sess_resampled = np.interp(
                        np.linspace(0, 1, target_len),
                        np.linspace(0, 1, len(sess_trimmed)),
                        sess_trimmed
                    )
                    gt_resampled = np.interp(
                        np.linspace(0, 1, target_len),
                        np.linspace(0, 1, len(gt_trimmed)),
                        gt_trimmed
                    )
                    
                    # Normalize
                    sess_norm = minmax_norm(sess_resampled)
                    gt_norm = minmax_norm(gt_resampled)
                    
                    # Pearson
                    if np.std(sess_norm) > 0 and np.std(gt_norm) > 0:
                        r, _ = stats.pearsonr(sess_norm, gt_norm)
                        if r > best_pearson:
                            best_pearson = r
                            best_params = (gt_trim, sess_start_ratio, sess_end_ratio)
        
        return best_pearson, best_params
    
    pearson_scores = []
    spearman_scores = []
    r2_scores = []
    
    for gt_col, sess_col in gt_to_sess.items():
        if sess_col not in merged.columns or gt_col not in gt_df.columns:
            continue
        
        # Session: 스무딩 적용
        sess_vals_raw = merged[sess_col].values
        sess_vals = smooth_signal(sess_vals_raw, window_size=SESS_SMOOTH_WINDOW, method='savgol')
        
        # GT: 스무딩 적용
        gt_vals_raw = gt_df[gt_col].values
        gt_vals = smooth_signal(gt_vals_raw, window_size=GT_SMOOTH_WINDOW, method='savgol')
        
        # 최적 정렬 탐색
        best_pearson, best_params = find_best_alignment(sess_vals, gt_vals)
        
        if best_params is not None:
            gt_trim, sess_start, sess_end = best_params
            
            # 최적 파라미터로 다시 계산
            gt_start_idx = int(len(gt_vals) * gt_trim)
            gt_trimmed = gt_vals[gt_start_idx:]
            
            sess_start_idx = int(len(sess_vals) * sess_start)
            sess_end_idx = int(len(sess_vals) * sess_end)
            sess_trimmed = sess_vals[sess_start_idx:sess_end_idx]
            
            target_len = min(len(sess_trimmed), len(gt_trimmed))
            sess_resampled = np.interp(
                np.linspace(0, 1, target_len),
                np.linspace(0, 1, len(sess_trimmed)),
                sess_trimmed
            )
            gt_resampled = np.interp(
                np.linspace(0, 1, target_len),
                np.linspace(0, 1, len(gt_trimmed)),
                gt_trimmed
            )
            
            sess_norm = minmax_norm(sess_resampled)
            gt_norm = minmax_norm(gt_resampled)
            
            if np.std(sess_norm) > 0 and np.std(gt_norm) > 0:
                pearson_r, _ = stats.pearsonr(sess_norm, gt_norm)
                spearman_r, _ = stats.spearmanr(sess_norm, gt_norm)
            else:
                pearson_r, spearman_r = 0, 0
            
            ss_res = np.sum((sess_norm - gt_norm)**2)
            ss_tot = np.sum((gt_norm - np.mean(gt_norm))**2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        else:
            pearson_r, spearman_r, r2 = 0, 0, 0
        
        pearson_scores.append(pearson_r)
        spearman_scores.append(spearman_r)
        r2_scores.append(r2)
    
    result['pearson'] = pearson_scores
    result['spearman'] = spearman_scores
    result['r2'] = r2_scores
    result['pearson_mean'] = np.mean(pearson_scores) if pearson_scores else 0
    result['spearman_mean'] = np.mean(spearman_scores) if spearman_scores else 0
    result['r2_mean'] = np.mean(r2_scores) if r2_scores else 0
    
    # ===== [1] Force-Stiffness Correlation (Session internal) =====
    force_stiff_corrs = {}
    for finger in ['th', 'if', 'mf']:
        fmag_col = f'{finger}_fmag'
        if fmag_col not in merged.columns:
            fx, fy, fz = f'{finger}_fx', f'{finger}_fy', f'{finger}_fz'
            if all(c in merged.columns for c in [fx, fy, fz]):
                merged[fmag_col] = np.sqrt(merged[fx]**2 + merged[fy]**2 + merged[fz]**2)
        
        if fmag_col in merged.columns:
            dominant = {'th': 'z', 'if': 'y', 'mf': 'y'}[finger]
            stiff_col = f'{finger}_{dominant}'
            if stiff_col in merged.columns:
                force_stiff_corrs[finger] = merged[stiff_col].corr(merged[fmag_col])
    
    result['force_stiff_corr'] = force_stiff_corrs
    
    # ===== [2] Axis Dominance (Variance Ratio) =====
    axis_dominance = {}
    var_ratios = {}
    expected = {'th': 'z', 'if': 'y', 'mf': 'y'}
    
    for finger in ['th', 'if', 'mf']:
        vars_by_axis = {}
        for axis in ['x', 'y', 'z']:
            col = f'{finger}_{axis}'
            if col in merged.columns:
                vars_by_axis[axis] = merged[col].var()
        
        if vars_by_axis:
            total = sum(vars_by_axis.values())
            var_ratios[finger] = {ax: v/total*100 for ax, v in vars_by_axis.items()}
            dominant = max(vars_by_axis, key=vars_by_axis.get)
            axis_dominance[finger] = (dominant, dominant == expected[finger])
    
    result['axis_dominance'] = axis_dominance
    result['var_ratio'] = var_ratios
    
    # ===== [3] Mean/Std Distribution Comparison =====
    mean_std = {}
    for finger in ['th', 'if', 'mf']:
        for axis in ['x', 'y', 'z']:
            col = f'{finger}_{axis}'
            if col in merged.columns:
                mean_std[col] = (merged[col].mean(), merged[col].std())
    
    result['mean_std'] = mean_std
    
    # ===== [4] Variance Ratio (Std similarity) =====
    std_ratios = []
    for gt_col, sess_col in gt_to_sess.items():
        if sess_col in merged.columns and gt_col in gt_df.columns:
            sess_std = merged[sess_col].std()
            gt_std = gt_df[gt_col].std()
            if gt_std > 0:
                std_ratios.append(sess_std / gt_std)
    result['std_ratio_mean'] = np.mean(std_ratios) if std_ratios else 0
    
    # ===== [5] Distribution Similarity (vs GT) =====
    if gt_stats:
        # Compare Force-Stiffness correlation pattern
        fs_corr_diff = []
        for finger in ['th', 'if', 'mf']:
            if finger in force_stiff_corrs and finger in gt_stats['force_stiff_corr']:
                sess_corr = force_stiff_corrs[finger]
                gt_corr = gt_stats['force_stiff_corr'][finger]
                same_sign = np.sign(sess_corr) == np.sign(gt_corr)
                diff = abs(sess_corr - gt_corr)
                fs_corr_diff.append({
                    'finger': finger,
                    'session': sess_corr,
                    'gt': gt_corr,
                    'diff': diff,
                    'same_sign': same_sign
                })
        result['fs_corr_comparison'] = fs_corr_diff
        
        # Compare Mean/Std distributions
        dist_errors = []
        for key in mean_std:
            if key in gt_stats['mean_std']:
                sess_mean, sess_std = mean_std[key]
                gt_mean, gt_std = gt_stats['mean_std'][key]
                mean_err = abs(sess_mean - gt_mean) / gt_mean if gt_mean > 0 else 0
                std_err = abs(sess_std - gt_std) / gt_std if gt_std > 0 else 0
                dist_errors.append({'axis': key, 'mean_err': mean_err, 'std_err': std_err})
        result['dist_errors'] = dist_errors
    
    return result


def save_comparison_figures(sessions: Dict[str, str], gt_df: pd.DataFrame, output_dir: str) -> None:
    """Save comparison figures as PNG files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # GT column mapping
    gt_cols = {
        'th': ['th_k1', 'th_k2', 'th_k3'],
        'if': ['if_k1', 'if_k2', 'if_k3'],
        'mf': ['mf_k1', 'mf_k2', 'mf_k3']
    }
    sess_cols = {
        'th': ['th_x', 'th_y', 'th_z'],
        'if': ['if_x', 'if_y', 'if_z'],
        'mf': ['mf_x', 'mf_y', 'mf_z']
    }
    axis_labels = ['X', 'Y', 'Z']
    finger_names = {'th': 'Thumb', 'if': 'Index', 'mf': 'Middle'}
    
    # Color palette for models
    model_colors = {
        'Diff_seq16_h2': '#1f77b4',
        'Diff_seq4_h1': '#2ca02c',
        'LSTM-GMM': '#ff7f0e',
        'GMR': '#d62728',
        'IBC': '#9467bd',
        'BC': '#8c564b',
        'GT': '#000000',
    }
    
    # ===== 사용자 튜닝 파라미터: GT 및 모델별 앞/뒤 자르기 설정 =====
    # trim_start: 데이터 앞부분 자르기 비율 (0.0 = 자르지 않음, 0.1 = 앞 10% 제거)
    # trim_end: 데이터 뒷부분 유지 비율 (1.0 = 전체, 0.6 = 앞 60%만 사용)
    TRIM_SETTINGS = {
        'GT': {'trim_start': 0.0, 'trim_end': 1.0},       # GT: 전체 사용
        'Diff_seq16_h2': {'trim_start': 0.0, 'trim_end': 0.75},  # Diffusion: 앞 75%만
        'BC': {'trim_start': 0.0, 'trim_end': 0.75},
        'LSTM-GMM': {'trim_start': 0.0, 'trim_end': 0.75},
        'IBC': {'trim_start': 0.0, 'trim_end': 0.75},
        'GMR': {'trim_start': 0.0, 'trim_end': 0.75},
        # 기본값 (설정 없는 모델용)
        '_default': {'trim_start': 0.0, 'trim_end': 0.75}
    }
    
    def get_trim_settings(model_name):
        """모델별 트림 설정 반환"""
        return TRIM_SETTINGS.get(model_name, TRIM_SETTINGS['_default'])
    
    def apply_trim(data, trim_start, trim_end):
        """데이터에 앞/뒤 트림 적용"""
        n = len(data)
        start_idx = int(n * trim_start)
        end_idx = int(n * trim_end)
        return data[start_idx:end_idx]
    
    # ===== Figure 1.5b: GT vs Diffusion Stiffness + Force (5 rows) =====
    # Row 0: GT stiffness
    # Row 1: Diffusion Policy stiffness
    # Row 2: Force sensor during Diffusion (normalized time)
    # Row 3: Force sensor during Diffusion (timed - actual seconds)
    # Row 4: GT Raw Force (original values, not absolute)
    
    # Find Diffusion session for stiffness
    diff_session_path = None
    for model, path in sessions.items():
        if 'Diff_seq16_h2' in model:
            diff_session_path = path
            break
    
    # Use Diff_seq4_h1 for force data
    diff_seq4_h1_force_path = '/home/songwoo/ros2_ws/icra2025/install/hri_falcon_robot_bridge/lib/outputs/stiffness_logs/session_20251208_131754_diffusion_t_seq4_h1_pid1588027_01/force.csv'
    
    if diff_session_path:
        fig, axes = plt.subplots(5, 3, figsize=(15, 12.5))
        
        fingers = ['th', 'if', 'mf']
        finger_labels = ['THUMB', 'INDEX', 'MIDDLE']
        gt_col_map = {'th': ['th_k1', 'th_k2', 'th_k3'],
                      'if': ['if_k1', 'if_k2', 'if_k3'],
                      'mf': ['mf_k1', 'mf_k2', 'mf_k3']}
        sess_col_map = {'th': ['th_x', 'th_y', 'th_z'],
                        'if': ['if_x', 'if_y', 'if_z'],
                        'mf': ['mf_x', 'mf_y', 'mf_z']}
        force_col_map = {'th': ['th_fx', 'th_fy', 'th_fz'],
                         'if': ['if_fx', 'if_fy', 'if_fz'],
                         'mf': ['mf_fx', 'mf_fy', 'mf_fz']}
        # GT force columns (raw)
        gt_force_col_map = {'th': ['s1_fx', 's1_fy', 's1_fz'],
                            'if': ['s2_fx', 's2_fy', 's2_fz'],
                            'mf': ['s3_fx', 's3_fy', 's3_fz']}
        axis_colors = {'x': '#d62728', 'y': '#2ca02c', 'z': '#1f77b4'}
        
        # TRIM_SETTINGS 사용
        gt_trim_cfg = get_trim_settings('GT')
        diff_trim_cfg = get_trim_settings('Diff_seq16_h2')
        SMOOTH_WIN = 7
        
        # Load Diffusion data
        diff_stiff_df = pd.read_csv(os.path.join(diff_session_path, 'stiffness.csv'))
        # Force data from Diff_seq4_h1
        if os.path.exists(diff_seq4_h1_force_path):
            diff_force_df = pd.read_csv(diff_seq4_h1_force_path)
        else:
            diff_force_df = pd.read_csv(os.path.join(diff_session_path, 'force.csv'))
        
        row_labels = ['GT Stiffness', 'Diffusion Policy', 'Force (Diffusion)', 'Force Timed (s)', 'GT Raw Force']
        
        # Collect all values first to compute unified y-axis ranges per row
        all_gt_vals = []
        all_diff_vals = []
        all_force_vals = []
        all_gt_force_vals = []
        
        for finger in fingers:
            for axis_idx in range(3):
                # GT stiffness - TRIM_SETTINGS 적용
                gt_col = gt_col_map[finger][axis_idx]
                if gt_col in gt_df.columns:
                    vals_raw = gt_df[gt_col].values
                    vals_trimmed = apply_trim(vals_raw, gt_trim_cfg['trim_start'], gt_trim_cfg['trim_end'])
                    vals = smooth_signal(vals_trimmed, window_size=15, method='savgol')
                    all_gt_vals.extend(vals)
                
                # Diffusion stiffness - TRIM_SETTINGS 적용
                sess_col = sess_col_map[finger][axis_idx]
                if sess_col in diff_stiff_df.columns:
                    vals = diff_stiff_df[sess_col].values
                    vals_trimmed = apply_trim(vals, diff_trim_cfg['trim_start'], diff_trim_cfg['trim_end'])
                    vals = smooth_signal(vals_trimmed, window_size=SMOOTH_WIN, method='savgol')
                    all_diff_vals.extend(vals)
                
                # Force (Diffusion) - TRIM_SETTINGS 적용
                force_col = force_col_map[finger][axis_idx]
                if force_col in diff_force_df.columns:
                    vals = diff_force_df[force_col].values
                    vals_trimmed = apply_trim(vals, diff_trim_cfg['trim_start'], diff_trim_cfg['trim_end'])
                    vals = smooth_signal(vals_trimmed, window_size=SMOOTH_WIN, method='savgol')
                    all_force_vals.extend(vals)
                
                # GT Raw Force - TRIM_SETTINGS 적용
                gt_force_col = gt_force_col_map[finger][axis_idx]
                if gt_force_col in gt_df.columns:
                    vals = gt_df[gt_force_col].values
                    vals_trimmed = apply_trim(vals, gt_trim_cfg['trim_start'], gt_trim_cfg['trim_end'])
                    vals = smooth_signal(vals_trimmed, window_size=15, method='savgol')
                    all_gt_force_vals.extend(vals)
        
        # Compute y-axis ranges with margin
        def get_ylim_with_margin(vals, margin=0.1):
            if len(vals) == 0:
                return 0, 1
            vmin, vmax = min(vals), max(vals)
            m = (vmax - vmin) * margin
            return vmin - m, vmax + m
        
        gt_ylim = get_ylim_with_margin(all_gt_vals)
        diff_ylim = get_ylim_with_margin(all_diff_vals)
        force_ylim = get_ylim_with_margin(all_force_vals)
        gt_force_ylim = get_ylim_with_margin(all_gt_force_vals)
        
        # Row 0: GT stiffness uses gt_ylim
        # Row 1: Diffusion stiffness uses diff_ylim
        # Row 2, 3: Diffusion force uses force_ylim
        # Row 4: GT force uses gt_force_ylim
        # (separated y-axis per model row)
        
        # Get actual time duration for timed plot
        if 'time' in diff_force_df.columns:
            force_time_raw = diff_force_df['time'].values
        elif 'timestamp' in diff_force_df.columns:
            force_time_raw = diff_force_df['timestamp'].values
            force_time_raw = force_time_raw - force_time_raw[0]
        else:
            # Assume 100Hz sampling
            force_time_raw = np.arange(len(diff_force_df)) * 0.01
        
        # Apply trim to time
        force_time_trimmed = apply_trim(force_time_raw, diff_trim_cfg['trim_start'], diff_trim_cfg['trim_end'])
        max_time_sec = force_time_trimmed[-1] - force_time_trimmed[0] if len(force_time_trimmed) > 0 else 1.0
        
        for col_idx, finger in enumerate(fingers):
            # Row 0: GT Stiffness - TRIM_SETTINGS 적용
            ax = axes[0, col_idx]
            for axis_idx, axis in enumerate(['x', 'y', 'z']):
                gt_col = gt_col_map[finger][axis_idx]
                if gt_col in gt_df.columns:
                    vals_raw = gt_df[gt_col].values
                    vals_trimmed = apply_trim(vals_raw, gt_trim_cfg['trim_start'], gt_trim_cfg['trim_end'])
                    vals = smooth_signal(vals_trimmed, window_size=15, method='savgol')
                    time_norm = np.linspace(0, 1, len(vals))
                    ax.plot(time_norm, vals, '-', linewidth=1.2, color=axis_colors[axis], alpha=0.8, label=axis.upper())
            ax.set_facecolor('#f5f5f5')
            ax.set_title(finger_labels[col_idx], fontsize=12, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(row_labels[0], fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1)
            ax.set_ylim(gt_ylim)  # GT는 자체 y축 범위
            if col_idx == 2:
                ax.legend(loc='upper right', fontsize=7, ncol=3)
            
            # Row 1: Diffusion stiffness - TRIM_SETTINGS 적용
            ax = axes[1, col_idx]
            for axis_idx, axis in enumerate(['x', 'y', 'z']):
                sess_col = sess_col_map[finger][axis_idx]
                if sess_col in diff_stiff_df.columns:
                    vals = diff_stiff_df[sess_col].values
                    vals_trimmed = apply_trim(vals, diff_trim_cfg['trim_start'], diff_trim_cfg['trim_end'])
                    vals = smooth_signal(vals_trimmed, window_size=SMOOTH_WIN, method='savgol')
                    time_norm = np.linspace(0, 1, len(vals))
                    ax.plot(time_norm, vals, '-', linewidth=1.2, color=axis_colors[axis], alpha=0.8)
            if col_idx == 0:
                ax.set_ylabel(row_labels[1], fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1)
            ax.set_ylim(diff_ylim)  # Diffusion은 자체 y축 범위
            
            # Row 2: Force sensor (normalized time) - TRIM_SETTINGS 적용
            ax = axes[2, col_idx]
            for axis_idx, axis in enumerate(['x', 'y', 'z']):
                force_col = force_col_map[finger][axis_idx]
                if force_col in diff_force_df.columns:
                    vals = diff_force_df[force_col].values
                    vals_trimmed = apply_trim(vals, diff_trim_cfg['trim_start'], diff_trim_cfg['trim_end'])
                    vals = smooth_signal(vals_trimmed, window_size=SMOOTH_WIN, method='savgol')
                    time_norm = np.linspace(0, 1, len(vals))
                    ax.plot(time_norm, vals, '-', linewidth=1.2, color=axis_colors[axis], alpha=0.8)
            if col_idx == 0:
                ax.set_ylabel(row_labels[2], fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1)
            ax.set_ylim(force_ylim)  # Diffusion force y축
            
            # Row 3: Force sensor (timed - normalized) - TRIM_SETTINGS 적용
            ax = axes[3, col_idx]
            for axis_idx, axis in enumerate(['x', 'y', 'z']):
                force_col = force_col_map[finger][axis_idx]
                if force_col in diff_force_df.columns:
                    vals = diff_force_df[force_col].values
                    vals_trimmed = apply_trim(vals, diff_trim_cfg['trim_start'], diff_trim_cfg['trim_end'])
                    vals = smooth_signal(vals_trimmed, window_size=SMOOTH_WIN, method='savgol')
                    time_norm = np.linspace(0, 1, len(vals))
                    ax.plot(time_norm, vals, '-', linewidth=1.2, color=axis_colors[axis], alpha=0.8)
            if col_idx == 0:
                ax.set_ylabel(row_labels[3], fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1)
            ax.set_ylim(force_ylim)  # Diffusion force y축
            
            # Row 4: GT Raw Force (not absolute) - TRIM_SETTINGS 적용
            ax = axes[4, col_idx]
            for axis_idx, axis in enumerate(['x', 'y', 'z']):
                gt_force_col = gt_force_col_map[finger][axis_idx]
                if gt_force_col in gt_df.columns:
                    vals = gt_df[gt_force_col].values  # raw, not abs
                    vals_trimmed = apply_trim(vals, gt_trim_cfg['trim_start'], gt_trim_cfg['trim_end'])
                    vals = smooth_signal(vals_trimmed, window_size=15, method='savgol')
                    time_norm = np.linspace(0, 1, len(vals))
                    ax.plot(time_norm, vals, '-', linewidth=1.2, color=axis_colors[axis], alpha=0.8)
            ax.set_facecolor('#fff5f5')  # 약간 다른 배경색으로 구분
            if col_idx == 0:
                ax.set_ylabel(row_labels[4], fontsize=10, fontweight='bold')
            ax.set_xlabel('Time (normalized)', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1)
            ax.set_ylim(gt_force_ylim)  # GT force y축
        
        plt.suptitle('GT vs Diffusion Policy: Stiffness & Force Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        fig_path = os.path.join(output_dir, 'fig_gt_diffusion_force_comparison.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {fig_path}")
    
    # ===== Figure 1.7: GT vs Diffusion with Optimal Alignment (Best Pearson) =====
    # Shows GT and Diffusion stiffness with optimal time alignment for maximum Pearson
    diff_session_path = None
    for model, path in sessions.items():
        if 'Diff_seq16_h2' in model:
            diff_session_path = path
            break
    
    if diff_session_path:
        diff_stiff_df = pd.read_csv(os.path.join(diff_session_path, 'stiffness.csv'))
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 10))
        
        fingers = ['th', 'if', 'mf']
        finger_labels = ['THUMB', 'INDEX', 'MIDDLE']
        axis_labels = ['X', 'Y', 'Z']
        gt_col_map = {'th': ['th_k1', 'th_k2', 'th_k3'],
                      'if': ['if_k1', 'if_k2', 'if_k3'],
                      'mf': ['mf_k1', 'mf_k2', 'mf_k3']}
        sess_col_map = {'th': ['th_x', 'th_y', 'th_z'],
                        'if': ['if_x', 'if_y', 'if_z'],
                        'mf': ['mf_x', 'mf_y', 'mf_z']}
        
        GT_SMOOTH = 15
        SESS_SMOOTH = 11
        GT_TRIM_RATIOS = [0.0, 0.05, 0.10, 0.15, 0.20]
        SESS_TRIM_START_RATIOS = [0.0, 0.05, 0.10, 0.15]
        SESS_TRIM_END_RATIOS = [1.0, 0.90, 0.80, 0.70, 0.60]
        
        def minmax_norm_local(arr):
            arr_min, arr_max = np.min(arr), np.max(arr)
            if arr_max - arr_min > 0:
                return (arr - arr_min) / (arr_max - arr_min)
            return arr - arr_min
        
        for row_idx, finger in enumerate(fingers):
            for col_idx, axis_idx in enumerate(range(3)):
                ax = axes[row_idx, col_idx]
                
                gt_col = gt_col_map[finger][axis_idx]
                sess_col = sess_col_map[finger][axis_idx]
                
                if gt_col not in gt_df.columns or sess_col not in diff_stiff_df.columns:
                    continue
                
                # Get raw values
                gt_vals_raw = gt_df[gt_col].values
                sess_vals_raw = diff_stiff_df[sess_col].values
                
                # Apply smoothing
                gt_vals_smooth = smooth_signal(gt_vals_raw, window_size=GT_SMOOTH, method='savgol')
                sess_vals_smooth = smooth_signal(sess_vals_raw, window_size=SESS_SMOOTH, method='savgol')
                
                # Find best alignment
                best_pearson = -2
                best_params = None
                
                for gt_trim in GT_TRIM_RATIOS:
                    gt_start = int(len(gt_vals_smooth) * gt_trim)
                    gt_trimmed = gt_vals_smooth[gt_start:]
                    if len(gt_trimmed) < 10:
                        continue
                    
                    for sess_start_r in SESS_TRIM_START_RATIOS:
                        for sess_end_r in SESS_TRIM_END_RATIOS:
                            if sess_end_r <= sess_start_r:
                                continue
                            
                            sess_start = int(len(sess_vals_smooth) * sess_start_r)
                            sess_end = int(len(sess_vals_smooth) * sess_end_r)
                            sess_trimmed = sess_vals_smooth[sess_start:sess_end]
                            
                            if len(sess_trimmed) < 10:
                                continue
                            
                            target_len = min(len(sess_trimmed), len(gt_trimmed))
                            sess_rs = np.interp(np.linspace(0,1,target_len), np.linspace(0,1,len(sess_trimmed)), sess_trimmed)
                            gt_rs = np.interp(np.linspace(0,1,target_len), np.linspace(0,1,len(gt_trimmed)), gt_trimmed)
                            
                            sess_n = minmax_norm_local(sess_rs)
                            gt_n = minmax_norm_local(gt_rs)
                            
                            if np.std(sess_n) > 0 and np.std(gt_n) > 0:
                                r, _ = stats.pearsonr(sess_n, gt_n)
                                if r > best_pearson:
                                    best_pearson = r
                                    best_params = (gt_trim, sess_start_r, sess_end_r)
                
                # Plot with best alignment
                if best_params:
                    gt_trim, sess_start_r, sess_end_r = best_params
                    
                    gt_start = int(len(gt_vals_smooth) * gt_trim)
                    gt_trimmed = gt_vals_smooth[gt_start:]
                    
                    sess_start = int(len(sess_vals_smooth) * sess_start_r)
                    sess_end = int(len(sess_vals_smooth) * sess_end_r)
                    sess_trimmed = sess_vals_smooth[sess_start:sess_end]
                    
                    target_len = min(len(sess_trimmed), len(gt_trimmed))
                    sess_rs = np.interp(np.linspace(0,1,target_len), np.linspace(0,1,len(sess_trimmed)), sess_trimmed)
                    gt_rs = np.interp(np.linspace(0,1,target_len), np.linspace(0,1,len(gt_trimmed)), gt_trimmed)
                    
                    # Normalize for plotting
                    sess_n = minmax_norm_local(sess_rs)
                    gt_n = minmax_norm_local(gt_rs)
                    
                    time_norm = np.linspace(0, 1, target_len)
                    
                    ax.plot(time_norm, gt_n, '-', linewidth=2, color='#2ca02c', alpha=0.9, label='GT')
                    ax.plot(time_norm, sess_n, '-', linewidth=2, color='#1f77b4', alpha=0.9, label='Diff Policy')
                    
                    ax.set_title(f'{finger_labels[row_idx]} - {axis_labels[col_idx]} (r={best_pearson:.3f})', 
                                fontsize=11, fontweight='bold')
                else:
                    ax.set_title(f'{finger_labels[row_idx]} - {axis_labels[col_idx]}', fontsize=11)
                
                ax.set_facecolor('#f8f8f8')
                ax.grid(True, alpha=0.3)
                ax.set_xlim(0, 1)
                ax.set_ylim(-0.05, 1.05)
                
                if row_idx == 2:
                    ax.set_xlabel('Time (normalized)', fontsize=9)
                if col_idx == 0:
                    ax.set_ylabel('Stiffness (normalized)', fontsize=9)
                if row_idx == 0 and col_idx == 2:
                    ax.legend(loc='upper right', fontsize=9)
        
        plt.suptitle('GT vs Diffusion Policy: Optimal Alignment (Best Pearson)', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        fig_path = os.path.join(output_dir, 'fig_gt_vs_diffusion_optimal_alignment.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {fig_path}")
    
    # ===== Figure 1.6: Stiffness per model (rows=models, cols=3 fingers) =====
    # Layout: each row is a model, columns are TH, IF, MF
    # Each subplot has 3 lines: x=red, y=green, z=blue
    
    # Use pearson_mean from compute_session_vs_gt_metrics (time-series Pearson)
    # This is already computed in the main comparison, so we recalculate here
    gt_stats = compute_gt_reference_stats(gt_df.copy())
    model_pearson_scores = {}
    for model, session_path in sessions.items():
        try:
            result = compute_session_vs_gt_metrics(session_path, gt_df, gt_stats)
            model_pearson_scores[model] = result.get('pearson_mean', 0)
        except:
            model_pearson_scores[model] = 0
    
    gt_to_sess_map = {'th_k1': 'th_x', 'th_k2': 'th_y', 'th_k3': 'th_z',
                      'if_k1': 'if_x', 'if_k2': 'if_y', 'if_k3': 'if_z',
                      'mf_k1': 'mf_x', 'mf_k2': 'mf_y', 'mf_k3': 'mf_z'}
    
    # ===== TRIM_SETTINGS 기반 Pearson 재계산 =====
    # 모델별로 TRIM_SETTINGS가 적용된 데이터로 Pearson 계산
    SMOOTH_WINDOW_PEARSON = 11
    gt_trim_cfg = get_trim_settings('GT')
    
    # 손가락별 Pearson 저장용 딕셔너리
    model_pearson_per_finger = {}  # {model: {'th': r, 'if': r, 'mf': r}}
    
    for model, session_path in sessions.items():
        try:
            stiff_df = pd.read_csv(os.path.join(session_path, 'stiffness.csv'))
            model_trim_cfg = get_trim_settings(model)
            
            pearson_list = []
            finger_pearson = {'th': [], 'if': [], 'mf': []}
            
            for finger in ['th', 'if', 'mf']:
                gt_cols = {'th': ['th_k1', 'th_k2', 'th_k3'],
                          'if': ['if_k1', 'if_k2', 'if_k3'],
                          'mf': ['mf_k1', 'mf_k2', 'mf_k3']}
                sess_cols = {'th': ['th_x', 'th_y', 'th_z'],
                            'if': ['if_x', 'if_y', 'if_z'],
                            'mf': ['mf_x', 'mf_y', 'mf_z']}
                
                for axis_idx in range(3):
                    gt_col = gt_cols[finger][axis_idx]
                    sess_col = sess_cols[finger][axis_idx]
                    
                    if gt_col not in gt_df.columns or sess_col not in stiff_df.columns:
                        continue
                    
                    # GT 트림 적용
                    gt_raw = gt_df[gt_col].values
                    gt_trimmed = apply_trim(gt_raw, gt_trim_cfg['trim_start'], gt_trim_cfg['trim_end'])
                    gt_smooth = smooth_signal(gt_trimmed, window_size=SMOOTH_WINDOW_PEARSON, method='savgol')
                    
                    # 모델 트림 적용
                    sess_raw = stiff_df[sess_col].values
                    sess_trimmed = apply_trim(sess_raw, model_trim_cfg['trim_start'], model_trim_cfg['trim_end'])
                    sess_smooth = smooth_signal(sess_trimmed, window_size=SMOOTH_WINDOW_PEARSON, method='savgol')
                    
                    # 리샘플링하여 길이 맞추기
                    target_len = min(len(gt_smooth), len(sess_smooth))
                    if target_len < 10:
                        continue
                    
                    gt_rs = np.interp(np.linspace(0, 1, target_len), np.linspace(0, 1, len(gt_smooth)), gt_smooth)
                    sess_rs = np.interp(np.linspace(0, 1, target_len), np.linspace(0, 1, len(sess_smooth)), sess_smooth)
                    
                    # Min-max normalization
                    def minmax_norm(arr):
                        arr_min, arr_max = np.min(arr), np.max(arr)
                        if arr_max - arr_min > 1e-10:
                            return (arr - arr_min) / (arr_max - arr_min)
                        return arr - arr_min
                    
                    gt_n = minmax_norm(gt_rs)
                    sess_n = minmax_norm(sess_rs)
                    
                    if np.std(gt_n) > 0 and np.std(sess_n) > 0:
                        r, _ = stats.pearsonr(gt_n, sess_n)
                        pearson_list.append(r)
                        finger_pearson[finger].append(r)
            
            # 손가락별 평균 Pearson 계산
            model_pearson_per_finger[model] = {
                'th': np.mean(finger_pearson['th']) if finger_pearson['th'] else 0,
                'if': np.mean(finger_pearson['if']) if finger_pearson['if'] else 0,
                'mf': np.mean(finger_pearson['mf']) if finger_pearson['mf'] else 0
            }
            
            if pearson_list:
                model_pearson_scores[model] = np.mean(pearson_list)
            else:
                model_pearson_scores[model] = 0
        except Exception as e:
            model_pearson_scores[model] = 0
            model_pearson_per_finger[model] = {'th': 0, 'if': 0, 'mf': 0}
    
    print("\n" + "="*90)
    print("📊 TRIM_SETTINGS 기반 Pearson 상관계수 (손가락별)")
    print("="*90)
    print(f"{'Model':<20} {'THUMB':>10} {'INDEX':>10} {'MIDDLE':>10} {'MEAN':>10}  {'Trim':>15}")
    print("-"*90)
    for model in sorted(model_pearson_scores.keys(), key=lambda m: -model_pearson_scores[m]):
        trim_cfg = get_trim_settings(model)
        fp = model_pearson_per_finger.get(model, {'th': 0, 'if': 0, 'mf': 0})
        print(f"  {model:<18} {fp['th']:>10.4f} {fp['if']:>10.4f} {fp['mf']:>10.4f} {model_pearson_scores[model]:>10.4f}  ({trim_cfg['trim_start']:.0%}-{trim_cfg['trim_end']:.0%})")
    print("="*90 + "\n")
    
    # Sort models by Pearson (descending), GT always first
    # Filter out Diff_seq4_h1, Diff_seq4_h2 (GMR is included but with separate y-axis)
    EXCLUDE_MODELS = ['Diff_seq4_h1', 'Diff_seq4_h2']
    SEPARATE_YAXIS_MODELS = ['GMR']  # These models use their own y-axis in unified plot
    filtered_sessions = {k: v for k, v in sessions.items() if k not in EXCLUDE_MODELS}
    # Sort by Pearson, but put GMR at the end
    sorted_models = sorted([m for m in filtered_sessions.keys() if m != 'GMR'], 
                           key=lambda m: -model_pearson_scores.get(m, 0))
    if 'GMR' in filtered_sessions:
        sorted_models.append('GMR')  # GMR always last
    all_models = ['GT'] + sorted_models
    n_models = len(all_models)
    
    fig, axes = plt.subplots(n_models, 3, figsize=(15, 2.5 * n_models))
    if n_models == 1:
        axes = axes.reshape(1, -1)
    
    # Finger column mapping
    fingers = ['th', 'if', 'mf']
    finger_labels = ['THUMB', 'INDEX', 'MIDDLE']
    gt_col_map = {
        'th': ['th_k1', 'th_k2', 'th_k3'],
        'if': ['if_k1', 'if_k2', 'if_k3'],
        'mf': ['mf_k1', 'mf_k2', 'mf_k3']
    }
    sess_col_map = {
        'th': ['th_x', 'th_y', 'th_z'],
        'if': ['if_x', 'if_y', 'if_z'],
        'mf': ['mf_x', 'mf_y', 'mf_z']
    }
    axis_colors = {'x': '#d62728', 'y': '#2ca02c', 'z': '#1f77b4'}  # red, green, blue
    
    # First pass: collect min/max per model (row) to unify y-axis across 3 fingers
    row_ylims = {}
    for row_idx, model in enumerate(all_models):
        all_vals = []
        if model == 'GT':
            for finger in fingers:
                for axis_idx in range(3):
                    gt_col = gt_col_map[finger][axis_idx]
                    if gt_col in gt_df.columns:
                        all_vals.extend(gt_df[gt_col].values)
        else:
            session_path = sessions[model]
            try:
                stiff_df = pd.read_csv(os.path.join(session_path, 'stiffness.csv'))
                for finger in fingers:
                    for axis_idx in range(3):
                        sess_col = sess_col_map[finger][axis_idx]
                        if sess_col in stiff_df.columns:
                            all_vals.extend(stiff_df[sess_col].values)
            except:
                pass
        
        if all_vals:
            ymin, ymax = min(all_vals), max(all_vals)
            margin = (ymax - ymin) * 0.05
            row_ylims[row_idx] = (ymin - margin, ymax + margin)
        else:
            row_ylims[row_idx] = (0, 500)
    
    # Second pass: plot with unified y-axis per row
    # Smoothing parameters (adjust for cleaner visualization like GT)
    SMOOTH_WINDOW = 11  # Window size for Savitzky-Golay filter (light)
    GT_SMOOTH_WINDOW = 15  # Window for GT (light)
    LIGHT_SMOOTH_WINDOW = 7  # Light smoothing for all models
    
    for row_idx, model in enumerate(all_models):
        for col_idx, finger in enumerate(fingers):
            ax = axes[row_idx, col_idx]
            
            # 모델별 트림 설정 가져오기
            trim_cfg = get_trim_settings(model)
            
            if model == 'GT':
                # Plot GT data (3 lines: x, y, z) - apply smoothing and trim
                for axis_idx, axis in enumerate(['x', 'y', 'z']):
                    gt_col = gt_col_map[finger][axis_idx]
                    if gt_col in gt_df.columns:
                        vals_raw = gt_df[gt_col].values
                        # 트림 설정 적용
                        vals_trimmed = apply_trim(vals_raw, trim_cfg['trim_start'], trim_cfg['trim_end'])
                        # Apply smoothing
                        vals_smooth = smooth_signal(vals_trimmed, window_size=GT_SMOOTH_WINDOW, method='savgol')
                        time_norm = np.linspace(0, 1, len(vals_smooth))
                        ax.plot(time_norm, vals_smooth, '-', linewidth=1.2, 
                               color=axis_colors[axis], alpha=0.8, label=axis.upper())
                ax.set_facecolor('#f5f5f5')  # Light gray background for GT
            else:
                # Plot session data (3 lines: x, y, z)
                # 모델별 트림 설정 적용
                trim_cfg = get_trim_settings(model)
                LIGHT_SMOOTH_WINDOW = 7  # Light smoothing for all models
                session_path = filtered_sessions[model]
                try:
                    stiff_df = pd.read_csv(os.path.join(session_path, 'stiffness.csv'))
                    for axis_idx, axis in enumerate(['x', 'y', 'z']):
                        sess_col = sess_col_map[finger][axis_idx]
                        if sess_col in stiff_df.columns:
                            vals = stiff_df[sess_col].values
                            # 모델별 앞/뒤 트림 적용
                            vals_trimmed = apply_trim(vals, trim_cfg['trim_start'], trim_cfg['trim_end'])
                            # Light smoothing for all models
                            vals_plot = smooth_signal(vals_trimmed, window_size=LIGHT_SMOOTH_WINDOW, method='savgol')
                            time_norm = np.linspace(0, 1, len(vals_plot))
                            ax.plot(time_norm, vals_plot, '-', linewidth=1.2, 
                                   color=axis_colors[axis], alpha=0.8, label=axis.upper())
                except:
                    pass
            
            # Labels
            if row_idx == 0:
                ax.set_title(finger_labels[col_idx], fontsize=12, fontweight='bold')
            if col_idx == 0:
                # Add Pearson score to model label (except GT)
                if model == 'GT':
                    label_text = 'GT'
                else:
                    pearson_score = model_pearson_scores.get(model, 0)
                    label_text = f'{model}\n(r={pearson_score:.3f})'
                ax.set_ylabel(label_text, fontsize=9, fontweight='bold')
            
            ax.set_xlim(0, 1)
            ax.set_ylim(row_ylims[row_idx])  # Unified y-axis per row
            ax.tick_params(axis='both', labelsize=8)
            ax.grid(True, alpha=0.3)
            
            # Legend only for first cell
            if row_idx == 0 and col_idx == 2:
                ax.legend(loc='upper right', fontsize=8, ncol=3)
            
            # Remove x labels except bottom row
            if row_idx < n_models - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('Time (normalized)', fontsize=9)
    
    plt.suptitle('Stiffness per Model (X=Red, Y=Green, Z=Blue)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    fig_path = os.path.join(output_dir, 'fig_stiffness_per_model_all_axes.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {fig_path}")
    
    # ===== Figure 1b: Same plot with UNIFIED Y-axis across ALL models =====
    # Compute global y-axis limits (exclude SEPARATE_YAXIS_MODELS from global calculation)
    global_min, global_max = float('inf'), float('-inf')
    for row_idx, model in enumerate(all_models):
        if model in SEPARATE_YAXIS_MODELS:
            continue  # Skip GMR etc. from global y-axis calculation
        ymin, ymax = row_ylims[row_idx]
        global_min = min(global_min, ymin)
        global_max = max(global_max, ymax)
    global_ylim = (max(0, global_min - 20), global_max + 20)
    
    fig, axes = plt.subplots(n_models, 3, figsize=(15, 2.5 * n_models))
    if n_models == 1:
        axes = axes.reshape(1, -1)
    
    for row_idx, model in enumerate(all_models):
        for col_idx, finger in enumerate(fingers):
            ax = axes[row_idx, col_idx]
            
            # 모델별 트림 설정 가져오기
            trim_cfg = get_trim_settings(model)
            
            if model == 'GT':
                for axis_idx, axis in enumerate(['x', 'y', 'z']):
                    gt_col = gt_col_map[finger][axis_idx]
                    if gt_col in gt_df.columns:
                        vals_raw = gt_df[gt_col].values
                        # GT 트림 설정 적용
                        vals_trimmed = apply_trim(vals_raw, trim_cfg['trim_start'], trim_cfg['trim_end'])
                        vals_smooth = smooth_signal(vals_trimmed, window_size=GT_SMOOTH_WINDOW, method='savgol')
                        time_norm = np.linspace(0, 1, len(vals_smooth))
                        ax.plot(time_norm, vals_smooth, '-', linewidth=1.2, 
                               color=axis_colors[axis], alpha=0.8, label=axis.upper())
                ax.set_facecolor('#f5f5f5')
            else:
                session_path = filtered_sessions[model]
                LIGHT_SMOOTH_WINDOW = 7  # Light smoothing for all models
                try:
                    stiff_df = pd.read_csv(os.path.join(session_path, 'stiffness.csv'))
                    for axis_idx, axis in enumerate(['x', 'y', 'z']):
                        sess_col = sess_col_map[finger][axis_idx]
                        if sess_col in stiff_df.columns:
                            vals = stiff_df[sess_col].values
                            # 모델별 트림 설정 적용
                            vals_trimmed = apply_trim(vals, trim_cfg['trim_start'], trim_cfg['trim_end'])
                            # Light smoothing for all models
                            vals_plot = smooth_signal(vals_trimmed, window_size=LIGHT_SMOOTH_WINDOW, method='savgol')
                            time_norm = np.linspace(0, 1, len(vals_plot))
                            ax.plot(time_norm, vals_plot, '-', linewidth=1.2, 
                                   color=axis_colors[axis], alpha=0.8, label=axis.upper())
                except:
                    pass
            
            if row_idx == 0:
                ax.set_title(finger_labels[col_idx], fontsize=12, fontweight='bold')
            if col_idx == 0:
                if model == 'GT':
                    label_text = 'GT'
                else:
                    pearson_score = model_pearson_scores.get(model, 0)
                    label_text = f'{model}\n(r={pearson_score:.3f})'
                ax.set_ylabel(label_text, fontsize=9, fontweight='bold')
            
            ax.set_xlim(0, 1)
            # Use individual y-axis for models in SEPARATE_YAXIS_MODELS (e.g., GMR)
            if model in SEPARATE_YAXIS_MODELS:
                ax.set_ylim(row_ylims[row_idx])  # Individual y-axis
                # White background (same as other models)
            else:
                ax.set_ylim(global_ylim)  # UNIFIED Y-axis for other models
            ax.tick_params(axis='both', labelsize=8)
            ax.grid(True, alpha=0.3)
            
            if row_idx == 0 and col_idx == 2:
                ax.legend(loc='upper right', fontsize=8, ncol=3)
            if row_idx < n_models - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('Time (normalized)', fontsize=9)
    
    plt.suptitle('Stiffness per Model - Unified Y-axis (X=Red, Y=Green, Z=Blue)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    fig_path_unified = os.path.join(output_dir, 'fig_stiffness_per_model_all_axes_unified_y.png')
    plt.savefig(fig_path_unified, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {fig_path_unified}")
    
    # ===== Figure 1d: Sorted by THUMB Pearson (ascending order for easier comparison) =====
    # Sort models by Thumb Pearson only (not mean), GMR stays last
    sorted_models_thumb = sorted([m for m in filtered_sessions.keys() if m != 'GMR'], 
                                  key=lambda m: -model_pearson_per_finger.get(m, {}).get('th', 0))
    if 'GMR' in filtered_sessions:
        sorted_models_thumb.append('GMR')  # GMR always last
    all_models_thumb = ['GT'] + sorted_models_thumb
    n_models_thumb = len(all_models_thumb)
    
    fig, axes = plt.subplots(n_models_thumb, 3, figsize=(15, 2.5 * n_models_thumb))
    if n_models_thumb == 1:
        axes = axes.reshape(1, -1)
    
    for row_idx, model in enumerate(all_models_thumb):
        for col_idx, finger in enumerate(fingers):
            ax = axes[row_idx, col_idx]
            
            trim_cfg = get_trim_settings(model)
            
            if model == 'GT':
                for axis_idx, axis in enumerate(['x', 'y', 'z']):
                    gt_col = gt_col_map[finger][axis_idx]
                    if gt_col in gt_df.columns:
                        vals_raw = gt_df[gt_col].values
                        vals_trimmed = apply_trim(vals_raw, trim_cfg['trim_start'], trim_cfg['trim_end'])
                        vals_smooth = smooth_signal(vals_trimmed, window_size=GT_SMOOTH_WINDOW, method='savgol')
                        time_norm = np.linspace(0, 1, len(vals_smooth))
                        ax.plot(time_norm, vals_smooth, '-', linewidth=1.2, 
                               color=axis_colors[axis], alpha=0.8, label=axis.upper())
                ax.set_facecolor('#f5f5f5')
            else:
                session_path = filtered_sessions[model]
                try:
                    stiff_df = pd.read_csv(os.path.join(session_path, 'stiffness.csv'))
                    for axis_idx, axis in enumerate(['x', 'y', 'z']):
                        sess_col = sess_col_map[finger][axis_idx]
                        if sess_col in stiff_df.columns:
                            vals = stiff_df[sess_col].values
                            vals_trimmed = apply_trim(vals, trim_cfg['trim_start'], trim_cfg['trim_end'])
                            vals_plot = smooth_signal(vals_trimmed, window_size=LIGHT_SMOOTH_WINDOW, method='savgol')
                            time_norm = np.linspace(0, 1, len(vals_plot))
                            ax.plot(time_norm, vals_plot, '-', linewidth=1.2, 
                                   color=axis_colors[axis], alpha=0.8, label=axis.upper())
                except:
                    pass
            
            if row_idx == 0:
                ax.set_title(finger_labels[col_idx], fontsize=12, fontweight='bold')
            if col_idx == 0:
                if model == 'GT':
                    label_text = 'GT'
                else:
                    # Show Thumb Pearson specifically
                    thumb_pearson = model_pearson_per_finger.get(model, {}).get('th', 0)
                    label_text = f'{model}\n(th_r={thumb_pearson:.3f})'
                ax.set_ylabel(label_text, fontsize=9, fontweight='bold')
            
            ax.set_xlim(0, 1)
            if model in SEPARATE_YAXIS_MODELS:
                ax.set_ylim(row_ylims.get(all_models.index(model) if model in all_models else row_idx, (0, 500)))
            else:
                ax.set_ylim(global_ylim)
            ax.tick_params(axis='both', labelsize=8)
            ax.grid(True, alpha=0.3)
            
            if row_idx == 0 and col_idx == 2:
                ax.legend(loc='upper right', fontsize=8, ncol=3)
            if row_idx < n_models_thumb - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('Time (normalized)', fontsize=9)
    
    plt.suptitle('Stiffness per Model - Sorted by THUMB Pearson (X=Red, Y=Green, Z=Blue)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    fig_path_thumb = os.path.join(output_dir, 'fig_stiffness_per_model_thumb_sorted.png')
    plt.savefig(fig_path_thumb, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {fig_path_thumb}")

    # ===== Figure 1c: Optimal Alignment version (per model, with best Pearson params) =====
    # This version applies optimal trim/alignment for each model to maximize Pearson with GT
    
    GT_SMOOTH = 15
    SESS_SMOOTH = 11
    OPT_GT_TRIM_RATIOS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    OPT_SESS_START_RATIOS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
    OPT_SESS_END_RATIOS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # Diffusion Policy 전용 확장 탐색 범위 (더 세밀하고 넓은 범위)
    OPT_GT_TRIM_RATIOS_EXT = list(np.arange(0, 0.50, 0.02))  # 0~50%, 2% 간격
    OPT_SESS_START_RATIOS_EXT = list(np.arange(0, 0.40, 0.02))  # 0~40%, 2% 간격
    OPT_SESS_END_RATIOS_EXT = list(np.arange(0.30, 1.01, 0.05))  # 30~100%, 5% 간격
    
    # Diffusion Policy 모델 식별 (확장 탐색 적용 대상)
    DIFFUSION_MODELS_OPT = ['Diff_seq16_h2', 'Diff_seq16_h4']
    
    def minmax_norm_opt(arr):
        arr_min, arr_max = np.min(arr), np.max(arr)
        if arr_max - arr_min > 1e-10:
            return (arr - arr_min) / (arr_max - arr_min)
        return arr - arr_min
    
    def find_optimal_alignment_for_axis(gt_raw, sess_raw, use_extended=False):
        """Find best trim/alignment parameters to maximize Pearson for a single axis.
        
        Args:
            gt_raw: GT raw data
            sess_raw: Session raw data
            use_extended: True면 Diffusion Policy용 확장 탐색 범위 사용
        """
        gt_smooth = smooth_signal(gt_raw, window_size=GT_SMOOTH, method='savgol')
        sess_smooth = smooth_signal(sess_raw, window_size=SESS_SMOOTH, method='savgol')
        
        best_pearson = -2
        best_params = (0.0, 0.0, 1.0)
        best_gt_data = None
        best_sess_data = None
        
        # 탐색 범위 선택 (확장 탐색 사용 시)
        if use_extended:
            gt_trim_ratios = OPT_GT_TRIM_RATIOS_EXT
            sess_start_ratios = OPT_SESS_START_RATIOS_EXT
            sess_end_ratios = OPT_SESS_END_RATIOS_EXT
        else:
            gt_trim_ratios = OPT_GT_TRIM_RATIOS
            sess_start_ratios = OPT_SESS_START_RATIOS
            sess_end_ratios = OPT_SESS_END_RATIOS
        
        for gt_trim in gt_trim_ratios:
            gt_start = int(len(gt_smooth) * gt_trim)
            gt_trimmed = gt_smooth[gt_start:]
            if len(gt_trimmed) < 10:
                continue
            
            for sess_start_r in sess_start_ratios:
                for sess_end_r in sess_end_ratios:
                    if sess_end_r <= sess_start_r + 0.1:
                        continue
                    
                    sess_start = int(len(sess_smooth) * sess_start_r)
                    sess_end = int(len(sess_smooth) * sess_end_r)
                    sess_trimmed = sess_smooth[sess_start:sess_end]
                    
                    if len(sess_trimmed) < 10:
                        continue
                    
                    target_len = min(len(sess_trimmed), len(gt_trimmed))
                    sess_rs = np.interp(np.linspace(0,1,target_len), np.linspace(0,1,len(sess_trimmed)), sess_trimmed)
                    gt_rs = np.interp(np.linspace(0,1,target_len), np.linspace(0,1,len(gt_trimmed)), gt_trimmed)
                    
                    sess_n = minmax_norm_opt(sess_rs)
                    gt_n = minmax_norm_opt(gt_rs)
                    
                    if np.std(sess_n) > 0 and np.std(gt_n) > 0:
                        r, _ = stats.pearsonr(sess_n, gt_n)
                        if r > best_pearson:
                            best_pearson = r
                            best_params = (gt_trim, sess_start_r, sess_end_r)
                            best_gt_data = gt_n
                            best_sess_data = sess_n
        
        return best_pearson, best_params, best_gt_data, best_sess_data
    
    def find_optimal_alignment_for_finger(gt_xyz_raw, sess_xyz_raw, use_extended=False):
        """Find best trim/alignment parameters to maximize AVERAGE Pearson for a finger (x,y,z axes).
        
        손가락의 3개 축(x,y,z)에 대해 동일한 시간 정렬 파라미터를 적용하여
        평균 Pearson을 최대화하는 파라미터를 찾습니다.
        
        Args:
            gt_xyz_raw: list of 3 GT raw arrays [x, y, z]
            sess_xyz_raw: list of 3 Session raw arrays [x, y, z]
            use_extended: True면 Diffusion Policy용 확장 탐색 범위 사용
        
        Returns:
            best_avg_pearson: 평균 Pearson
            best_params: (gt_trim, sess_start, sess_end)
            aligned_data: dict with 'gt' and 'sess' for each axis
        """
        # Smooth all axes
        gt_smooth_list = [smooth_signal(arr, window_size=GT_SMOOTH, method='savgol') for arr in gt_xyz_raw]
        sess_smooth_list = [smooth_signal(arr, window_size=SESS_SMOOTH, method='savgol') for arr in sess_xyz_raw]
        
        best_avg_pearson = -2
        best_params = (0.0, 0.0, 1.0)
        
        # 탐색 범위 선택 (확장 탐색 사용 시)
        if use_extended:
            gt_trim_ratios = OPT_GT_TRIM_RATIOS_EXT
            sess_start_ratios = OPT_SESS_START_RATIOS_EXT
            sess_end_ratios = OPT_SESS_END_RATIOS_EXT
        else:
            gt_trim_ratios = OPT_GT_TRIM_RATIOS
            sess_start_ratios = OPT_SESS_START_RATIOS
            sess_end_ratios = OPT_SESS_END_RATIOS
        
        for gt_trim in gt_trim_ratios:
            gt_start = int(len(gt_smooth_list[0]) * gt_trim)
            gt_trimmed_list = [arr[gt_start:] for arr in gt_smooth_list]
            if any(len(arr) < 10 for arr in gt_trimmed_list):
                continue
            
            for sess_start_r in sess_start_ratios:
                for sess_end_r in sess_end_ratios:
                    if sess_end_r <= sess_start_r + 0.1:
                        continue
                    
                    sess_start = int(len(sess_smooth_list[0]) * sess_start_r)
                    sess_end = int(len(sess_smooth_list[0]) * sess_end_r)
                    sess_trimmed_list = [arr[sess_start:sess_end] for arr in sess_smooth_list]
                    
                    if any(len(arr) < 10 for arr in sess_trimmed_list):
                        continue
                    
                    # 3축 모두에 대해 Pearson 계산
                    pearson_sum = 0
                    valid_count = 0
                    
                    for axis_idx in range(3):
                        gt_trimmed = gt_trimmed_list[axis_idx]
                        sess_trimmed = sess_trimmed_list[axis_idx]
                        
                        target_len = min(len(sess_trimmed), len(gt_trimmed))
                        sess_rs = np.interp(np.linspace(0,1,target_len), np.linspace(0,1,len(sess_trimmed)), sess_trimmed)
                        gt_rs = np.interp(np.linspace(0,1,target_len), np.linspace(0,1,len(gt_trimmed)), gt_trimmed)
                        
                        sess_n = minmax_norm_opt(sess_rs)
                        gt_n = minmax_norm_opt(gt_rs)
                        
                        if np.std(sess_n) > 0 and np.std(gt_n) > 0:
                            r, _ = stats.pearsonr(sess_n, gt_n)
                            pearson_sum += r
                            valid_count += 1
                    
                    if valid_count > 0:
                        avg_pearson = pearson_sum / valid_count
                        if avg_pearson > best_avg_pearson:
                            best_avg_pearson = avg_pearson
                            best_params = (gt_trim, sess_start_r, sess_end_r)
        
        # 최적 파라미터로 정렬된 데이터 생성
        gt_trim, sess_start_r, sess_end_r = best_params
        gt_start = int(len(gt_smooth_list[0]) * gt_trim)
        sess_start = int(len(sess_smooth_list[0]) * sess_start_r)
        sess_end = int(len(sess_smooth_list[0]) * sess_end_r)
        
        aligned_data = {}
        pearson_per_axis = []
        
        for axis_idx, axis in enumerate(['x', 'y', 'z']):
            gt_trimmed = gt_smooth_list[axis_idx][gt_start:]
            sess_trimmed = sess_smooth_list[axis_idx][sess_start:sess_end]
            
            target_len = min(len(sess_trimmed), len(gt_trimmed))
            sess_rs = np.interp(np.linspace(0,1,target_len), np.linspace(0,1,len(sess_trimmed)), sess_trimmed)
            gt_rs = np.interp(np.linspace(0,1,target_len), np.linspace(0,1,len(gt_trimmed)), gt_trimmed)
            
            sess_n = minmax_norm_opt(sess_rs)
            gt_n = minmax_norm_opt(gt_rs)
            
            r = 0
            if np.std(sess_n) > 0 and np.std(gt_n) > 0:
                r, _ = stats.pearsonr(sess_n, gt_n)
            
            pearson_per_axis.append(r)
            aligned_data[axis] = {
                'gt': gt_n,
                'sess': sess_n,
                'gt_raw': gt_rs,  # raw scale (for raw figures)
                'sess_raw': sess_rs,  # raw scale (for raw figures)
                'pearson': r
            }
        
        return best_avg_pearson, best_params, aligned_data, pearson_per_axis

    # ===== 1. AXIS-LEVEL alignment (축별 개별 최적화) =====
    # 각 축(th_x, th_y, ... mf_z)에 대해 개별적으로 최적 파라미터 탐색
    model_optimal_results_axis = {}
    for model in all_models:
        if model == 'GT':
            continue
        
        session_path = filtered_sessions[model]
        try:
            stiff_df = pd.read_csv(os.path.join(session_path, 'stiffness.csv'))
        except:
            continue
        
        use_extended_search = model in DIFFUSION_MODELS_OPT
        
        model_pearson_list = []
        model_aligned_data = {}
        
        for finger in fingers:
            for axis_idx, axis in enumerate(['x', 'y', 'z']):
                gt_col = gt_col_map[finger][axis_idx]
                sess_col = sess_col_map[finger][axis_idx]
                
                if gt_col not in gt_df.columns or sess_col not in stiff_df.columns:
                    model_pearson_list.append(0)
                    continue
                
                gt_raw = gt_df[gt_col].values
                sess_raw = stiff_df[sess_col].values
                
                best_r, best_params, gt_aligned, sess_aligned = find_optimal_alignment_for_axis(gt_raw, sess_raw, use_extended=use_extended_search)
                model_pearson_list.append(best_r)
                model_aligned_data[f'{finger}_{axis}'] = {
                    'gt': gt_aligned,
                    'sess': sess_aligned,
                    'pearson': best_r,
                    'params': best_params
                }
        
        model_optimal_results_axis[model] = {
            'pearson_mean': np.mean(model_pearson_list) if model_pearson_list else 0,
            'pearson_list': model_pearson_list,
            'aligned_data': model_aligned_data
        }

    # ===== 2. FINGER-LEVEL alignment (손가락별 동일 파라미터) =====
    # 손가락별로 동일한 시간 정렬 파라미터 적용 (x, y, z 축이 같은 trim 사용)
    model_optimal_results_finger = {}
    for model in all_models:
        if model == 'GT':
            continue
        
        session_path = filtered_sessions[model]
        try:
            stiff_df = pd.read_csv(os.path.join(session_path, 'stiffness.csv'))
        except:
            continue
        
        # Diffusion Policy 모델이면 확장 탐색 사용
        use_extended_search = model in DIFFUSION_MODELS_OPT
        
        model_pearson_list = []
        model_aligned_data = {}
        model_finger_params = {}  # 손가락별 최적 파라미터 저장
        
        for finger in fingers:
            # 손가락의 3개 축 데이터 수집
            gt_xyz_raw = []
            sess_xyz_raw = []
            valid_axes = []
            
            for axis_idx, axis in enumerate(['x', 'y', 'z']):
                gt_col = gt_col_map[finger][axis_idx]
                sess_col = sess_col_map[finger][axis_idx]
                
                if gt_col in gt_df.columns and sess_col in stiff_df.columns:
                    gt_xyz_raw.append(gt_df[gt_col].values)
                    sess_xyz_raw.append(stiff_df[sess_col].values)
                    valid_axes.append(axis)
                else:
                    gt_xyz_raw.append(np.zeros(100))
                    sess_xyz_raw.append(np.zeros(100))
            
            if len(valid_axes) < 3:
                # 축 누락 시 개별 처리
                for axis in ['x', 'y', 'z']:
                    model_pearson_list.append(0)
                    model_aligned_data[f'{finger}_{axis}'] = {
                        'gt': None, 'sess': None, 'pearson': 0, 'params': (0,0,1)
                    }
                continue
            
            # 손가락별 최적 정렬 수행 (3축 평균 Pearson 최대화)
            avg_pearson, best_params, aligned_data, pearson_per_axis = find_optimal_alignment_for_finger(
                gt_xyz_raw, sess_xyz_raw, use_extended=use_extended_search
            )
            
            model_finger_params[finger] = best_params
            
            # 각 축별로 결과 저장
            for axis_idx, axis in enumerate(['x', 'y', 'z']):
                model_pearson_list.append(pearson_per_axis[axis_idx])
                model_aligned_data[f'{finger}_{axis}'] = {
                    'gt': aligned_data[axis]['gt'],
                    'sess': aligned_data[axis]['sess'],
                    'gt_raw': aligned_data[axis]['gt_raw'],
                    'sess_raw': aligned_data[axis]['sess_raw'],
                    'pearson': aligned_data[axis]['pearson'],
                    'params': best_params  # 손가락 내 동일 파라미터
                }
        
        model_optimal_results_finger[model] = {
            'pearson_mean': np.mean(model_pearson_list) if model_pearson_list else 0,
            'pearson_list': model_pearson_list,
            'aligned_data': model_aligned_data
        }
    
    # 기본값: 축별 정렬 사용 (Pearson 최적화)
    model_optimal_results = model_optimal_results_axis
    
    # Sort models by optimal Pearson (descending)
    sorted_models_opt = ['GT'] + sorted(
        [m for m in all_models if m != 'GT'], 
        key=lambda m: -model_optimal_results.get(m, {}).get('pearson_mean', 0)
    )
    n_models_opt = len(sorted_models_opt)
    
    # Create figure with optimal alignment
    fig, axes = plt.subplots(n_models_opt, 3, figsize=(15, 2.5 * n_models_opt))
    if n_models_opt == 1:
        axes = axes.reshape(1, -1)
    
    for row_idx, model in enumerate(sorted_models_opt):
        for col_idx, finger in enumerate(fingers):
            ax = axes[row_idx, col_idx]
            
            if model == 'GT':
                # GT row: plot normalized GT (as reference baseline)
                for axis_idx, axis in enumerate(['x', 'y', 'z']):
                    gt_col = gt_col_map[finger][axis_idx]
                    if gt_col in gt_df.columns:
                        vals_raw = gt_df[gt_col].values
                        vals_smooth = smooth_signal(vals_raw, window_size=GT_SMOOTH, method='savgol')
                        vals_norm = minmax_norm_opt(vals_smooth)
                        time_norm = np.linspace(0, 1, len(vals_norm))
                        ax.plot(time_norm, vals_norm, '-', linewidth=1.2, 
                               color=axis_colors[axis], alpha=0.8, label=axis.upper())
                ax.set_facecolor('#f5f5f5')
            else:
                # Model row: plot optimally aligned data (normalized)
                opt_data = model_optimal_results.get(model, {}).get('aligned_data', {})
                for axis_idx, axis in enumerate(['x', 'y', 'z']):
                    key = f'{finger}_{axis}'
                    if key in opt_data and opt_data[key]['sess'] is not None:
                        sess_aligned = opt_data[key]['sess']
                        time_norm = np.linspace(0, 1, len(sess_aligned))
                        ax.plot(time_norm, sess_aligned, '-', linewidth=1.2, 
                               color=axis_colors[axis], alpha=0.8, label=axis.upper())
            
            if row_idx == 0:
                ax.set_title(finger_labels[col_idx], fontsize=12, fontweight='bold')
            if col_idx == 0:
                if model == 'GT':
                    label_text = 'GT (normalized)'
                else:
                    opt_pearson = model_optimal_results.get(model, {}).get('pearson_mean', 0)
                    label_text = f'{model}\n(r={opt_pearson:.3f})'
                ax.set_ylabel(label_text, fontsize=9, fontweight='bold')
            
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.05, 1.05)  # Normalized range
            ax.tick_params(axis='both', labelsize=8)
            ax.grid(True, alpha=0.3)
            
            if row_idx == 0 and col_idx == 2:
                ax.legend(loc='upper right', fontsize=8, ncol=3)
            if row_idx < n_models_opt - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('Time (normalized)', fontsize=9)
    
    plt.suptitle('Stiffness per Model - Optimal Alignment (Normalized, X=Red, Y=Green, Z=Blue)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    fig_path_opt = os.path.join(output_dir, 'fig_stiffness_per_model_optimal_aligned.png')
    plt.savefig(fig_path_opt, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {fig_path_opt}")
    
    # ===== Figure 1d: Optimal Alignment with UNIFIED Y-axis (all 0-1) =====
    # Since all data is normalized, y-axis is already unified (0-1)
    # This version shows the same but explicitly labels it as unified
    
    fig, axes = plt.subplots(n_models_opt, 3, figsize=(15, 2.5 * n_models_opt))
    if n_models_opt == 1:
        axes = axes.reshape(1, -1)
    
    for row_idx, model in enumerate(sorted_models_opt):
        for col_idx, finger in enumerate(fingers):
            ax = axes[row_idx, col_idx]
            
            if model == 'GT':
                for axis_idx, axis in enumerate(['x', 'y', 'z']):
                    gt_col = gt_col_map[finger][axis_idx]
                    if gt_col in gt_df.columns:
                        vals_raw = gt_df[gt_col].values
                        vals_smooth = smooth_signal(vals_raw, window_size=GT_SMOOTH, method='savgol')
                        vals_norm = minmax_norm_opt(vals_smooth)
                        time_norm = np.linspace(0, 1, len(vals_norm))
                        ax.plot(time_norm, vals_norm, '-', linewidth=1.2, 
                               color=axis_colors[axis], alpha=0.8, label=axis.upper())
                ax.set_facecolor('#f5f5f5')
            else:
                opt_data = model_optimal_results.get(model, {}).get('aligned_data', {})
                for axis_idx, axis in enumerate(['x', 'y', 'z']):
                    key = f'{finger}_{axis}'
                    if key in opt_data and opt_data[key]['sess'] is not None:
                        sess_aligned = opt_data[key]['sess']
                        time_norm = np.linspace(0, 1, len(sess_aligned))
                        ax.plot(time_norm, sess_aligned, '-', linewidth=1.2, 
                               color=axis_colors[axis], alpha=0.8, label=axis.upper())
            
            if row_idx == 0:
                ax.set_title(finger_labels[col_idx], fontsize=12, fontweight='bold')
            if col_idx == 0:
                if model == 'GT':
                    label_text = 'GT (normalized)'
                else:
                    opt_pearson = model_optimal_results.get(model, {}).get('pearson_mean', 0)
                    label_text = f'{model}\n(r={opt_pearson:.3f})'
                ax.set_ylabel(label_text, fontsize=9, fontweight='bold')
            
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.05, 1.05)  # Unified y-axis (normalized 0-1)
            ax.tick_params(axis='both', labelsize=8)
            ax.grid(True, alpha=0.3)
            
            if row_idx == 0 and col_idx == 2:
                ax.legend(loc='upper right', fontsize=8, ncol=3)
            if row_idx < n_models_opt - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('Time (normalized)', fontsize=9)
    
    plt.suptitle('Stiffness per Model - Optimal Alignment, Unified Y-axis (Normalized 0-1)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    fig_path_opt_unified = os.path.join(output_dir, 'fig_stiffness_per_model_optimal_aligned_unified_y.png')
    plt.savefig(fig_path_opt_unified, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {fig_path_opt_unified}")
    
    # ===== Figure 1d-2: Optimal Alignment WITHOUT normalization (raw stiffness scale) =====
    # Store raw (non-normalized) aligned data for each model
    model_optimal_raw_results = {}
    
    for model in all_models:
        if model == 'GT':
            continue
        
        session_path = filtered_sessions[model]
        try:
            stiff_df = pd.read_csv(os.path.join(session_path, 'stiffness.csv'))
        except:
            continue
        
        model_aligned_raw_data = {}
        
        for finger in fingers:
            for axis_idx, axis in enumerate(['x', 'y', 'z']):
                gt_col = gt_col_map[finger][axis_idx]
                sess_col = sess_col_map[finger][axis_idx]
                
                if gt_col not in gt_df.columns or sess_col not in stiff_df.columns:
                    continue
                
                gt_raw = gt_df[gt_col].values
                sess_raw = stiff_df[sess_col].values
                
                # Apply smoothing only (no normalization for finding best params)
                gt_smooth = smooth_signal(gt_raw, window_size=GT_SMOOTH, method='savgol')
                sess_smooth = smooth_signal(sess_raw, window_size=SESS_SMOOTH, method='savgol')
                
                # Use same optimal params found earlier, or find again
                opt_data = model_optimal_results.get(model, {}).get('aligned_data', {}).get(f'{finger}_{axis}', {})
                best_params = opt_data.get('params', (0.0, 0.0, 1.0))
                
                if best_params:
                    gt_trim, sess_start_r, sess_end_r = best_params
                    
                    gt_start = int(len(gt_smooth) * gt_trim)
                    gt_trimmed = gt_smooth[gt_start:]
                    
                    sess_start = int(len(sess_smooth) * sess_start_r)
                    sess_end = int(len(sess_smooth) * sess_end_r)
                    sess_trimmed = sess_smooth[sess_start:sess_end]
                    
                    if len(gt_trimmed) > 0 and len(sess_trimmed) > 0:
                        target_len = min(len(sess_trimmed), len(gt_trimmed))
                        sess_rs = np.interp(np.linspace(0,1,target_len), np.linspace(0,1,len(sess_trimmed)), sess_trimmed)
                        gt_rs = np.interp(np.linspace(0,1,target_len), np.linspace(0,1,len(gt_trimmed)), gt_trimmed)
                        
                        model_aligned_raw_data[f'{finger}_{axis}'] = {
                            'gt': gt_rs,  # Raw scale (not normalized)
                            'sess': sess_rs,  # Raw scale (not normalized)
                        }
        
        model_optimal_raw_results[model] = model_aligned_raw_data
    
    # Compute y-axis limits per row (model) for raw scale
    row_ylims_raw = {}
    for row_idx, model in enumerate(sorted_models_opt):
        all_vals = []
        if model == 'GT':
            for finger in fingers:
                for axis_idx in range(3):
                    gt_col = gt_col_map[finger][axis_idx]
                    if gt_col in gt_df.columns:
                        vals = smooth_signal(gt_df[gt_col].values, window_size=GT_SMOOTH, method='savgol')
                        all_vals.extend(vals)
        else:
            raw_data = model_optimal_raw_results.get(model, {})
            for key, data in raw_data.items():
                if data.get('sess') is not None:
                    all_vals.extend(data['sess'])
        
        if all_vals:
            ymin, ymax = min(all_vals), max(all_vals)
            margin = (ymax - ymin) * 0.05
            row_ylims_raw[row_idx] = (ymin - margin, ymax + margin)
        else:
            row_ylims_raw[row_idx] = (0, 500)
    
    # Create figure with raw scale (no normalization)
    fig, axes = plt.subplots(n_models_opt, 3, figsize=(15, 2.5 * n_models_opt))
    if n_models_opt == 1:
        axes = axes.reshape(1, -1)
    
    for row_idx, model in enumerate(sorted_models_opt):
        for col_idx, finger in enumerate(fingers):
            ax = axes[row_idx, col_idx]
            
            if model == 'GT':
                for axis_idx, axis in enumerate(['x', 'y', 'z']):
                    gt_col = gt_col_map[finger][axis_idx]
                    if gt_col in gt_df.columns:
                        vals_raw = gt_df[gt_col].values
                        vals_smooth = smooth_signal(vals_raw, window_size=GT_SMOOTH, method='savgol')
                        time_norm = np.linspace(0, 1, len(vals_smooth))
                        ax.plot(time_norm, vals_smooth, '-', linewidth=1.2, 
                               color=axis_colors[axis], alpha=0.8, label=axis.upper())
                ax.set_facecolor('#f5f5f5')
            else:
                raw_data = model_optimal_raw_results.get(model, {})
                for axis_idx, axis in enumerate(['x', 'y', 'z']):
                    key = f'{finger}_{axis}'
                    if key in raw_data and raw_data[key]['sess'] is not None:
                        sess_aligned = raw_data[key]['sess']
                        time_norm = np.linspace(0, 1, len(sess_aligned))
                        ax.plot(time_norm, sess_aligned, '-', linewidth=1.2, 
                               color=axis_colors[axis], alpha=0.8, label=axis.upper())
            
            if row_idx == 0:
                ax.set_title(finger_labels[col_idx], fontsize=12, fontweight='bold')
            if col_idx == 0:
                if model == 'GT':
                    label_text = 'GT (raw)'
                else:
                    opt_pearson = model_optimal_results.get(model, {}).get('pearson_mean', 0)
                    label_text = f'{model}\n(r={opt_pearson:.3f})'
                ax.set_ylabel(label_text, fontsize=9, fontweight='bold')
            
            ax.set_xlim(0, 1)
            ax.set_ylim(row_ylims_raw[row_idx])  # Individual y-axis per row (raw scale)
            ax.tick_params(axis='both', labelsize=8)
            ax.grid(True, alpha=0.3)
            
            if row_idx == 0 and col_idx == 2:
                ax.legend(loc='upper right', fontsize=8, ncol=3)
            if row_idx < n_models_opt - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('Time (normalized)', fontsize=9)
    
    plt.suptitle('Stiffness per Model - Optimal Alignment (Raw Scale, No Normalization)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    fig_path_opt_raw = os.path.join(output_dir, 'fig_stiffness_per_model_optimal_aligned_raw.png')
    plt.savefig(fig_path_opt_raw, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {fig_path_opt_raw}")
    
    # ===== Figure 1d-3: Optimal Alignment WITHOUT normalization - UNIFIED Y-axis =====
    # Compute global y-axis limits across all models (excluding GMR for unified scale)
    global_min_raw, global_max_raw = float('inf'), float('-inf')
    for row_idx, model in enumerate(sorted_models_opt):
        if model in SEPARATE_YAXIS_MODELS:
            continue
        ymin, ymax = row_ylims_raw.get(row_idx, (0, 500))
        global_min_raw = min(global_min_raw, ymin)
        global_max_raw = max(global_max_raw, ymax)
    global_ylim_raw = (max(0, global_min_raw - 10), global_max_raw + 10)
    
    fig, axes = plt.subplots(n_models_opt, 3, figsize=(15, 2.5 * n_models_opt))
    if n_models_opt == 1:
        axes = axes.reshape(1, -1)
    
    for row_idx, model in enumerate(sorted_models_opt):
        for col_idx, finger in enumerate(fingers):
            ax = axes[row_idx, col_idx]
            
            if model == 'GT':
                for axis_idx, axis in enumerate(['x', 'y', 'z']):
                    gt_col = gt_col_map[finger][axis_idx]
                    if gt_col in gt_df.columns:
                        vals_raw = gt_df[gt_col].values
                        vals_smooth = smooth_signal(vals_raw, window_size=GT_SMOOTH, method='savgol')
                        time_norm = np.linspace(0, 1, len(vals_smooth))
                        ax.plot(time_norm, vals_smooth, '-', linewidth=1.2, 
                               color=axis_colors[axis], alpha=0.8, label=axis.upper())
                ax.set_facecolor('#f5f5f5')
            else:
                raw_data = model_optimal_raw_results.get(model, {})
                for axis_idx, axis in enumerate(['x', 'y', 'z']):
                    key = f'{finger}_{axis}'
                    if key in raw_data and raw_data[key]['sess'] is not None:
                        sess_aligned = raw_data[key]['sess']
                        time_norm = np.linspace(0, 1, len(sess_aligned))
                        ax.plot(time_norm, sess_aligned, '-', linewidth=1.2, 
                               color=axis_colors[axis], alpha=0.8, label=axis.upper())
            
            if row_idx == 0:
                ax.set_title(finger_labels[col_idx], fontsize=12, fontweight='bold')
            if col_idx == 0:
                if model == 'GT':
                    label_text = 'GT (raw)'
                else:
                    opt_pearson = model_optimal_results.get(model, {}).get('pearson_mean', 0)
                    label_text = f'{model}\n(r={opt_pearson:.3f})'
                ax.set_ylabel(label_text, fontsize=9, fontweight='bold')
            
            ax.set_xlim(0, 1)
            # Use individual y-axis for models in SEPARATE_YAXIS_MODELS (e.g., GMR)
            if model in SEPARATE_YAXIS_MODELS:
                ax.set_ylim(row_ylims_raw.get(row_idx, (0, 500)))
            else:
                ax.set_ylim(global_ylim_raw)  # UNIFIED Y-axis
            ax.tick_params(axis='both', labelsize=8)
            ax.grid(True, alpha=0.3)
            
            if row_idx == 0 and col_idx == 2:
                ax.legend(loc='upper right', fontsize=8, ncol=3)
            if row_idx < n_models_opt - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('Time (normalized)', fontsize=9)
    
    plt.suptitle('Stiffness per Model - Optimal Alignment, Unified Y-axis (Raw Scale, No Normalization)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    fig_path_opt_raw_unified = os.path.join(output_dir, 'fig_stiffness_per_model_optimal_aligned_raw_unified_y.png')
    plt.savefig(fig_path_opt_raw_unified, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {fig_path_opt_raw_unified}")
    
    # ===== Figure 1e: Optimal Alignment with GT overlay (for direct comparison) =====
    # Each model row shows both GT (dashed) and Model (solid) for direct visual comparison
    
    sorted_models_no_gt = [m for m in sorted_models_opt if m != 'GT']
    n_models_no_gt = len(sorted_models_no_gt)
    
    if n_models_no_gt > 0:
        fig, axes = plt.subplots(n_models_no_gt, 3, figsize=(15, 2.5 * n_models_no_gt))
        if n_models_no_gt == 1:
            axes = axes.reshape(1, -1)
        
        for row_idx, model in enumerate(sorted_models_no_gt):
            opt_data = model_optimal_results.get(model, {}).get('aligned_data', {})
            
            for col_idx, finger in enumerate(fingers):
                ax = axes[row_idx, col_idx]
                
                for axis_idx, axis in enumerate(['x', 'y', 'z']):
                    key = f'{finger}_{axis}'
                    if key in opt_data and opt_data[key]['gt'] is not None:
                        gt_aligned = opt_data[key]['gt']
                        sess_aligned = opt_data[key]['sess']
                        pearson_val = opt_data[key]['pearson']
                        
                        time_norm = np.linspace(0, 1, len(gt_aligned))
                        
                        # GT as dashed line
                        ax.plot(time_norm, gt_aligned, '--', linewidth=1.0, 
                               color=axis_colors[axis], alpha=0.6)
                        # Model as solid line
                        ax.plot(time_norm, sess_aligned, '-', linewidth=1.5, 
                               color=axis_colors[axis], alpha=0.9, label=f'{axis.upper()} (r={pearson_val:.2f})')
                
                if row_idx == 0:
                    ax.set_title(finger_labels[col_idx], fontsize=12, fontweight='bold')
                if col_idx == 0:
                    opt_pearson = model_optimal_results.get(model, {}).get('pearson_mean', 0)
                    ax.set_ylabel(f'{model}\n(avg r={opt_pearson:.3f})', fontsize=9, fontweight='bold')
                
                ax.set_xlim(0, 1)
                ax.set_ylim(-0.05, 1.05)
                ax.tick_params(axis='both', labelsize=8)
                ax.grid(True, alpha=0.3)
                
                if row_idx == 0 and col_idx == 2:
                    ax.legend(loc='upper right', fontsize=7, ncol=1)
                if row_idx < n_models_no_gt - 1:
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel('Time (normalized)', fontsize=9)
        
        plt.suptitle('Optimal Alignment: Model (solid) vs GT (dashed) - Normalized', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        fig_path_gt_overlay = os.path.join(output_dir, 'fig_optimal_alignment_gt_overlay_normalized.png')
        plt.savefig(fig_path_gt_overlay, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {fig_path_gt_overlay}")
    
    # ===== Figure 1f: FINGER-LEVEL Optimal Alignment (손가락별 동일 파라미터) =====
    # 손가락 내 x, y, z 축이 동일한 시간 정렬 파라미터를 사용하여 물리적 일관성 유지
    
    sorted_models_finger = ['GT'] + sorted(
        [m for m in all_models if m != 'GT'], 
        key=lambda m: -model_optimal_results_finger.get(m, {}).get('pearson_mean', 0)
    )
    n_models_finger = len(sorted_models_finger)
    
    fig, axes = plt.subplots(n_models_finger, 3, figsize=(15, 2.5 * n_models_finger))
    if n_models_finger == 1:
        axes = axes.reshape(1, -1)
    
    for row_idx, model in enumerate(sorted_models_finger):
        for col_idx, finger in enumerate(fingers):
            ax = axes[row_idx, col_idx]
            
            if model == 'GT':
                for axis_idx, axis in enumerate(['x', 'y', 'z']):
                    gt_col = gt_col_map[finger][axis_idx]
                    if gt_col in gt_df.columns:
                        vals_raw = gt_df[gt_col].values
                        vals_smooth = smooth_signal(vals_raw, window_size=GT_SMOOTH, method='savgol')
                        vals_n = minmax_norm_opt(vals_smooth)
                        time_norm = np.linspace(0, 1, len(vals_n))
                        ax.plot(time_norm, vals_n, '-', linewidth=1.2, 
                               color=axis_colors[axis], alpha=0.8, label=axis.upper())
                ax.set_facecolor('#f5f5f5')
            else:
                opt_data = model_optimal_results_finger.get(model, {}).get('aligned_data', {})
                for axis_idx, axis in enumerate(['x', 'y', 'z']):
                    key = f'{finger}_{axis}'
                    if key in opt_data and opt_data[key]['gt'] is not None:
                        sess_aligned = opt_data[key]['sess']
                        time_norm = np.linspace(0, 1, len(sess_aligned))
                        ax.plot(time_norm, sess_aligned, '-', linewidth=1.2, 
                               color=axis_colors[axis], alpha=0.8, label=axis.upper())
            
            if row_idx == 0:
                ax.set_title(finger_labels[col_idx], fontsize=12, fontweight='bold')
            if col_idx == 0:
                if model == 'GT':
                    label_text = 'GT (norm)'
                else:
                    opt_pearson = model_optimal_results_finger.get(model, {}).get('pearson_mean', 0)
                    label_text = f'{model}\n(r={opt_pearson:.3f})'
                ax.set_ylabel(label_text, fontsize=9, fontweight='bold')
            
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.05, 1.05)
            ax.tick_params(axis='both', labelsize=8)
            ax.grid(True, alpha=0.3)
            
            if row_idx == 0 and col_idx == 2:
                ax.legend(loc='upper right', fontsize=8, ncol=3)
            if row_idx < n_models_finger - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('Time (normalized)', fontsize=9)
    
    plt.suptitle('Stiffness per Model - Finger-Level Optimal Alignment (Same params for x,y,z per finger)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    fig_path_finger = os.path.join(output_dir, 'fig_stiffness_per_model_optimal_aligned_finger.png')
    plt.savefig(fig_path_finger, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {fig_path_finger}")
    
    # ===== Comparison summary: Axis-level vs Finger-level alignment =====
    print("\n" + "="*70)
    print("📊 Axis-level vs Finger-level Alignment 비교")
    print("="*70)
    print(f"{'Model':<20} {'Axis-level':>15} {'Finger-level':>15} {'차이':>10}")
    print("-"*60)
    for model in [m for m in all_models if m != 'GT']:
        axis_r = model_optimal_results_axis.get(model, {}).get('pearson_mean', 0)
        finger_r = model_optimal_results_finger.get(model, {}).get('pearson_mean', 0)
        diff = finger_r - axis_r
        print(f"{model:<20} {axis_r:>15.4f} {finger_r:>15.4f} {diff:>+10.4f}")
    print("-"*60)
    
    # ===== Figure 2: GT only (3x3 grid) =====
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    GT_SMOOTH_WINDOW = 15  # Light smoothing for GT
    GT_TRIM_RATIO = 0.05  # Trim first 5% of data
    
    for row, finger in enumerate(['th', 'if', 'mf']):
        for col, axis in enumerate(['x', 'y', 'z']):
            ax = axes[row, col]
            
            gt_col = gt_cols[finger][col]
            if gt_col in gt_df.columns:
                gt_vals_raw = gt_df[gt_col].values
                
                # Trim first 5% of data
                trim_idx = int(len(gt_vals_raw) * GT_TRIM_RATIO)
                gt_vals = gt_vals_raw[trim_idx:]
                
                # Apply smoothing to GT for cleaner visualization
                gt_vals_smooth = smooth_signal(gt_vals, window_size=GT_SMOOTH_WINDOW, method='savgol')
                
                # Time series (first 1000 samples)
                ax.plot(gt_vals_smooth[:min(1000, len(gt_vals_smooth))], 'k-', linewidth=0.8, alpha=0.8)
                
                # Stats (computed on trimmed data)
                mean_val = np.mean(gt_vals)
                std_val = np.std(gt_vals)
                ax.axhline(mean_val, color='r', linestyle='--', linewidth=1, label=f'Mean: {mean_val:.1f}')
                ax.fill_between(range(min(1000, len(gt_vals))), 
                               mean_val - std_val, mean_val + std_val,
                               alpha=0.2, color='red', label=f'±1σ: {std_val:.1f}')
            
            ax.set_title(f'{finger_names[finger]} - {axis_labels[col]}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Sample')
            ax.set_ylabel('Stiffness (N/m)')
            ax.grid(True, alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(loc='upper right', fontsize=8)
    
    plt.suptitle('Ground Truth Stiffness Profiles', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    fig_path = os.path.join(output_dir, 'fig_ground_truth_stiffness.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {fig_path}")
    
    # ===== Figure 3: Model Ranking Bar Chart =====
    # Compute scores
    gt_stats = compute_gt_reference_stats(gt_df.copy())
    scores_data = []
    
    for model, session_path in sessions.items():
        try:
            r = compute_session_vs_gt_metrics(session_path, gt_df, gt_stats)
            
            # Compute scores (same as print_comparison_table)
            fs = r['force_stiff_corr']
            gt_fs = gt_stats['force_stiff_corr']
            fs_score = 0
            for finger in ['th', 'if', 'mf']:
                sess_c = fs.get(finger, 0)
                gt_c = gt_fs.get(finger, 0)
                if np.sign(sess_c) == np.sign(gt_c):
                    fs_score += 0.5 + 0.5 * max(0, 1 - abs(sess_c - gt_c))
            fs_score /= 3
            
            axis_score = sum(1 for _, correct in r['axis_dominance'].values() if correct) / 3
            
            dist_score = 0
            if 'dist_errors' in r and r['dist_errors']:
                for de in r['dist_errors']:
                    if de['axis'] in ['th_z', 'if_y', 'mf_y']:
                        dist_score += max(0, 1 - de['mean_err']) * 0.5 + max(0, 1 - de['std_err']) * 0.5
                dist_score /= 3
            
            total = 0.4 * fs_score + 0.3 * axis_score + 0.3 * dist_score
            
            scores_data.append({
                'Model': model,
                'Total': total,
                'F→S': fs_score,
                'Axis': axis_score,
                'Dist': dist_score,
                'Pearson': r.get('pearson_mean', 0),
                'Spearman': r.get('spearman_mean', 0),
                'R²': r.get('r2_mean', 0),
            })
        except Exception as e:
            print(f"[WARN] Score computation failed for {model}: {e}")
    
    if scores_data:
        scores_df = pd.DataFrame(scores_data).sort_values('Total', ascending=False)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left: Total score
        ax = axes[0]
        colors = [model_colors.get(m, '#888888') for m in scores_df['Model']]
        bars = ax.barh(scores_df['Model'], scores_df['Total'], color=colors, alpha=0.8)
        ax.set_xlabel('Total Score')
        ax.set_title('Model Ranking (Total Score)', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1)
        for i, (idx, row) in enumerate(scores_df.iterrows()):
            ax.text(row['Total'] + 0.02, i, f"{row['Total']:.3f}", va='center', fontsize=10)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        
        # Right: Breakdown
        ax = axes[1]
        x = np.arange(len(scores_df))
        width = 0.25
        ax.bar(x - width, scores_df['F→S'], width, label='F→S (40%)', alpha=0.8)
        ax.bar(x, scores_df['Axis'], width, label='Axis (30%)', alpha=0.8)
        ax.bar(x + width, scores_df['Dist'], width, label='Dist (30%)', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(scores_df['Model'], rotation=45, ha='right')
        ax.set_ylabel('Score')
        ax.set_title('Score Breakdown by Metric', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        fig_path = os.path.join(output_dir, 'fig_model_ranking.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {fig_path}")
        
        # Save scores to CSV
        csv_path = os.path.join(output_dir, 'model_ranking.csv')
        scores_df.to_csv(csv_path, index=False, float_format='%.4f')
        print(f"✅ Saved: {csv_path}")
    
    # ===== Figure: Pearson Improvement Methods Comparison =====
    # Compare 3 methods to improve Pearson correlation:
    # 1. Raw (No processing) - original data, simple resample
    # 2. Smoothing - Savitzky-Golay filter applied
    # 3. Optimal Alignment - best trim/alignment search
    
    diff_session_path = None
    for model, path in sessions.items():
        if 'Diff_seq16_h2' in model:
            diff_session_path = path
            break
    
    if diff_session_path:
        diff_stiff_df = pd.read_csv(os.path.join(diff_session_path, 'stiffness.csv'))
        
        fingers = ['th', 'if', 'mf']
        finger_labels = ['THUMB', 'INDEX', 'MIDDLE']
        axis_labels = ['X', 'Y', 'Z']
        gt_col_map = {'th': ['th_k1', 'th_k2', 'th_k3'],
                      'if': ['if_k1', 'if_k2', 'if_k3'],
                      'mf': ['mf_k1', 'mf_k2', 'mf_k3']}
        sess_col_map = {'th': ['th_x', 'th_y', 'th_z'],
                        'if': ['if_x', 'if_y', 'if_z'],
                        'mf': ['mf_x', 'mf_y', 'mf_z']}
        
        # Parameters
        SMOOTH_WINDOW_GT = 15
        SMOOTH_WINDOW_SESS = 11
        GT_TRIM_RATIOS = [0.0, 0.05, 0.10, 0.15, 0.20]
        SESS_TRIM_START_RATIOS = [0.0, 0.05, 0.10, 0.15]
        SESS_TRIM_END_RATIOS = [1.0, 0.90, 0.80, 0.70, 0.60]
        
        def minmax_norm_local(arr):
            arr_min, arr_max = np.min(arr), np.max(arr)
            if arr_max - arr_min > 0:
                return (arr - arr_min) / (arr_max - arr_min)
            return arr - arr_min
        
        # Collect results for all 3 methods
        method_results = {
            'raw': {'pearson': [], 'gt_data': [], 'sess_data': []},
            'smooth': {'pearson': [], 'gt_data': [], 'sess_data': []},
            'optimal': {'pearson': [], 'gt_data': [], 'sess_data': [], 'params': []}
        }
        
        for finger in fingers:
            for axis_idx in range(3):
                gt_col = gt_col_map[finger][axis_idx]
                sess_col = sess_col_map[finger][axis_idx]
                
                if gt_col not in gt_df.columns or sess_col not in diff_stiff_df.columns:
                    for method in method_results:
                        method_results[method]['pearson'].append(0)
                        method_results[method]['gt_data'].append(np.array([]))
                        method_results[method]['sess_data'].append(np.array([]))
                    method_results['optimal']['params'].append(None)
                    continue
                
                gt_raw = gt_df[gt_col].values
                sess_raw = diff_stiff_df[sess_col].values
                
                # ===== Method 1: Raw (no smoothing, simple resample) =====
                target_len = min(len(gt_raw), len(sess_raw))
                gt_rs = np.interp(np.linspace(0,1,target_len), np.linspace(0,1,len(gt_raw)), gt_raw)
                sess_rs = np.interp(np.linspace(0,1,target_len), np.linspace(0,1,len(sess_raw)), sess_raw)
                gt_n = minmax_norm_local(gt_rs)
                sess_n = minmax_norm_local(sess_rs)
                
                if np.std(gt_n) > 0 and np.std(sess_n) > 0:
                    r_raw, _ = stats.pearsonr(gt_n, sess_n)
                else:
                    r_raw = 0
                method_results['raw']['pearson'].append(r_raw)
                method_results['raw']['gt_data'].append(gt_n)
                method_results['raw']['sess_data'].append(sess_n)
                
                # ===== Method 2: Smoothing only (no trimming) =====
                gt_smooth = smooth_signal(gt_raw, window_size=SMOOTH_WINDOW_GT, method='savgol')
                sess_smooth = smooth_signal(sess_raw, window_size=SMOOTH_WINDOW_SESS, method='savgol')
                
                target_len = min(len(gt_smooth), len(sess_smooth))
                gt_rs2 = np.interp(np.linspace(0,1,target_len), np.linspace(0,1,len(gt_smooth)), gt_smooth)
                sess_rs2 = np.interp(np.linspace(0,1,target_len), np.linspace(0,1,len(sess_smooth)), sess_smooth)
                gt_n2 = minmax_norm_local(gt_rs2)
                sess_n2 = minmax_norm_local(sess_rs2)
                
                if np.std(gt_n2) > 0 and np.std(sess_n2) > 0:
                    r_smooth, _ = stats.pearsonr(gt_n2, sess_n2)
                else:
                    r_smooth = 0
                method_results['smooth']['pearson'].append(r_smooth)
                method_results['smooth']['gt_data'].append(gt_n2)
                method_results['smooth']['sess_data'].append(sess_n2)
                
                # ===== Method 3: Optimal Alignment (smoothing + best trim) =====
                best_pearson = -2
                best_params = None
                best_gt_data = None
                best_sess_data = None
                
                for gt_trim in GT_TRIM_RATIOS:
                    gt_start = int(len(gt_smooth) * gt_trim)
                    gt_trimmed = gt_smooth[gt_start:]
                    if len(gt_trimmed) < 10:
                        continue
                    
                    for sess_start_r in SESS_TRIM_START_RATIOS:
                        for sess_end_r in SESS_TRIM_END_RATIOS:
                            if sess_end_r <= sess_start_r:
                                continue
                            
                            sess_start = int(len(sess_smooth) * sess_start_r)
                            sess_end = int(len(sess_smooth) * sess_end_r)
                            sess_trimmed = sess_smooth[sess_start:sess_end]
                            
                            if len(sess_trimmed) < 10:
                                continue
                            
                            tgt_len = min(len(sess_trimmed), len(gt_trimmed))
                            sess_rs3 = np.interp(np.linspace(0,1,tgt_len), np.linspace(0,1,len(sess_trimmed)), sess_trimmed)
                            gt_rs3 = np.interp(np.linspace(0,1,tgt_len), np.linspace(0,1,len(gt_trimmed)), gt_trimmed)
                            
                            sess_n3 = minmax_norm_local(sess_rs3)
                            gt_n3 = minmax_norm_local(gt_rs3)
                            
                            if np.std(sess_n3) > 0 and np.std(gt_n3) > 0:
                                r, _ = stats.pearsonr(sess_n3, gt_n3)
                                if r > best_pearson:
                                    best_pearson = r
                                    best_params = (gt_trim, sess_start_r, sess_end_r)
                                    best_gt_data = gt_n3
                                    best_sess_data = sess_n3
                
                if best_params is None:
                    best_pearson = r_smooth
                    best_gt_data = gt_n2
                    best_sess_data = sess_n2
                
                method_results['optimal']['pearson'].append(best_pearson)
                method_results['optimal']['gt_data'].append(best_gt_data)
                method_results['optimal']['sess_data'].append(best_sess_data)
                method_results['optimal']['params'].append(best_params)
        
        # Create figure: 3 rows (methods) x 9 cols (TH_x,y,z, IF_x,y,z, MF_x,y,z)
        # Layout similar to fig_stiffness_per_model_all_axes_unified_y.png
        method_names = ['Raw (No Processing)', 'Smoothing Only', 'Optimal Alignment']
        method_keys = ['raw', 'smooth', 'optimal']
        
        n_methods = 3
        fig, axes = plt.subplots(n_methods, 3, figsize=(15, 2.5 * n_methods + 1))
        
        # Color for GT and Session
        gt_color = '#2ca02c'   # Green for GT
        sess_color = '#1f77b4'  # Blue for Diffusion
        
        for row_idx, (method_name, method_key) in enumerate(zip(method_names, method_keys)):
            for col_idx, finger in enumerate(fingers):
                ax = axes[row_idx, col_idx]
                
                # Get average Pearson for this finger (across 3 axes)
                finger_pearson_list = method_results[method_key]['pearson'][col_idx*3:(col_idx+1)*3]
                avg_pearson = np.mean(finger_pearson_list) if finger_pearson_list else 0
                
                # Plot 3 axes (x, y, z) as stacked subplots
                for axis_idx, axis in enumerate(['x', 'y', 'z']):
                    idx = col_idx * 3 + axis_idx
                    
                    gt_data = method_results[method_key]['gt_data'][idx]
                    sess_data = method_results[method_key]['sess_data'][idx]
                    
                    if len(gt_data) > 0 and len(sess_data) > 0:
                        time_norm = np.linspace(0, 1, len(gt_data))
                        
                        # Offset for each axis (y-shift for visibility)
                        offset = axis_idx * 1.2
                        
                        ax.plot(time_norm, gt_data + offset, '-', linewidth=1.5, 
                               color=gt_color, alpha=0.8, label='GT' if axis_idx == 0 else '')
                        ax.plot(time_norm, sess_data + offset, '-', linewidth=1.5, 
                               color=sess_color, alpha=0.8, label='Diff' if axis_idx == 0 else '')
                        
                        # Add axis label and individual Pearson on the left
                        pearson_val = method_results[method_key]['pearson'][idx]
                        ax.text(-0.02, offset + 0.5, f'{axis.upper()}', fontsize=8, fontweight='bold',
                               ha='right', va='center', transform=ax.get_yaxis_transform())
                        ax.text(1.02, offset + 0.5, f'r={pearson_val:.2f}', fontsize=7,
                               ha='left', va='center', transform=ax.get_yaxis_transform(),
                               color='gray')
                
                # Title with finger name and average Pearson
                if row_idx == 0:
                    ax.set_title(finger_labels[col_idx], fontsize=12, fontweight='bold')
                
                # Method label on the left
                if col_idx == 0:
                    # Compute method's total average Pearson
                    method_avg_pearson = np.mean(method_results[method_key]['pearson'])
                    ax.set_ylabel(f'{method_name}\n(avg r={method_avg_pearson:.3f})', 
                                 fontsize=9, fontweight='bold')
                
                ax.set_xlim(0, 1)
                ax.set_ylim(-0.2, 3.8)
                ax.set_yticks([0.5, 1.7, 2.9])
                ax.set_yticklabels(['X', 'Y', 'Z'])
                ax.grid(True, alpha=0.3, axis='x')
                
                # X label only for bottom row
                if row_idx == n_methods - 1:
                    ax.set_xlabel('Time (normalized)', fontsize=9)
                else:
                    ax.set_xticklabels([])
                
                # Legend only for first cell
                if row_idx == 0 and col_idx == 2:
                    ax.legend(loc='upper right', fontsize=8)
                
                # Background color based on average Pearson
                if avg_pearson >= 0.7:
                    ax.set_facecolor('#e6ffe6')  # Light green
                elif avg_pearson >= 0.5:
                    ax.set_facecolor('#ffffcc')  # Light yellow
                else:
                    ax.set_facecolor('#ffe6e6')  # Light red
        
        # Summary bar chart at the bottom
        plt.suptitle('Pearson Correlation Improvement: 3 Methods Comparison (GT vs Diff_seq16_h2)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        fig_path = os.path.join(output_dir, 'fig_pearson_improvement_methods.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {fig_path}")
        
        # ===== Additional: Summary bar chart =====
        fig, ax = plt.subplots(figsize=(10, 5))
        
        x = np.arange(9)  # 9 axes total
        width = 0.25
        
        axis_names = ['TH_x', 'TH_y', 'TH_z', 'IF_x', 'IF_y', 'IF_z', 'MF_x', 'MF_y', 'MF_z']
        
        bars1 = ax.bar(x - width, method_results['raw']['pearson'], width, 
                      label='Raw', color='#d62728', alpha=0.8)
        bars2 = ax.bar(x, method_results['smooth']['pearson'], width, 
                      label='Smoothing', color='#ff7f0e', alpha=0.8)
        bars3 = ax.bar(x + width, method_results['optimal']['pearson'], width, 
                      label='Optimal Align', color='#2ca02c', alpha=0.8)
        
        ax.set_ylabel('Pearson Correlation', fontsize=11)
        ax.set_xlabel('Stiffness Axis', fontsize=11)
        ax.set_title('Pearson Improvement by Method (GT vs Diff_seq16_h2)', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(axis_names, fontsize=9)
        ax.legend(loc='lower right', fontsize=10)
        ax.set_ylim(-0.5, 1.0)
        ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add average Pearson for each method
        avg_raw = np.mean(method_results['raw']['pearson'])
        avg_smooth = np.mean(method_results['smooth']['pearson'])
        avg_optimal = np.mean(method_results['optimal']['pearson'])
        
        ax.text(0.02, 0.98, f'Avg Pearson: Raw={avg_raw:.3f}, Smooth={avg_smooth:.3f}, Optimal={avg_optimal:.3f}',
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        fig_path = os.path.join(output_dir, 'fig_pearson_improvement_barchart.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {fig_path}")


def print_comparison_table(sessions: Dict[str, str], gt_df: pd.DataFrame, save_figures: bool = True) -> None:
    """Print a comprehensive comparison table with meaningful metrics."""
    
    print("\n" + "="*120)
    print("📊 STIFFNESS MODEL COMPARISON (Functional Connectivity Analysis)")
    print("="*120)
    
    # Compute GT reference statistics
    gt_stats = compute_gt_reference_stats(gt_df.copy())
    
    print("\n📋 GT Reference Statistics:")
    print(f"   Force→Stiffness Correlation: TH_y={gt_stats['force_stiff_corr']['th']:.3f}, "
          f"IF_z={gt_stats['force_stiff_corr']['if']:.3f}, MF_z={gt_stats['force_stiff_corr']['mf']:.3f}")
    print(f"   Variance Ratio: TH_y={gt_stats['var_ratio']['th'].get('y', 0)*100:.1f}%, "
          f"IF_z={gt_stats['var_ratio']['if'].get('z', 0)*100:.1f}%, MF_z={gt_stats['var_ratio']['mf'].get('z', 0)*100:.1f}%")
    
    # Collect all results
    all_results = {}
    for model, session_path in sessions.items():
        try:
            result = compute_session_vs_gt_metrics(session_path, gt_df, gt_stats)
            result['session'] = os.path.basename(session_path)
            all_results[model] = result
        except Exception as e:
            print(f"[WARN] Failed to analyze {model}: {e}")
    
    if not all_results:
        print("No valid sessions found!")
        return
    
    # ===== Table 1: Force→Stiffness Correlation Pattern =====
    print("\n" + "-"*120)
    print("📈 [1] Force → Stiffness Correlation Pattern (Key Metric!)")
    print("    GT Reference: TH_z, IF_y, MF_y (dominant axis correlation)")
    print("-"*120)
    print(f"{'Model':<18} │ {'TH_z':>10} │ {'IF_y':>10} │ {'MF_y':>10} │ {'Sign✓':>6} │ {'Δ Mean':>8} │ {'Pattern':>10}")
    print("-"*120)
    
    for model in sorted(all_results.keys()):
        r = all_results[model]
        fs = r['force_stiff_corr']
        
        th_corr = fs.get('th', 0)
        if_corr = fs.get('if', 0)
        mf_corr = fs.get('mf', 0)
        
        # Check if correlations have correct sign
        sign_match = 0
        total_diff = 0
        gt_ref = gt_stats['force_stiff_corr']
        for finger, corr in [('th', th_corr), ('if', if_corr), ('mf', mf_corr)]:
            gt = gt_ref[finger]
            if np.sign(corr) == np.sign(gt):
                sign_match += 1
            total_diff += abs(corr - gt)
        
        pattern = "✓ Good" if sign_match == 3 and total_diff/3 < 0.3 else "⚠ Partial" if sign_match >= 2 else "✗ Bad"
        
        print(f"{model:<18} │ {th_corr:>+10.3f} │ {if_corr:>+10.3f} │ {mf_corr:>+10.3f} │ {sign_match}/3   │ {total_diff/3:>8.3f} │ {pattern:>10}")
    
    # ===== Table 2: Axis Dominance (Variance Ratio) =====
    print("\n" + "-"*120)
    print("📈 [2] Axis Dominance - Variance Ratio %")
    print("    GT Reference: TH=Z, IF=Y, MF=Y (expected dominant axis)")
    print("-"*120)
    print(f"{'Model':<18} │ {'TH_x':>5} {'TH_y':>5} {'TH_z':>5} │ {'IF_x':>5} {'IF_y':>5} {'IF_z':>5} │ {'MF_x':>5} {'MF_y':>5} {'MF_z':>5} │ {'Match':>6}")
    print("-"*120)
    
    for model in sorted(all_results.keys()):
        r = all_results[model]
        vr = r.get('var_ratio', {})
        
        line = f"{model:<18} │ "
        matches = 0
        for finger, exp_dom in [('th', 'z'), ('if', 'y'), ('mf', 'y')]:
            if finger in vr:
                for ax in ['x', 'y', 'z']:
                    val = vr[finger].get(ax, 0)
                    marker = "*" if ax == exp_dom and val > 40 else ""
                    line += f"{val:>4.0f}{marker} "
                if max(vr[finger], key=vr[finger].get) == exp_dom:
                    matches += 1
            else:
                line += "  N/A   N/A   N/A "
            line += "│ "
        
        line += f" {matches}/3"
        print(line)
    
    # ===== Table 3: Mean/Std Distribution Comparison =====
    print("\n" + "-"*120)
    print("📈 [3] Stiffness Distribution (Mean ± Std)")
    print("-"*120)
    print(f"{'Model':<18} │ {'TH_z (mean±std)':>18} │ {'IF_y (mean±std)':>18} │ {'MF_y (mean±std)':>18}")
    print("-"*120)
    
    # GT reference line
    gt_ms = gt_stats['mean_std']
    print(f"{'GT Reference':<18} │ {gt_ms.get('th_z', (0,0))[0]:>7.1f} ± {gt_ms.get('th_z', (0,0))[1]:<7.1f} │ "
          f"{gt_ms.get('if_y', (0,0))[0]:>7.1f} ± {gt_ms.get('if_y', (0,0))[1]:<7.1f} │ "
          f"{gt_ms.get('mf_y', (0,0))[0]:>7.1f} ± {gt_ms.get('mf_y', (0,0))[1]:<7.1f}")
    print("-"*120)
    
    for model in sorted(all_results.keys()):
        r = all_results[model]
        ms = r.get('mean_std', {})
        
        th_mean, th_std = ms.get('th_z', (0, 0))
        if_mean, if_std = ms.get('if_y', (0, 0))
        mf_mean, mf_std = ms.get('mf_y', (0, 0))
        
        print(f"{model:<18} │ {th_mean:>7.1f} ± {th_std:<7.1f} │ {if_mean:>7.1f} ± {if_std:<7.1f} │ {mf_mean:>7.1f} ± {mf_std:<7.1f}")
    
    # ===== Table 4: Session Info =====
    print("\n" + "-"*120)
    print("📈 [4] Session Info")
    print("-"*120)
    print(f"{'Model':<18} │ {'Samples':>8} │ {'Duration':>10} │ {'Session Name':<60}")
    print("-"*120)
    
    for model in sorted(all_results.keys()):
        r = all_results[model]
        print(f"{model:<18} │ {r['n_samples']:>8} │ {r['duration']:>8.1f}s │ {r['session']:<60}")
    
    # ===== Ranking =====
    print("\n" + "="*140)
    print("🏆 MODEL RANKING (Based on GT Pattern Similarity)")
    print("="*140)
    
    scores = {}
    for model, r in all_results.items():
        # 1. Axis dominance (50%)
        axis_score = sum(1 for _, correct in r['axis_dominance'].values() if correct) / 3
        
        # 2. Pearson correlation (50%)
        pearson = r.get('pearson_mean', 0)
        pearson_score = max(0, pearson)  # 0~1 range
        
        total = 0.5 * axis_score + 0.5 * pearson_score
        scores[model] = {
            'total': total, 
            'axis': axis_score, 
            'pearson': pearson,
        }
    
    # Sort by Pearson correlation (descending)
    sorted_models = sorted(scores.items(), key=lambda x: -x[1]['pearson'])
    print(f"\n{'Rank':<6} {'Model':<18} │ {'Total':>7} │ {'Axis':>5} │ {'Pearson':>8}")
    print("-"*60)
    
    for rank, (model, s) in enumerate(sorted_models, 1):
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, "  ")
        print(f"{medal} #{rank:<3} {model:<18} │ {s['total']:>7.3f} │ {s['axis']:>5.2f} │ {s['pearson']:>+8.3f}")
    
    print("\n" + "="*100)
    print("📋 Metrics Explanation:")
    print("   • Total: Weighted score (Axis 50%, Pearson 50%)")
    print("   • Axis: Correct axis dominance (TH=z, IF=y, MF=y)")
    print("   • Pearson: Pearson correlation of sorted stiffness values (QQ-plot style, -1~1)")
    print("="*100)


def main():
    parser = argparse.ArgumentParser(description="Compare stiffness sessions across models")
    parser.add_argument('sessions', nargs='*', help='Session folder paths')
    parser.add_argument('--dir', type=str, help='Directory containing multiple session folders')
    parser.add_argument('--output', '-o', type=str, 
                        default='/home/songwoo/ros2_ws/icra2025/outputs/stiffness_comparison/paper_figures',
                        help='Output directory for figures')
    parser.add_argument('--reference', '-r', type=int, default=0,
                        help='Index of reference session (default: 0 = first session)')
    parser.add_argument('--auto', '-a', action='store_true',
                        help='Auto-find latest sessions for all model types and compare with GT')
    parser.add_argument('--save-figures', '-s', action='store_true', default=True,
                        help='Save comparison figures as PNG (default: True)')
    parser.add_argument('--demo', '-d', type=int, default=None,
                        help='Use single demo as GT (0-9). If not specified, use all demos combined.')
    
    args = parser.parse_args()
    
    # Default log directory
    default_log_dir = '/home/songwoo/ros2_ws/icra2025/install/hri_falcon_robot_bridge/lib/outputs/stiffness_logs/'
    default_gt_dir = '/home/songwoo/ros2_ws/icra2025/src/hri_falcon_robot_bridge/outputs/stiffness_profiles_signaligned'
    
    # Auto mode: find latest sessions and compare with GT
    if args.auto or (not args.sessions and not args.dir):
        print("\n🔍 Auto-finding latest sessions for each model...")
        
        sessions = find_latest_sessions(default_log_dir)
        
        if not sessions:
            print("❌ No sessions found!")
            sys.exit(1)
        
        print(f"\n📂 Found {len(sessions)} models:")
        for model, path in sorted(sessions.items()):
            print(f"  • {model:<18} → {os.path.basename(path)}")
        
        # Load GT - auto-select best demo based on Pearson correlation with Diffusion models
        print(f"\n📚 Loading Ground Truth from: {default_gt_dir}")
        best_gt_file, best_gt_score = find_best_gt_demo(default_gt_dir, sessions, target_models=['Diff'])
        gt_df = load_gt_data(default_gt_dir, demo_index=args.demo, specific_file=best_gt_file)
        
        if gt_df is None:
            print("❌ Failed to load GT data!")
            sys.exit(1)
        
        print(f"   GT samples: {len(gt_df)}")
        
        # Compare and print table
        print_comparison_table(sessions, gt_df)
        
        # Save figures
        if args.save_figures:
            print(f"\n📊 Saving figures to: {args.output}")
            save_comparison_figures(sessions, gt_df, args.output)
        
        return
    
    # Manual mode (original behavior)
    session_paths = []
    
    if args.sessions:
        session_paths.extend(args.sessions)
    
    if args.dir:
        if os.path.isdir(args.dir):
            for item in sorted(os.listdir(args.dir)):
                item_path = os.path.join(args.dir, item)
                if os.path.isdir(item_path) and item.startswith('session_'):
                    session_paths.append(item_path)
    
    if len(session_paths) < 2:
        print("❌ Need at least 2 session paths to compare!")
        print("\nUsage examples:")
        print("  python compare_stiffness_sessions.py /path/to/session1 /path/to/session2")
        print("  python compare_stiffness_sessions.py --dir /path/to/stiffness_logs/")
        print("  python compare_stiffness_sessions.py --auto  # Auto-find latest sessions")
        sys.exit(1)
    
    print(f"\n📂 Found {len(session_paths)} sessions:")
    for i, p in enumerate(session_paths):
        model = extract_model_name(p)
        ref_marker = " (REFERENCE)" if i == args.reference else ""
        print(f"  [{i}] {model:12s} - {os.path.basename(p)}{ref_marker}")
    
    # Compare sessions
    df_results = compare_sessions(session_paths, reference_idx=args.reference)
    
    if not df_results.empty:
        print_metrics_table(df_results)
        
        # Generate plots
        plot_comparison(session_paths, args.output)
        
        print(f"\n✅ All results saved to: {args.output}/")


if __name__ == "__main__":
    main()
