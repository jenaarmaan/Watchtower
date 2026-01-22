"""
Signal Computation Module

Computes label-free signals for model risk detection:
- Feature distribution drift
- Prediction distribution drift
- Confidence collapse
- Out-of-range/unseen inputs
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class SignalComputer:
    """Computes various risk signals from prediction logs."""
    
    def __init__(self, baseline_window_days: int = 7, min_samples: int = 30):
        """
        Initialize signal computer.
        
        Args:
            baseline_window_days: Days to use for baseline statistics
            min_samples: Minimum samples required for reliable computation
                NOTE: Demo threshold (30). Production systems should tune this per signal.
        """
        self.baseline_window_days = baseline_window_days
        self.min_samples = min_samples
        self.baseline_stats = {}
        self.baseline_features = None
        self.baseline_predictions = None
        self.baseline_confidence = None
        
    def update_baseline(self, df: pd.DataFrame, timestamp_col: str = 'timestamp'):
        """
        Update baseline statistics from historical data.
        
        Args:
            df: DataFrame with prediction logs
            timestamp_col: Name of timestamp column
        """
        if len(df) < self.min_samples:
            return False
            
        # Get baseline period (most recent baseline_window_days)
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = df.sort_values(timestamp_col)
        
        baseline_end = df[timestamp_col].max()
        baseline_start = baseline_end - pd.Timedelta(days=self.baseline_window_days)
        baseline_df = df[df[timestamp_col] >= baseline_start]
        
        if len(baseline_df) < self.min_samples:
            return False
        
        # Store baseline data
        self.baseline_features = self._extract_features(baseline_df)
        self.baseline_predictions = baseline_df['prediction'].values if 'prediction' in baseline_df else None
        self.baseline_confidence = baseline_df['confidence'].values if 'confidence' in baseline_df else None
        
        # Compute baseline statistics
        self.baseline_stats = {
            'feature_means': self.baseline_features.mean(),
            'feature_stds': self.baseline_features.std(),
            'feature_ranges': {
                col: (self.baseline_features[col].min(), self.baseline_features[col].max())
                for col in self.baseline_features.columns
            },
            'confidence_mean': np.mean(self.baseline_confidence) if self.baseline_confidence is not None else None,
            'confidence_std': np.std(self.baseline_confidence) if self.baseline_confidence is not None else None,
            'prediction_dist': self._compute_prediction_distribution(self.baseline_predictions) if self.baseline_predictions is not None else None,
        }
        
        return True
    
    def compute_signals(self, df: pd.DataFrame, timestamp_col: str = 'timestamp') -> Dict:
        """
        Compute all risk signals for current window.
        
        Args:
            df: DataFrame with prediction logs
            timestamp_col: Name of timestamp column
            
        Returns:
            Dictionary of signal values and metadata
        """
        if self.baseline_stats == {}:
            return {
                'error': 'Baseline not initialized',
                'signals': {}
            }
        
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = df.sort_values(timestamp_col)
        
        # Get current window (last 24 hours)
        current_end = df[timestamp_col].max()
        current_start = current_end - pd.Timedelta(hours=24)
        current_df = df[df[timestamp_col] >= current_start]
        
        if len(current_df) < self.min_samples:
            return {
                'status': 'insufficient_samples',
                'message': f'{len(current_df)} < {self.min_samples}',
                'signals': {}
            }
        
        current_features = self._extract_features(current_df)
        current_predictions = current_df['prediction'].values if 'prediction' in current_df else None
        current_confidence = current_df['confidence'].values if 'confidence' in current_df else None
        
        signals = {}
        
        # 1. Feature Distribution Drift
        feature_drift = self._compute_feature_drift(current_features)
        signals['feature_drift'] = feature_drift
        
        # 2. Prediction Distribution Drift
        if current_predictions is not None:
            pred_drift = self._compute_prediction_drift(current_predictions)
            signals['prediction_drift'] = pred_drift
        else:
            signals['prediction_drift'] = {'score': 0.0, 'method': 'N/A'}
        
        # 3. Confidence Collapse
        if current_confidence is not None:
            conf_collapse = self._compute_confidence_collapse(current_confidence)
            signals['confidence_collapse'] = conf_collapse
        else:
            signals['confidence_collapse'] = {'score': 0.0, 'method': 'N/A'}
        
        # 4. Out-of-Range/Unseen Inputs
        oor_score = self._compute_out_of_range(current_features)
        signals['out_of_range'] = oor_score
        
        return {
            'timestamp': current_end,
            'window_size': len(current_df),
            'signals': signals
        }
    
    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract feature columns from dataframe."""
        feature_cols = [col for col in df.columns 
                       if col not in ['timestamp', 'prediction', 'confidence', 'model_version', 'id']]
        return df[feature_cols].select_dtypes(include=[np.number])
    
    def _compute_feature_drift(self, current_features: pd.DataFrame) -> Dict:
        """
        Compute feature distribution drift using multiple methods.
        
        Returns:
            Dictionary with drift scores and explanations
        """
        if self.baseline_features is None:
            return {'score': 0.0, 'method': 'N/A', 'details': {}}
        
        # Method 1: Kolmogorov-Smirnov test for each feature
        ks_scores = []
        drifted_features = []
        
        for col in current_features.columns:
            if col not in self.baseline_features.columns:
                continue
            
            baseline_vals = self.baseline_features[col].dropna()
            current_vals = current_features[col].dropna()
            
            if len(baseline_vals) < 10 or len(current_vals) < 10:
                continue
            
            # KS test
            ks_stat, p_value = stats.ks_2samp(baseline_vals, current_vals)
            ks_scores.append(ks_stat)
            
            if p_value < 0.05:  # Significant drift
                drifted_features.append({
                    'feature': col,
                    'ks_statistic': float(ks_stat),
                    'p_value': float(p_value),
                    'baseline_mean': float(baseline_vals.mean()),
                    'current_mean': float(current_vals.mean())
                })
        
        # Aggregate score (max KS statistic, normalized to 0-1)
        max_ks = max(ks_scores) if ks_scores else 0.0
        
        # Method 2: Mean shift magnitude
        mean_shifts = []
        for col in current_features.columns:
            if col not in self.baseline_features.columns:
                continue
            baseline_mean = self.baseline_features[col].mean()
            current_mean = current_features[col].mean()
            baseline_std = self.baseline_features[col].std()
            if baseline_std > 0:
                shift = abs(current_mean - baseline_mean) / baseline_std
                mean_shifts.append(shift)
        
        mean_shift_score = np.mean(mean_shifts) if mean_shifts else 0.0
        
        # Combined score (0-1 scale)
        combined_score = min(1.0, (max_ks * 0.6 + mean_shift_score * 0.4))
        
        return {
            'score': float(combined_score),
            'method': 'KS_test + mean_shift',
            'details': {
                'max_ks_statistic': float(max_ks),
                'mean_shift_zscore': float(mean_shift_score),
                'drifted_features': drifted_features[:5]  # Top 5
            }
        }
    
    def _compute_prediction_drift(self, current_predictions: np.ndarray) -> Dict:
        """
        Compute prediction distribution drift.
        
        Returns:
            Dictionary with drift score and explanation
        """
        if self.baseline_predictions is None:
            return {'score': 0.0, 'method': 'N/A'}
        
        # Compute distributions
        baseline_dist = self._compute_prediction_distribution(self.baseline_predictions)
        current_dist = self._compute_prediction_distribution(current_predictions)
        
        # Jensen-Shannon divergence (symmetric, bounded 0-1)
        js_div = jensenshannon(baseline_dist, current_dist)
        
        # Also compute class shift (if applicable)
        if len(baseline_dist) == len(current_dist):
            class_shifts = np.abs(current_dist - baseline_dist)
            max_class_shift = np.max(class_shifts)
        else:
            max_class_shift = 0.0
        
        return {
            'score': float(js_div),
            'method': 'Jensen-Shannon_divergence',
            'details': {
                'js_divergence': float(js_div),
                'max_class_shift': float(max_class_shift),
                'baseline_dist': baseline_dist.tolist(),
                'current_dist': current_dist.tolist()
            }
        }
    
    def _compute_prediction_distribution(self, predictions: np.ndarray) -> np.ndarray:
        """Compute normalized prediction class distribution."""
        if predictions is None or len(predictions) == 0:
            return np.array([])
        
        # Handle both categorical and continuous predictions
        if predictions.dtype in [np.int64, np.int32, np.object_]:
            # Categorical: count classes
            unique, counts = np.unique(predictions, return_counts=True)
            dist = np.zeros(len(unique))
            for i, val in enumerate(unique):
                dist[i] = np.sum(predictions == val)
            return dist / dist.sum() if dist.sum() > 0 else dist
        else:
            # Continuous: bin into histogram
            hist, _ = np.histogram(predictions, bins=10)
            return hist / hist.sum() if hist.sum() > 0 else hist
    
    def _compute_confidence_collapse(self, current_confidence: np.ndarray) -> Dict:
        """
        Detect confidence collapse (sudden drop in model confidence).
        
        Returns:
            Dictionary with collapse score and explanation
        """
        if self.baseline_confidence is None:
            return {'score': 0.0, 'method': 'N/A'}
        
        baseline_mean = self.baseline_stats['confidence_mean']
        baseline_std = self.baseline_stats['confidence_std']
        current_mean = np.mean(current_confidence)
        current_std = np.std(current_confidence)
        
        # Z-score of mean drop
        if baseline_std > 0:
            mean_zscore = (baseline_mean - current_mean) / baseline_std
        else:
            mean_zscore = 0.0
        
        # Variance increase (uncertainty spike)
        if baseline_std > 0:
            variance_ratio = current_std / baseline_std
        else:
            variance_ratio = 1.0
        
        # Low confidence rate
        low_conf_threshold = baseline_mean - 2 * baseline_std
        low_conf_rate = np.mean(current_confidence < low_conf_threshold)
        
        # Combined score
        # Mean drop contributes 50%, variance increase 30%, low conf rate 20%
        mean_drop_score = min(1.0, max(0.0, mean_zscore / 3.0))  # Normalize to 0-1
        variance_score = min(1.0, max(0.0, (variance_ratio - 1.0) / 2.0))
        low_conf_score = low_conf_rate
        
        combined_score = (mean_drop_score * 0.5 + variance_score * 0.3 + low_conf_score * 0.2)
        
        return {
            'score': float(combined_score),
            'method': 'mean_drop + variance_increase + low_conf_rate',
            'details': {
                'baseline_mean': float(baseline_mean),
                'current_mean': float(current_mean),
                'mean_drop_zscore': float(mean_zscore),
                'variance_ratio': float(variance_ratio),
                'low_confidence_rate': float(low_conf_rate)
            }
        }
    
    def _compute_out_of_range(self, current_features: pd.DataFrame) -> Dict:
        """
        Detect out-of-range or unseen input values.
        
        Returns:
            Dictionary with OOR score and explanation
        """
        if 'feature_ranges' not in self.baseline_stats:
            return {'score': 0.0, 'method': 'N/A'}
        
        oor_counts = []
        unseen_features = []
        
        for col in current_features.columns:
            if col not in self.baseline_stats['feature_ranges']:
                continue
            
            min_val, max_val = self.baseline_stats['feature_ranges'][col]
            current_vals = current_features[col].dropna()
            
            if len(current_vals) == 0:
                continue
            
            # Count out-of-range values
            oor_count = np.sum((current_vals < min_val) | (current_vals > max_val))
            oor_rate = oor_count / len(current_vals)
            
            if oor_rate > 0.05:  # More than 5% OOR
                unseen_features.append({
                    'feature': col,
                    'oor_rate': float(oor_rate),
                    'baseline_range': [float(min_val), float(max_val)],
                    'current_min': float(current_vals.min()),
                    'current_max': float(current_vals.max())
                })
            
            oor_counts.append(oor_rate)
        
        # Aggregate score
        max_oor_rate = max(oor_counts) if oor_counts else 0.0
        mean_oor_rate = np.mean(oor_counts) if oor_counts else 0.0
        
        # Combined: max contributes 70%, mean 30%
        combined_score = min(1.0, max_oor_rate * 0.7 + mean_oor_rate * 0.3)
        
        return {
            'score': float(combined_score),
            'method': 'out_of_range_rate',
            'details': {
                'max_oor_rate': float(max_oor_rate),
                'mean_oor_rate': float(mean_oor_rate),
                'unseen_features': unseen_features[:5]  # Top 5
            }
        }
