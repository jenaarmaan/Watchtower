"""
Risk Scoring and Aggregation Module

Aggregates multiple signals into a unified Model Risk Score (0-100)
and tracks risk trends over time.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque


class RiskScorer:
    """
    Aggregates signals into Model Risk Score and tracks trends.
    """
    
    def __init__(self, 
                 signal_weights: Optional[Dict[str, float]] = None,
                 risk_threshold_high: float = 70.0,
                 risk_threshold_medium: float = 40.0):
        """
        Initialize risk scorer.
        
        Args:
            signal_weights: Weights for each signal type (default: equal weights)
            risk_threshold_high: Risk score threshold for high risk (0-100)
            risk_threshold_medium: Risk score threshold for medium risk (0-100)
        """
        self.signal_weights = signal_weights or {
            'feature_drift': 0.30,
            'prediction_drift': 0.25,
            'confidence_collapse': 0.25,
            'out_of_range': 0.20
        }
        
        # Normalize weights
        total_weight = sum(self.signal_weights.values())
        self.signal_weights = {k: v/total_weight for k, v in self.signal_weights.items()}
        
        self.risk_threshold_high = risk_threshold_high
        self.risk_threshold_medium = risk_threshold_medium
        
        # Risk history tracking
        self.risk_history = []  # List of (timestamp, risk_score, signals)
        self.last_stable_checkpoint = None
        self.baseline_risk_score = None
    
    def compute_risk_score(self, signals: Dict) -> Dict:
        """
        Compute Model Risk Score from signal dictionary.
        
        Args:
            signals: Dictionary from SignalComputer.compute_signals()
            
        Returns:
            Dictionary with risk score, level, and breakdown
        """
        if 'signals' not in signals or not signals['signals']:
            return {
                'risk_score': 0.0,
                'risk_level': 'unknown',
                'error': signals.get('error', 'No signals available')
            }
        
        signal_dict = signals['signals']
        
        # Extract signal scores
        signal_scores = {}
        for signal_name, weight in self.signal_weights.items():
            if signal_name in signal_dict:
                signal_data = signal_dict[signal_name]
                if isinstance(signal_data, dict) and 'score' in signal_data:
                    signal_scores[signal_name] = signal_data['score']
                else:
                    signal_scores[signal_name] = 0.0
            else:
                signal_scores[signal_name] = 0.0
        
        # Weighted aggregation (convert 0-1 scores to 0-100)
        weighted_sum = sum(
            signal_scores[signal_name] * weight * 100
            for signal_name, weight in self.signal_weights.items()
        )
        
        risk_score = min(100.0, max(0.0, weighted_sum))
        
        # Determine risk level
        if risk_score >= self.risk_threshold_high:
            risk_level = 'high'
        elif risk_score >= self.risk_threshold_medium:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        # Compute trend
        trend = self._compute_trend(risk_score)
        
        # Update history
        timestamp = signals.get('timestamp', datetime.now())
        self.risk_history.append({
            'timestamp': timestamp,
            'risk_score': risk_score,
            'signals': signal_scores,
            'risk_level': risk_level
        })
        
        # Update baseline if this is first computation
        if self.baseline_risk_score is None:
            self.baseline_risk_score = risk_score
            self.last_stable_checkpoint = {
                'timestamp': timestamp,
                'risk_score': risk_score,
                'signals': signal_scores.copy()
            }
        
        # Update stable checkpoint if risk is low
        if risk_level == 'low' and len(self.risk_history) > 0:
            # Check if we've been stable for a while
            recent_scores = [h['risk_score'] for h in self.risk_history[-5:]]
            if len(recent_scores) >= 3 and all(s < self.risk_threshold_medium for s in recent_scores):
                self.last_stable_checkpoint = {
                    'timestamp': timestamp,
                    'risk_score': risk_score,
                    'signals': signal_scores.copy()
                }
        
        return {
            'risk_score': float(risk_score),
            'risk_level': risk_level,
            'trend': trend,
            'signal_breakdown': {
                name: {
                    'score': float(signal_scores[name]),
                    'contribution': float(signal_scores[name] * weight * 100),
                    'weight': float(weight)
                }
                for name, weight in self.signal_weights.items()
            },
            'timestamp': timestamp,
            'baseline_risk_score': self.baseline_risk_score,
            'last_stable_checkpoint': self.last_stable_checkpoint
        }
    
    def _compute_trend(self, current_score: float) -> str:
        """
        Compute risk trend from recent history.
        
        Returns:
            'increasing', 'decreasing', 'stable', or 'unknown'
        """
        if len(self.risk_history) < 3:
            return 'unknown'
        
        recent_scores = [h['risk_score'] for h in self.risk_history[-5:]]
        
        # Simple linear trend
        if len(recent_scores) >= 3:
            x = np.arange(len(recent_scores))
            slope = np.polyfit(x, recent_scores, 1)[0]
            
            if slope > 2.0:  # Increasing by more than 2 points per period
                return 'increasing'
            elif slope < -2.0:  # Decreasing by more than 2 points per period
                return 'decreasing'
            else:
                return 'stable'
        
        return 'unknown'
    
    def get_risk_history(self, days: int = 7) -> pd.DataFrame:
        """
        Get risk history as DataFrame.
        
        Args:
            days: Number of days to retrieve
            
        Returns:
            DataFrame with risk history
        """
        if not self.risk_history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.risk_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filter by days
        cutoff = df['timestamp'].max() - pd.Timedelta(days=days)
        df = df[df['timestamp'] >= cutoff]
        
        return df
    
    def get_risk_summary(self) -> Dict:
        """
        Get summary statistics of risk history.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.risk_history:
            return {
                'total_observations': 0,
                'current_risk_score': None,
                'average_risk_score': None,
                'max_risk_score': None,
                'risk_level_distribution': {}
            }
        
        df = pd.DataFrame(self.risk_history)
        
        return {
            'total_observations': len(df),
            'current_risk_score': df['risk_score'].iloc[-1] if len(df) > 0 else None,
            'average_risk_score': float(df['risk_score'].mean()),
            'max_risk_score': float(df['risk_score'].max()),
            'min_risk_score': float(df['risk_score'].min()),
            'risk_level_distribution': df['risk_level'].value_counts().to_dict(),
            'last_stable_checkpoint': self.last_stable_checkpoint
        }
