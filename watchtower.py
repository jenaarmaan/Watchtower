"""
Watchtower: Main Orchestrator

Coordinates signal computation, risk scoring, and alert generation.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

from signals import SignalComputer
from risk_scorer import RiskScorer
from alerts import AlertGenerator


class Watchtower:
    """
    Main Watchtower system orchestrator.
    
    Coordinates signal computation, risk scoring, and alert generation
    to provide early warning of model decay.
    """
    
    def __init__(self,
                 baseline_window_days: int = 7,
                 min_samples: int = 30,
                 signal_weights: Optional[Dict[str, float]] = None,
                 risk_threshold_high: float = 70.0,
                 risk_threshold_medium: float = 40.0):
        """
        Initialize Watchtower system.
        
        Args:
            baseline_window_days: Days to use for baseline statistics
            min_samples: Minimum samples required for reliable computation
                NOTE: Demo threshold (30). Production systems should tune this per signal.
            signal_weights: Weights for signal aggregation
            risk_threshold_high: High risk threshold (0-100)
            risk_threshold_medium: Medium risk threshold (0-100)
        """
        self.signal_computer = SignalComputer(
            baseline_window_days=baseline_window_days,
            min_samples=min_samples
        )
        
        self.risk_scorer = RiskScorer(
            signal_weights=signal_weights,
            risk_threshold_high=risk_threshold_high,
            risk_threshold_medium=risk_threshold_medium
        )
        
        self.alert_generator = AlertGenerator(
            high_risk_threshold=risk_threshold_high,
            medium_risk_threshold=risk_threshold_medium
        )
        
        self.alerts_history = []
        self.is_baseline_initialized = False
    
    def initialize_baseline(self, df: pd.DataFrame, timestamp_col: str = 'timestamp') -> bool:
        """
        Initialize baseline statistics from historical data.
        
        Args:
            df: DataFrame with prediction logs
            timestamp_col: Name of timestamp column
            
        Returns:
            True if baseline initialized successfully
        """
        success = self.signal_computer.update_baseline(df, timestamp_col)
        self.is_baseline_initialized = success
        return success
    
    def assess_risk(self, df: pd.DataFrame, timestamp_col: str = 'timestamp') -> Dict:
        """
        Perform complete risk assessment on current data.
        
        Args:
            df: DataFrame with prediction logs
            timestamp_col: Name of timestamp column
            
        Returns:
            Dictionary with risk assessment, signals, and alerts
        """
        if not self.is_baseline_initialized:
            return {
                'error': 'Baseline not initialized. Call initialize_baseline() first.',
                'risk_assessment': None,
                'signals': None,
                'alert': None
            }
        
        # Compute signals
        signals = self.signal_computer.compute_signals(df, timestamp_col)
        
        # Handle insufficient samples - convert to system risk
        if signals.get('status') == 'insufficient_samples':
            return {
                'risk_assessment': {
                    'risk_score': 65.0,
                    'risk_level': 'medium',
                    'trend': 'unknown',
                    'signal_breakdown': {},
                    'baseline_risk_score': None,
                    'last_stable_checkpoint': None
                },
                'signals': signals,
                'alert': {
                    'severity': 'medium',
                    'message': f"Low sample volume detected ({signals.get('message', 'unknown')})",
                    'reason_labels': ['insufficient_samples'],
                    'recommendations': [
                        f"Collect more data: {signals.get('message', 'unknown')}",
                        'Risk assessment may be unreliable with low sample count',
                        'Consider increasing data collection window'
                    ]
                },
                'meta': {
                    'status': 'insufficient_samples'
                }
            }
        
        # Handle other errors (backward compatibility)
        if 'error' in signals:
            return {
                'error': signals['error'],
                'risk_assessment': None,
                'signals': signals,
                'alert': None
            }
        
        # Compute risk score
        risk_assessment = self.risk_scorer.compute_risk_score(signals)
        
        # Generate alert if needed
        alert = self.alert_generator.generate_alert(risk_assessment, signals)
        
        if alert:
            self.alerts_history.append(alert)
        
        return {
            'risk_assessment': risk_assessment,
            'signals': signals,
            'alert': alert,
            'timestamp': datetime.now()
        }
    
    def get_risk_trend(self, days: int = 7) -> pd.DataFrame:
        """
        Get risk trend over time.
        
        Args:
            days: Number of days to retrieve
            
        Returns:
            DataFrame with risk history
        """
        return self.risk_scorer.get_risk_history(days)
    
    def get_alerts_history(self, days: int = 7) -> List[Dict]:
        """
        Get alerts history.
        
        Args:
            days: Number of days to retrieve
            
        Returns:
            List of alert dictionaries
        """
        if not self.alerts_history:
            return []
        
        cutoff = datetime.now() - timedelta(days=days)
        return [
            alert for alert in self.alerts_history
            if alert['timestamp'] >= cutoff
        ]
    
    def get_system_status(self) -> Dict:
        """
        Get overall system status and summary.
        
        Returns:
            Dictionary with system status
        """
        risk_summary = self.risk_scorer.get_risk_summary()
        
        return {
            'baseline_initialized': self.is_baseline_initialized,
            'risk_summary': risk_summary,
            'total_alerts': len(self.alerts_history),
            'recent_alerts': len(self.get_alerts_history(days=1)),
            'last_stable_checkpoint': self.risk_scorer.last_stable_checkpoint
        }
    
    def export_report(self, output_path: str = 'watchtower_report.json'):
        """
        Export comprehensive report to JSON.
        
        Args:
            output_path: Path to save report
        """
        risk_trend = self.get_risk_trend(days=30)
        alerts = self.get_alerts_history(days=30)
        status = self.get_system_status()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_status': status,
            'risk_trend': risk_trend.to_dict('records') if not risk_trend.empty else [],
            'alerts': alerts,
            'configuration': {
                'baseline_window_days': self.signal_computer.baseline_window_days,
                'min_samples': self.signal_computer.min_samples,
                'signal_weights': self.risk_scorer.signal_weights,
                'risk_thresholds': {
                    'high': self.risk_scorer.risk_threshold_high,
                    'medium': self.risk_scorer.risk_threshold_medium
                }
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report


def run(data: pd.DataFrame) -> Dict:
    """
    Public entrypoint for Watchtower.
    Used by Streamlit and external callers.
    
    Args:
        data: DataFrame with prediction logs (must have 'timestamp', 'prediction', 'confidence', and feature columns)
        
    Returns:
        Dictionary with risk assessment, signals, and alerts
    """
    # Create Watchtower instance
    engine = Watchtower()
    
    # Initialize baseline from the data
    baseline_initialized = engine.initialize_baseline(data, timestamp_col='timestamp')
    
    if not baseline_initialized:
        return {
            'error': 'Failed to initialize baseline. Insufficient data.',
            'risk_assessment': None,
            'signals': None,
            'alert': None
        }
    
    # Assess risk on the data
    results = engine.assess_risk(data, timestamp_col='timestamp')
    
    # Ensure results is always a dict (safety check)
    if not isinstance(results, dict):
        return {
            'error': 'Unexpected result type from assess_risk()',
            'risk_assessment': None,
            'signals': None,
            'alert': None,
            'system_status': None
        }
    
    # Add system status for dashboard
    results['system_status'] = engine.get_system_status()
    
    # Explicitly return dict (guaranteed contract)
    return results
