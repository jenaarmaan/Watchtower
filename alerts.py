"""
Alert Generation Module

Generates explainable alerts with actionable recommendations
based on risk signals and scores.
"""

from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum


class AlertSeverity(Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertGenerator:
    """
    Generates explainable alerts from risk signals.
    """
    
    def __init__(self, 
                 high_risk_threshold: float = 70.0,
                 medium_risk_threshold: float = 40.0):
        """
        Initialize alert generator.
        
        Args:
            high_risk_threshold: Risk score threshold for high severity
            medium_risk_threshold: Risk score threshold for medium severity
        """
        self.high_risk_threshold = high_risk_threshold
        self.medium_risk_threshold = medium_risk_threshold
    
    def generate_alert(self, risk_assessment: Dict, signals: Dict) -> Optional[Dict]:
        """
        Generate alert from risk assessment and signals.
        
        Args:
            risk_assessment: Output from RiskScorer.compute_risk_score()
            signals: Output from SignalComputer.compute_signals()
            
        Returns:
            Alert dictionary or None if no alert needed
        """
        risk_score = risk_assessment.get('risk_score', 0.0)
        risk_level = risk_assessment.get('risk_level', 'low')
        
        # Only generate alerts for medium+ risk
        if risk_level == 'low':
            return None
        
        # Determine severity
        if risk_score >= self.high_risk_threshold:
            severity = AlertSeverity.HIGH
        elif risk_score >= self.medium_risk_threshold:
            severity = AlertSeverity.MEDIUM
        else:
            return None
        
        # Identify contributing signals
        signal_breakdown = risk_assessment.get('signal_breakdown', {})
        contributing_signals = self._identify_contributing_signals(signal_breakdown)
        
        # Generate reason labels
        reason_labels = self._generate_reason_labels(contributing_signals, signals)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(contributing_signals, signals)
        
        # Compose alert message
        message = self._compose_alert_message(
            risk_score, 
            risk_level, 
            reason_labels, 
            contributing_signals
        )
        
        return {
            'timestamp': risk_assessment.get('timestamp', datetime.now()),
            'severity': severity.value,
            'risk_score': risk_score,
            'risk_level': risk_level,
            'message': message,
            'reason_labels': reason_labels,
            'contributing_signals': contributing_signals,
            'recommendations': recommendations,
            'signal_details': self._extract_signal_details(signals)
        }
    
    def _identify_contributing_signals(self, signal_breakdown: Dict) -> List[str]:
        """
        Identify which signals are contributing most to risk.
        
        Returns:
            List of signal names ordered by contribution
        """
        # Sort by contribution (descending)
        sorted_signals = sorted(
            signal_breakdown.items(),
            key=lambda x: x[1]['contribution'],
            reverse=True
        )
        
        # Return signals with contribution > 10% of total risk
        contributing = []
        for signal_name, data in sorted_signals:
            if data['contribution'] > 10.0:  # More than 10 points
                contributing.append(signal_name)
        
        return contributing
    
    def _generate_reason_labels(self, contributing_signals: List[str], signals: Dict) -> List[str]:
        """
        Generate human-readable reason labels.
        
        Returns:
            List of reason label strings
        """
        label_mapping = {
            'feature_drift': 'data drift',
            'prediction_drift': 'prediction shift',
            'confidence_collapse': 'confidence collapse',
            'out_of_range': 'unseen inputs'
        }
        
        labels = []
        for signal_name in contributing_signals:
            if signal_name in label_mapping:
                labels.append(label_mapping[signal_name])
        
        # Add specific details from signal data
        signal_details = signals.get('signals', {})
        
        # Check for specific patterns
        if 'confidence_collapse' in contributing_signals:
            conf_data = signal_details.get('confidence_collapse', {})
            details = conf_data.get('details', {})
            if details.get('mean_drop_zscore', 0) > 2.0:
                labels.append('significant confidence drop')
        
        if 'out_of_range' in contributing_signals:
            oor_data = signal_details.get('out_of_range', {})
            details = oor_data.get('details', {})
            if details.get('max_oor_rate', 0) > 0.2:
                labels.append('high out-of-range rate')
        
        return labels
    
    def _generate_recommendations(self, contributing_signals: List[str], signals: Dict) -> List[str]:
        """
        Generate actionable recommendations based on signals.
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        signal_details = signals.get('signals', {})
        
        if 'feature_drift' in contributing_signals:
            drift_data = signal_details.get('feature_drift', {})
            details = drift_data.get('details', {})
            drifted_features = details.get('drifted_features', [])
            
            if drifted_features:
                top_feature = drifted_features[0]['feature']
                recommendations.append(
                    f"Review data pipeline for feature '{top_feature}' - "
                    f"detected distribution shift (KS={drifted_features[0]['ks_statistic']:.2f})"
                )
            else:
                recommendations.append(
                    "Review data collection pipeline - detected feature distribution drift"
                )
        
        if 'prediction_drift' in contributing_signals:
            recommendations.append(
                "Monitor prediction distribution - model output patterns have shifted. "
                "Consider retraining if drift persists."
            )
        
        if 'confidence_collapse' in contributing_signals:
            conf_data = signal_details.get('confidence_collapse', {})
            details = conf_data.get('details', {})
            mean_drop = details.get('mean_drop_zscore', 0)
            
            if mean_drop > 2.0:
                recommendations.append(
                    f"Model confidence has dropped significantly (z-score: {mean_drop:.2f}). "
                    "Investigate input data quality and consider model refresh."
                )
            else:
                recommendations.append(
                    "Model confidence is declining. Review input data quality and "
                    "consider recalibration or retraining."
                )
        
        if 'out_of_range' in contributing_signals:
            oor_data = signal_details.get('out_of_range', {})
            details = oor_data.get('details', {})
            unseen_features = details.get('unseen_features', [])
            
            if unseen_features:
                top_feature = unseen_features[0]['feature']
                recommendations.append(
                    f"Input validation issue detected for '{top_feature}' - "
                    f"{unseen_features[0]['oor_rate']*100:.1f}% of values are out of training range. "
                    "Review data preprocessing and input validation."
                )
            else:
                recommendations.append(
                    "High rate of out-of-range inputs detected. "
                    "Review data preprocessing pipeline and input validation rules."
                )
        
        # General recommendations
        if len(contributing_signals) >= 3:
            recommendations.append(
                "Multiple risk signals detected simultaneously. "
                "Consider comprehensive model review and potential retraining."
            )
        
        if not recommendations:
            recommendations.append(
                "Monitor model performance closely. Review prediction logs for anomalies."
            )
        
        return recommendations
    
    def _compose_alert_message(self, 
                              risk_score: float, 
                              risk_level: str,
                              reason_labels: List[str],
                              contributing_signals: List[str]) -> str:
        """
        Compose human-readable alert message.
        
        Returns:
            Alert message string
        """
        reasons_str = " + ".join(reason_labels) if reason_labels else "multiple risk factors"
        
        message = (
            f"Model Risk Alert ({risk_level.upper()}): Risk score = {risk_score:.1f}/100. "
            f"Detected: {reasons_str}."
        )
        
        return message
    
    def _extract_signal_details(self, signals: Dict) -> Dict:
        """
        Extract key details from signals for alert context.
        
        Returns:
            Dictionary with signal details
        """
        signal_dict = signals.get('signals', {})
        details = {}
        
        for signal_name, signal_data in signal_dict.items():
            if isinstance(signal_data, dict) and 'details' in signal_data:
                # Extract key metrics
                signal_details = signal_data['details']
                details[signal_name] = {
                    'score': signal_data.get('score', 0.0),
                    'key_metrics': {}
                }
                
                # Extract specific metrics based on signal type
                if signal_name == 'feature_drift':
                    details[signal_name]['key_metrics'] = {
                        'max_ks_statistic': signal_details.get('max_ks_statistic', 0.0),
                        'mean_shift_zscore': signal_details.get('mean_shift_zscore', 0.0),
                        'num_drifted_features': len(signal_details.get('drifted_features', []))
                    }
                elif signal_name == 'prediction_drift':
                    details[signal_name]['key_metrics'] = {
                        'js_divergence': signal_details.get('js_divergence', 0.0),
                        'max_class_shift': signal_details.get('max_class_shift', 0.0)
                    }
                elif signal_name == 'confidence_collapse':
                    details[signal_name]['key_metrics'] = {
                        'mean_drop_zscore': signal_details.get('mean_drop_zscore', 0.0),
                        'variance_ratio': signal_details.get('variance_ratio', 1.0),
                        'low_confidence_rate': signal_details.get('low_confidence_rate', 0.0)
                    }
                elif signal_name == 'out_of_range':
                    details[signal_name]['key_metrics'] = {
                        'max_oor_rate': signal_details.get('max_oor_rate', 0.0),
                        'mean_oor_rate': signal_details.get('mean_oor_rate', 0.0),
                        'num_unseen_features': len(signal_details.get('unseen_features', []))
                    }
        
        return details
