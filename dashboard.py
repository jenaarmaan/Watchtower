"""
Dashboard Module

Simple dashboard for visualizing risk trends and alerts.
Uses matplotlib for static plots (can be extended with Streamlit).
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os


class WatchtowerDashboard:
    """
    Dashboard for visualizing Watchtower outputs.
    """
    
    def __init__(self, output_dir: str = 'dashboard_output'):
        """
        Initialize dashboard.
        
        Args:
            output_dir: Directory to save dashboard outputs
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_risk_trend_plot(self, 
                                 risk_history: pd.DataFrame,
                                 baseline_score: Optional[float] = None,
                                 last_stable: Optional[Dict] = None,
                                 save_path: Optional[str] = None) -> str:
        """
        Generate risk trend visualization.
        
        Args:
            risk_history: DataFrame with risk history
            baseline_score: Baseline risk score
            last_stable: Last stable checkpoint info
            save_path: Path to save plot
            
        Returns:
            Path to saved plot
        """
        if risk_history.empty:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Convert timestamp to datetime if needed
        if 'timestamp' in risk_history.columns:
            timestamps = pd.to_datetime(risk_history['timestamp'])
        else:
            timestamps = risk_history.index
        
        risk_scores = risk_history['risk_score'].values
        
        # Plot risk score
        ax.plot(timestamps, risk_scores, 'b-', linewidth=2, label='Risk Score', marker='o', markersize=4)
        
        # Add threshold lines
        ax.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='High Risk Threshold (70)')
        ax.axhline(y=40, color='orange', linestyle='--', alpha=0.5, label='Medium Risk Threshold (40)')
        
        # Add baseline line
        if baseline_score is not None:
            ax.axhline(y=baseline_score, color='g', linestyle=':', alpha=0.7, label=f'Baseline ({baseline_score:.1f})')
        
        # Highlight last stable checkpoint
        if last_stable and 'timestamp' in last_stable:
            stable_time = pd.to_datetime(last_stable['timestamp'])
            stable_score = last_stable.get('risk_score', baseline_score)
            ax.plot(stable_time, stable_score, 'go', markersize=10, label='Last Stable Checkpoint')
        
        # Color code by risk level
        if 'risk_level' in risk_history.columns:
            for level, color in [('high', 'red'), ('medium', 'orange'), ('low', 'green')]:
                mask = risk_history['risk_level'] == level
                if mask.any():
                    ax.scatter(timestamps[mask], risk_scores[mask], 
                             c=color, alpha=0.3, s=50, label=f'{level.capitalize()} Risk')
        
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Risk Score (0-100)', fontsize=12)
        ax.set_title('Model Risk Trend Over Time', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'risk_trend.png')
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def generate_signal_breakdown_plot(self,
                                      risk_assessment: Dict,
                                      save_path: Optional[str] = None) -> str:
        """
        Generate signal contribution breakdown visualization.
        
        Args:
            risk_assessment: Risk assessment dictionary
            save_path: Path to save plot
            
        Returns:
            Path to saved plot
        """
        signal_breakdown = risk_assessment.get('signal_breakdown', {})
        
        if not signal_breakdown:
            return None
        
        # Prepare data
        signal_names = []
        contributions = []
        colors = []
        
        color_map = {
            'feature_drift': '#FF6B6B',
            'prediction_drift': '#4ECDC4',
            'confidence_collapse': '#FFE66D',
            'out_of_range': '#95E1D3'
        }
        
        for name, data in signal_breakdown.items():
            signal_names.append(name.replace('_', ' ').title())
            contributions.append(data['contribution'])
            colors.append(color_map.get(name, '#95A5A6'))
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Bar chart
        bars = ax1.barh(signal_names, contributions, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Contribution to Risk Score', fontsize=11)
        ax1.set_title('Signal Contributions to Risk Score', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, contrib) in enumerate(zip(bars, contributions)):
            ax1.text(contrib + 1, i, f'{contrib:.1f}', 
                    va='center', fontsize=10, fontweight='bold')
        
        # Pie chart
        ax2.pie(contributions, labels=signal_names, colors=colors, autopct='%1.1f%%',
               startangle=90, textprops={'fontsize': 10})
        ax2.set_title('Signal Contribution Distribution', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'signal_breakdown.png')
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def generate_html_report(self,
                            risk_assessment: Dict,
                            signals: Dict,
                            alert: Optional[Dict],
                            risk_history: pd.DataFrame,
                            system_status: Dict,
                            output_path: Optional[str] = None) -> str:
        """
        Generate HTML dashboard report.
        
        Args:
            risk_assessment: Current risk assessment
            signals: Signal computation results
            alert: Current alert (if any)
            risk_history: Risk history DataFrame
            system_status: System status dictionary
            output_path: Path to save HTML file
            
        Returns:
            Path to saved HTML file
        """
        if output_path is None:
            output_path = os.path.join(self.output_dir, 'watchtower_dashboard.html')
        
        # Generate plots
        trend_plot = self.generate_risk_trend_plot(
            risk_history,
            baseline_score=risk_assessment.get('baseline_risk_score'),
            last_stable=risk_assessment.get('last_stable_checkpoint')
        )
        
        signal_plot = self.generate_signal_breakdown_plot(risk_assessment)
        
        # Get relative paths for HTML
        trend_plot_rel = os.path.basename(trend_plot) if trend_plot else None
        signal_plot_rel = os.path.basename(signal_plot) if signal_plot else None
        
        # Build HTML
        html_content = self._build_html_content(
            risk_assessment, signals, alert, system_status,
            trend_plot_rel, signal_plot_rel
        )
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        return output_path
    
    def _build_html_content(self,
                           risk_assessment: Dict,
                           signals: Dict,
                           alert: Optional[Dict],
                           system_status: Dict,
                           trend_plot: Optional[str],
                           signal_plot: Optional[str]) -> str:
        """Build HTML content for dashboard."""
        
        risk_score = risk_assessment.get('risk_score', 0.0)
        risk_level = risk_assessment.get('risk_level', 'unknown')
        trend = risk_assessment.get('trend', 'unknown')
        
        # Risk level color
        risk_colors = {
            'high': '#FF4444',
            'medium': '#FF8800',
            'low': '#44AA44',
            'unknown': '#888888'
        }
        risk_color = risk_colors.get(risk_level, '#888888')
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Watchtower Dashboard</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        .risk-score {{
            font-size: 48px;
            font-weight: bold;
            color: {risk_color};
            text-align: center;
            margin: 20px 0;
        }}
        .risk-level {{
            text-align: center;
            font-size: 24px;
            color: {risk_color};
            font-weight: bold;
            margin-bottom: 30px;
        }}
        .alert-box {{
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }}
        .alert-critical {{
            background-color: #f8d7da;
            border-left-color: #dc3545;
        }}
        .alert-high {{
            background-color: #fff3cd;
            border-left-color: #ffc107;
        }}
        .section {{
            margin: 30px 0;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 4px;
        }}
        .section h2 {{
            color: #2c3e50;
            margin-top: 0;
        }}
        .signal-item {{
            margin: 10px 0;
            padding: 10px;
            background: white;
            border-radius: 4px;
            border-left: 3px solid #3498db;
        }}
        .recommendation {{
            margin: 8px 0;
            padding: 10px;
            background: white;
            border-radius: 4px;
            border-left: 3px solid #27ae60;
        }}
        .plot-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .plot-container img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🛡️ Watchtower: Model Risk Dashboard</h1>
        <p class="timestamp">Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="risk-score">{risk_score:.1f}/100</div>
        <div class="risk-level">Risk Level: {risk_level.upper()} | Trend: {trend.upper()}</div>
        
        {self._generate_alert_html(alert) if alert else ''}
        
        <div class="section">
            <h2>📊 Risk Trend</h2>
            {f'<div class="plot-container"><img src="{trend_plot}" alt="Risk Trend"></div>' if trend_plot else '<p>No trend data available</p>'}
        </div>
        
        <div class="section">
            <h2>🔍 Signal Breakdown</h2>
            {f'<div class="plot-container"><img src="{signal_plot}" alt="Signal Breakdown"></div>' if signal_plot else '<p>No signal data available</p>'}
            {self._generate_signal_details_html(risk_assessment)}
        </div>
        
        <div class="section">
            <h2>📈 System Status</h2>
            {self._generate_status_html(system_status)}
        </div>
        
        {self._generate_recommendations_html(alert) if alert else ''}
        
        <div class="section">
            <h2>⚠️ Scientific Assumptions & Limitations</h2>
            <ul>
                <li><strong>Early Warning System:</strong> Watchtower detects risk signals, not perfect accuracy. 
                It is designed as a preventive radar, not a diagnostic oracle.</li>
                <li><strong>Label-Free:</strong> System operates without ground-truth labels, using only prediction logs and metadata.</li>
                <li><strong>Statistical Signals:</strong> Risk scores are based on distribution shifts and statistical anomalies, 
                which may not always correlate with accuracy degradation.</li>
                <li><strong>Baseline Dependency:</strong> System requires a stable baseline period. 
                Performance depends on baseline quality and representativeness.</li>
                <li><strong>False Positives:</strong> Some alerts may be false positives due to natural data variation 
                or temporary system changes.</li>
                <li><strong>Black Box Model:</strong> System treats the model as a black box and cannot diagnose 
                internal model issues directly.</li>
            </ul>
        </div>
    </div>
</body>
</html>
"""
        return html
    
    def _generate_alert_html(self, alert: Dict) -> str:
        """Generate HTML for alert box."""
        severity = alert.get('severity', 'medium')
        message = alert.get('message', '')
        reason_labels = alert.get('reason_labels', [])
        
        alert_class = f'alert-{severity}'
        reasons = ' + '.join(reason_labels) if reason_labels else 'multiple factors'
        
        return f"""
        <div class="alert-box {alert_class}">
            <h3>⚠️ Alert: {severity.upper()}</h3>
            <p><strong>{message}</strong></p>
            <p><strong>Contributing Factors:</strong> {reasons}</p>
        </div>
        """
    
    def _generate_signal_details_html(self, risk_assessment: Dict) -> str:
        """Generate HTML for signal details table."""
        signal_breakdown = risk_assessment.get('signal_breakdown', {})
        
        if not signal_breakdown:
            return '<p>No signal details available</p>'
        
        rows = []
        for name, data in signal_breakdown.items():
            rows.append(f"""
            <tr>
                <td>{name.replace('_', ' ').title()}</td>
                <td>{data['score']:.3f}</td>
                <td>{data['contribution']:.1f}</td>
                <td>{data['weight']*100:.1f}%</td>
            </tr>
            """)
        
        return f"""
        <table>
            <thead>
                <tr>
                    <th>Signal</th>
                    <th>Score (0-1)</th>
                    <th>Contribution</th>
                    <th>Weight</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
        """
    
    def _generate_status_html(self, system_status: Dict) -> str:
        """Generate HTML for system status."""
        risk_summary = system_status.get('risk_summary', {})
        
        return f"""
        <table>
            <tr>
                <td><strong>Baseline Initialized:</strong></td>
                <td>{'✅ Yes' if system_status.get('baseline_initialized') else '❌ No'}</td>
            </tr>
            <tr>
                <td><strong>Current Risk Score:</strong></td>
                <td>{risk_summary.get('current_risk_score', 'N/A'):.1f}</td>
            </tr>
            <tr>
                <td><strong>Average Risk Score:</strong></td>
                <td>{risk_summary.get('average_risk_score', 'N/A'):.1f}</td>
            </tr>
            <tr>
                <td><strong>Total Alerts:</strong></td>
                <td>{system_status.get('total_alerts', 0)}</td>
            </tr>
            <tr>
                <td><strong>Recent Alerts (24h):</strong></td>
                <td>{system_status.get('recent_alerts', 0)}</td>
            </tr>
        </table>
        """
    
    def _generate_recommendations_html(self, alert: Dict) -> str:
        """Generate HTML for recommendations."""
        recommendations = alert.get('recommendations', [])
        
        if not recommendations:
            return ''
        
        rec_items = ''.join([
            f'<div class="recommendation">{i+1}. {rec}</div>'
            for i, rec in enumerate(recommendations)
        ])
        
        return f"""
        <div class="section">
            <h2>💡 Recommended Actions</h2>
            {rec_items}
        </div>
        """


import streamlit as st

def render(results):
    """
    Public dashboard renderer for Watchtower.
    """
    # Defensive check: ensure results is a dict
    if results is None:
        st.error("⚠️ Watchtower returned no results.")
        return
    
    if not isinstance(results, dict):
        st.error(f"⚠️ Watchtower returned invalid results type: {type(results)}")
        return
    
    # Check for errors in results
    if 'error' in results:
        st.error(f"❌ Error: {results['error']}")
        return
    
    st.subheader("📊 Model Health Overview")

    # Extract risk score from nested structure
    risk_assessment = results.get("risk_assessment", {})
    risk_score = risk_assessment.get("risk_score", None)
    risk_level = risk_assessment.get("risk_level", "unknown")
    
    if risk_score is not None:
        # Color code based on risk level
        if risk_level == "high":
            delta_color = "inverse"
        elif risk_level == "medium":
            delta_color = "normal"
        else:
            delta_color = "off"
        
        st.metric("Model Risk Score", f"{risk_score:.1f} / 100", 
                 delta=f"{risk_level.upper()} RISK", delta_color=delta_color)
    else:
        st.warning("Risk score not available")

    # Alerts
    alert = results.get("alert", None)
    if alert:
        st.subheader("🚨 Active Alerts")
        severity = alert.get("severity", "medium")
        message = alert.get("message", "Alert detected")
        
        if severity == "high" or severity == "critical":
            st.error(f"**{severity.upper()}**: {message}")
        else:
            st.warning(f"**{severity.upper()}**: {message}")
        
        # Show recommendations if available
        recommendations = alert.get("recommendations", [])
        if recommendations:
            st.subheader("💡 Recommended Actions")
            for i, rec in enumerate(recommendations, 1):
                st.info(f"{i}. {rec}")
    else:
        st.success("✅ No critical alerts detected.")

    # Signal details (drift and confidence)
    signals = results.get("signals", {})
    signal_data = signals.get("signals", {})
    
    if signal_data:
        st.subheader("📈 Drift Signals")
        
        # Feature drift
        if "feature_drift" in signal_data:
            drift_info = signal_data["feature_drift"]
            drift_score = drift_info.get("score", 0.0)
            st.metric("Feature Drift Score", f"{drift_score:.3f}", 
                     help="Measures distribution shift in input features")
            if "details" in drift_info:
                with st.expander("Feature Drift Details"):
                    st.json(drift_info["details"])
        
        # Prediction drift
        if "prediction_drift" in signal_data:
            pred_drift = signal_data["prediction_drift"]
            pred_score = pred_drift.get("score", 0.0)
            st.metric("Prediction Drift Score", f"{pred_score:.3f}",
                     help="Measures shift in prediction distribution")
            if "details" in pred_drift:
                with st.expander("Prediction Drift Details"):
                    st.json(pred_drift["details"])

        # Confidence signals
        st.subheader("🔍 Confidence Signals")
        
        if "confidence_collapse" in signal_data:
            conf_info = signal_data["confidence_collapse"]
            conf_score = conf_info.get("score", 0.0)
            st.metric("Confidence Collapse Score", f"{conf_score:.3f}",
                     help="Measures drop in model confidence")
            if "details" in conf_info:
                with st.expander("Confidence Details"):
                    st.json(conf_info["details"])
        
        # Out of range
        if "out_of_range" in signal_data:
            oor_info = signal_data["out_of_range"]
            oor_score = oor_info.get("score", 0.0)
            st.metric("Out-of-Range Score", f"{oor_score:.3f}",
                     help="Measures rate of unseen input values")
            if "details" in oor_info:
                with st.expander("Out-of-Range Details"):
                    st.json(oor_info["details"])

    # Signal breakdown from risk assessment
    signal_breakdown = risk_assessment.get("signal_breakdown", {})
    if signal_breakdown:
        st.subheader("📊 Signal Contribution Breakdown")
        breakdown_data = {
            name.replace("_", " ").title(): data.get("contribution", 0.0)
            for name, data in signal_breakdown.items()
        }
        st.bar_chart(breakdown_data)