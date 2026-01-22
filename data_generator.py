"""
Sample Data Generator

Generates synthetic prediction logs for demonstration purposes.
Simulates various decay scenarios.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict


class PredictionLogGenerator:
    """
    Generates synthetic prediction logs for testing and demonstration.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        self.seed = seed
    
    def generate_baseline_data(self,
                              n_samples: int = 1000,
                              n_features: int = 5,
                              start_date: Optional[datetime] = None,
                              days: int = 7) -> pd.DataFrame:
        """
        Generate baseline data (stable, healthy model).
        
        Args:
            n_samples: Number of samples
            n_features: Number of feature columns
            start_date: Start date (default: 7 days ago)
            days: Number of days to generate
            
        Returns:
            DataFrame with prediction logs
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=days)
        
        # Generate timestamps
        timestamps = pd.date_range(
            start=start_date,
            periods=n_samples,
            freq=f'{days*24*60/n_samples:.0f}min'
        )
        
        # Generate features (normal distributions)
        feature_data = {}
        for i in range(n_features):
            mean = np.random.uniform(-2, 2)
            std = np.random.uniform(0.5, 2.0)
            feature_data[f'feature_{i+1}'] = np.random.normal(mean, std, n_samples)
        
        # Generate predictions (categorical, balanced)
        n_classes = 3
        predictions = np.random.choice(n_classes, n_samples, p=[0.4, 0.35, 0.25])
        
        # Generate confidence scores (high, stable)
        confidence = np.random.beta(8, 2, n_samples)  # Skewed toward high confidence
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'prediction': predictions,
            'confidence': confidence,
            'model_version': 'v1.0',
            **feature_data
        })
        
        return df
    
    def generate_decay_scenario(self,
                               baseline_df: pd.DataFrame,
                               scenario: str = 'data_drift',
                               severity: float = 0.5,
                               n_samples: int = 200) -> pd.DataFrame:
        """
        Generate data with decay scenario.
        
        Args:
            baseline_df: Baseline DataFrame
            scenario: Type of decay ('data_drift', 'confidence_collapse', 'out_of_range', 'prediction_drift')
            severity: Severity level (0.0 to 1.0)
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame with decay scenario
        """
        last_timestamp = pd.to_datetime(baseline_df['timestamp']).max()
        timestamps = pd.date_range(
            start=last_timestamp + timedelta(hours=1),
            periods=n_samples,
            freq='30min'
        )
        
        n_features = len([c for c in baseline_df.columns if c.startswith('feature_')])
        
        if scenario == 'data_drift':
            # Feature distribution shift
            feature_data = {}
            for i in range(n_features):
                col = f'feature_{i+1}'
                baseline_mean = baseline_df[col].mean()
                baseline_std = baseline_df[col].std()
                
                # Shift mean
                shift = severity * baseline_std * 2
                new_mean = baseline_mean + shift
                
                feature_data[col] = np.random.normal(new_mean, baseline_std, n_samples)
            
            predictions = np.random.choice(3, n_samples, p=[0.4, 0.35, 0.25])
            confidence = np.random.beta(8, 2, n_samples)
        
        elif scenario == 'confidence_collapse':
            # Confidence drops significantly
            feature_data = {}
            for i in range(n_features):
                col = f'feature_{i+1}'
                baseline_mean = baseline_df[col].mean()
                baseline_std = baseline_df[col].std()
                feature_data[col] = np.random.normal(baseline_mean, baseline_std, n_samples)
            
            predictions = np.random.choice(3, n_samples, p=[0.4, 0.35, 0.25])
            # Lower confidence (beta with lower alpha)
            confidence = np.random.beta(2, 3, n_samples)  # Lower confidence
        
        elif scenario == 'out_of_range':
            # Out-of-range feature values
            feature_data = {}
            for i in range(n_features):
                col = f'feature_{i+1}'
                baseline_min = baseline_df[col].min()
                baseline_max = baseline_df[col].max()
                baseline_range = baseline_max - baseline_min
                
                # Generate values outside range
                oor_rate = severity
                n_oor = int(n_samples * oor_rate)
                
                # Out-of-range values
                oor_values = np.random.choice([
                    np.random.uniform(baseline_min - baseline_range, baseline_min),
                    np.random.uniform(baseline_max, baseline_max + baseline_range)
                ], n_oor)
                
                # In-range values
                in_range_values = np.random.uniform(baseline_min, baseline_max, n_samples - n_oor)
                
                feature_data[col] = np.concatenate([oor_values, in_range_values])
                np.random.shuffle(feature_data[col])
            
            predictions = np.random.choice(3, n_samples, p=[0.4, 0.35, 0.25])
            confidence = np.random.beta(8, 2, n_samples)
        
        elif scenario == 'prediction_drift':
            # Prediction distribution shifts
            feature_data = {}
            for i in range(n_features):
                col = f'feature_{i+1}'
                baseline_mean = baseline_df[col].mean()
                baseline_std = baseline_df[col].std()
                feature_data[col] = np.random.normal(baseline_mean, baseline_std, n_samples)
            
            # Shift prediction distribution
            shift = severity
            new_probs = [0.4 - shift, 0.35, 0.25 + shift]
            new_probs = np.clip(new_probs, 0.1, 0.9)
            new_probs = new_probs / new_probs.sum()
            
            predictions = np.random.choice(3, n_samples, p=new_probs)
            confidence = np.random.beta(8, 2, n_samples)
        
        else:
            # Default: no decay
            feature_data = {}
            for i in range(n_features):
                col = f'feature_{i+1}'
                baseline_mean = baseline_df[col].mean()
                baseline_std = baseline_df[col].std()
                feature_data[col] = np.random.normal(baseline_mean, baseline_std, n_samples)
            
            predictions = np.random.choice(3, n_samples, p=[0.4, 0.35, 0.25])
            confidence = np.random.beta(8, 2, n_samples)
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'prediction': predictions,
            'confidence': confidence,
            'model_version': 'v1.0',
            **feature_data
        })
        
        return df
    
    def generate_combined_data(self,
                               baseline_df: pd.DataFrame,
                               scenarios: Dict[str, float],
                               n_samples: int = 200) -> pd.DataFrame:
        """
        Generate data with multiple decay scenarios combined.
        
        Args:
            baseline_df: Baseline DataFrame
            scenarios: Dictionary of {scenario: severity}
            n_samples: Number of samples
            
        Returns:
            DataFrame with combined decay
        """
        # Start with baseline-like data
        last_timestamp = pd.to_datetime(baseline_df['timestamp']).max()
        timestamps = pd.date_range(
            start=last_timestamp + timedelta(hours=1),
            periods=n_samples,
            freq='30min'
        )
        
        n_features = len([c for c in baseline_df.columns if c.startswith('feature_')])
        
        feature_data = {}
        for i in range(n_features):
            col = f'feature_{i+1}'
            baseline_mean = baseline_df[col].mean()
            baseline_std = baseline_df[col].std()
            baseline_min = baseline_df[col].min()
            baseline_max = baseline_df[col].max()
            baseline_range = baseline_max - baseline_min
            
            values = np.random.normal(baseline_mean, baseline_std, n_samples)
            
            # Apply data drift
            if 'data_drift' in scenarios:
                shift = scenarios['data_drift'] * baseline_std * 2
                values = values + shift
            
            # Apply out-of-range
            if 'out_of_range' in scenarios:
                oor_rate = scenarios['out_of_range']
                n_oor = int(n_samples * oor_rate)
                oor_indices = np.random.choice(n_samples, n_oor, replace=False)
                oor_values = np.random.choice([
                    np.random.uniform(baseline_min - baseline_range, baseline_min),
                    np.random.uniform(baseline_max, baseline_max + baseline_range)
                ], n_oor)
                values[oor_indices] = oor_values
            
            feature_data[col] = values
        
        # Predictions
        if 'prediction_drift' in scenarios:
            shift = scenarios['prediction_drift']
            new_probs = [0.4 - shift, 0.35, 0.25 + shift]
            new_probs = np.clip(new_probs, 0.1, 0.9)
            new_probs = new_probs / new_probs.sum()
            predictions = np.random.choice(3, n_samples, p=new_probs)
        else:
            predictions = np.random.choice(3, n_samples, p=[0.4, 0.35, 0.25])
        
        # Confidence
        if 'confidence_collapse' in scenarios:
            collapse_severity = scenarios['confidence_collapse']
            # Interpolate between high and low confidence
            alpha = 8 - collapse_severity * 6
            beta = 2 + collapse_severity * 1
            confidence = np.random.beta(alpha, beta, n_samples)
        else:
            confidence = np.random.beta(8, 2, n_samples)
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'prediction': predictions,
            'confidence': confidence,
            'model_version': 'v1.0',
            **feature_data
        })
        
        return df


def generate_data(n: int, drift_level: float = 0.3) -> pd.DataFrame:
    """
    Generate synthetic prediction logs with optional drift.
    
    Args:
        n: Total number of samples to generate
        drift_level: Drift intensity (0.0 to 1.0). Higher values = more drift.
        
    Returns:
        DataFrame with prediction logs including baseline and drift data
    """
    generator = PredictionLogGenerator(seed=42)
    
    # Generate baseline data (60% of samples)
    n_baseline = int(n * 0.6)
    baseline_df = generator.generate_baseline_data(
        n_samples=n_baseline,
        n_features=5,
        days=7
    )
    
    # Generate decay scenario data (40% of samples) if drift_level > 0
    if drift_level > 0:
        n_decay = n - n_baseline
        # Use combined scenarios based on drift_level
        scenarios = {
            'data_drift': drift_level * 0.4,
            'confidence_collapse': drift_level * 0.3,
            'out_of_range': drift_level * 0.2,
            'prediction_drift': drift_level * 0.1
        }
        decay_df = generator.generate_combined_data(
            baseline_df=baseline_df,
            scenarios=scenarios,
            n_samples=n_decay
        )
        
        # Combine baseline and decay data
        combined_df = pd.concat([baseline_df, decay_df], ignore_index=True)
        return combined_df
    else:
        # No drift, return just baseline
        return baseline_df