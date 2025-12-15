"""
Statistical anomaly detection methods.
Simple but effective techniques that don't require ML.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from dataclasses import dataclass

from ..utils.config import STATISTICAL_ZSCORE_THRESHOLD
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AnomalyResult:
    """Result of anomaly detection."""
    is_anomaly: bool
    score: float
    method: str
    details: dict


class StatisticalDetector:
    """
    Statistical anomaly detection using Z-score and IQR methods.
    These methods are simple but effective for detecting outliers.
    """
    
    def __init__(self, zscore_threshold: float = None):
        """
        Initialize the statistical detector.
        
        Args:
            zscore_threshold: Threshold for Z-score method (default: 3.0)
        """
        self.zscore_threshold = zscore_threshold or STATISTICAL_ZSCORE_THRESHOLD
        self.mean = None
        self.std = None
        self.q1 = None
        self.q3 = None
        self.iqr = None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray):
        """
        Fit the detector on normal traffic data.
        
        Args:
            X: Training data (normal traffic)
        """
        logger.info(f"Fitting statistical detector on {len(X)} samples")
        
        # Calculate statistics for Z-score
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        
        # Handle zero standard deviation
        self.std = np.where(self.std == 0, 1, self.std)
        
        # Calculate statistics for IQR
        self.q1 = np.percentile(X, 25, axis=0)
        self.q3 = np.percentile(X, 75, axis=0)
        self.iqr = self.q3 - self.q1
        
        # Handle zero IQR
        self.iqr = np.where(self.iqr == 0, 1, self.iqr)
        
        self.is_fitted = True
        logger.info("Statistical detector fitted successfully")
    
    def zscore_detect(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using Z-score method.
        
        Args:
            X: Data to analyze
            
        Returns:
            Tuple of (is_anomaly array, scores array)
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before detection")
        
        # Calculate Z-scores for each feature
        z_scores = np.abs((X - self.mean) / self.std)
        
        # Max Z-score across features for each sample
        max_z_scores = np.max(z_scores, axis=1)
        
        # Anomaly if max Z-score exceeds threshold
        is_anomaly = max_z_scores > self.zscore_threshold
        
        return is_anomaly, max_z_scores
    
    def iqr_detect(self, X: np.ndarray, multiplier: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using IQR (Interquartile Range) method.
        
        Args:
            X: Data to analyze
            multiplier: IQR multiplier for bounds (default: 1.5)
            
        Returns:
            Tuple of (is_anomaly array, scores array)
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before detection")
        
        # Calculate bounds
        lower_bound = self.q1 - multiplier * self.iqr
        upper_bound = self.q3 + multiplier * self.iqr
        
        # Check if values are outside bounds
        below = X < lower_bound
        above = X > upper_bound
        
        # Calculate how far outside bounds (normalized by IQR)
        below_score = np.abs((X - lower_bound) / self.iqr) * below
        above_score = np.abs((X - upper_bound) / self.iqr) * above
        
        # Max deviation score across features
        scores = np.max(below_score + above_score, axis=1)
        
        # Anomaly if any feature is outside bounds
        is_anomaly = np.any(below | above, axis=1)
        
        return is_anomaly, scores
    
    def detect(self, X: np.ndarray, method: str = 'zscore') -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using specified method.
        
        Args:
            X: Data to analyze
            method: 'zscore' or 'iqr'
            
        Returns:
            Tuple of (is_anomaly array, scores array)
        """
        if method == 'zscore':
            return self.zscore_detect(X)
        elif method == 'iqr':
            return self.iqr_detect(X)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'zscore' or 'iqr'")
    
    def detect_single(self, x: np.ndarray, method: str = 'zscore') -> AnomalyResult:
        """
        Detect if a single sample is anomalous.
        
        Args:
            x: Single sample (1D array)
            method: 'zscore' or 'iqr'
            
        Returns:
            AnomalyResult with detection details
        """
        # Reshape to 2D
        X = x.reshape(1, -1)
        
        is_anomaly, scores = self.detect(X, method)
        
        return AnomalyResult(
            is_anomaly=bool(is_anomaly[0]),
            score=float(scores[0]),
            method=method,
            details={
                'threshold': self.zscore_threshold if method == 'zscore' else 1.5
            }
        )


def calculate_statistics(X: np.ndarray) -> dict:
    """
    Calculate basic statistics for the data.
    
    Args:
        X: Data array
        
    Returns:
        Dictionary with statistics
    """
    return {
        'mean': np.mean(X, axis=0).tolist(),
        'std': np.std(X, axis=0).tolist(),
        'min': np.min(X, axis=0).tolist(),
        'max': np.max(X, axis=0).tolist(),
        'median': np.median(X, axis=0).tolist(),
        'q1': np.percentile(X, 25, axis=0).tolist(),
        'q3': np.percentile(X, 75, axis=0).tolist(),
    }
