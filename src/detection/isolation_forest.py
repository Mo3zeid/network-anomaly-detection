"""
Isolation Forest anomaly detection.
A simple but powerful ML algorithm for detecting outliers.
"""
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from sklearn.ensemble import IsolationForest
import joblib

from ..utils.config import ISOLATION_FOREST_CONTAMINATION, MODELS_DIR
from ..utils.logger import get_logger

logger = get_logger(__name__)


class IsolationForestDetector:
    """
    Anomaly detection using Isolation Forest algorithm.
    
    Isolation Forest works by randomly selecting features and splitting
    the data. Anomalies are isolated faster than normal points because
    they are different from the majority of the data.
    """
    
    def __init__(
        self, 
        contamination: float = None,
        n_estimators: int = 100,
        random_state: int = 42
    ):
        """
        Initialize the Isolation Forest detector.
        
        Args:
            contamination: Expected proportion of anomalies (0.0 to 0.5)
            n_estimators: Number of trees in the forest
            random_state: Random seed for reproducibility
        """
        self.contamination = contamination or ISOLATION_FOREST_CONTAMINATION
        self.n_estimators = n_estimators
        self.random_state = random_state
        
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1  # Use all available cores
        )
        self.is_fitted = False
    
    def fit(self, X: np.ndarray):
        """
        Fit the model on normal traffic data.
        
        For best results, train only on normal (benign) traffic.
        The model will learn what "normal" looks like and flag
        anything different as anomalous.
        
        Args:
            X: Training data (ideally only normal traffic)
        """
        logger.info(f"Training Isolation Forest on {len(X)} samples...")
        logger.info(f"Parameters: n_estimators={self.n_estimators}, contamination={self.contamination}")
        
        self.model.fit(X)
        self.is_fitted = True
        
        logger.info("Isolation Forest training complete")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict if samples are anomalous.
        
        Args:
            X: Data to analyze
            
        Returns:
            Array of predictions: 1 = normal, -1 = anomaly
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.model.predict(X)
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores for samples.
        
        Lower scores indicate more anomalous samples.
        
        Args:
            X: Data to analyze
            
        Returns:
            Array of anomaly scores (lower = more anomalous)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")
        
        return self.model.score_samples(X)
    
    def detect(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies and return boolean array with scores.
        
        Args:
            X: Data to analyze
            
        Returns:
            Tuple of (is_anomaly array, scores array)
        """
        predictions = self.predict(X)
        scores = self.score_samples(X)
        
        # Convert: 1 -> False (normal), -1 -> True (anomaly)
        is_anomaly = predictions == -1
        
        # Invert scores so higher = more anomalous
        normalized_scores = -scores
        
        return is_anomaly, normalized_scores
    
    def detect_single(self, x: np.ndarray) -> dict:
        """
        Detect if a single sample is anomalous.
        
        Args:
            x: Single sample (1D array)
            
        Returns:
            Dictionary with detection result
        """
        X = x.reshape(1, -1)
        is_anomaly, scores = self.detect(X)
        
        return {
            'is_anomaly': bool(is_anomaly[0]),
            'score': float(scores[0]),
            'method': 'isolation_forest',
            'threshold': self.contamination
        }
    
    def save(self, filepath: Optional[Path] = None):
        """
        Save the model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if filepath is None:
            filepath = MODELS_DIR / "isolation_forest.joblib"
        
        joblib.dump({
            'model': self.model,
            'contamination': self.contamination,
            'n_estimators': self.n_estimators,
            'is_fitted': self.is_fitted
        }, filepath)
        
        logger.info(f"Isolation Forest model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: Optional[Path] = None) -> 'IsolationForestDetector':
        """
        Load a model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded IsolationForestDetector instance
        """
        if filepath is None:
            filepath = MODELS_DIR / "isolation_forest.joblib"
        
        data = joblib.load(filepath)
        
        detector = cls(
            contamination=data['contamination'],
            n_estimators=data['n_estimators']
        )
        detector.model = data['model']
        detector.is_fitted = data['is_fitted']
        
        logger.info(f"Isolation Forest model loaded from {filepath}")
        return detector


def train_isolation_forest(
    X_train: np.ndarray,
    contamination: float = None,
    save_model: bool = True
) -> IsolationForestDetector:
    """
    Train an Isolation Forest model and optionally save it.
    
    Args:
        X_train: Training data (normal traffic)
        contamination: Expected anomaly proportion
        save_model: Whether to save the model
        
    Returns:
        Trained IsolationForestDetector
    """
    detector = IsolationForestDetector(contamination=contamination)
    detector.fit(X_train)
    
    if save_model:
        detector.save()
    
    return detector
