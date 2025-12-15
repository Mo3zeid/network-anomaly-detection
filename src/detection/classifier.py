
"""
AttackClassifier Wrapper
Wraps the modern TwoStageDetector (XGBoost) into the legacy AttackClassifier API.
This ensures drop-in compatibility with the existing system while using advanced models.
"""
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any

from ..utils.logger import get_logger
from ..utils.config import MODELS_DIR
from .two_stage import TwoStageDetector

logger = get_logger(__name__)

class AttackClassifier:
    """
    Legacy-compatible wrapper for the Two-Stage XGBoost Detector.
    Maintains the API expected by the rest of the application.
    """
    
    def __init__(self, n_estimators=100, max_depth=20, random_state=42):
        """
        Initialize the AttackClassifier.
        Args are kept for API compatibility but ignored in favor of pre-trained XGBoost models.
        """
        self.detector = TwoStageDetector()
        self.model_loaded = False
        # Attempt to load immediately
        try:
            self.load()
        except Exception as e:
            logger.warning(f"Could not load models on init: {e}")

    def load(self, filepath: Optional[Path] = None):
        """Load the underlying TwoStageDetector models."""
        try:
            self.detector.load_models()
            self.model_loaded = True
            logger.info("TwoStageDetector models loaded successfully via AttackClassifier.")
        except Exception as e:
            logger.error(f"Failed to load TwoStageDetector: {e}")
            self.model_loaded = False
            raise

    def classify_single(self, x: np.ndarray) -> dict:
        """
        Classify a single sample using the Two-Stage Detector.
        
        Args:
            x: Single sample (1D numpy array of features).
            
        Returns:
            dict: {
                'attack_type': str,
                'confidence': float,
                'top_predictions': list[dict]
            }
        """
        if not self.model_loaded:
            # Try lazy loading
            self.load()

        result = self.detector.detect_single(x)
        
        # Map TwoStageDetector output to legacy format
        attack_type = result.get('type', 'Unknown')
        confidence = result.get('confidence', 0.0)
        
        # Construct top_predictions (simulate if not available)
        top_predictions = []
        if 'all_probs' in result:
            # Sort by probability descending
            # result['all_probs'] is {class: prob}
            sorted_probs = sorted(
                result['all_probs'].items(), 
                key=lambda item: item[1], 
                reverse=True
            )
            top_predictions = [
                {'attack_type': k, 'probability': v} 
                for k, v in sorted_probs[:3]
            ]
        else:
            # Fallback if probability distribution is not available (e.g. Benign in Stage 1)
            top_predictions = [{'attack_type': attack_type, 'probability': confidence}]

        return {
            'attack_type': attack_type,
            'confidence': confidence,
            'top_predictions': top_predictions
        }

    # Legacy methods stubbed for compatibility
    def predict(self, X: np.ndarray) -> np.ndarray:
        # Not optimized for batch, but implemented via loop if needed
        # Or better yet, implement batch predict in TwoStageDetector later
        raise NotImplementedError("Batch predict not yet implemented for TwoStageDetector wrapper")

    def save(self, filepath: Optional[Path] = None):
        logger.warning("Save is disabled for TwoStageDetector wrapper (models are immutable).")
