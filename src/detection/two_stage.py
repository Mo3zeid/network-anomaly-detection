
"""
Two-Stage XGBoost Detector (Inference)
Implements the detection pipeline: 
Stage 1 (Binary) -> if Attack -> Stage 2 (Attack Type)
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import json
from pathlib import Path
from typing import Dict, Any, Union

from ..utils.logger import get_logger
from ..utils.config import MODELS_DIR

logger = get_logger(__name__)

class TwoStageDetector:
    def __init__(self):
        self.model_bin = None
        self.model_multi = None
        self.label_encoder = None        # For global usage if needed
        self.label_encoder_stage2 = None # For decoding Stage 2 predictions
        self.features = None
        self.is_loaded = False
        
    def load_models(self):
        """Load all artifacts required for inference."""
        try:
            # 1. Load Feature List
            feature_path = MODELS_DIR / "model_features.json"
            if not feature_path.exists():
                raise FileNotFoundError(f"Feature list not found at {feature_path}")
            
            with open(feature_path, 'r') as f:
                self.features = json.load(f)
                
            # 2. Load Stage 1 (Binary)
            stage1_path = MODELS_DIR / "stage1_xgboost.json"
            if not stage1_path.exists():
                raise FileNotFoundError("Stage 1 model missing")
            
            self.model_bin = xgb.Booster()
            self.model_bin.load_model(stage1_path)
            
            # 3. Load Stage 2 (Multi)
            stage2_path = MODELS_DIR / "stage2_xgboost.json"
            if not stage2_path.exists():
                raise FileNotFoundError("Stage 2 model missing")
            
            self.model_multi = xgb.Booster()
            self.model_multi.load_model(stage2_path)
            
            # 4. Load Label Encoder
            le_path = MODELS_DIR / "label_encoder.joblib"
            if not le_path.exists():
                raise FileNotFoundError("Label encoder missing")
                
            self.label_encoder = joblib.load(le_path)
            
            # 5. Load Stage 2 Label Encoder (Critical for correct mapping)
            le_stage2_path = MODELS_DIR / "stage2_label_encoder.joblib"
            if le_stage2_path.exists():
                self.label_encoder_stage2 = joblib.load(le_stage2_path)
            else:
                logger.warning("Stage 2 encoder not found! Falling back to global (May be inaccurate).")
                self.label_encoder_stage2 = self.label_encoder
            
            self.is_loaded = True
            logger.info("TwoStageDetector models loaded successfully.")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.is_loaded = False
            raise

    def preprocess(self, x: Union[Dict, pd.Series, np.ndarray]) -> pd.DataFrame:
        """
        Convert input to DataFrame with correct feature order.
        """
        if self.features is None:
            raise ValueError("Models not loaded. Call load_models() first.")
            
        # Case 1: Dict
        if isinstance(x, dict):
            df = pd.DataFrame([x])
            
        # Case 2: Series
        elif isinstance(x, pd.Series):
            df = pd.DataFrame([x])
            
        # Case 3: Numpy Array (Risky - assumptions made about order)
        elif isinstance(x, np.ndarray):
            # Reshape if 1D
            if x.ndim == 1:
                x = x.reshape(1, -1)
            
            # If shape matches feature count, assign names
            if x.shape[1] == len(self.features):
                df = pd.DataFrame(x, columns=self.features)
            else:
                # If mismatch, slice or pad
                # Use generic features if we can't trust order?
                # For now, assume training order.
                df = pd.DataFrame(x)
                if x.shape[1] > len(self.features):
                    df = df.iloc[:, :len(self.features)]
                df.columns = self.features
        else:
            raise ValueError("Unsupported input type")

        # Ensure all columns exist (fill 0) and are in order
        for col in self.features:
            if col not in df.columns:
                df[col] = 0
                
        return df[self.features].astype(float)

    def detect_single(self, x) -> Dict[str, Any]:
        """
        Predict for a single sample.
        Returns detailed dict: {status: 'Benign'|'Attack', type: 'DDoS', confidence: 0.99}
        """
        if not self.is_loaded:
            self.load_models()
            
        try:
            df = self.preprocess(x)
            dmatrix = xgb.DMatrix(df)
            
            # Stage 1: Binary
            # Output is probability of Class 1 (Attack)
            prob_attack = self.model_bin.predict(dmatrix)[0]
            
            threshold = 0.5
            if prob_attack < threshold:
                return {
                    'status': 'Benign',
                    'type': 'Benign',
                    'confidence': float(1 - prob_attack),
                    'stage1_prob': float(prob_attack)
                }
            
            # Stage 2: Multi-class
            # Output is array of probabilities for each class
            prob_multi = self.model_multi.predict(dmatrix)[0] # Shape (n_classes,)
            class_idx = np.argmax(prob_multi)
            confidence = prob_multi[class_idx]
            
            # Use Stage 2 Encoder to Decode
            attack_type = self.label_encoder_stage2.inverse_transform([class_idx])[0]
            
            return {
                'status': 'Attack',
                'type': attack_type,
                'confidence': float(confidence),
                'stage1_prob': float(prob_attack),
                'all_probs': {
                    cls: float(prob) 
                    for cls, prob in zip(self.label_encoder_stage2.classes_, prob_multi)
                }
            }
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return {'status': 'Error', 'type': 'Unknown', 'confidence': 0.0}
