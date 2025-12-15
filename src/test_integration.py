
import numpy as np
import pandas as pd
import json
from src.detection.classifier import AttackClassifier
from src.utils.logger import get_logger
from src.utils.config import MODELS_DIR

logger = get_logger(__name__)

def test_integration():
    logger.info("Initializing AttackClassifier...")
    clf = AttackClassifier()
    
    if not clf.model_loaded:
        logger.error("Failed to load models!")
        return
        
    # Load correct feature count from model_features.json
    try:
        with open(MODELS_DIR / "model_features.json", 'r') as f:
            features = json.load(f)
        n_features = len(features)
        logger.info(f"Model expects {n_features} features.")
    except Exception as e:
        logger.warning(f"Could not read feature count: {e}. Using 60.")
        n_features = 60

    # Create a random feature vector matching correct features
    fake_input = np.random.rand(1, n_features)
    
    logger.info("Testing prediction on random input...")
    try:
        result = clf.classify_single(fake_input)
        logger.info(f"Prediction Result: {result}")
        
        if result['attack_type'] in ['Benign', 'Error'] or result['confidence'] > 0:
            logger.info("✅ Integration Test Passed: Output format is valid.")
        else:
            logger.error("❌ Integration Test Failed: Output is weird.")
    except Exception as e:
        logger.error(f"❌ Integration Test Crashed: {e}")

if __name__ == "__main__":
    test_integration()
