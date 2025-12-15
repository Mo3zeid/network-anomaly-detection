"""
Training script for all models.
Run this script to train the detection models on the CICIDS2017 dataset.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_dataset, get_dataset_info
from src.data.preprocessor import DataPreprocessor, get_normal_traffic_data
from src.detection.statistical import StatisticalDetector
from src.detection.isolation_forest import IsolationForestDetector
from src.detection.classifier import AttackClassifier
from src.utils.config import MODELS_DIR, LABEL_COLUMN
from src.utils.logger import get_logger

import numpy as np
from sklearn.model_selection import train_test_split
import joblib

logger = get_logger(__name__)


def train_all_models(sample_size: int = None):
    """
    Train all detection models.
    
    Args:
        sample_size: Number of samples to use (None = all data)
    """
    logger.info("=" * 60)
    logger.info("NETWORK ANOMALY DETECTION - MODEL TRAINING")
    logger.info("=" * 60)
    
    # Step 1: Load data
    logger.info("\n[1/6] Loading dataset...")
    df = load_dataset()
    
    if sample_size and len(df) > sample_size:
        logger.info(f"Sampling {sample_size} records for faster training...")
        df = df.sample(n=sample_size, random_state=42)
    
    info = get_dataset_info(df)
    logger.info(f"Dataset: {info['total_records']} records, {info['features']} features")
    logger.info(f"Attack types: {info.get('attack_types', 'N/A')}")
    
    # Step 2: Preprocess data
    logger.info("\n[2/6] Preprocessing data...")
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.clean_data(df)
    
    # Get labels
    labels = df_clean[LABEL_COLUMN]
    y_binary = (labels != 'BENIGN').astype(int)
    
    # Select features
    df_features = preprocessor.select_features(df_clean)
    
    # Split data
    X_train_raw, X_test_raw, y_train, y_test, labels_train, labels_test = train_test_split(
        df_features, y_binary, labels,
        test_size=0.2,
        random_state=42,
        stratify=y_binary
    )
    
    # Normalize
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    
    # Save preprocessor
    preprocessor.save()
    logger.info(f"Preprocessor saved. Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Step 3: Train Statistical Detector
    logger.info("\n[3/6] Training Statistical Detector...")
    stat_detector = StatisticalDetector()
    
    # Train on normal traffic only
    normal_mask = y_train == 0
    X_normal = X_train[normal_mask]
    stat_detector.fit(X_normal)
    
    # Test
    is_anomaly, scores = stat_detector.detect(X_test, method='zscore')
    stat_accuracy = np.mean(is_anomaly == y_test)
    logger.info(f"Statistical Detector accuracy: {stat_accuracy:.2%}")
    
    # Save
    joblib.dump(stat_detector, MODELS_DIR / "statistical_detector.joblib")
    
    # Step 4: Train Isolation Forest
    logger.info("\n[4/6] Training Isolation Forest...")
    iso_detector = IsolationForestDetector(contamination=0.1)
    iso_detector.fit(X_normal)
    
    # Test
    is_anomaly, scores = iso_detector.detect(X_test)
    iso_accuracy = np.mean(is_anomaly == y_test)
    logger.info(f"Isolation Forest accuracy: {iso_accuracy:.2%}")
    
    # Save
    iso_detector.save()
    
    # Step 5: Train Attack Classifier
    logger.info("\n[5/6] Training Attack Classifier...")
    classifier = AttackClassifier()
    
    # Train on all data (including attacks)
    classifier.fit(X_train, labels_train.values)
    
    # Test
    predictions = classifier.predict(X_test)
    cls_accuracy = np.mean(predictions == classifier._map_labels(labels_test.values))
    logger.info(f"Attack Classifier accuracy: {cls_accuracy:.2%}")
    
    # Save
    classifier.save()
    
    # Step 6: Summary
    logger.info("\n[6/6] Training Complete!")
    logger.info("=" * 60)
    logger.info("MODEL PERFORMANCE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Statistical Detector:  {stat_accuracy:.2%}")
    logger.info(f"Isolation Forest:      {iso_accuracy:.2%}")
    logger.info(f"Attack Classifier:     {cls_accuracy:.2%}")
    logger.info("=" * 60)
    logger.info(f"Models saved to: {MODELS_DIR}")
    
    return {
        'statistical_accuracy': stat_accuracy,
        'isolation_forest_accuracy': iso_accuracy,
        'classifier_accuracy': cls_accuracy
    }


if __name__ == "__main__":
    # Train with limited samples for testing, remove for full training
    train_all_models(sample_size=50000)
