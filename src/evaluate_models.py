"""
Comprehensive Model Evaluation Script.
Trains models and captures all real metrics including accuracy, precision, 
recall, F1, confusion matrices, and detailed per-class statistics.

Run with: python -m src.evaluate_models
"""
import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, roc_curve
)
import joblib

from src.data.loader import load_dataset, get_dataset_info
from src.data.preprocessor import DataPreprocessor
from src.detection.statistical import StatisticalDetector
from src.detection.isolation_forest import IsolationForestDetector
from src.detection.classifier import AttackClassifier
from src.utils.config import MODELS_DIR, LABEL_COLUMN, SELECTED_FEATURES
from src.utils.logger import get_logger

logger = get_logger(__name__)


def print_separator(title=""):
    """Print a section separator."""
    print("\n" + "=" * 80)
    if title:
        print(f"  {title}")
        print("=" * 80)


def calculate_metrics(y_true, y_pred, y_scores=None, prefix=""):
    """Calculate comprehensive metrics."""
    metrics = {
        f'{prefix}accuracy': float(accuracy_score(y_true, y_pred)),
        f'{prefix}precision': float(precision_score(y_true, y_pred, zero_division=0)),
        f'{prefix}recall': float(recall_score(y_true, y_pred, zero_division=0)),
        f'{prefix}f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics[f'{prefix}confusion_matrix'] = cm.tolist()
    
    # Extract TP, TN, FP, FN for binary classification
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics[f'{prefix}true_positives'] = int(tp)
        metrics[f'{prefix}true_negatives'] = int(tn)
        metrics[f'{prefix}false_positives'] = int(fp)
        metrics[f'{prefix}false_negatives'] = int(fn)
        
        # Additional metrics
        metrics[f'{prefix}specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        metrics[f'{prefix}false_positive_rate'] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        metrics[f'{prefix}false_negative_rate'] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    
    # ROC AUC if scores available
    if y_scores is not None:
        try:
            metrics[f'{prefix}roc_auc'] = float(roc_auc_score(y_true, y_scores))
        except:
            pass
    
    return metrics


def evaluate_all_models(sample_size=None, save_results=True):
    """
    Train and evaluate all models with comprehensive metrics.
    
    Args:
        sample_size: Number of samples to use (None = all data, may take a long time)
        save_results: Whether to save results to JSON file
    """
    results = {
        'timestamp': datetime.now().isoformat(),
        'dataset': {},
        'preprocessing': {},
        'statistical_detector': {},
        'isolation_forest': {},
        'combined_detection': {},
        'attack_classifier': {},
        'feature_importance': {},
        'per_attack_metrics': {}
    }
    
    print_separator("NETWORK ANOMALY DETECTION - COMPREHENSIVE MODEL EVALUATION")
    print(f"Started at: {results['timestamp']}")
    
    # =========================================================================
    # STEP 1: LOAD DATASET
    # =========================================================================
    print_separator("STEP 1: Loading Dataset")
    
    df = load_dataset()
    original_size = len(df)
    
    if sample_size and len(df) > sample_size:
        print(f"Sampling {sample_size:,} records from {len(df):,} total...")
        df = df.sample(n=sample_size, random_state=42)
    
    info = get_dataset_info(df)
    
    results['dataset'] = {
        'original_records': original_size,
        'used_records': len(df),
        'features_available': len(df.columns),
        'features_selected': len(SELECTED_FEATURES),
        'label_distribution': info.get('label_distribution', {}),
        'attack_types_count': info.get('attack_types', 0),
        'benign_count': info.get('benign_count', 0),
        'attack_count': info.get('attack_count', 0),
        'attack_ratio': info.get('attack_count', 0) / len(df) if len(df) > 0 else 0
    }
    
    print(f"Total records: {len(df):,}")
    print(f"Features: {len(SELECTED_FEATURES)}")
    print(f"Label distribution:")
    for label, count in sorted(info.get('label_distribution', {}).items(), key=lambda x: -x[1])[:10]:
        pct = count / len(df) * 100
        print(f"  {label}: {count:,} ({pct:.2f}%)")
    
    # =========================================================================
    # STEP 2: PREPROCESS DATA
    # =========================================================================
    print_separator("STEP 2: Preprocessing Data")
    
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.clean_data(df)
    
    # Get labels
    labels = df_clean[LABEL_COLUMN]
    y_binary = (labels != 'BENIGN').astype(int)  # 0 = normal, 1 = attack
    
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
    
    results['preprocessing'] = {
        'records_after_cleaning': len(df_clean),
        'records_removed': len(df) - len(df_clean),
        'train_size': len(X_train),
        'test_size': len(X_test),
        'train_attack_ratio': float(y_train.sum() / len(y_train)),
        'test_attack_ratio': float(y_test.sum() / len(y_test)),
        'features_used': SELECTED_FEATURES
    }
    
    print(f"After cleaning: {len(df_clean):,} records")
    print(f"Train set: {len(X_train):,} samples")
    print(f"Test set: {len(X_test):,} samples")
    print(f"Train attack ratio: {results['preprocessing']['train_attack_ratio']:.2%}")
    print(f"Test attack ratio: {results['preprocessing']['test_attack_ratio']:.2%}")
    
    # Save preprocessor
    preprocessor.save()
    
    # =========================================================================
    # STEP 3: TRAIN AND EVALUATE STATISTICAL DETECTOR
    # =========================================================================
    print_separator("STEP 3: Statistical Detector (Z-score)")
    
    # Train on normal traffic only
    normal_mask = y_train == 0
    X_normal = X_train[normal_mask]
    
    stat_detector = StatisticalDetector(zscore_threshold=3.0)
    stat_detector.fit(X_normal)
    
    # Predict on test set
    stat_anomaly, stat_scores = stat_detector.detect(X_test, method='zscore')
    stat_pred = stat_anomaly.astype(int)
    
    stat_metrics = calculate_metrics(y_test.values, stat_pred, stat_scores, prefix='stat_')
    results['statistical_detector'] = stat_metrics
    results['statistical_detector']['threshold'] = stat_detector.zscore_threshold
    results['statistical_detector']['training_samples'] = len(X_normal)
    
    print(f"Training samples (normal only): {len(X_normal):,}")
    print(f"Accuracy:  {stat_metrics['stat_accuracy']:.4f} ({stat_metrics['stat_accuracy']*100:.2f}%)")
    print(f"Precision: {stat_metrics['stat_precision']:.4f}")
    print(f"Recall:    {stat_metrics['stat_recall']:.4f}")
    print(f"F1 Score:  {stat_metrics['stat_f1_score']:.4f}")
    print(f"Specificity: {stat_metrics.get('stat_specificity', 0):.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN={stat_metrics.get('stat_true_negatives', 0):,}  FP={stat_metrics.get('stat_false_positives', 0):,}")
    print(f"  FN={stat_metrics.get('stat_false_negatives', 0):,}  TP={stat_metrics.get('stat_true_positives', 0):,}")
    
    # Save model
    joblib.dump(stat_detector, MODELS_DIR / "statistical_detector.joblib")
    
    # =========================================================================
    # STEP 4: TRAIN AND EVALUATE ISOLATION FOREST
    # =========================================================================
    print_separator("STEP 4: Isolation Forest")
    
    iso_detector = IsolationForestDetector(contamination=0.1, n_estimators=100)
    iso_detector.fit(X_normal)
    
    # Predict on test set
    iso_anomaly, iso_scores = iso_detector.detect(X_test)
    iso_pred = iso_anomaly.astype(int)
    
    iso_metrics = calculate_metrics(y_test.values, iso_pred, iso_scores, prefix='iso_')
    results['isolation_forest'] = iso_metrics
    results['isolation_forest']['contamination'] = iso_detector.contamination
    results['isolation_forest']['n_estimators'] = iso_detector.n_estimators
    results['isolation_forest']['training_samples'] = len(X_normal)
    
    print(f"Parameters: contamination={iso_detector.contamination}, n_estimators={iso_detector.n_estimators}")
    print(f"Training samples (normal only): {len(X_normal):,}")
    print(f"Accuracy:  {iso_metrics['iso_accuracy']:.4f} ({iso_metrics['iso_accuracy']*100:.2f}%)")
    print(f"Precision: {iso_metrics['iso_precision']:.4f}")
    print(f"Recall:    {iso_metrics['iso_recall']:.4f}")
    print(f"F1 Score:  {iso_metrics['iso_f1_score']:.4f}")
    print(f"Specificity: {iso_metrics.get('iso_specificity', 0):.4f}")
    if 'iso_roc_auc' in iso_metrics:
        print(f"ROC AUC:   {iso_metrics['iso_roc_auc']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN={iso_metrics.get('iso_true_negatives', 0):,}  FP={iso_metrics.get('iso_false_positives', 0):,}")
    print(f"  FN={iso_metrics.get('iso_false_negatives', 0):,}  TP={iso_metrics.get('iso_true_positives', 0):,}")
    
    # Save model
    iso_detector.save()
    
    # =========================================================================
    # STEP 5: COMBINED DETECTION (Statistical OR Isolation Forest)
    # =========================================================================
    print_separator("STEP 5: Combined Detection (Statistical OR Isolation Forest)")
    
    # Combined: anomaly if EITHER method flags it
    combined_pred = (stat_pred | iso_pred).astype(int)
    combined_scores = np.maximum(stat_scores, iso_scores)
    
    combined_metrics = calculate_metrics(y_test.values, combined_pred, combined_scores, prefix='combined_')
    results['combined_detection'] = combined_metrics
    
    print(f"Logic: Statistical OR Isolation Forest")
    print(f"Accuracy:  {combined_metrics['combined_accuracy']:.4f} ({combined_metrics['combined_accuracy']*100:.2f}%)")
    print(f"Precision: {combined_metrics['combined_precision']:.4f}")
    print(f"Recall:    {combined_metrics['combined_recall']:.4f}")
    print(f"F1 Score:  {combined_metrics['combined_f1_score']:.4f}")
    print(f"Specificity: {combined_metrics.get('combined_specificity', 0):.4f}")
    if 'combined_roc_auc' in combined_metrics:
        print(f"ROC AUC:   {combined_metrics['combined_roc_auc']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN={combined_metrics.get('combined_true_negatives', 0):,}  FP={combined_metrics.get('combined_false_positives', 0):,}")
    print(f"  FN={combined_metrics.get('combined_false_negatives', 0):,}  TP={combined_metrics.get('combined_true_positives', 0):,}")
    
    # =========================================================================
    # STEP 6: TRAIN AND EVALUATE ATTACK CLASSIFIER
    # =========================================================================
    print_separator("STEP 6: Attack Classifier (Random Forest)")
    
    classifier = AttackClassifier(n_estimators=100, max_depth=20)
    classifier.fit(X_train, labels_train.values)
    
    # Predict on test set
    cls_pred = classifier.predict(X_test)
    cls_true = classifier._map_labels(labels_test.values)
    
    # Overall classification metrics
    cls_accuracy = accuracy_score(cls_true, cls_pred)
    results['attack_classifier'] = {
        'accuracy': float(cls_accuracy),
        'n_estimators': classifier.n_estimators,
        'max_depth': classifier.max_depth,
        'classes': list(classifier.classes_),
        'training_samples': len(X_train)
    }
    
    print(f"Parameters: n_estimators={classifier.n_estimators}, max_depth={classifier.max_depth}")
    print(f"Training samples: {len(X_train):,}")
    print(f"Classes: {list(classifier.classes_)}")
    print(f"\nOverall Accuracy: {cls_accuracy:.4f} ({cls_accuracy*100:.2f}%)")
    
    # Per-class metrics
    print(f"\nPer-Class Classification Report:")
    print("-" * 70)
    report = classification_report(cls_true, cls_pred, output_dict=True, zero_division=0)
    results['attack_classifier']['classification_report'] = report
    
    # Print formatted report
    print(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("-" * 70)
    for class_name in classifier.classes_:
        if class_name in report:
            r = report[class_name]
            print(f"{class_name:<20} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1-score']:>10.4f} {int(r['support']):>10,}")
    print("-" * 70)
    print(f"{'Macro Avg':<20} {report['macro avg']['precision']:>10.4f} {report['macro avg']['recall']:>10.4f} {report['macro avg']['f1-score']:>10.4f}")
    print(f"{'Weighted Avg':<20} {report['weighted avg']['precision']:>10.4f} {report['weighted avg']['recall']:>10.4f} {report['weighted avg']['f1-score']:>10.4f}")
    
    # Confusion matrix for classifier
    cls_cm = confusion_matrix(cls_true, cls_pred, labels=list(classifier.classes_))
    results['attack_classifier']['confusion_matrix'] = cls_cm.tolist()
    results['attack_classifier']['confusion_matrix_labels'] = list(classifier.classes_)
    
    print(f"\nConfusion Matrix (rows=actual, cols=predicted):")
    print(f"Labels: {list(classifier.classes_)}")
    print(cls_cm)
    
    # Save model
    classifier.save()
    
    # =========================================================================
    # STEP 7: FEATURE IMPORTANCE
    # =========================================================================
    print_separator("STEP 7: Feature Importance Analysis")
    
    importances = classifier.get_feature_importance()
    feature_importance_dict = {}
    
    print(f"{'Rank':<5} {'Feature':<35} {'Importance':>12}")
    print("-" * 55)
    for rank, (idx, importance) in enumerate(importances, 1):
        feature_name = SELECTED_FEATURES[idx] if idx < len(SELECTED_FEATURES) else f"Feature_{idx}"
        feature_importance_dict[feature_name] = float(importance)
        print(f"{rank:<5} {feature_name:<35} {importance:>12.6f}")
    
    results['feature_importance'] = feature_importance_dict
    
    # =========================================================================
    # STEP 8: PER-ATTACK TYPE DETECTION METRICS
    # =========================================================================
    print_separator("STEP 8: Per-Attack Type Detection Rates")
    
    print(f"{'Attack Type':<25} {'Total':>8} {'Detected':>10} {'Rate':>8}")
    print("-" * 55)
    
    per_attack = {}
    for attack_type in labels_test.unique():
        mask = labels_test == attack_type
        if mask.sum() == 0:
            continue
        
        total = mask.sum()
        detected = combined_pred[mask.values].sum()
        rate = detected / total if total > 0 else 0
        
        per_attack[attack_type] = {
            'total': int(total),
            'detected': int(detected),
            'detection_rate': float(rate)
        }
        
        print(f"{attack_type:<25} {total:>8,} {detected:>10,} {rate:>7.2%}")
    
    results['per_attack_metrics'] = per_attack
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print_separator("EVALUATION SUMMARY")
    
    print(f"\n{'Model':<30} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 75)
    print(f"{'Statistical (Z-score)':<30} {stat_metrics['stat_accuracy']:>10.4f} {stat_metrics['stat_precision']:>10.4f} {stat_metrics['stat_recall']:>10.4f} {stat_metrics['stat_f1_score']:>10.4f}")
    print(f"{'Isolation Forest':<30} {iso_metrics['iso_accuracy']:>10.4f} {iso_metrics['iso_precision']:>10.4f} {iso_metrics['iso_recall']:>10.4f} {iso_metrics['iso_f1_score']:>10.4f}")
    print(f"{'Combined (Stat OR IF)':<30} {combined_metrics['combined_accuracy']:>10.4f} {combined_metrics['combined_precision']:>10.4f} {combined_metrics['combined_recall']:>10.4f} {combined_metrics['combined_f1_score']:>10.4f}")
    print(f"{'Attack Classifier':<30} {cls_accuracy:>10.4f} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
    print("-" * 75)
    
    # Save results to JSON
    if save_results:
        results_file = MODELS_DIR / "evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {results_file}")
    
    print(f"\nAll models saved to: {MODELS_DIR}")
    print(f"Evaluation completed at: {datetime.now().isoformat()}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train and evaluate anomaly detection models")
    parser.add_argument("--sample", type=int, default=100000, 
                        help="Number of samples to use (default: 100000, use 0 for all)")
    args = parser.parse_args()
    
    sample_size = args.sample if args.sample > 0 else None
    
    print(f"Sample size: {sample_size if sample_size else 'ALL DATA'}")
    results = evaluate_all_models(sample_size=sample_size, save_results=True)
