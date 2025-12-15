
"""
Generate Visualization Graphs (Cloud)
Run this AFTER training to generate Confusion Matrices and ROC Curves.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize

from src.utils.logger import get_logger
from src.utils.config import MODELS_DIR, PROCESSED_DATA_DIR, ATTACK_TYPES

logger = get_logger(__name__)

def generate_graphs():
    logger.info("🎨 Starting Graph Generation...")
    
    # 1. Load Data (Must match training split directly)
    data_path = PROCESSED_DATA_DIR / "merged_dataset.csv"
    if not data_path.exists():
        logger.error("merged_dataset.csv not found!")
        return

    logger.info("Loading dataset for evaluation...")
    # Load a sample if full data is too big for plotting memory, but for metrics we want full or large subset
    # Let's load full, assuming VM has RAM.
    df = pd.read_csv(data_path, on_bad_lines='skip', na_filter=False)
    
    # Preprocessing (Same as Train)
    def normalize(l):
        l = str(l).strip()
        if l in ['0', '0.0', 'Benign', 'BENIGN']: return 'Benign'
        for k, v in ATTACK_TYPES.items():
            if k.lower() in l.lower(): return v
        return l
    
    df.columns = df.columns.str.strip()
    df['Label'] = df['Label'].apply(normalize)
    for c in df.columns:
        if c not in ['Label', 'Dataset_Source', 'Timestamp']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    
    # INFINITY FIX: Same as training
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    features = [c for c in df.columns if c not in ['Label', 'Dataset_Source', 'Timestamp']]
    X = df[features]
    
    # Encoders
    le = joblib.load(MODELS_DIR / "label_encoder.joblib")
    y_bin = df['Label'].apply(lambda x: 0 if x == 'Benign' else 1)
    y_multi = le.transform(df['Label'])
    
    # Consistency Split
    _, X_test, _, y_bin_test, _, y_multi_test = train_test_split(
        X, y_bin, y_multi, test_size=0.1, random_state=42, stratify=y_bin
    )
    
    logger.info(f"Evaluating on {len(X_test)} samples...")
    
    # Load Models
    clf_bin = xgb.XGBClassifier()
    clf_bin.load_model(MODELS_DIR / "stage1_xgboost.json")
    
    clf_multi = xgb.XGBClassifier()
    clf_multi.load_model(MODELS_DIR / "stage2_xgboost.json")
    
    # --- PLOT 1: CONFUSION MATRIX (Stage 1) ---
    y_pred_bin = clf_bin.predict(X_test)
    cm_bin = confusion_matrix(y_bin_test, y_pred_bin)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_bin, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'])
    plt.title('Stage 1: Binary Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix_stage1.png')
    logger.info("Saved confusion_matrix_stage1.png")
    
    # --- PLOT 2: CONFUSION MATRIX (Stage 2) ---
    # Only evaluate on Attacks (Waterfall logic)
    mask = y_bin_test == 1
    if mask.sum() > 0:
        X_test_attacks = X_test[mask]
        
        # Load Stage 2 Encoder
        try:
            le_stage2 = joblib.load(MODELS_DIR / "stage2_label_encoder.joblib")
            logger.info("Loaded Stage 2 specialized encoder.")
        except:
            logger.warning("Stage 2 encoder not found, falling back to global (Might fail if CPU script was used)")
            le_stage2 = le

        # Map global labels to Stage 2 labels
        y_test_attacks_global = y_multi_test[mask]
        # Transform: Global Int -> String -> Stage 2 Int
        y_test_attacks = le_stage2.transform(le.inverse_transform(y_test_attacks_global))
        
        # Bypass sklearn wrapper: Use Booster directly
        booster_multi = clf_multi.get_booster()
        # DMatrix from DataFrame (using columns as logic)
        dtest_multi = xgb.DMatrix(X_test_attacks, feature_names=features)
        y_prob_multi = booster_multi.predict(dtest_multi)
        y_pred_multi = np.argmax(y_prob_multi, axis=1)

        cm_multi = confusion_matrix(y_test_attacks, y_pred_multi)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_multi, annot=True, fmt='d', cmap='Reds',
                    xticklabels=le_stage2.classes_, yticklabels=le_stage2.classes_)
        plt.title('Stage 2: Multi-Class Confusion Matrix (Attacks Only)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('confusion_matrix_stage2.png')
        logger.info("Saved confusion_matrix_stage2.png")
    
    # --- PLOT 3: ROC CURVE (Stage 1) ---
    booster_bin = clf_bin.get_booster()
    dtest_bin = xgb.DMatrix(X_test, feature_names=features)
    y_prob_bin = booster_bin.predict(dtest_bin) # Returns prob of class 1 directly for binary
    
    fpr, tpr, _ = roc_curve(y_bin_test, y_prob_bin)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (Stage 1)')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    logger.info("Saved roc_curve.png")

    # --- PLOT 4: FEATURE IMPORTANCE (Top 10) ---
    # Get importance from Stage 1 (The Sentry)
    importance = clf_bin.feature_importances_
    # Map to feature names
    feat_imp = pd.Series(importance, index=features).sort_values(ascending=False).head(10)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feat_imp.values, y=feat_imp.index, palette='viridis')
    plt.title('Top 10 Features (Stage 1 XGBoost)')
    plt.xlabel('Feature Importance (Gain)')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    logger.info("Saved feature_importance.png")

    # --- PLOT 5: ATTACK DISTRIBUTION (Test Set) ---
    plt.figure(figsize=(10, 6))
    counts = pd.Series(y_multi_test).map(lambda x: le.classes_[x]).value_counts()
    sns.barplot(x=counts.index, y=counts.values, palette='magma')
    plt.title('Distribution of Attacks in Test Set')
    plt.xticks(rotation=45)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('attack_distribution.png')
    logger.info("Saved attack_distribution.png")


if __name__ == "__main__":
    generate_graphs()
