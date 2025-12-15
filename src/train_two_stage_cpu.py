
"""
Cloud / CPU Fallback Training Script
Use this if GPU Quota is 0.
Optimized for:
1. High-CPU Instances (e.g., c2-standard-16, n1-highmem-16)
2. XGBoost 'hist' method (Fastest on CPU)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from src.data.preprocessor import DataPreprocessor
from src.utils.logger import get_logger
from src.utils.config import MODELS_DIR, PROCESSED_DATA_DIR, ATTACK_TYPES

# Force CPU config
PROCESSED_DATA_DIR = Path(".") 
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

logger = get_logger(__name__)

def normalize_label(label):
    label = str(label).strip()
    if label in ['0', '0.0', 'Benign', 'BENIGN']:
        return 'Benign'
    for key, category in ATTACK_TYPES.items():
        if key.lower() in label.lower():
            return category
    return label 

def train_cpu():
    logger.info("⚠️ Starting CPU Fallback Training (This will take longer)...")
    
    # 1. Load Data
    data_path = PROCESSED_DATA_DIR / "merged_dataset.csv"
    if not data_path.exists():
        logger.error(f"Dataset not found at {data_path}")
        return

    logger.info(f"Loading dataset... (Please wait)")
    df = pd.read_csv(data_path, on_bad_lines='skip', na_filter=False)
    
    # Quick Clean
    df.columns = df.columns.str.strip()
    df['Label'] = df['Label'].apply(normalize_label)
    for col in df.columns:
        if col != 'Label' and col != 'Dataset_Source' and col != 'Timestamp':
             df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # XGBoost 'hist' method crashes on Infinite values. Replace them with NaN.
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    # 2. Features
    features = [c for c in df.columns if c not in ['Label', 'Dataset_Source', 'Timestamp']]
    
    # 3. Prepare Labels
    df['Binary_Label'] = df['Label'].apply(lambda x: 0 if x == 'Benign' else 1)
    le = LabelEncoder()
    df['Multi_Label'] = le.fit_transform(df['Label'])
    joblib.dump(le, MODELS_DIR / "label_encoder.joblib")
    
    # 4. Split
    X = df[features]
    y_bin = df['Binary_Label']
    y_multi = df['Multi_Label']
    
    X_train, X_test, y_bin_train, y_bin_test, y_multi_train, y_multi_test = train_test_split(
        X, y_bin, y_multi, test_size=0.1, random_state=42, stratify=y_bin
    )
    
    # --- STAGE 1: BINARY (CPU) ---
    logger.info("Training Stage 1 (Binary) on CPU (n_jobs=-1)...")
    clf_bin = xgb.XGBClassifier(
        n_estimators=100, # Reduced slightly for CPU speed
        max_depth=6,      # Reduced depth slightly
        learning_rate=0.1,
        objective='binary:logistic',
        tree_method='hist', # <--- CPU OPTIMIZED
        n_jobs=-1,          # Use ALL vCPUs
        eval_metric='logloss'
    )
    clf_bin.fit(X_train, y_bin_train)
    
    acc_bin = accuracy_score(y_bin_test, clf_bin.predict(X_test))
    logger.info(f"Stage 1 CPU Accuracy: {acc_bin:.5f}")
    clf_bin.save_model(MODELS_DIR / "stage1_xgboost.json")
    
    # --- STAGE 2: MULTI (CPU) ---
    logger.info("Training Stage 2 (Multi) on CPU...")
    # Filter for attacks only
    train_mask = y_bin_train == 1
    X_train_attacks = X_train[train_mask]
    y_train_attacks_global = y_multi_train[train_mask]
    
    # Create a specialized encoder for Stage 2 (Attacks Only) to ensure 0..N-1 labels
    le_stage2 = LabelEncoder()
    attack_names = le.inverse_transform(y_train_attacks_global) # Decode to strings
    y_train_attacks = le_stage2.fit_transform(attack_names)     # Re-encode to 0..N-1
    
    # Save the Stage 2 encoder
    joblib.dump(le_stage2, MODELS_DIR / "stage2_label_encoder.joblib")
    
    clf_multi = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective='multi:softprob',
        num_class=len(le_stage2.classes_),
        tree_method='hist', # <--- CPU OPTIMIZED
        n_jobs=-1,
        eval_metric='mlogloss'
    )
    clf_multi.fit(X_train_attacks, y_train_attacks)
    
    test_mask = y_bin_test == 1
    # We must transform the TEST labels too using the new Stage 2 encoder
    y_test_attacks_global = y_multi_test[test_mask]
    # Handle unseen labels in test set if necessary (though unlikely with StratifiedSplit)
    y_test_attacks = le_stage2.transform(le.inverse_transform(y_test_attacks_global))
    
    acc_multi = accuracy_score(y_test_attacks, clf_multi.predict(X_test[test_mask]))
    logger.info(f"Stage 2 CPU Accuracy: {acc_multi:.5f}")
    
    clf_multi.save_model(MODELS_DIR / "stage2_xgboost.json")
    
    with open(MODELS_DIR / "model_features.json", 'w') as f:
        json.dump(features, f)

    logger.info("DONE! CPU Training Complete.")

if __name__ == "__main__":
    train_cpu()
