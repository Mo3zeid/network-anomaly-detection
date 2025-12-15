"""
Cloud / Heavy-Duty Training Script
Optimized for:
1. Full 67M Dataset (or Merged 12M)
2. NVIDIA GPU Acceleration (tree_method='gpu_hist')
3. Maximum Speed (No SHAP generation to save time)
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

# Update paths for Cloud environment (Current directory usually)
# If running strictly on cloud with files in root:
# PROCESSED_DATA_DIR = Path(".") 
# MODELS_DIR = Path("models")
# MODELS_DIR.mkdir(exist_ok=True)

logger = get_logger(__name__)

def normalize_label(label):
    label = str(label).strip()
    if label in ['0', '0.0', 'Benign', 'BENIGN']:
        return 'Benign'
    for key, category in ATTACK_TYPES.items():
        if key.lower() in label.lower():
            return category
    return label 

def train_cloud():
    logger.info("🚀 Starting Cloud Training Optimization...")
    
    # 1. Load FULL Dataset (Merged)
    # Assuming the user uploaded 'merged_dataset.csv'
    data_path = PROCESSED_DATA_DIR / "merged_dataset.csv"
    
    if not data_path.exists():
        logger.error(f"Dataset not found at {data_path}. Did you upload it?")
        return

    logger.info(f"Loading FULL dataset from {data_path} (This may take time)...")
    # Read CSV without low_memory=False to allow chunking if needed, but for High-RAM VM just load it.
    df = pd.read_csv(data_path, on_bad_lines='skip', na_filter=False)
    
    # Quick Clean
    df.columns = df.columns.str.strip()
    df['Label'] = df['Label'].apply(normalize_label)
    
    # Convert numeric
    for col in df.columns:
        if col != 'Label' and col != 'Dataset_Source' and col != 'Timestamp':
             df[col] = pd.to_numeric(df[col], errors='coerce')
             
    logger.info(f"Loaded {len(df)} rows. Cleaning NaNs...")
    df.dropna(inplace=True)
    
    # 2. Features
    preprocessor = DataPreprocessor()
    # Skip correlation check for speed on 12M rows (it's O(N^2))
    # Just remove non-numeric
    features = [c for c in df.columns if c not in ['Label', 'Dataset_Source', 'Timestamp']]
    
    # 3. Prepare Labels
    df['Binary_Label'] = df['Label'].apply(lambda x: 0 if x == 'Benign' else 1)
    le = LabelEncoder()
    # We fit label encoder on EVERYTHING now
    df['Multi_Label'] = le.fit_transform(df['Label'])
    
    # Save encoder immediately
    joblib.dump(le, MODELS_DIR / "label_encoder.joblib")
    
    # 4. Split
    X = df[features]
    y_bin = df['Binary_Label']
    y_multi = df['Multi_Label']
    
    # Smaller test set (10% is enough for 12M rows)
    X_train, X_test, y_bin_train, y_bin_test, y_multi_train, y_multi_test = train_test_split(
        X, y_bin, y_multi, test_size=0.1, random_state=42, stratify=y_bin
    )
    
    # --- STAGE 1: BINARY (GPU) ---
    logger.info("Training Stage 1 (Binary) on GPU...")
    clf_bin = xgb.XGBClassifier(
        n_estimators=200, # More trees for big data
        max_depth=8,      # Deeper trees
        learning_rate=0.05,
        objective='binary:logistic',
        tree_method='gpu_hist', # <--- CRITICAL FOR SPEED
        predictor='gpu_predictor',
        eval_metric='logloss'
    )
    clf_bin.fit(X_train, y_bin_train)
    
    acc_bin = accuracy_score(y_bin_test, clf_bin.predict(X_test))
    logger.info(f"Stage 1 Cloud Accuracy: {acc_bin:.5f}")
    clf_bin.save_model(MODELS_DIR / "stage1_xgboost.json")
    
    # --- STAGE 2: MULTI (GPU) ---
    logger.info("Training Stage 2 (Multi) on GPU...")
    
    # Filter Attacks
    train_mask = y_bin_train == 1
    X_train_attacks = X_train[train_mask]
    y_train_attacks = y_multi_train[train_mask]
    
    clf_multi = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        objective='multi:softprob',
        num_class=len(le.classes_),
        tree_method='gpu_hist', # <--- CRITICAL
        predictor='gpu_predictor',
        eval_metric='mlogloss'
    )
    clf_multi.fit(X_train_attacks, y_train_attacks)
    
    # Test on attacks
    test_mask = y_bin_test == 1
    acc_multi = accuracy_score(y_multi_test[test_mask], clf_multi.predict(X_test[test_mask]))
    logger.info(f"Stage 2 Cloud Accuracy: {acc_multi:.5f}")
    
    clf_multi.save_model(MODELS_DIR / "stage2_xgboost.json")
    
    # Save Feature List
    with open(MODELS_DIR / "model_features.json", 'w') as f:
        json.dump(features, f)

    logger.info("DONE! Download your models now.")

if __name__ == "__main__":
    train_cloud()
