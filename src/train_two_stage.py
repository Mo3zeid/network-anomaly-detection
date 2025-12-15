"""
Two-Stage Training Script
Implements the Cascade XGBoost Approach:
1. Binary Classification (Benign vs Attack)
2. Multi-class Classification (Attack Type)
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from src.data.preprocessor import DataPreprocessor
from src.utils.logger import get_logger
from src.utils.config import MODELS_DIR, PROCESSED_DATA_DIR, SELECTED_FEATURES, ATTACK_TYPES

logger = get_logger(__name__)

def normalize_label(label):
    label = str(label).strip()
    if label in ['0', '0.0', 'Benign', 'BENIGN']:
        return 'Benign'
    
    # Check specific mappings
    for key, category in ATTACK_TYPES.items():
        if key.lower() in label.lower():
            return category
            
    return label # Fallback

def train_two_stage():
    # Use the pre-shuffled sample to ensure class balance
    data_path = PROCESSED_DATA_DIR / "training_aligned.csv"
    if not data_path.exists():
         logger.error("Training sample not found!")
         return

    logger.info(f"Loading shuffled training sample from {data_path}...")
    # Load the full sample (already limited to 500k by shuf)
    # na_filter=False ensures '0.0' or 'nan' string is read as string, not Float NaN
    df = pd.read_csv(data_path, low_memory=False, on_bad_lines='skip', na_filter=False)
    
    # Fix column names - strip whitespace
    df.columns = df.columns.str.strip()
    
    logger.info(f"Raw Label Unique Values (First 20): {df['Label'].unique()[:20]}")
    logger.info(f"Raw Label Types: {df['Label'].apply(type).unique()}")
    
    # Normalize Labels IMMEDIATELY
    df['Label'] = df['Label'].apply(normalize_label)
    
    # Ensure all numeric columns are actually numeric
    # (Some might be objects if "labels" or headers were read in body)
    for col in df.columns:
        if col != 'Label' and col != 'Dataset_Source' and col != 'Timestamp':
             df[col] = pd.to_numeric(df[col], errors='coerce')
    
    logger.info(f"Loaded {len(df)} rows for training") 
    logger.info(f"Label Distribution:\n{df['Label'].value_counts()}") 
    
    # 2. Preprocess (Strict)
    preprocessor = DataPreprocessor()
    df = preprocessor.clean_data(df)
    
    # 3. Feature Selection (Correlation)
    # Disabled for speed and to keep generic alignment stable
    # df = preprocessor.remove_highly_correlated_features(df, threshold=0.95)
    
    # Save current feature list for reference
    features = [c for c in df.columns if c not in ['Label', 'Dataset_Source', 'Timestamp']]
    
    # 4. Prepare Labels
    # Binary Label: 0=Benign, 1=Attack
    df['Binary_Label'] = df['Label'].apply(lambda x: 0 if x == 'Benign' else 1)
    
    # Multi-class Label: Encode
    le = LabelEncoder()
    # Filter for Stage 2 first to get classes? 
    # Or just encode all, but Stage 2 train only on Attacks
    
    # 5. Split
    X = df[features]
    y_bin = df['Binary_Label']
    y_multi = df['Label']
    
    X_train, X_test, y_bin_train, y_bin_test, y_multi_train, y_multi_test = train_test_split(
        X, y_bin, y_multi, test_size=0.2, random_state=42, stratify=y_bin
    )
    
    # --- STAGE 1: BINARY XGBOOST ---
    logger.info("Training Stage 1 (Binary)...")
    clf_bin = xgb.XGBClassifier(
        n_estimators=100, 
        max_depth=6, 
        learning_rate=0.1,
        objective='binary:logistic',
        eval_metric='logloss',
        n_jobs=-1
    )
    clf_bin.fit(X_train, y_bin_train)
    
    # Evaluate Stage 1
    y_bin_pred = clf_bin.predict(X_test)
    acc_bin = accuracy_score(y_bin_test, y_bin_pred)
    logger.info(f"Stage 1 Accuracy: {acc_bin:.4f}")
    
    # Save Stage 1
    # Use get_booster() to avoid sklearn compatibility issues
    clf_bin.get_booster().save_model(MODELS_DIR / "stage1_xgboost.json")
    
    # --- STAGE 2: MULTI-CLASS XGBOOST ---
    logger.info("Training Stage 2 (Multi-class)...")
    
    # Filter Training Data: Only Attacks
    train_mask = y_bin_train == 1
    X_train_attacks = X_train[train_mask]
    y_train_attacks = y_multi_train[train_mask]
    
    # Filter Test Data: Only Attacks (for evaluation)
    test_mask = y_bin_test == 1
    X_test_attacks = X_test[test_mask]
    y_test_attacks = y_multi_test[test_mask]
    
    if len(X_train_attacks) == 0:
        logger.warning("No attacks in training set! Skipping Stage 2.")
        return

    # Encode Multi-labels
    y_train_attacks_enc = le.fit_transform(y_train_attacks)
    y_test_attacks_enc = le.transform(y_test_attacks) # transform test using fit from train
    
    # Save encoder
    joblib.dump(le, MODELS_DIR / "label_encoder.joblib")
    
    clf_multi = xgb.XGBClassifier(
        n_estimators=100, 
        max_depth=6, 
        learning_rate=0.1,
        objective='multi:softprob',
        num_class=len(le.classes_),
        eval_metric='mlogloss',
        n_jobs=-1
    )
    clf_multi.fit(X_train_attacks, y_train_attacks_enc)
    
    # Evaluate Stage 2
    # Ensure we get class indices even if model returns probabilities
    y_multi_prob = clf_multi.predict_proba(X_test_attacks)
    y_multi_pred_enc = np.argmax(y_multi_prob, axis=1)
    
    acc_multi = accuracy_score(y_test_attacks_enc, y_multi_pred_enc)
    logger.info(f"Stage 2 Accuracy (on Attacks): {acc_multi:.4f}")
    
    # Save Stage 2
    clf_multi.get_booster().save_model(MODELS_DIR / "stage2_xgboost.json")
    
    # --- STAGE 3: EXPLAINABILITY (SHAP) ---
    logger.info("Generating SHAP Explanations...")
    generate_shap_plots(clf_bin, X_test, "Binary_Stage1")
    generate_shap_plots(clf_multi, X_test_attacks, "Multiclass_Stage2")

    # Save Feature List
    import json
    with open(MODELS_DIR / "model_features.json", 'w') as f:
        json.dump(features, f)
        
    logger.info("Training Complete!")

def generate_shap_plots(model, X, name):
    try:
        import shap
        import matplotlib.pyplot as plt
        
        # Subsample for speed
        if len(X) > 1000:
            X_sample = X.sample(1000, random_state=42)
        else:
            X_sample = X
            
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        plt.figure()
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.savefig(PROCESSED_DATA_DIR / f"shap_summary_{name}.png", bbox_inches='tight')
        plt.close()
        logger.info(f"Saved SHAP plot for {name}")
    except Exception as e:
        logger.error(f"Failed to generate SHAP plots for {name}: {e}")

if __name__ == "__main__":
    train_two_stage()
