import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.preprocessing import LabelEncoder
from src.data.preprocessor import DataPreprocessor
from src.utils.logger import get_logger
from src.utils.config import MODELS_DIR, PROCESSED_DATA_DIR, ATTACK_TYPES

logger = get_logger(__name__)

def normalize_label(label):
    label = str(label).strip()
    if label in ['0', '0.0', 'Benign', 'BENIGN']:
        return 'Benign'
    
    # Check specific mappings
    for key, category in ATTACK_TYPES.items():
        if key.lower() in label.lower():
            return category
            
    return label 

def generate_assets():
    # 1. Load Data (Same Logic as Training)
    data_path = PROCESSED_DATA_DIR / "training_aligned.csv"
    if not data_path.exists():
         logger.error("Training sample not found!")
         return

    logger.info("Loading data for evaluation...")
    df = pd.read_csv(data_path, low_memory=False, on_bad_lines='skip', na_filter=False)
    
    # Normalize Labels (Crucial step missed before)
    df['Label'] = df['Label'].apply(normalize_label)
    
    df.columns = df.columns.str.strip()
    
    # Drop rows with missing labels (they cause the 'nan' class issue)
    df.dropna(subset=['Label'], inplace=True)
    
    for col in df.columns:
        if col != 'Label' and col != 'Dataset_Source' and col != 'Timestamp':
             df[col] = pd.to_numeric(df[col], errors='coerce')

    # Preprocess
    preprocessor = DataPreprocessor()
    df = preprocessor.clean_data(df)
    # DO NOT run correlation reduction dynamically here! It might drop different features.
    
    # Load exact features used in training
    with open(MODELS_DIR / "model_features.json", 'r') as f:
        features = json.load(f)
        
    logger.info(f"Using {len(features)} features from training configuration.")
    
    # Prepare Labels
    df['Binary_Label'] = df['Label'].apply(lambda x: 0 if x == 'Benign' else 1)
    
    # Split (Same Seed = Same Split)
    X = df[features]
    y_bin = df['Binary_Label']
    y_multi = df['Label']
    
    _, X_test, _, y_bin_test, _, y_multi_test = train_test_split(
        X, y_bin, y_multi, test_size=0.2, random_state=42, stratify=y_bin
    )
    
    logger.info(f"Test Set Size: {len(X_test)} samples")

    # --- STAGE 1 EVALUATION ---
    logger.info("Evaluating Stage 1 (Binary)...")
    clf_bin = xgb.XGBClassifier()
    clf_bin.load_model(MODELS_DIR / "stage1_xgboost.json")
    
    y_bin_pred = clf_bin.predict(X_test)
    y_bin_prob = clf_bin.predict_proba(X_test)[:, 1]
    
    # Report
    bin_report = classification_report(y_bin_test, y_bin_pred, target_names=['Benign', 'Attack'])
    logger.info(f"\nStage 1 Report:\n{bin_report}")
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_bin_test, y_bin_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Stage 1 ROC Curve')
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(PROCESSED_DATA_DIR / "presentation_roc_curve.png")
    plt.close()
    
    # --- STAGE 2 EVALUATION ---
    logger.info("Evaluating Stage 2 (Multi-class)...")
    clf_multi = xgb.XGBClassifier()
    clf_multi.load_model(MODELS_DIR / "stage2_xgboost.json")
    le = joblib.load(MODELS_DIR / "label_encoder.joblib")
    
    # Filter for attacks only (Stage 2 logic)
    mask = y_bin_test == 1
    X_test_attacks = X_test[mask]
    y_test_attacks = y_multi_test[mask]
    
    y_test_attacks_enc = le.transform(y_test_attacks)
    
    y_multi_prob = clf_multi.predict_proba(X_test_attacks)
    y_multi_pred_enc = np.argmax(y_multi_prob, axis=1)
    
    # Report
    # Ensure classes are strings and match labels explicitly
    target_names = [str(c) for c in le.classes_]
    all_labels = range(len(le.classes_))
    
    multi_report = classification_report(
        y_test_attacks_enc, 
        y_multi_pred_enc, 
        labels=all_labels,
        target_names=target_names,
        zero_division=0
    )
    logger.info(f"\nStage 2 Report:\n{multi_report}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test_attacks_enc, y_multi_pred_enc)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Stage 2 Confusion Matrix (Attacks Only)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(PROCESSED_DATA_DIR / "presentation_confusion_matrix.png")
    plt.close()

    # Save Text Report
    with open(PROCESSED_DATA_DIR / "presentation_stats.txt", "w") as f:
        f.write("=== Stage 1 (Binary) ===\n")
        f.write(bin_report)
        f.write("\n\n=== Stage 2 (Multi-class) ===\n")
        f.write(multi_report)
        
    logger.info("Assets Generated in data/processed/")

if __name__ == "__main__":
    generate_assets()
