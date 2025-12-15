"""
Data preprocessing module for network traffic data.
Handles cleaning, feature selection, and normalization.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

from ..utils.config import (
    SELECTED_FEATURES, 
    LABEL_COLUMN, 
    PROCESSED_DATA_DIR,
    MODELS_DIR
)
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DataPreprocessor:
    """
    Preprocessor for network traffic data.
    Handles cleaning, feature selection, and normalization.
    """
    
    def __init__(self, features: Optional[list] = None):
        """
        Initialize the preprocessor.
        
        Args:
            features: List of features to use (default: SELECTED_FEATURES)
        """
        self.features = features or SELECTED_FEATURES
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset using Strict Rules (Slides Requirement).
        1. Replace Inf with NaN
        2. Fill NaN with 0 (Not Median)
        3. Drop Duplicates
        """
        logger.info("Cleaning data (Strict Mode)...")
        df_clean = df.copy()
        
        # 1. Replace infinity values with NaN
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
        
        # 2. Fill missing values with 0
        # (Slides requirement: "Zero-filling preferred for network flows")
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(0)
        
        # Fill non-numeric with "Unknown" or similar if needed
        # df_clean = df_clean.fillna("Unknown")
        
        # 3. Deduplication
        initial_len = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        dedup_len = len(df_clean)
        if initial_len - dedup_len > 0:
            logger.info(f"Removed {initial_len - dedup_len} duplicate rows")
        
        logger.info(f"Cleaned dataset has {len(df_clean)} records")
        return df_clean

    def remove_highly_correlated_features(self, df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
        """
        Remove features with correlation > threshold (Slides Requirement).
        """
        logger.info(f"Removing features with correlation > {threshold}...")
        
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        if 'Label' in numeric_df.columns:
            numeric_df = numeric_df.drop(columns=['Label'])
            
        corr_matrix = numeric_df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        # NEVER drop the Label column
        to_drop = [c for c in to_drop if c != 'Label' and c != ' Label']
        
        if to_drop:
            logger.info(f"Dropping {len(to_drop)} correlated features: {to_drop}")
            df_reduced = df.drop(columns=to_drop, errors='ignore')
            return df_reduced
        
        return df
    
    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select only the specified features from the dataset.
        
        Args:
            df: DataFrame with all features
            
        Returns:
            DataFrame with selected features only
        """
        # Check which features are available
        available_features = [f for f in self.features if f in df.columns]
        missing_features = [f for f in self.features if f not in df.columns]
        
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
        
        if not available_features:
            raise ValueError("None of the selected features are available in the dataset")
        
        logger.info(f"Selected {len(available_features)} features")
        self.features = available_features
        
        return df[available_features]
    
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Fit the scaler and transform the data.
        
        Args:
            df: DataFrame with selected features
            
        Returns:
            Normalized numpy array
        """
        logger.info("Fitting and transforming data...")
        self.is_fitted = True
        return self.scaler.fit_transform(df)
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform data using the fitted scaler.
        
        Args:
            df: DataFrame with selected features
            
        Returns:
            Normalized numpy array
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transforming")
        return self.scaler.transform(df)
    
    def preprocess(self, df: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """
        Full preprocessing pipeline: clean, select, and normalize.
        
        Args:
            df: Raw DataFrame
            fit: Whether to fit the scaler (True for training data)
            
        Returns:
            Preprocessed numpy array
        """
        # Clean data
        df_clean = self.clean_data(df)
        
        # Select features
        df_features = self.select_features(df_clean)
        
        # Normalize
        if fit:
            X = self.fit_transform(df_features)
        else:
            X = self.transform(df_features)
        
        logger.info(f"Preprocessing complete. Shape: {X.shape}")
        return X
    
    def save(self, filepath: Optional[Path] = None):
        """
        Save the preprocessor (scaler) to disk.
        
        Args:
            filepath: Path to save the preprocessor
        """
        if filepath is None:
            filepath = MODELS_DIR / "preprocessor.joblib"
        
        joblib.dump({
            'scaler': self.scaler,
            'features': self.features,
            'is_fitted': self.is_fitted
        }, filepath)
        logger.info(f"Preprocessor saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: Optional[Path] = None) -> 'DataPreprocessor':
        """
        Load a preprocessor from disk.
        
        Args:
            filepath: Path to the saved preprocessor
            
        Returns:
            Loaded DataPreprocessor instance
        """
        if filepath is None:
            filepath = MODELS_DIR / "preprocessor.joblib"
        
        data = joblib.load(filepath)
        
        preprocessor = cls(features=data['features'])
        preprocessor.scaler = data['scaler']
        preprocessor.is_fitted = data['is_fitted']
        
        logger.info(f"Preprocessor loaded from {filepath}")
        return preprocessor


def prepare_training_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.Series, pd.Series]:
    """
    Prepare data for training with train/test split.
    
    Args:
        df: Cleaned DataFrame with features and labels
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, labels_train, labels_test)
    """
    preprocessor = DataPreprocessor()
    
    # Clean and select features
    df_clean = preprocessor.clean_data(df)
    
    # Extract labels
    if LABEL_COLUMN in df_clean.columns:
        labels = df_clean[LABEL_COLUMN]
        # Binary labels: 0 for BENIGN, 1 for attack
        y = (labels != 'BENIGN').astype(int)
    else:
        raise ValueError(f"Label column '{LABEL_COLUMN}' not found in dataset")
    
    # Select features
    df_features = preprocessor.select_features(df_clean)
    
    # Split data
    X_train_raw, X_test_raw, y_train, y_test, labels_train, labels_test = train_test_split(
        df_features, y, labels, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y
    )
    
    logger.info(f"Train set: {len(X_train_raw)} samples")
    logger.info(f"Test set: {len(X_test_raw)} samples")
    
    # Fit preprocessor on training data
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    
    # Save preprocessor
    preprocessor.save()
    
    return X_train, X_test, y_train.values, y_test.values, labels_train, labels_test


def get_normal_traffic_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract only normal (benign) traffic from the dataset.
    Used for training unsupervised anomaly detection models.
    
    Args:
        df: Full dataset
        
    Returns:
        DataFrame with only benign traffic
    """
    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"Label column '{LABEL_COLUMN}' not found")
    
    normal_df = df[df[LABEL_COLUMN] == 'BENIGN'].copy()
    logger.info(f"Extracted {len(normal_df)} normal traffic samples")
    
    return normal_df
