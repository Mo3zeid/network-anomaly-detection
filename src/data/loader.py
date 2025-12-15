"""
Data loader module for CICIDS2017 dataset.
Handles loading and basic validation of network traffic data.
"""
import pandas as pd
from pathlib import Path
from typing import Optional, List

from ..utils.config import RAW_DATA_DIR, LABEL_COLUMN
from ..utils.logger import get_logger

logger = get_logger(__name__)


def load_single_file(filepath: Path) -> pd.DataFrame:
    """
    Load a single CSV file from the CICIDS2017 dataset.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame with the loaded data
    """
    logger.info(f"Loading file: {filepath.name}")
    
    # CICIDS2017 uses different encodings, try multiple
    encodings = ['utf-8', 'latin-1', 'cp1252']
    
    for encoding in encodings:
        try:
            df = pd.read_csv(filepath, encoding=encoding, low_memory=False)
            # Clean column names (remove leading/trailing spaces)
            df.columns = df.columns.str.strip()
            logger.info(f"Successfully loaded {len(df)} records from {filepath.name}")
            return df
        except UnicodeDecodeError:
            continue
    
    raise ValueError(f"Could not load file {filepath} with any known encoding")


def load_dataset(data_dir: Optional[Path] = None, files: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Load the complete CICIDS2017 dataset from multiple CSV files.
    
    Args:
        data_dir: Directory containing the CSV files (default: RAW_DATA_DIR)
        files: Specific files to load (default: all CSV files)
        
    Returns:
        Combined DataFrame with all data
    """
    if data_dir is None:
        data_dir = RAW_DATA_DIR
    
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            f"Please download CICIDS2017 dataset and place CSV files in {data_dir}"
        )
    
    # Get all CSV files
    if files is None:
        csv_files = list(data_dir.glob("*.csv"))
    else:
        csv_files = [data_dir / f for f in files]
    
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {data_dir}\n"
            f"Please download CICIDS2017 dataset from:\n"
            f"https://www.unb.ca/cic/datasets/ids-2017.html"
        )
    
    logger.info(f"Found {len(csv_files)} CSV files to load")
    
    # Load and combine all files
    dataframes = []
    for filepath in csv_files:
        try:
            df = load_single_file(filepath)
            dataframes.append(df)
        except Exception as e:
            logger.warning(f"Failed to load {filepath}: {e}")
    
    if not dataframes:
        raise ValueError("No data could be loaded from any files")
    
    # Combine all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    logger.info(f"Total records loaded: {len(combined_df)}")
    
    return combined_df


def get_dataset_info(df: pd.DataFrame) -> dict:
    """
    Get summary information about the dataset.
    
    Args:
        df: The loaded DataFrame
        
    Returns:
        Dictionary with dataset statistics
    """
    info = {
        'total_records': len(df),
        'features': len(df.columns),
        'columns': list(df.columns),
    }
    
    # Label distribution if available
    if LABEL_COLUMN in df.columns:
        label_counts = df[LABEL_COLUMN].value_counts().to_dict()
        info['label_distribution'] = label_counts
        info['attack_types'] = len(label_counts)
        
        # Calculate attack vs benign ratio
        benign_count = label_counts.get('BENIGN', 0)
        attack_count = sum(v for k, v in label_counts.items() if k != 'BENIGN')
        info['benign_count'] = benign_count
        info['attack_count'] = attack_count
    
    return info


def load_sample_data(n_samples: int = 10000, random_state: int = 42) -> pd.DataFrame:
    """
    Load a sample of the dataset for quick testing.
    
    Args:
        n_samples: Number of samples to load
        random_state: Random seed for reproducibility
        
    Returns:
        Sampled DataFrame
    """
    df = load_dataset()
    
    if len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=random_state)
        logger.info(f"Sampled {n_samples} records from dataset")
    
    return df
