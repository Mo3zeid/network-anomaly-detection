
import pandas as pd
import numpy as np
from src.utils.logger import get_logger
from src.utils.config import PROCESSED_DATA_DIR

logger = get_logger(__name__)

def align_and_merge():
    # Source files
    files = {
        'balanced': PROCESSED_DATA_DIR / "training_sample_balanced.csv",
        'benign': PROCESSED_DATA_DIR / "benign_sample.csv",
        'ddos': PROCESSED_DATA_DIR / "ddos_sample.csv"
    }
    
    aligned_dfs = []
    
    for name, path in files.items():
        if not path.exists():
            continue
            
        logger.info(f"Processing {name}...")
        
        # Read without header
        # benign/ddos have NO header. 
        # balanced HAS header (line 1). We should skip it.
        skip = 1 if name == 'balanced' else 0
        
        try:
            df = pd.read_csv(path, header=None, skiprows=skip, low_memory=False, on_bad_lines='skip')
        except Exception as e:
            logger.error(f"Failed to read {name}: {e}")
            continue
            
        # Strategy:
        # Drop first 7 columns (Meta: FlowID, IPs, Port, Time)
        # Keep next 65 columns (Main features)
        # Keep last column (Label)
        
        # 1. Drop meta
        df_feats = df.iloc[:, 7:]
        
        # 2. Separate Label
        # Label is consistently at -2 (Last col is Source or Junk)
        labels = df_feats.iloc[:, -2]
        features = df_feats.iloc[:, :-2]
        
        # 3. Truncate features to common width (65)
        # Malware has ~73 features. Benign ~78.
        # We keep first 60 to be safe/consistent?
        # Let's keep 60.
        features = features.iloc[:, :60]
        
        # 4. Re-attach Label
        features['Label'] = labels
        
        aligned_dfs.append(features)
        logger.info(f"Aligned {name}: {features.shape}")

    # Merge
    final_df = pd.concat(aligned_dfs)
    
    # Create valid header
    cols = [f"Feature_{i}" for i in range(60)] + ['Label']
    final_df.columns = cols
    
    output_path = PROCESSED_DATA_DIR / "training_aligned.csv"
    final_df.to_csv(output_path, index=False)
    logger.info(f"Saved aligned dataset ({len(final_df)} rows) to {output_path}")

if __name__ == "__main__":
    align_and_merge()
