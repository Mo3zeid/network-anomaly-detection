"""
Dataset Merger Module
Unifies CICIDS2017, ToN-IoT, Bot-IoT, and UNSW-NB15 into a single standard format.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from ..utils.logger import get_logger

logger = get_logger(__name__)

class DatasetMerger:
    """
    Merges multiple network datasets into a unified format.
    Standardizes column names and labels.
    """
    
    # Target Schema: CICIDS2017
    
    # Mappings for other datasets
    MAPPINGS = {
        "CICIDS2018": {
            "Dst Port": "Destination Port",
            "Flow Duration": "Flow Duration",
            "Tot Fwd Pkts": "Total Fwd Packets",
            "Tot Bwd Pkts": "Total Backward Packets",
            "TotLen Fwd Pkts": "Total Length of Fwd Packets",
            "TotLen Bwd Pkts": "Total Length of Bwd Packets",
            "Fwd Pkt Len Mean": "Fwd Packet Length Mean",
            "Bwd Pkt Len Mean": "Bwd Packet Length Mean",
            "Flow Byts/s": "Flow Bytes/s",
            "Flow Pkts/s": "Flow Packets/s",
            "Flow IAT Mean": "Flow IAT Mean",
            "Flow IAT Std": "Flow IAT Std",
            "Flow IAT Max": "Flow IAT Max",
            "Flow IAT Min": "Flow IAT Min",
            "Fwd Header Len": "Fwd Header Length",
            "Bwd Header Len": "Bwd Header Length",
            "Fwd Pkts/s": "Fwd Packets/s",
            "Bwd Pkts/s": "Bwd Packets/s",
            "Pkt Len Mean": "Packet Length Mean",
            "Pkt Len Std": "Packet Length Std",
            "Pkt Len Var": "Packet Length Variance",
            "Label": "Label"
        },
        "UNSW_NB15": {
            "dur": "Flow Duration", # Needs scaling (s -> us)
            "spkts": "Total Fwd Packets",
            "dpkts": "Total Backward Packets",
            "sbytes": "Total Length of Fwd Packets",
            "dbytes": "Total Length of Bwd Packets",
            "smean": "Fwd Packet Length Mean",
            "dmean": "Bwd Packet Length Mean",
            "sload": "Flow Bytes/s",
            "dload": "Flow Packets/s", # Proxy
            "sttl": "Fwd Header Length", # Proxy (TTL != Header Len but correlated)
            "dttl": "Bwd Header Length",
            "sjit": "Flow IAT Std", # Proxy
            "attack_cat": "Label"
        },
        "ToN_IoT": {
            "src_pkts": "Total Fwd Packets",
            "dst_pkts": "Total Backward Packets",
            "src_bytes": "Total Length of Fwd Packets",
            "dst_bytes": "Total Length of Bwd Packets",
            "duration": "Flow Duration",
            "label": "Label" # 0/1 or string?
        }
        # Bot-IoT often lacks packet-level details, might need omission or heavy padding
    }

    def __init__(self, raw_dir: Path, output_dir: Path):
        self.raw_dir = raw_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def normalize_label(self, label: str) -> str:
        """Map diverse labels to the 6 target classes."""
        label = str(label).lower()
        if 'benign' in label or label == '0': return 'Benign'
        if 'ddos' in label: return 'DDoS'
        if 'dos' in label: return 'DoS'
        if 'brute' in label or 'password' in label or 'patator' in label: return 'Bruteforce'
        if 'portscan' in label or 'scan' in label or 'recon' in label: return 'Reconnaissance'
        if 'bot' in label or 'malware' in label or 'backdoor' in label: return 'Malware'
        if 'injection' in label or 'sql' in label or 'xss' in label: return 'Injection'
        if 'infiltration' in label: return 'Infiltration'
        return 'Other'

    def load_and_standardize(self, filepath: Path, source_type: str) -> pd.DataFrame:
        """Load a CSV and rename columns to standard schema."""
        logger.info(f"Loading {source_type} from {filepath.name}...")
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            logger.error(f"Failed to load {filepath}: {e}")
            return pd.DataFrame()

        # Rename columns
        if source_type in self.MAPPINGS:
            # Create a reverse map for safety? No, just rename
            mapping = self.MAPPINGS[source_type]
            # Verify columns exist before renaming
            valid_map = {k: v for k, v in mapping.items() if k in df.columns}
            df = df.rename(columns=valid_map)
            
            # Special handling for UNSW Duration (Seconds to Microseconds)
            if source_type == "UNSW_NB15" and "Flow Duration" in df.columns:
                 df["Flow Duration"] = df["Flow Duration"] * 1e6
        
        # Standardize Labels
        label_col = "Label"
        if "attack_cat" in df.columns: # UNSW
            df["Label"] = df["attack_cat"]
        elif "type" in df.columns: # ToN-IoT
            df["Label"] = df["type"]
            
        if "Label" in df.columns:
            df["Label"] = df["Label"].apply(self.normalize_label)
        else:
             logger.warning(f"No label column found in {filepath.name}")

        df['Dataset_Source'] = source_type
        return df

    def merge_all(self, chunksize=100000):
        """Main execution flow with chunking."""
        logger.info("Starting merge process with chunking...")
        output_path = self.output_dir / "merged_dataset.csv"
        
        # Initialize output file with header
        first_chunk = True
        
        # Helper to process and append
        def process_file(filepath, source_type):
            nonlocal first_chunk
            logger.info(f"Processing {filepath.name}...")
            try:
                # Iterate in chunks
                for chunk in pd.read_csv(filepath, chunksize=chunksize, low_memory=False):
                    
                    # 1. Rename columns
                    if source_type in self.MAPPINGS:
                        mapping = self.MAPPINGS[source_type]
                        valid_map = {k: v for k, v in mapping.items() if k in chunk.columns}
                        chunk = chunk.rename(columns=valid_map)
                        
                        # Unit conversion
                        if source_type == "UNSW_NB15" and "Flow Duration" in chunk.columns:
                            chunk["Flow Duration"] = chunk["Flow Duration"] * 1e6
                    
                    # 2. Standardize Labels
                    if "attack_cat" in chunk.columns:
                        chunk["Label"] = chunk["attack_cat"]
                    elif "type" in chunk.columns:
                        chunk["Label"] = chunk["type"]
                        
                    if "Label" in chunk.columns:
                        chunk["Label"] = chunk["Label"].apply(self.normalize_label)
                    
                    chunk['Dataset_Source'] = source_type
                    
                    # 3. Filter to Target Columns Only
                    # (To keep the output clean, we likely want to intersect with CICIDS2017 columns)
                    # For now, we keep all and let preprocessor select? 
                    # No, let's try to keep the key ones from CICIDS2017 mapping + Label
                    # If columns are missing, fill with 0?
                    # Let's just save for now.
                    
                    # 4. Append to CSV
                    mode = 'w' if first_chunk else 'a'
                    header = first_chunk
                    chunk.to_csv(output_path, mode=mode, header=header, index=False)
                    first_chunk = False
                    
            except Exception as e:
                logger.error(f"Error processing {filepath}: {e}")

        # 1. CICIDS2017
        c17_path = self.raw_dir / "cicids2017"
        if c17_path.exists():
            for f in c17_path.glob("*.csv"):
                process_file(f, "CICIDS2017")

        # 2. CICIDS2018
        c18_path = self.raw_dir / "cicids2018"
        if c18_path.exists():
            for f in c18_path.glob("*.csv"):
                 process_file(f, "CICIDS2018")

        # 3. UNSW-NB15
        unsw_path = self.raw_dir / "unsw_nb15"
        if unsw_path.exists():
             for f in unsw_path.glob("*.csv"):
                 if "LIST_EVENTS" not in f.name and "features" not in f.name:
                     process_file(f, "UNSW_NB15")

        # 4. ToN-IoT
        ton_path = self.raw_dir / "ton_iot"
        if ton_path.exists():
             for f in ton_path.glob("*.csv"):
                 process_file(f, "ToN_IoT")
                 
        logger.info(f"Finished merging. Output: {output_path}")

if __name__ == "__main__":
    from ..utils.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
    merger = DatasetMerger(RAW_DATA_DIR, PROCESSED_DATA_DIR)
    merger.merge_all()
