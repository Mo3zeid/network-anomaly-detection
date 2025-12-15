
import pandas as pd
import numpy as np
import gc
from pathlib import Path
from src.utils.logger import get_logger
from src.utils.config import PROCESSED_DATA_DIR, ATTACK_TYPES

logger = get_logger(__name__)

# Define Normalization Logic (Copied from merger.py to ensure consistency)
def normalize_attack(label):
    label = str(label).strip()
    if label == 'Benign':
        return 'Benign'
    
    # Check specific mappings first
    for key, category in ATTACK_TYPES.items():
        if key.lower() in label.lower():
            return category
            
    # Fallback rules
    if 'dos' in label.lower(): return 'DoS'
    if 'ddos' in label.lower(): return 'DDoS'
    if 'portscan' in label.lower(): return 'Reconnaissance'
    if 'bruteforce' in label.lower() or 'force' in label.lower(): return 'Bruteforce'
    if 'xss' in label.lower() or 'sql' in label.lower() or 'injection' in label.lower(): return 'Injection'
    if 'bot' in label.lower(): return 'Botnet'
    
    return 'Malware' # Default catch-all for unknown attacks

def create_balanced_sample():
    # Use the pre-shuffled subset for speed (3M rows)
    input_path = PROCESSED_DATA_DIR / "subset_with_header.csv"
    output_path = PROCESSED_DATA_DIR / "training_sample_balanced.csv"
    
    if not input_path.exists():
        logger.error("Subset dataset not found!")
        return

    logger.info("Scanning dataset to build balanced sample...")
    
    # Target counts per class
    TARGET_PER_CLASS = 50000 
    
    # Storage for sampled dataframes
    samples = {}
    
    # Chunk processing
    chunk_size = 500000
    total_processed = 0
    
    # We need to find: Benign, DDoS, DoS, Injection, Reconnaissance, Bruteforce, Malware
    needed = ['Benign', 'DDoS', 'DoS', 'Injection', 'Reconnaissance', 'Bruteforce', 'Malware']
    collected_counts = {k: 0 for k in needed}
    
    try:
        with pd.read_csv(input_path, chunksize=chunk_size, low_memory=False, on_bad_lines='skip') as reader:
            for chunk in reader:
                chunk.columns = chunk.columns.str.strip()
                
                # Normalize labels
                chunk['Normalized_Label'] = chunk['Label'].apply(normalize_attack)
                
                # Group by normalized label
                for label, group in chunk.groupby('Normalized_Label'):
                    if label not in needed:
                        continue
                        
                    current_count = collected_counts[label]
                    if current_count >= TARGET_PER_CLASS:
                        continue
                        
                    # Calculate how many we still need
                    remaining = TARGET_PER_CLASS - current_count
                    
                    # Take random sample from this chunk
                    if len(group) > remaining:
                        sample = group.sample(remaining)
                    else:
                        sample = group
                        
                    # Add to collection
                    if label not in samples:
                        samples[label] = []
                    samples[label].append(sample)
                    collected_counts[label] += len(sample)
                    
                total_processed += len(chunk)
                logger.info(f"Processed {total_processed} rows. Collected: {collected_counts}")
                
                # Stop if we found enough of everything
                if all(c >= TARGET_PER_CLASS for c in collected_counts.values()):
                    logger.info("Collected enough samples for all classes!")
                    break
        
        # Combine everything
        logger.info("Concatenating samples...")
        final_dfs = []
        for label in samples:
            if samples[label]:
                final_dfs.append(pd.concat(samples[label]))
        
        if not final_dfs:
            logger.error("No samples collected!")
            return

        final_df = pd.concat(final_dfs)
        
        # Shuffle
        final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Replace the raw Label with Normalized for training
        final_df['Label'] = final_df['Normalized_Label']
        final_df.drop(columns=['Normalized_Label'], inplace=True)
        
        logger.info(f"Saving balanced sample ({len(final_df)} rows) to {output_path}...")
        final_df.to_csv(output_path, index=False)
        logger.info("Done!")
        
        # Verify
        print(final_df['Label'].value_counts())

    except Exception as e:
        logger.error(f"Failed: {e}")

if __name__ == "__main__":
    create_balanced_sample()
