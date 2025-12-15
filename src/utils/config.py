"""
Configuration settings for the Network Anomaly Detection system.
"""
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Features to use for detection
SELECTED_FEATURES = [
    'Flow Duration',
    'Total Fwd Packets',
    'Total Backward Packets',
    'Flow Bytes/s',
    'Flow Packets/s',
    'Fwd Packet Length Mean',
    'Bwd Packet Length Mean',
    'Flow IAT Mean',
    'Fwd IAT Mean',
    'Bwd IAT Mean',
    'Fwd PSH Flags',
    'SYN Flag Count',
    'ACK Flag Count',
    'Packet Length Variance',
    'Average Packet Size',
]

# Label column in the dataset
LABEL_COLUMN = 'Label'

# Anomaly detection settings
ISOLATION_FOREST_CONTAMINATION = 0.1  # Expected proportion of anomalies
STATISTICAL_ZSCORE_THRESHOLD = 3.0     # Z-score threshold for anomaly

# Attack type mapping
ATTACK_TYPES = {
    'BENIGN': 'Normal',
    'DDoS': 'DDoS Attack',
    'DoS Hulk': 'DoS Attack',
    'DoS GoldenEye': 'DoS Attack',
    'DoS slowloris': 'DoS Attack',
    'DoS Slowhttptest': 'DoS Attack',
    'PortScan': 'Port Scan',
    'FTP-Patator': 'Brute Force',
    'SSH-Patator': 'Brute Force',
    'Web Attack – Brute Force': 'Web Attack',
    'Web Attack – XSS': 'Web Attack',
    'Web Attack – Sql Injection': 'Web Attack',
    'Infiltration': 'Infiltration',
    'Bot': 'Botnet',
    'Heartbleed': 'Heartbleed',
}

# API settings
API_HOST = "0.0.0.0"
API_PORT = 8000
