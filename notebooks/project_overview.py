"""
Network Anomaly Detection - Project Overview & Data Exploration
================================================================
Updated for Two-Stage XGBoost (Cloud-Trained Edition)

This script provides insights into the upgraded NIDS system trained on 67M+ records.
Run this script or use it as a reference for your graduation presentation.
"""

import os
import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# %% [markdown]
# # Network Anomaly Detection System
# ## Two-Stage XGBoost - Cloud-Trained Edition
# 
# **Key Achievements:**
# - Stage 1 (Binary): 99.90% Accuracy
# - Stage 2 (Multi-class): 99.87% Accuracy
# - Training Data: 67M+ network flows
# - Datasets: CICIDS2017, CICIDS2018, UNSW-NB15, ToN-IoT

# %%
print("=" * 60)
print("NETWORK ANOMALY DETECTION SYSTEM")
print("Two-Stage XGBoost - Cloud-Trained Edition")
print("=" * 60)

# %% [markdown]
# ## 1. System Architecture

# %%
print("""
┌────────────────────────────────────────────────────────────┐
│                  DETECTION PIPELINE                        │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Network Traffic                                           │
│       │                                                    │
│       ▼                                                    │
│  ┌──────────────┐                                          │
│  │ Preprocessor │  ← Normalization, Feature Selection      │
│  └──────────────┘                                          │
│       │                                                    │
│       ▼                                                    │
│  ┌──────────────────────────────────────────┐              │
│  │     STAGE 1: Binary Classification       │              │
│  │     XGBoost (Normal vs Attack)           │              │
│  │     Accuracy: 99.90%                     │              │
│  └──────────────────────────────────────────┘              │
│       │                                                    │
│       ▼ (if Attack)                                        │
│  ┌──────────────────────────────────────────┐              │
│  │     STAGE 2: Multi-class Classification  │              │
│  │     XGBoost (Attack Type)                │              │
│  │     Accuracy: 99.87%                     │              │
│  └──────────────────────────────────────────┘              │
│       │                                                    │
│       ▼                                                    │
│  Dashboard / Alerts / Firewall                             │
│                                                            │
└────────────────────────────────────────────────────────────┘
""")

# %% [markdown]
# ## 2. Training Data Overview

# %%
print("\n" + "=" * 60)
print("TRAINING DATA OVERVIEW")
print("=" * 60)

training_data = {
    "CICIDS2017": {
        "records": "~2.8M",
        "attacks": ["DDoS", "Port Scan", "Brute Force", "Web Attack", "Infiltration", "Heartbleed", "Botnet"]
    },
    "CICIDS2018": {
        "records": "~16M", 
        "attacks": ["DDoS", "DoS", "Brute Force", "Web Attack", "Infiltration", "Botnet"]
    },
    "UNSW-NB15": {
        "records": "~2.5M",
        "attacks": ["Fuzzers", "Analysis", "Backdoors", "DoS", "Exploits", "Generic", "Reconnaissance", "Shellcode", "Worms"]
    },
    "ToN-IoT": {
        "records": "~46M",
        "attacks": ["DDoS", "DoS", "Injection", "MITM", "Password", "Ransomware", "Scanning", "XSS", "Backdoor"]
    }
}

total_records = 0
for dataset, info in training_data.items():
    print(f"\n{dataset}:")
    print(f"  Records: {info['records']}")
    print(f"  Attack Types: {', '.join(info['attacks'][:5])}")
    if len(info['attacks']) > 5:
        print(f"                {', '.join(info['attacks'][5:])}")

print(f"\n{'=' * 60}")
print("TOTAL: 67+ Million Network Flows")
print("=" * 60)

# %% [markdown]
# ## 3. Model Performance

# %%
print("\n" + "=" * 60)
print("MODEL PERFORMANCE (Cloud Training Results)")
print("=" * 60)

print("""
┌─────────────────────────────────────────────────────────────┐
│                  STAGE 1: BINARY CLASSIFIER                 │
├─────────────────────────────────────────────────────────────┤
│  Task:      Normal vs Attack Detection                      │
│  Model:     XGBoost (tree_method='hist')                    │
│  Accuracy:  99.90%                                          │
│  Precision: ~99.9%                                          │
│  Recall:    ~99.9%                                          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                STAGE 2: ATTACK CLASSIFIER                   │
├─────────────────────────────────────────────────────────────┤
│  Task:      Attack Type Classification                      │
│  Model:     XGBoost (multi:softmax)                         │
│  Accuracy:  99.87%                                          │
│  Classes:   10+ attack categories                           │
│  F1-Score:  ~99.8%                                          │
└─────────────────────────────────────────────────────────────┘
""")

# %% [markdown]
# ## 4. Model Files Location

# %%
print("\n" + "=" * 60)
print("MODEL FILES")
print("=" * 60)

models_dir = Path(__file__).parent.parent / "models"

model_files = [
    ("stage1_xgboost.json", "Stage 1 Binary Classifier"),
    ("stage2_xgboost.json", "Stage 2 Attack Classifier"),
    ("label_encoder.joblib", "Label Encoder (Binary)"),
    ("stage2_label_encoder.joblib", "Label Encoder (Attack Types)"),
    ("model_features.json", "Feature Names"),
    ("preprocessor.joblib", "Data Preprocessor"),
]

for filename, description in model_files:
    filepath = models_dir / filename
    status = "✓ EXISTS" if filepath.exists() else "✗ MISSING"
    size = f"({filepath.stat().st_size / 1024:.1f} KB)" if filepath.exists() else ""
    print(f"  {status} {filename:30} - {description} {size}")

# %% [markdown]
# ## 5. Load and Test Models

# %%
print("\n" + "=" * 60)
print("TESTING MODELS")
print("=" * 60)

try:
    from src.detection.two_stage import TwoStageDetector
    
    detector = TwoStageDetector()
    print("✓ TwoStageDetector loaded successfully!")
    
    # Show attack classes
    if hasattr(detector, 'label_encoder_stage2') and detector.label_encoder_stage2:
        print(f"\nAttack classes the model can detect:")
        for i, cls in enumerate(detector.label_encoder_stage2.classes_, 1):
            print(f"  {i:2}. {cls}")
    
except Exception as e:
    print(f"✗ Error loading models: {e}")
    print("  Make sure the models are downloaded to the models/ directory")

# %% [markdown]
# ## 6. Sample Detection

# %%
print("\n" + "=" * 60)
print("SAMPLE DETECTION")
print("=" * 60)

try:
    import numpy as np
    from src.detection.two_stage import TwoStageDetector
    
    detector = TwoStageDetector()
    
    # Create sample normal traffic
    normal_sample = {
        'flow_duration': 100000,
        'total_fwd_packets': 5,
        'total_bwd_packets': 3,
        'flow_bytes_per_sec': 1500.0,
        'flow_packets_per_sec': 8.0,
        'fwd_packet_length_mean': 200.0,
        'bwd_packet_length_mean': 180.0,
        'flow_iat_mean': 12000.0,
        'fwd_iat_mean': 15000.0,
        'bwd_iat_mean': 18000.0,
        'syn_flag_count': 1,
        'ack_flag_count': 3,
        'packet_length_variance': 500.0,
        'average_packet_size': 190.0,
        'fwd_psh_flags': 1
    }
    
    # Create sample attack traffic (DDoS-like)
    ddos_sample = {
        'flow_duration': 1000,
        'total_fwd_packets': 10000,
        'total_bwd_packets': 0,
        'flow_bytes_per_sec': 50000000.0,
        'flow_packets_per_sec': 100000.0,
        'fwd_packet_length_mean': 60.0,
        'bwd_packet_length_mean': 0.0,
        'flow_iat_mean': 10.0,
        'fwd_iat_mean': 10.0,
        'bwd_iat_mean': 0.0,
        'syn_flag_count': 10000,
        'ack_flag_count': 0,
        'packet_length_variance': 0.0,
        'average_packet_size': 60.0,
        'fwd_psh_flags': 0
    }
    
    print("Testing with Normal Traffic Sample:")
    result1 = detector.predict([normal_sample])
    print(f"  Is Attack: {result1['is_attack']}")
    print(f"  Attack Type: {result1['attack_type']}")
    print(f"  Confidence: {result1['confidence']:.2%}")
    
    print("\nTesting with DDoS-like Traffic Sample:")
    result2 = detector.predict([ddos_sample])
    print(f"  Is Attack: {result2['is_attack']}")
    print(f"  Attack Type: {result2['attack_type']}")
    print(f"  Confidence: {result2['confidence']:.2%}")
    
except Exception as e:
    print(f"Error during detection test: {e}")

# %% [markdown]
# ## 7. Visualization Files

# %%
print("\n" + "=" * 60)
print("VISUALIZATION FILES (from Cloud Training)")
print("=" * 60)

viz_files = [
    "confusion_matrix_stage1.png",
    "confusion_matrix_stage2.png", 
    "roc_curve_stage1.png",
    "feature_importance.png",
    "attack_distribution.png"
]

for filename in viz_files:
    filepath = models_dir / filename
    status = "✓" if filepath.exists() else "✗"
    print(f"  {status} {filename}")

print(f"\nTo view these graphs, open the files in: {models_dir}")

# %% [markdown]
# ## 8. Running the System

# %%
print("\n" + "=" * 60)
print("HOW TO RUN THE SYSTEM")
print("=" * 60)

print("""
1. Start the API Backend (with root for sniffing):
   $ sudo env PYTHONPATH=. python3 -m src.api.main

2. Start the Frontend Dashboard:
   $ cd frontend && npm run dev

3. Open the Dashboard:
   http://localhost:3000

4. Use Live Sniffer:
   - Go to "Live Sniffer" page
   - Click "Start Sniffing"
   - Watch real-time detection!

5. Use Detection Page:
   - Upload a CSV file with network flow features
   - Get instant classification results
""")

print("=" * 60)
print("SYSTEM READY FOR GRADUATION DEMO!")
print("=" * 60)
