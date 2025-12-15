
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time
import sys
import os
from pathlib import Path

# Add project root to path so we can import src
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import PROCESSED_DATA_DIR

# Page Config
st.set_page_config(page_title="Sentinel NIDS", layout="wide", initial_sidebar_state="expanded")

# --- Constants & Utility Functions ---
SIM_DATA_PATH = Path("data/processed/merged_dataset.csv")  # We will use our merged file

def normalize_attack(label):
    """Normalize attack labels to standard 6 classes + Benign."""
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

@st.cache_data
def load_real_dataset():
    """Load a sample of the dataset."""
    if not SIM_DATA_PATH.exists():
        return None
        
    # Read first 10k rows for simulation to be fast
    # on_bad_lines='skip' to be safe
    df = pd.read_csv(SIM_DATA_PATH, nrows=10000, on_bad_lines='skip')
    
    # Fix column names - strip whitespace (Crucial!)
    df.columns = df.columns.str.strip()
    
    # Normalize
    df['Normalized_Attack'] = df['Label'].apply(normalize_attack)
    return df

# --- Sidebar Controls ---
st.sidebar.title("🛡️ Sentinel NIDS (Two-Stage)")
st.sidebar.markdown("---")

data_source = st.sidebar.radio("Data Source", ["Real Test Data (CSV)", "Live Packet Capture (Not Implemented)"])
simulation_speed = st.sidebar.slider("Simulation Speed (sec)", 0.05, 2.0, 0.5)
run_simulation = st.sidebar.checkbox("Start Live Simulation", value=False)

st.sidebar.markdown("---")
st.sidebar.subheader("Attack Injection")
traffic_type = st.sidebar.selectbox(
    "Inject Traffic Type", 
    ["Benign", "DDoS", "Bruteforce", "Malware", "DoS", "Reconnaissance", "Injection"]
)

# --- Main Dashboard ---
st.title("Network Intrusion Detection System")
st.markdown("### Real-Time Network Traffic Monitoring")

# Metrics Placeholders
col1, col2, col3, col4 = st.columns(4)
metric_total = col1.empty()
metric_benign = col2.empty()
metric_attack = col3.empty()
metric_status = col4.empty()

# Charts Placeholders
st.markdown("---")
chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    st.subheader("Traffic Class Distribution")
    plot_dist = st.empty()

with chart_col2:
    st.subheader("Explainable AI (SHAP)")
    # Load static SHAP images if available
    shap_bin_path = PROCESSED_DATA_DIR / "shap_summary_Binary_Stage1.png"
    shap_multi_path = PROCESSED_DATA_DIR / "shap_summary_Multiclass_Stage2.png"
    
    tab1, tab2 = st.tabs(["Binary Model", "Multiclass Model"])
    
    with tab1:
        if shap_bin_path.exists():
            st.image(str(shap_bin_path), caption="Feature Importance (Binary Stage)")
        else:
            st.info("SHAP plots not found. Run training first.")
            
    with tab2:
        if shap_multi_path.exists():
            st.image(str(shap_multi_path), caption="Feature Importance (Multiclass Stage)")
        else:
            st.info("SHAP plots not found. Run training first.")

# --- Simulation Logic ---
if data_source == "Real Test Data (CSV)":
    df = load_real_dataset()
    
    if df is None:
        st.error("Dataset not found! Please run the merger script first.")
        st.stop()

    # Load Models & Features
    try:
        import xgboost as xgb
        import joblib
        import json
        from src.utils.config import MODELS_DIR
        
        model_s1 = xgb.Booster()
        model_s1.load_model(MODELS_DIR / "stage1_xgboost.json")
        
        model_s2 = xgb.Booster()
        model_s2.load_model(MODELS_DIR / "stage2_xgboost.json")
        
        le = joblib.load(MODELS_DIR / "label_encoder.joblib")
        # Load Stage 2 Encoder for correct attack mapping
        try:
            le_s2 = joblib.load(MODELS_DIR / "stage2_label_encoder.joblib")
        except:
            le_s2 = le # Fallback (risky but better than crash)
        
        with open(MODELS_DIR / "model_features.json", 'r') as f:
            features = json.load(f)
            
        models_loaded = True
        st.sidebar.success("✅ Models Loaded")
    except Exception as e:
        models_loaded = False
        st.sidebar.warning("⚠️ Models not trained yet. Showing dataset labels only.")
        
    if run_simulation:
        batch_size = 5
        
        # Keep track of counts
        total_counts = {"Benign": 0, "Attack": 0}
        
        # For charts
        history_df = pd.DataFrame(columns=["Time", "Benign", "Attack"])
        
        for i in range(0, len(df), batch_size):
            if not run_simulation:
                break
                
            batch = df.iloc[i:i+batch_size]
            
            # INFERENCE
            preds = []
            if models_loaded:
                # Prepare features (ensure order matches training)
                # Fill missing cols with 0
                X_batch = batch.reindex(columns=features, fill_value=0)
                X_batch.replace([np.inf, -np.inf], np.nan, inplace=True) # FIX: Sanitize Inf values
                dmatrix = xgb.DMatrix(X_batch)
                
                # Stage 1
                s1_probs = model_s1.predict(dmatrix)
                s1_preds = (s1_probs > 0.5).astype(int)
                
                for idx, is_attack in enumerate(s1_preds):
                    if is_attack == 0:
                        preds.append("Benign")
                    else:
                        # Stage 2 (Only if Attack)
                        # Need to isolate this single row's features?
                        # Or predict all batch and mask.
                        # For simplicity, predict all through S2 then pick
                        s2_probs = model_s2.predict(dmatrix) 
                        # s2_probs is (N, Classes)
                        # s2_probs is (N, Classes)
                        pred_class_idx = np.argmax(s2_probs[idx])
                        # Use Stage 2 encoder to decode
                        pred_label = le_s2.inverse_transform([pred_class_idx])[0]
                        preds.append(pred_label)
            else:
                # Fallback to ground truth if no model
                preds = batch['Normalized_Attack'].tolist()
            
            # Update metrics based on PREDICTIONS
            for p in preds:
                if p == 'Benign':
                    total_counts['Benign'] += 1
                else:
                    total_counts['Attack'] += 1
            
            metric_total.metric("Total Packets", i + batch_size)
            metric_benign.metric("Benign Packets", total_counts['Benign'])
            metric_attack.metric("Malicious Packets", total_counts['Attack'], delta_color="inverse")
            
            # Status Indicator
            last_pred = preds[-1]
            state_color = "green" if last_pred == 'Benign' else "red"
            metric_status.markdown(f"**Latest Classification:** :{state_color}[{last_pred.upper()}]")
            
            # Update Plot
            counts_df = pd.DataFrame({
                "Type": ["Benign", "Attack"],
                "Count": [total_counts['Benign'], total_counts['Attack']]
            })
            fig = px.pie(counts_df, values='Count', names='Type', color='Type', 
                         color_discrete_map={'Benign':'green', 'Attack':'red'},
                         hole=0.4)
            plot_dist.plotly_chart(fig, use_container_width=True)
            
            time.sleep(simulation_speed)
    else:
        st.info("Click 'Start Live Simulation' to begin monitoring.")
