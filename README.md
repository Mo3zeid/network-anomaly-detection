# 🛡️ Network Anomaly Detection System

A real-time **Intrusion Detection System (IDS)** powered by **Two-Stage XGBoost** machine learning, trained on **67+ million network flows** from 4 major cybersecurity datasets.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Cloud--Trained-green.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-99.9%25-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 🎯 Key Features

| Feature | Description |
|---------|-------------|
| 🧠 **Two-Stage XGBoost** | Stage 1: Binary (99.90%), Stage 2: Multi-class (99.87%) |
| 📡 **Live Packet Sniffer** | Real-time network traffic analysis using Scapy |
| 🎛️ **React Dashboard** | Interactive monitoring with alerts, analytics, rules |
| 🔥 **Rule Engine** | Rate limiting, IP whitelist/blacklist, port scan detection |
| ☁️ **Cloud-Trained** | Trained on Google Cloud with 67M+ records |

---

## 📊 Training Data

| Dataset | Records | Attack Types |
|---------|---------|--------------|
| CICIDS2017 | ~2.8M | DDoS, Port Scan, Brute Force, Web Attack, Botnet |
| CICIDS2018 | ~16M | DDoS, DoS, Brute Force, Infiltration |
| UNSW-NB15 | ~2.5M | Exploits, Fuzzers, Backdoors, Reconnaissance |
| ToN-IoT | ~46M | DDoS, Ransomware, Injection, XSS, Scanning |

**Total: 67+ Million Network Flows**

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/network-anomaly-detection.git
cd network-anomaly-detection

# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend && npm install && cd ..
```

### 2. Download Pre-trained Models

Download the trained models from [Releases](https://github.com/YOUR_USERNAME/network-anomaly-detection/releases) and place in `models/` folder:
- `stage1_xgboost.json`
- `stage2_xgboost.json`
- `label_encoder.joblib`
- `stage2_label_encoder.joblib`
- `preprocessor.joblib`

### 3. Start the System

```bash
# Terminal 1: Start API (requires sudo for packet sniffing)
sudo env PYTHONPATH=. python3 -m src.api.main

# Terminal 2: Start Dashboard
cd frontend && npm run dev
```

### 4. Open Dashboard

Navigate to: **http://localhost:3000**

---

## 📁 Project Structure

```
network-anomaly-detection/
├── src/
│   ├── api/              # FastAPI backend
│   ├── detection/        # ML detection (Two-Stage XGBoost)
│   ├── sniffing/         # Packet capture & feature extraction
│   └── data/             # Data preprocessing
├── frontend/             # React/Next.js dashboard
├── models/               # Trained ML models (download separately)
├── notebooks/            # Analysis scripts
└── requirements.txt
```

---

## 🏗️ System Architecture

```
Network Traffic → Scapy Sniffer → Feature Extraction (15 features)
                                          ↓
                    ┌─────────────────────────────────────┐
                    │     STAGE 1: Binary Classifier      │
                    │     (Normal vs Attack) - 99.90%     │
                    └─────────────────────────────────────┘
                                          ↓ (if Attack)
                    ┌─────────────────────────────────────┐
                    │   STAGE 2: Attack Type Classifier   │
                    │    (10+ attack categories) - 99.87% │
                    └─────────────────────────────────────┘
                                          ↓
                    Dashboard → Alerts → Firewall Rules
```

---

## 🖥️ Dashboard Screenshots

- **Live Sniffer**: Real-time packet capture with ML detection
- **Analytics**: Top talkers, attack distribution, model performance
- **Rules**: Rate limits, whitelist/blacklist, thresholds
- **Alerts**: Historical attack notifications

---

## 🔧 Tech Stack

| Component | Technology |
|-----------|------------|
| ML Models | XGBoost, Scikit-learn |
| Backend | Python, FastAPI |
| Packet Capture | Scapy |
| Frontend | React, Next.js |
| Training | Google Cloud (Vertex AI) |

---

## 📈 Model Performance

| Metric | Stage 1 (Binary) | Stage 2 (Multi-class) |
|--------|------------------|----------------------|
| Accuracy | 99.90% | 99.87% |
| Precision | 99.85% | 99.80% |
| Recall | 99.87% | 99.85% |
| F1-Score | 99.86% | 99.82% |

---

## 🎓 Academic Project

This project was developed as a **Graduation Project** for Network Security / Cybersecurity program.

**Key Contributions:**
- Two-stage detection architecture
- Multi-dataset training (4 datasets, 67M+ records)
- Real-time packet analysis with feature extraction
- Interactive React dashboard

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

## 👤 Author

**Moaz Elmorsy**
- GitHub: [@mo3z3id](https://github.com/mo3z3id)
