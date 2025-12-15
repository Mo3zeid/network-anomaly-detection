# Machine Learning Models & Dataset Documentation

## Complete Technical Reference for Network Anomaly Detection

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Dataset: CICIDS2017](#2-dataset-cicids2017)
3. [Feature Engineering](#3-feature-engineering)
4. [Detection Models](#4-detection-models)
5. [Training Pipeline](#5-training-pipeline)
6. [Model Evaluation](#6-model-evaluation)
7. [Model Comparison & Selection Rationale](#7-model-comparison--selection-rationale)
8. [Preprocessing & Normalization](#8-preprocessing--normalization)
9. [Hyperparameters](#9-hyperparameters)
10. [Serialization & Deployment](#10-serialization--deployment)
11. [Limitations & Future Work](#11-limitations--future-work)

---

## 1. Executive Summary

This document provides complete technical documentation of the machine learning components used in the Network Anomaly Detection System.

### System Overview

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Dataset** | CICIDS2017 | Training & evaluation data |
| **Preprocessing** | StandardScaler (sklearn) | Feature normalization |
| **Anomaly Detection** | Statistical (Z-score/IQR) + Isolation Forest | Binary anomaly detection |
| **Classification** | Random Forest Classifier | Multi-class attack type identification |

### Model Performance Summary (Real Evaluation Results)

> **Evaluated on:** 100,000 samples from CICIDS2017 dataset  
> **Test set:** 20,000 samples (80/20 stratified split)  
> **Evaluation date:** December 10, 2025

| Model | Accuracy | Precision | Recall | F1 Score | Training Data |
|-------|----------|-----------|--------|----------|---------------|
| Statistical Detector (Z-score) | **76.71%** | 42.14% | 48.29% | 45.01% | Normal traffic only (64,210 samples) |
| Isolation Forest | **80.97%** | 52.05% | 45.07% | 48.31% | Normal traffic only (64,210 samples) |
| Combined (Statistical OR IF) | **76.54%** | 41.83% | 48.29% | 44.83% | - |
| Attack Classifier | **98.35%** | 98.31% (weighted) | 98.35% | 98.30% | All labeled traffic (80,000 samples) |

---

## 2. Dataset: CICIDS2017

### 2.1 Why CICIDS2017?

The **Canadian Institute for Cybersecurity Intrusion Detection System 2017** dataset was chosen for the following reasons:

| Criterion | CICIDS2017 Advantage | Comparison to Alternatives |
|-----------|---------------------|---------------------------|
| **Recency** | 2017 (relatively modern) | KDD99 (1999) is outdated with obsolete attack patterns |
| **Realism** | Captured from real network infrastructure | Synthetic datasets lack real-world traffic patterns |
| **Completeness** | 80+ features extracted from pcap files | Many datasets have limited features |
| **Labeled Data** | Fully labeled with 15 attack types | Essential for supervised learning |
| **Size** | ~2.8 million samples | Large enough for robust training |
| **Availability** | Free and publicly available | Commercial datasets are expensive |
| **Academic Recognition** | Widely cited in research papers | Enables comparison with published results |

### 2.2 Alternatives Considered

| Dataset | Year | Why Not Chosen |
|---------|------|----------------|
| **KDD Cup 1999** | 1999 | Outdated attack patterns, unrealistic traffic distribution |
| **NSL-KDD** | 2009 | Improved KDD but still outdated, lacks modern attack types |
| **UNSW-NB15** | 2015 | Good alternative, but fewer attack categories than CICIDS2017 |
| **ISCX 2012** | 2012 | Predecessor to CICIDS2017, less comprehensive |
| **CICIDS2018** | 2018 | Newer but more complex structure, similar attack coverage |

### 2.4 Attack Types Distribution

| Attack Type | Count | Percentage | Category |
|-------------|-------|------------|----------|
| **BENIGN** | ~2,273,097 | 80.3% | Normal Traffic |
| **DDoS** | ~128,027 | 4.5% | Denial of Service |
| **PortScan** | ~158,930 | 5.6% | Reconnaissance |
| **DoS Hulk** | ~231,073 | 8.2% | Denial of Service |
| **DoS GoldenEye** | ~10,293 | 0.4% | Denial of Service |
| **DoS Slowloris** | ~5,796 | 0.2% | Denial of Service |
| **DoS Slowhttptest** | ~5,499 | 0.2% | Denial of Service |
| **FTP-Patator** | ~7,938 | 0.3% | Brute Force |
| **SSH-Patator** | ~5,897 | 0.2% | Brute Force |
| **Web Attack - Brute Force** | ~1,507 | 0.05% | Web Attack |
| **Web Attack - XSS** | ~652 | 0.02% | Web Attack |
| **Web Attack - SQL Injection** | ~21 | <0.01% | Web Attack |
| **Infiltration** | ~36 | <0.01% | Advanced Persistent |
| **Bot** | ~1,966 | 0.07% | Botnet |
| **Heartbleed** | ~11 | <0.01% | Vulnerability Exploit |

### 2.5 Data Collection Methodology

The CICIDS2017 dataset was created using a realistic testbed:

```
┌─────────────────────────────────────────────────────────────────────┐
│                      DATA COLLECTION SETUP                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│   [Attack Network]          [Victim Network]                         │
│         │                         │                                   │
│    ┌────┴────┐              ┌─────┴─────┐                            │
│    │ Attack  │     ────►    │  Victim   │                            │
│    │ Machine │   Network    │  Servers  │                            │
│    │ (Kali)  │   Traffic    │ (Various) │                            │
│    └─────────┘              └─────┬─────┘                            │
│                                   │                                   │
│                           ┌───────┴───────┐                          │
│                           │   CICFlowMeter │                          │
│                           │ (Feature       │                          │
│                           │  Extraction)   │                          │
│                           └───────┬───────┘                          │
│                                   │                                   │
│                           ┌───────┴───────┐                          │
│                           │   CSV Output   │                          │
│                           │   (78 features) │                          │
│                           └───────────────┘                          │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.6 Dataset Files

| Day | File Name | Attack Types Included |
|-----|-----------|----------------------|
| Monday | `Monday-WorkingHours.pcap_ISCX.csv` | BENIGN only |
| Tuesday | `Tuesday-WorkingHours.pcap_ISCX.csv` | FTP-Patator, SSH-Patator |
| Wednesday | `Wednesday-workingHours.pcap_ISCX.csv` | DoS (Slowloris, Slowhttptest, Hulk, GoldenEye) |
| Thursday AM | `Thursday-WorkingHours-Morning...csv` | Web Attacks (Brute Force, XSS, SQL Injection) |
| Thursday PM | `Thursday-WorkingHours-Afternoon...csv` | Infiltration |
| Friday AM | `Friday-WorkingHours-Morning.pcap_ISCX.csv` | Bot |
| Friday PM1 | `Friday-WorkingHours-Afternoon-PortScan.csv` | PortScan |
| Friday PM2 | `Friday-WorkingHours-Afternoon-DDos.csv` | DDoS |

---

## 3. Feature Engineering

### 3.1 Why Only 15 Features?

The original CICIDS2017 dataset contains 78 features. We selected 15 features for the following reasons:

| Reason | Explanation |
|--------|-------------|
| **Dimensionality Reduction** | Too many features cause overfitting and slow inference |
| **Feature Correlation** | Many original features are highly correlated (redundant) |
| **Real-time Computation** | Selected features can be computed from live traffic |
| **Research-backed Selection** | Features chosen based on published IDS research |
| **Interpretability** | Fewer features = easier to understand model decisions |

### 3.2 Feature Selection Methodology

Features were selected based on three criteria:

1. **Importance Ranking** - Features with high importance scores in initial Random Forest training
2. **Extractability** - Must be computable from raw network packets in real-time
3. **Distinctiveness** - Features that differ significantly between attack and normal traffic

### 3.3 The 15 Selected Features - Complete Mathematical Reference

---

## CATEGORY 1: Flow Timing Features (4 features)

### Feature 1: Flow Duration

**Definition:** Total time elapsed from the first packet to the last packet in a network flow.

**Formula:**
```
Flow Duration = t_last - t_first
```

Where:
- `t_first` = Timestamp of the first packet in the flow
- `t_last` = Timestamp of the last packet in the flow

**Unit:** Microseconds (μs)

**Calculation in Code:**
```python
start_time = packets[0].time          # First packet timestamp
end_time = packets[-1].time           # Last packet timestamp
duration = end_time - start_time      # Duration in seconds
duration_micros = duration * 1_000_000  # Convert to microseconds
```

**Example:**
```
First packet:  10:00:00.000000
Last packet:   10:00:00.500000
Flow Duration = 0.5 seconds = 500,000 μs
```

**Attack Detection Significance:**
| Attack Type | Typical Duration |
|-------------|------------------|
| SYN Flood | Very short (< 1ms per connection) |
| Slowloris | Extremely long (minutes to hours) |
| Normal HTTP | Medium (100ms - 5s) |
| Port Scan | Very short (< 100ms per probe) |

---

### Feature 2: Flow IAT Mean (Inter-Arrival Time)

**Definition:** Average time between consecutive packets in the entire flow, regardless of direction.

**Formula:**
```
                    n-1
                    Σ (t[i+1] - t[i])
                   i=1
Flow IAT Mean = ────────────────────
                      n - 1
```

Where:
- `n` = Total number of packets in the flow
- `t[i]` = Timestamp of packet i

**Unit:** Microseconds (μs)

**Calculation in Code:**
```python
flow_iats = []
last_time = packets[0].time

for i, pkt in enumerate(packets[1:], 1):
    iat = pkt.time - last_time
    flow_iats.append(iat)
    last_time = pkt.time

flow_iat_mean = np.mean(flow_iats) * 1_000_000  # Convert to μs
```

**Example:**
```
Packet times: [0.000, 0.100, 0.150, 0.300]
IATs:         [0.100, 0.050, 0.150]
Flow IAT Mean = (0.100 + 0.050 + 0.150) / 3 = 0.100 sec = 100,000 μs
```

**Attack Detection Significance:**
| Traffic Type | IAT Pattern |
|--------------|-------------|
| Normal browsing | Irregular, human-paced |
| Bot/Script | Very regular, automated |
| DDoS flood | Extremely low IAT (< 1ms) |
| Slowloris | Very high IAT (seconds) |

---

### Feature 3: Fwd IAT Mean (Forward Inter-Arrival Time)

**Definition:** Average time between consecutive packets traveling from source to destination only.

**Formula:**
```
                       n_fwd-1
                         Σ (t_fwd[i+1] - t_fwd[i])
                        i=1
Fwd IAT Mean = ─────────────────────────────────
                         n_fwd - 1
```

Where:
- `n_fwd` = Number of forward packets
- `t_fwd[i]` = Timestamp of forward packet i

**Unit:** Microseconds (μs)

**Calculation in Code:**
```python
fwd_iats = []
last_fwd_time = 0
src_ip_fwd = packets[0][IP].src  # First packet defines "forward" direction

for pkt in packets:
    if pkt[IP].src == src_ip_fwd:  # Forward direction
        if last_fwd_time > 0:
            fwd_iats.append(pkt.time - last_fwd_time)
        last_fwd_time = pkt.time

fwd_iat_mean = np.mean(fwd_iats) * 1_000_000  # Convert to μs
```

**Attack Detection Significance:**
- **Brute Force:** Very regular Fwd IAT (automated login attempts)
- **Normal user:** Irregular Fwd IAT (typing, clicking)

---

### Feature 4: Bwd IAT Mean (Backward Inter-Arrival Time)

**Definition:** Average time between consecutive packets traveling from destination back to source.

**Formula:**
```
                       n_bwd-1
                         Σ (t_bwd[i+1] - t_bwd[i])
                        i=1
Bwd IAT Mean = ─────────────────────────────────
                         n_bwd - 1
```

**Unit:** Microseconds (μs)

**Attack Detection Significance:**
- **DoS victim:** No backward packets (server overwhelmed)
- **Normal:** Regular responses from server

---

## CATEGORY 2: Packet Count Features (2 features)

### Feature 5: Total Fwd Packets

**Definition:** Count of all packets traveling from source IP to destination IP.

**Formula:**
```
Total Fwd Packets = Σ 1, for each packet where packet.src_ip == flow.src_ip
```

**Unit:** Count (integer)

**Calculation in Code:**
```python
fwd_pkts = 0
src_ip_fwd = packets[0][IP].src  # First packet determines forward direction

for pkt in packets:
    if pkt[IP].src == src_ip_fwd:
        fwd_pkts += 1
```

**Attack Detection Significance:**
| Attack Type | Fwd Packets Pattern |
|-------------|---------------------|
| Port Scan | Many (1 per port probed) |
| SYN Flood | Extremely high |
| Normal HTTP | Moderate (10-50) |

---

### Feature 6: Total Backward Packets

**Definition:** Count of all packets traveling from destination back to source.

**Formula:**
```
Total Bwd Packets = Total Packets - Total Fwd Packets
```

**Unit:** Count (integer)

**Attack Detection Significance:**
- **DDoS:** Fwd >> Bwd (asymmetric, server can't respond)
- **Normal:** Fwd ≈ Bwd (request-response balance)
- **Port Scan:** Few Bwd packets (RST or no response)

---

## CATEGORY 3: Flow Rate Features (2 features)

### Feature 7: Flow Bytes/s

**Definition:** Total bytes transferred per second during the flow.

**Formula:**
```
                    Total Bytes Fwd + Total Bytes Bwd
Flow Bytes/s = ─────────────────────────────────────────
                         Flow Duration (seconds)
```

**Unit:** Bytes per second

**Calculation in Code:**
```python
fwd_len_sum = sum(len(pkt) for pkt in packets if pkt[IP].src == src_ip_fwd)
bwd_len_sum = sum(len(pkt) for pkt in packets if pkt[IP].src != src_ip_fwd)
total_bytes = fwd_len_sum + bwd_len_sum

flow_bytes_per_sec = total_bytes / duration  # duration in seconds
```

**Example:**
```
Total bytes: 50,000 bytes
Duration: 0.5 seconds
Flow Bytes/s = 50,000 / 0.5 = 100,000 bytes/s = 100 KB/s
```

**Attack Detection Significance:**
| Traffic Type | Typical Bytes/s |
|--------------|-----------------|
| Normal web browsing | 10-500 KB/s |
| Video streaming | 1-10 MB/s |
| DDoS attack | 10+ MB/s |
| Port scan | Low (small probes) |

---

### Feature 8: Flow Packets/s

**Definition:** Number of packets transmitted per second during the flow.

**Formula:**
```
                         Total Packets
Flow Packets/s = ────────────────────────
                    Flow Duration (seconds)
```

**Unit:** Packets per second

**Calculation in Code:**
```python
flow_packets_per_sec = len(packets) / duration
```

**Attack Detection Significance:**
| Traffic Type | Packets/s |
|--------------|-----------|
| Normal TCP | 10-100 pps |
| Port Scan | 100-1000 pps |
| SYN Flood | 1000+ pps |
| DDoS | 10,000+ pps |

---

## CATEGORY 4: Packet Size Features (4 features)

### Feature 9: Fwd Packet Length Mean

**Definition:** Average size (in bytes) of packets sent from source to destination.

**Formula:**
```
                              Σ len(fwd_packet[i])
Fwd Packet Length Mean = ─────────────────────────
                              n_fwd
```

Where:
- `len(fwd_packet[i])` = Size of forward packet i in bytes
- `n_fwd` = Total number of forward packets

**Unit:** Bytes

**Calculation in Code:**
```python
fwd_lens = [len(pkt) for pkt in packets if pkt[IP].src == src_ip_fwd]
fwd_len_mean = np.mean(fwd_lens) if fwd_lens else 0
```

**Attack Detection Significance:**
- **Attack tools:** Consistent packet sizes (low variance)
- **Normal traffic:** Variable sizes (different content)

---

### Feature 10: Bwd Packet Length Mean

**Definition:** Average size (in bytes) of packets sent from destination back to source.

**Formula:**
```
                              Σ len(bwd_packet[i])
Bwd Packet Length Mean = ─────────────────────────
                              n_bwd
```

---

### Feature 11: Packet Length Variance

**Definition:** Statistical variance of all packet sizes in the flow. Measures how much packet sizes differ from each other.

**Formula:**
```
                          Σ (len[i] - μ)²
Variance (σ²) = ────────────────────────
                          n
```

Where:
- `len[i]` = Size of packet i
- `μ` = Mean packet size (Average Packet Size)
- `n` = Total number of packets

**Unit:** Bytes² (squared bytes)

**Calculation in Code:**
```python
pkt_sizes = [len(pkt) for pkt in packets]
variance = np.var(pkt_sizes) if pkt_sizes else 0
```

**Example:**
```
Packet sizes: [100, 100, 100, 100]  → Variance = 0 (identical)
Packet sizes: [50, 100, 150, 200]  → Variance = 2500 (variable)
```

**Attack Detection Significance:**
| Traffic Type | Variance |
|--------------|----------|
| Attack tool (scripted) | **Low** (consistent packet sizes) |
| Normal human traffic | **High** (different requests/responses) |
| SYN flood | **Very low** (all SYN packets same size) |

> **Key Insight:** This is the #1 most important feature (16.4% importance). Automated attacks have suspiciously low variance!

---

### Feature 12: Average Packet Size

**Definition:** Mean size of all packets in the flow.

**Formula:**
```
                         Σ len(packet[i])
Average Packet Size = ────────────────────
                              n
```

**Unit:** Bytes

**Calculation in Code:**
```python
pkt_sizes = [len(pkt) for pkt in packets]
avg_size = np.mean(pkt_sizes) if pkt_sizes else 0
```

---

## CATEGORY 5: TCP Flag Features (3 features)

### Feature 13: Fwd PSH Flags

**Definition:** Count of TCP PUSH flags in forward direction packets.

**What is PSH flag?**
The TCP PSH (Push) flag tells the receiving end to push data to the application immediately, rather than buffering it.

**Formula:**
```
Fwd PSH Flags = Σ 1, for each fwd packet where TCP.flags contains 'P'
```

**Unit:** Count

**Calculation in Code:**
```python
fwd_psh_flags = 0
for pkt in packets:
    if pkt[IP].src == src_ip_fwd and TCP in pkt:
        if 'P' in pkt[TCP].flags:
            fwd_psh_flags += 1
```

**Attack Detection Significance:**
- **Web attacks:** High PSH count (pushing HTTP requests)
- **Normal browsing:** Moderate PSH count

---

### Feature 14: SYN Flag Count

**Definition:** Total number of TCP SYN flags in all packets of the flow.

**What is SYN flag?**
The TCP SYN (Synchronize) flag is used to initiate a TCP connection (first step of 3-way handshake).

**Formula:**
```
SYN Flag Count = Σ 1, for each packet where TCP.flags contains 'S'
```

**Unit:** Count

**Calculation in Code:**
```python
syn_flags = 0
for pkt in packets:
    if TCP in pkt and 'S' in pkt[TCP].flags:
        syn_flags += 1
```

**Attack Detection Significance:**
| Traffic Type | SYN Count |
|--------------|-----------|
| Normal connection | 1-2 (one handshake) |
| Port scan | Many (1 per port) |
| SYN flood | Extremely high |

---

### Feature 15: ACK Flag Count

**Definition:** Total number of TCP ACK flags in all packets of the flow.

**What is ACK flag?**
The TCP ACK (Acknowledge) flag confirms receipt of data. Normal TCP connections have ACK in most packets.

**Formula:**
```
ACK Flag Count = Σ 1, for each packet where TCP.flags contains 'A'
```

**Unit:** Count

**Attack Detection Significance:**
- **Normal traffic:** High ACK count (every response has ACK)
- **SYN flood:** Low ACK count (no handshake completion)
- **Ratio indicator:** `SYN / ACK ratio` high = potential attack

### 3.4 Feature Importance Analysis (Real Data from Training)

Based on our trained Random Forest model (evaluated December 10, 2025):

| Rank | Feature | Importance Score |
|------|---------|-----------------|
| 1 | **Packet Length Variance** | 0.1640 (16.40%) |
| 2 | **Bwd Packet Length Mean** | 0.1321 (13.21%) |
| 3 | **Average Packet Size** | 0.1225 (12.25%) |
| 4 | **Fwd Packet Length Mean** | 0.1121 (11.21%) |
| 5 | **Fwd IAT Mean** | 0.0721 (7.21%) |
| 6 | **Flow Packets/s** | 0.0639 (6.39%) |
| 7 | **Total Fwd Packets** | 0.0624 (6.24%) |
| 8 | **Flow Bytes/s** | 0.0575 (5.75%) |
| 9 | **Flow Duration** | 0.0526 (5.26%) |
| 10 | **Flow IAT Mean** | 0.0476 (4.76%) |
| 11 | **ACK Flag Count** | 0.0468 (4.68%) |
| 12 | **Total Backward Packets** | 0.0453 (4.53%) |
| 13 | **Bwd IAT Mean** | 0.0166 (1.66%) |
| 14 | **Fwd PSH Flags** | 0.0023 (0.23%) |
| 15 | **SYN Flag Count** | 0.0021 (0.21%) |

```
Feature Importance Ranking (from trained model)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Packet Length Variance ████████████████████████████████████████  16.40%
Bwd Packet Length Mean ████████████████████████████████          13.21%
Average Packet Size    ██████████████████████████████            12.25%
Fwd Packet Length Mean ████████████████████████████              11.21%
Fwd IAT Mean           ██████████████████                         7.21%
Flow Packets/s         ████████████████                           6.39%
Total Fwd Packets      ███████████████                            6.24%
Flow Bytes/s           ██████████████                             5.75%
Flow Duration          █████████████                              5.26%
Flow IAT Mean          ████████████                               4.76%
ACK Flag Count         ███████████                                4.68%
Total Backward Packets ███████████                                4.53%
Bwd IAT Mean           ████                                       1.66%
Fwd PSH Flags          █                                          0.23%
SYN Flag Count         █                                          0.21%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

> **Key Insight:** Packet size features (variance, mean) are the most important, followed by timing features. TCP flags (SYN, PSH) have surprisingly low importance in this dataset.

### 3.5 Why Each Feature Was Chosen

| Feature | Attack Detection Use Case |
|---------|---------------------------|
| **Flow Duration** | DDoS attacks: very short or very long flows; Slowloris: extremely long |
| **Total Fwd Packets** | Port scans: many SYN packets; DDoS: flooding patterns |
| **Total Backward Packets** | DDoS: few responses; Brute Force: many responses |
| **Flow Bytes/s** | DDoS: extremely high; Infiltration: low and slow |
| **Flow Packets/s** | Port scan: high packet rate; DoS: flooding |
| **Fwd Packet Length Mean** | Attack tools use consistent packet sizes |
| **Bwd Packet Length Mean** | Servers respond with error messages (small) or blocked |
| **Flow IAT Mean** | Bots and scanners have regular, automated timing |
| **Fwd IAT Mean** | Brute force: rapid repeated requests |
| **Bwd IAT Mean** | DoS targets: delayed or no responses |
| **Fwd PSH Flags** | Web attacks: HTTP requests push data |
| **SYN Flag Count** | Port scans: SYN flood signature |
| **ACK Flag Count** | Incomplete handshakes = attack indicator |
| **Packet Length Variance** | Attack tools have low variance; normal traffic varies |
| **Average Packet Size** | Different attack profiles have distinct averages |

---

## 4. Detection Models

### 4.1 System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DETECTION SYSTEM ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│   Input: Network Flow (15 features)                                  │
│              │                                                        │
│              ▼                                                        │
│   ┌──────────────────────┐                                           │
│   │    Preprocessor       │ ◄─── StandardScaler normalization        │
│   │    (StandardScaler)   │                                          │
│   └──────────┬───────────┘                                           │
│              │                                                        │
│              ▼                                                        │
│   ┌──────────────────────────────────────────────────┐               │
│   │              DETECTION LAYER                      │               │
│   │  ┌────────────────┐    ┌───────────────────────┐ │               │
│   │  │  Statistical   │    │   Isolation Forest    │ │               │
│   │  │   Detector     │    │      Detector         │ │               │
│   │  │                │    │                       │ │               │
│   │  │  • Z-score     │    │  • 100 trees          │ │               │
│   │  │  • IQR         │    │  • 10% contamination  │ │               │
│   │  │  • Threshold:3 │    │  • Trained on normal  │ │               │
│   │  └───────┬────────┘    └───────────┬───────────┘ │               │
│   │          │                         │              │               │
│   │          └────────────┬────────────┘              │               │
│   │                       │                           │               │
│   │                       ▼                           │               │
│   │               ┌───────────────┐                   │               │
│   │               │ Combine:      │                   │               │
│   │               │ OR logic +    │                   │               │
│   │               │ max(scores)   │                   │               │
│   │               └───────┬───────┘                   │               │
│   └─────────────────────────────────────────────────────────────────┘ │               │
│              │                                                        │
│              ▼                                                        │
│   ┌──────────────────────────────────────────────────┐               │
│   │           [If Anomaly Detected]                   │               │
│   │                                                   │               │
│   │   ┌──────────────────────────────────────────┐   │               │
│   │   │       Attack Classifier                   │   │               │
│   │   │       (Random Forest)                     │   │               │
│   │   │                                           │   │               │
│   │   │   • 100 trees, max_depth=20               │   │               │
│   │   │   • 8 attack categories                   │   │               │
│   │   │   • Returns type + confidence             │   │               │
│   │   └──────────────────────────────────────────┘   │               │
│   └──────────────────────────────────────────────────┘               │
│                       │                                               │
│                       ▼                                               │
│   Output: {is_anomaly, score, attack_type, confidence}               │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Model 1: Statistical Detector

#### Purpose
Fast, interpretable baseline detection using classical statistical methods.

#### Algorithms

**Z-Score Method:**
```
                    X - μ
            Z = ─────────
                    σ

Where:
    X = feature value
    μ = mean (from training data)
    σ = standard deviation (from training data)

Decision Rule:
    is_anomaly = max(|Z| across all features) > 3.0
```

**IQR (Interquartile Range) Method:**
```
    IQR = Q3 - Q1
    Lower Bound = Q1 - 1.5 × IQR
    Upper Bound = Q3 + 1.5 × IQR

Decision Rule:
    is_anomaly = any(X < Lower) OR any(X > Upper)
```

#### Why Z-Score and IQR?

| Advantage | Explanation |
|-----------|-------------|
| **Speed** | O(n) computation - suitable for real-time |
| **No Training Required** | Just compute mean/std from normal data |
| **Interpretable** | "This flow is 5 standard deviations from normal" |
| **Memory Efficient** | Only store mean/std per feature |
| **Robust to Distribution** | IQR works even for non-Gaussian data |

#### Implementation Details

```python
# File: src/detection/statistical.py

class StatisticalDetector:
    def __init__(self, zscore_threshold=3.0):
        self.zscore_threshold = zscore_threshold
        self.mean = None      # Learned from normal traffic
        self.std = None       # Learned from normal traffic
        self.q1 = None        # 25th percentile
        self.q3 = None        # 75th percentile
        self.iqr = None       # Q3 - Q1
    
    def fit(self, X_normal):
        """Learn statistics from normal traffic only."""
        self.mean = np.mean(X_normal, axis=0)
        self.std = np.std(X_normal, axis=0)
        self.std[self.std == 0] = 1  # Avoid division by zero
        
        self.q1 = np.percentile(X_normal, 25, axis=0)
        self.q3 = np.percentile(X_normal, 75, axis=0)
        self.iqr = self.q3 - self.q1
        self.iqr[self.iqr == 0] = 1  # Avoid division by zero
```

### 4.3 Model 2: Isolation Forest

#### Purpose
Machine learning-based anomaly detection that learns complex patterns.

#### How Isolation Forest Works

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ISOLATION FOREST ALGORITHM                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  1. BUILD FOREST (Training):                                         │
│     ┌─────────────────────────────────────────────────────────┐     │
│     │  For each tree (100 trees):                              │     │
│     │    • Sample data points randomly                         │     │
│     │    • Recursively partition:                              │     │
│     │      - Pick random feature                               │     │
│     │      - Pick random split value                           │     │
│     │      - Create left/right child nodes                     │     │
│     │    • Stop at max depth or single point                   │     │
│     └─────────────────────────────────────────────────────────┘     │
│                                                                       │
│  2. KEY INSIGHT:                                                     │
│     ┌─────────────────────────────────────────────────────────┐     │
│     │                                                          │     │
│     │  Normal Points:    ●━━━━━━━━━━━━━━━━━━━●                │     │
│     │                    (deep in tree, long path)             │     │
│     │                                                          │     │
│     │  Anomalies:        ●━━━━●                                │     │
│     │                    (isolated quickly, short path)        │     │
│     │                                                          │     │
│     │  WHY? Anomalies are "few and different" - they get      │     │
│     │       separated from the majority with fewer splits      │     │
│     │                                                          │     │
│     └─────────────────────────────────────────────────────────┘     │
│                                                                       │
│  3. SCORING (Prediction):                                            │
│     ┌─────────────────────────────────────────────────────────┐     │
│     │  • Traverse point through all trees                      │     │
│     │  • Calculate average path length                         │     │
│     │  • Normalize by expected path length                     │     │
│     │  • Score ≈ 2^(-avg_path_length / expected_path_length)  │     │
│     │                                                          │     │
│     │  Score close to 1 → Anomaly                             │     │
│     │  Score close to 0 → Normal                              │     │
│     └─────────────────────────────────────────────────────────┘     │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

#### Why Isolation Forest?

| Advantage | Explanation |
|-----------|-------------|
| **Unsupervised** | Only needs normal data - no labeled attacks required |
| **Scalable** | O(n log n) training, O(log n) prediction |
| **Robust** | Works well with high-dimensional data |
| **No Assumptions** | Doesn't assume data distribution shape |
| **Few Hyperparameters** | Only contamination and n_estimators |
| **Widely Validated** | Proven effective in network security research |

#### Alternatives Considered

| Algorithm | Why Not Chosen |
|-----------|----------------|
| **One-Class SVM** | Slow training, scales poorly to large datasets |
| **Local Outlier Factor (LOF)** | Memory-intensive, slow for real-time |
| **DBSCAN** | Struggles with varying density, needs tuning |
| **Autoencoder** | Complex, requires GPU, harder to interpret |

#### Implementation Details

```python
# File: src/detection/isolation_forest.py

class IsolationForestDetector:
    def __init__(self, contamination=0.1, n_estimators=100, random_state=42):
        self.contamination = contamination  # Expected % of anomalies
        self.n_estimators = n_estimators    # Number of trees
        self.random_state = random_state    # Reproducibility
        
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1  # Use all CPU cores
        )
    
    def fit(self, X_normal):
        """Train on normal traffic only."""
        self.model.fit(X_normal)
    
    def detect(self, X):
        """
        Returns:
            is_anomaly: array of boolean (True = anomaly)
            scores: array of floats (higher = more anomalous)
        """
        predictions = self.model.predict(X)  # 1=normal, -1=anomaly
        scores = -self.model.score_samples(X)  # Invert so higher=anomaly
        
        return predictions == -1, scores
```

### 4.4 Model 3: Random Forest Classifier

#### Purpose
Multi-class classification to identify specific attack types after anomaly detection.

#### How Random Forest Classification Works

```
┌─────────────────────────────────────────────────────────────────────┐
│                  RANDOM FOREST CLASSIFIER                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  1. ENSEMBLE OF DECISION TREES:                                      │
│                                                                       │
│     Tree 1        Tree 2        Tree 3         ...       Tree 100   │
│       │             │             │                          │       │
│    ┌──┴──┐       ┌──┴──┐       ┌──┴──┐                   ┌──┴──┐   │
│    │DDoS │       │DDoS │       │Scan │                   │DDoS │   │
│    └─────┘       └─────┘       └─────┘                   └─────┘   │
│                                                                       │
│  2. BOOTSTRAP AGGREGATING (Bagging):                                 │
│     • Each tree trains on random subset of data                      │
│     • Each split considers random subset of features                 │
│     • Reduces overfitting, increases generalization                  │
│                                                                       │
│  3. VOTING:                                                          │
│                                                                       │
│     Votes:  DDoS: 58   PortScan: 30   BruteForce: 12               │
│                  │                                                    │
│                  ▼                                                    │
│     Final: DDoS (58% confidence)                                     │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

#### Attack Categories

The classifier maps the 14 original attack types to 8 broader categories for better generalization:

| Original Label | Mapped Category |
|----------------|-----------------|
| BENIGN | Normal |
| DDoS | DDoS Attack |
| DoS Hulk, DoS GoldenEye, DoS Slowloris, DoS Slowhttptest | DoS Attack |
| PortScan | Port Scan |
| FTP-Patator, SSH-Patator | Brute Force |
| Web Attack - Brute Force, Web Attack - XSS, Web Attack - SQL Injection | Web Attack |
| Infiltration | Infiltration |
| Bot | Botnet |
| Heartbleed | Heartbleed |

#### Why Random Forest?

| Advantage | Explanation |
|-----------|-------------|
| **High Accuracy** | Typically 95%+ on network traffic |
| **Handles Imbalance** | Works with skewed class distribution |
| **Feature Importance** | Provides interpretable importance scores |
| **No Feature Scaling Required** | Tree-based, scale-invariant |
| **Robust to Overfitting** | Ensemble reduces variance |
| **Fast Prediction** | O(log n) per tree, parallelizable |

#### Alternatives Considered

| Algorithm | Why Not Chosen |
|-----------|----------------|
| **Gradient Boosting (XGBoost)** | Slower, more hyperparameters, marginal improvement |
| **SVM (multi-class)** | Slow training on large datasets |
| **Neural Network (MLP)** | Requires GPU, harder to interpret, needs more data |
| **Naive Bayes** | Lower accuracy, assumes feature independence |
| **k-NN** | Slow prediction, memory-intensive |

#### Implementation Details

```python
# File: src/detection/classifier.py

class AttackClassifier:
    def __init__(self, n_estimators=100, max_depth=20, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1  # Parallel processing
        )
        self.label_encoder = LabelEncoder()
    
    def fit(self, X, y):
        """Train on all traffic including attacks."""
        y_mapped = self._map_labels(y)  # Map to 8 categories
        y_encoded = self.label_encoder.fit_transform(y_mapped)
        self.model.fit(X, y_encoded)
    
    def classify_single(self, x):
        """Classify a single sample."""
        prediction = self.predict(x.reshape(1, -1))[0]
        probabilities = self.predict_proba(x.reshape(1, -1))[0]
        confidence = max(probabilities)
        
        return {
            'attack_type': prediction,
            'confidence': confidence,
            'top_predictions': self._get_top_k(probabilities, k=3)
        }
```

---

## 5. Training Pipeline

### 5.1 Training Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                      TRAINING PIPELINE                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  [1/6] LOAD DATA                                                     │
│  ────────────────                                                    │
│  • Read all CSV files from data/raw/                                 │
│  • Handle encoding issues (UTF-8, Latin-1)                           │
│  • Combine into single DataFrame                                     │
│  • Optional: Sample for faster training                              │
│                                                                       │
│         │                                                             │
│         ▼                                                             │
│                                                                       │
│  [2/6] PREPROCESS                                                    │
│  ─────────────────                                                   │
│  • Replace Inf with NaN                                              │
│  • Fill NaN with median values                                       │
│  • Select 15 features                                                │
│  • Split: 80% train, 20% test (stratified)                          │
│  • Fit StandardScaler on training data                               │
│  • Transform both train and test                                     │
│  • Save preprocessor to models/preprocessor.joblib                   │
│                                                                       │
│         │                                                             │
│         ▼                                                             │
│                                                                       │
│  [3/6] TRAIN STATISTICAL DETECTOR                                    │
│  ─────────────────────────────────                                   │
│  • Filter: BENIGN traffic only                                       │
│  • Compute mean, std for Z-score                                     │
│  • Compute Q1, Q3, IQR                                               │
│  • Test on full test set                                             │
│  • Save to models/statistical_detector.joblib                        │
│                                                                       │
│         │                                                             │
│         ▼                                                             │
│                                                                       │
│  [4/6] TRAIN ISOLATION FOREST                                        │
│  ─────────────────────────────                                       │
│  • Filter: BENIGN traffic only                                       │
│  • Build 100 isolation trees                                         │
│  • contamination = 0.1 (10% anomaly threshold)                       │
│  • Test on full test set                                             │
│  • Save to models/isolation_forest.joblib                            │
│                                                                       │
│         │                                                             │
│         ▼                                                             │
│                                                                       │
│  [5/6] TRAIN ATTACK CLASSIFIER                                       │
│  ─────────────────────────────                                       │
│  • Use ALL training data (attacks + benign)                          │
│  • Map 14 labels → 8 categories                                      │
│  • Build 100 decision trees, max_depth=20                            │
│  • Test on full test set                                             │
│  • Save to models/attack_classifier.joblib                           │
│                                                                       │
│         │                                                             │
│         ▼                                                             │
│                                                                       │
│  [6/6] SUMMARY                                                       │
│  ───────────────                                                     │
│  • Print accuracy for each model                                     │
│  • All models saved to models/ directory                             │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 Training Command

```bash
python -m src.train
```

### 5.3 Training Script

```python
# File: src/train.py

def train_all_models(sample_size=None):
    """
    Train all detection models.
    
    Args:
        sample_size: Number of samples to use (None = all data)
    """
    # Step 1: Load data
    df = load_dataset()
    if sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    # Step 2: Preprocess
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.clean_data(df)
    labels = df_clean['Label']
    y_binary = (labels != 'BENIGN').astype(int)
    df_features = preprocessor.select_features(df_clean)
    
    X_train_raw, X_test_raw, y_train, y_test, labels_train, labels_test = \
        train_test_split(df_features, y_binary, labels, test_size=0.2, 
                        random_state=42, stratify=y_binary)
    
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    preprocessor.save()
    
    # Step 3: Train Statistical Detector (normal traffic only)
    normal_mask = y_train == 0
    X_normal = X_train[normal_mask]
    
    stat_detector = StatisticalDetector()
    stat_detector.fit(X_normal)
    joblib.dump(stat_detector, MODELS_DIR / "statistical_detector.joblib")
    
    # Step 4: Train Isolation Forest (normal traffic only)
    iso_detector = IsolationForestDetector(contamination=0.1)
    iso_detector.fit(X_normal)
    iso_detector.save()
    
    # Step 5: Train Classifier (all traffic)
    classifier = AttackClassifier()
    classifier.fit(X_train, labels_train.values)
    classifier.save()
```

### 5.4 Data Split Strategy

```
┌─────────────────────────────────────────────────────────────────────┐
│                      DATA SPLITTING                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Total Data: 100%                                                    │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                       │
│  ┌────────────────────────────────────┬────────────────────────────┐│
│  │         TRAINING SET               │        TEST SET            ││
│  │             80%                    │           20%              ││
│  │                                    │                            ││
│  │   Used for:                        │   Used for:                ││
│  │   • Statistical stats calc         │   • Model evaluation       ││
│  │   • Isolation Forest fitting       │   • Accuracy metrics       ││
│  │   • Classifier training            │   • Final performance      ││
│  │                                    │                            ││
│  │   Split type: Stratified           │   Never seen during        ││
│  │   (maintains attack ratios)        │   training                 ││
│  │                                    │                            ││
│  └────────────────────────────────────┴────────────────────────────┘│
│                                                                       │
│  STRATIFICATION: Ensures each split has same attack type ratios     │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                       │
│  Example (if total has 80% BENIGN, 15% DDoS, 5% PortScan):          │
│    Train set: 80% BENIGN, 15% DDoS, 5% PortScan                     │
│    Test set:  80% BENIGN, 15% DDoS, 5% PortScan                     │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 6. Model Evaluation

### 6.1 Evaluation Metrics

| Metric | Formula | Use Case |
|--------|---------|----------|
| **Accuracy** | (TP + TN) / Total | Overall correctness |
| **Precision** | TP / (TP + FP) | "Of detected attacks, how many are real?" |
| **Recall (TPR)** | TP / (TP + FN) | "Of real attacks, how many did we catch?" |
| **F1 Score** | 2 × (Precision × Recall) / (Precision + Recall) | Balanced metric |
| **FPR** | FP / (FP + TN) | False alarm rate |

Where:
- TP = True Positive (attack correctly detected)
- TN = True Negative (normal correctly identified)
- FP = False Positive (normal flagged as attack)
- FN = False Negative (attack missed)

### 6.2 Real Evaluation Results (CICIDS2017 Dataset)

> **Evaluation Date:** December 10, 2025  
> **Dataset:** CICIDS2017 (100,000 samples)  
> **Train/Test Split:** 80,000 / 20,000 (stratified)

#### Anomaly Detection Models (Binary Classification)

| Model | Accuracy | Precision | Recall | F1 Score | Specificity | ROC AUC |
|-------|----------|-----------|--------|----------|-------------|--------|
| **Statistical (Z-score)** | 76.71% | 42.14% | 48.29% | 45.01% | 83.70% | 0.688 |
| **Isolation Forest** | 80.97% | 52.05% | 45.07% | 48.31% | 89.79% | 0.724 |
| **Combined (OR)** | 76.54% | 41.83% | 48.29% | 44.83% | 83.49% | 0.688 |

#### Attack Classifier (Multi-class Classification)

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | 98.35% |
| **Weighted Precision** | 98.31% |
| **Weighted Recall** | 98.35% |
| **Weighted F1 Score** | 98.30% |
| **Macro Precision** | 67.51% |
| **Macro Recall** | 60.88% |
| **Macro F1 Score** | 62.77% |

### 6.3 Confusion Matrices (Real Data)

#### Statistical Detector Confusion Matrix

```
                    PREDICTED
              ┌─────────┬─────────┐
              │ Normal  │ Attack  │
     ─────────┼─────────┼─────────┤
     Normal   │ 13,436  │  2,617  │   ← False Positives: 16.30%
Actual ───────┼─────────┼─────────┤
     Attack   │  2,041  │  1,906  │   ← Missed 51.71% of attacks
              └─────────┴─────────┘

 True Positives:  1,906   |   False Positives: 2,617
 True Negatives: 13,436   |   False Negatives: 2,041
```

#### Isolation Forest Confusion Matrix

```
                    PREDICTED
              ┌─────────┬─────────┐
              │ Normal  │ Attack  │
     ─────────┼─────────┼─────────┤
     Normal   │ 14,414  │  1,639  │   ← False Positives: 10.21%
Actual ───────┼─────────┼─────────┤
     Attack   │  2,168  │  1,779  │   ← Missed 54.93% of attacks
              └─────────┴─────────┘

 True Positives:  1,779   |   False Positives: 1,639
 True Negatives: 14,414   |   False Negatives: 2,168
```

### 6.4 Attack Classifier Per-Class Performance

| Attack Category | Precision | Recall | F1 Score | Support (Test) |
|-----------------|-----------|--------|----------|----------------|
| **Normal** | 99.29% | 98.66% | 98.98% | 16,053 |
| **Port Scan** | 99.47% | 99.64% | 99.55% | 1,122 |
| **DDoS Attack** | 99.46% | 99.14% | 99.30% | 931 |
| **DoS Attack** | 89.76% | 97.39% | 93.42% | 1,764 |
| **Brute Force** | 92.11% | 72.16% | 80.92% | 97 |
| **Botnet** | 60.00% | 20.00% | 30.00% | 15 |
| **Heartbleed** | 0.00% | 0.00% | 0.00% | 1 |
| **Unknown Attack** | 0.00% | 0.00% | 0.00% | 17 |

> **Note:** Low performance on Heartbleed and Unknown Attack is due to extremely small sample sizes (1 and 17 samples respectively in test set).

### 6.5 Per-Attack Type Detection Rates (Combined Detector)

| Attack Type | Total in Test | Detected | Detection Rate |
|-------------|---------------|----------|----------------|
| **DoS GoldenEye** | 81 | 60 | 74.07% |
| **DoS Hulk** | 1,611 | 1,185 | 73.57% |
| **DoS Slowloris** | 36 | 24 | 66.67% |
| **DDoS** | 931 | 584 | 62.73% |
| **FTP-Patator** | 50 | 24 | 48.00% |
| **DoS Slowhttptest** | 36 | 17 | 47.22% |
| **Heartbleed** | 1 | 1 | 100.00% |
| **PortScan** | 1,122 | 11 | 0.98% | ⚠️ Low detection |
| **Bot** | 15 | 0 | 0.00% | ⚠️ Missed |
| **SSH-Patator** | 47 | 0 | 0.00% | ⚠️ Missed |
| **Web Attack** | 17 | 0 | 0.00% | ⚠️ Missed |
| **BENIGN (False Alarms)** | 16,053 | 2,651 | 16.51% |

---

## 7. Model Comparison & Selection Rationale

### 7.1 Why This Combination?

```
┌─────────────────────────────────────────────────────────────────────┐
│              MODEL COMBINATION RATIONALE                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  LAYER 1: DETECTION (Binary: Normal vs Anomaly)                      │
│  ───────────────────────────────────────────────                     │
│                                                                       │
│  ┌─────────────────────┐         ┌─────────────────────┐            │
│  │  Statistical        │   +     │  Isolation Forest   │            │
│  │  (Z-score / IQR)    │         │  (ML-based)         │            │
│  │                     │         │                     │            │
│  │  ✓ Fast (< 1ms)     │         │  ✓ Anomaly-focused  │            │
│  │  ✓ Interpretable    │         │  ✓ Captures complex │            │
│  │  ✓ No training      │         │    patterns         │            │
│  │  ✗ High FP rate     │         │  ✗ Less interpretable│           │
│  └─────────────────────┘         └─────────────────────┘            │
│                                                                       │
│  Combined with OR logic: Catches more attacks (high recall)         │
│                                                                       │
│                                                                       │
│  LAYER 2: CLASSIFICATION (Attack Type Identification)               │
│  ─────────────────────────────────────────────────────               │
│                                                                       │
│  ┌───────────────────────────────────────────────────────┐          │
│  │  Random Forest Classifier                              │          │
│  │                                                        │          │
│  │  ✓ High accuracy (95%+)                               │          │
│  │  ✓ Provides confidence scores                          │          │
│  │  ✓ Feature importance for interpretability            │          │
│  │  ✓ Handles class imbalance                            │          │
│  │  ✓ Works on CPU (no GPU needed)                       │          │
│  └───────────────────────────────────────────────────────┘          │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.2 Detection Logic

```python
def detect(flow):
    # Step 1: Run both detectors
    stat_anomaly, stat_score = statistical_detector.detect(flow)
    iso_anomaly, iso_score = isolation_forest.detect(flow)
    
    # Step 2: Combine with OR (catch more attacks)
    is_anomaly = stat_anomaly OR iso_anomaly
    score = max(stat_score, iso_score)
    
    # Step 3: If anomaly, classify attack type
    if is_anomaly:
        result = classifier.classify(flow)
        
        # IMPORTANT: Classifier has final say
        # If classifier says "Normal", override detection
        if result['attack_type'] == 'Normal':
            is_anomaly = False
    
    return is_anomaly, score, result['attack_type'], result['confidence']
```

### 7.3 Why Two-Stage Detection?

| Single-Stage Problems | Two-Stage Solution |
|-----------------------|--------------------|
| Classifier trained on historical attacks | Anomaly detectors catch unknown attacks |
| Missing attacks not in training data | Anomaly detection is unsupervised |
| "Unknown" class is hard to define | Detectors flag anything unusual |
| Class imbalance (few attack samples) | Binary detection balances problem |

---

## 8. Preprocessing & Normalization

### 8.1 StandardScaler

#### Why StandardScaler?

```
Before Scaling:                    After Scaling:
─────────────────                  ─────────────────
Flow Duration: 0 - 120,000,000 μs  Mean = 0, Std = 1
Flow Bytes/s:  0 - 1,000,000,000   Mean = 0, Std = 1
SYN Flags:     0 - 1000            Mean = 0, Std = 1

Problem: Features have vastly different scales
         → Some algorithms (Z-score, Isolation Forest) are affected
         → Larger values dominate distance calculations

Solution: StandardScaler normalizes all features to mean=0, std=1
```

#### Formula

```
            X - μ
X_scaled = ───────
              σ

Where:
    X = original value
    μ = mean (learned from training data)
    σ = standard deviation (learned from training data)
```

#### Implementation

```python
# File: src/data/preprocessor.py

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.features = SELECTED_FEATURES
    
    def fit_transform(self, X_train):
        """Fit on training data, return transformed."""
        return self.scaler.fit_transform(X_train)
    
    def transform(self, X):
        """Apply same transformation to new data."""
        return self.scaler.transform(X)
```

### 8.2 Data Cleaning

```python
def clean_data(df):
    """
    Cleaning steps:
    1. Replace infinity with NaN
    2. Fill NaN with median values
    3. Drop any remaining rows with NaN
    """
    df = df.replace([np.inf, -np.inf], np.nan)
    
    for col in numeric_columns:
        median = df[col].median()
        df[col].fillna(median, inplace=True)
    
    df = df.dropna()
    return df
```

---

## 9. Hyperparameters

### 9.1 All Configurable Parameters

| Component | Parameter | Value | Location |
|-----------|-----------|-------|----------|
| **Statistical** | zscore_threshold | 3.0 | config.py |
| **Statistical** | iqr_multiplier | 1.5 | statistical.py |
| **Isolation Forest** | contamination | 0.1 | config.py |
| **Isolation Forest** | n_estimators | 100 | isolation_forest.py |
| **Isolation Forest** | random_state | 42 | isolation_forest.py |
| **Classifier** | n_estimators | 100 | classifier.py |
| **Classifier** | max_depth | 20 | classifier.py |
| **Classifier** | random_state | 42 | classifier.py |
| **Training** | test_size | 0.2 | train.py |
| **Training** | sample_size | 50000 | train.py |

### 9.2 Parameter Justification

| Parameter | Value | Justification |
|-----------|-------|---------------|
| **zscore_threshold=3.0** | 3 std deviations | Statistical standard: 99.7% of normal data falls within ±3σ |
| **contamination=0.1** | 10% | Dataset has ~20% attacks; 10% is conservative |
| **n_estimators=100** | 100 trees | Diminishing returns after 100; balance of accuracy vs speed |
| **max_depth=20** | 20 levels | Prevents overfitting while capturing complex patterns |
| **test_size=0.2** | 80/20 split | Standard ML practice; enough test data for reliable metrics |

---

## 10. Serialization & Deployment

### 10.1 Saved Model Files

```
models/
├── preprocessor.joblib         # StandardScaler + feature list
├── statistical_detector.joblib # Mean, std, Q1, Q3, IQR arrays
├── isolation_forest.joblib     # Sklearn IsolationForest model
└── attack_classifier.joblib    # RandomForest + LabelEncoder
```

### 10.2 Model Loading

```python
# File: src/api/routes.py

def load_models():
    """Load all trained models."""
    models['preprocessor'] = DataPreprocessor.load()
    models['statistical'] = joblib.load(MODELS_DIR / "statistical_detector.joblib")
    models['isolation_forest'] = IsolationForestDetector.load()
    models['classifier'] = AttackClassifier.load()
```

### 10.3 Inference Pipeline

```python
def detect_anomaly(flow_features):
    # 1. Preprocess
    X = preprocessor.transform(flow_features.reshape(1, -1))
    
    # 2. Statistical detection
    stat_anomaly, stat_score = statistical_detector.detect(X)
    
    # 3. Isolation Forest detection
    iso_anomaly, iso_score = isolation_forest.detect(X)
    
    # 4. Combine results
    is_anomaly = stat_anomaly[0] or iso_anomaly[0]
    score = max(stat_score[0], iso_score[0])
    
    # 5. Classify if anomaly
    attack_type = None
    if is_anomaly:
        result = classifier.classify_single(X[0])
        attack_type = result['attack_type']
        
        # Classifier override
        if attack_type == 'Normal':
            is_anomaly = False
    
    return {
        'is_anomaly': is_anomaly,
        'score': score,
        'attack_type': attack_type
    }
```

---

## 11. Limitations & Future Work

### 11.1 Current Limitations

| Limitation | Impact | Potential Solution |
|------------|--------|-------------------|
| **Dataset Age** | 2017 attacks may not represent 2024 threats | Collect new data, use CICIDS2018/2019 |
| **Fixed Features** | Can't adapt to new attack patterns | Online learning, feature auto-selection |
| **Binary+Multi-class** | Two-stage adds latency | End-to-end neural network |
| **Class Imbalance** | Rare attacks may be missed | SMOTE, cost-sensitive learning |
| **Concept Drift** | Model degrades over time | Periodic retraining, drift detection |

### 11.2 Future Improvements

| Enhancement | Description |
|-------------|-------------|
| **Deep Learning** | LSTM/Transformer for sequence modeling |
| **Federated Learning** | Train on distributed data without sharing |
| **Explainable AI** | SHAP/LIME for prediction explanations |
| **Active Learning** | Request labels for uncertain predictions |
| **Ensemble Expansion** | Add Autoencoder, one-class SVM |

---

## Document Information

| Field | Value |
|-------|-------|
| **Version** | 1.0 |
| **Last Updated** | December 2024 |
| **Author** | Network Anomaly Detection System |
| **Related Files** | `src/train.py`, `src/detection/*`, `src/data/*`, `src/utils/config.py` |

---

*This document provides complete technical reference for the machine learning components. For API documentation, see `DOCUMENTATION.md`. For quick start guide, see `README.md`.*
