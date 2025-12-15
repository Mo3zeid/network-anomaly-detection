# Demo Scenarios Guide

Step-by-step attack simulation scenarios for demonstrating the Network Anomaly Detection System.

---

## Prerequisites

1. **Start Backend API** (as Administrator for firewall features):
   ```powershell
   cd d:\Materials\Grade 4\Semester 1\Graduation I\Project\network-anomaly-detection
   uvicorn src.api.main:app --reload
   ```

2. **Start Frontend Dashboard**:
   ```powershell
   cd frontend
   npm run dev
   ```

3. **Access Dashboard**: Open `http://localhost:3000`

---

## Scenario 1: Port Scan Detection

**Attack Tool**: Nmap from Kali Linux VM

### Steps:

1. **Start Live Sniffer**
   - Navigate to `Live Sniffer` page
   - Click `Start Sniffing`

2. **Execute Port Scan** (from attacker VM):
   ```bash
   nmap -sS -p 1-1000 <target_ip>
   ```

3. **Expected Detection**:
   - Alert type: `Port Scan`
   - Severity: `Critical`
   - Trigger: Multiple ports accessed from same source

4. **Dashboard Response**:
   - Alert appears in Alerts panel
   - IP appears in malicious IPs list
   - Option to block attacker IP

---

## Scenario 2: SYN Flood Attack (DoS)

**Attack Tool**: hping3 from Kali Linux VM

### Steps:

1. **Execute SYN Flood**:
   ```bash
   hping3 -S --flood -p 80 <target_ip>
   ```

2. **Expected Detection**:
   - Alert type: `DDoS Attack` or `SYN Flood`
   - High SYN flag count detected
   - Rule engine triggers on SYN flood threshold

3. **Mitigation**:
   - Click `Block IP` in Live Sniffer
   - Verify Windows Firewall rule created

---

## Scenario 3: Brute Force (SSH/FTP)

**Attack Tool**: Hydra from Kali Linux VM

### Steps:

1. **Execute Brute Force**:
   ```bash
   hydra -l admin -P /usr/share/wordlists/rockyou.txt ssh://<target_ip>
   ```

2. **Expected Detection**:
   - Alert type: `Brute Force`
   - Multiple connection attempts
   - High connection rate from single IP

---

## Scenario 4: Traffic Simulation (No VM Required)

Use the built-in simulation for demos without attack VMs.

### Steps:

1. **Navigate to Dashboard** (`http://localhost:3000`)

2. **Use API to Simulate Traffic**:
   ```powershell
   curl -X POST "http://localhost:8000/api/simulate?num_samples=50&anomaly_ratio=0.3"
   ```

3. **Observe Dashboard**:
   - Stats cards update with flows analyzed
   - Attack distribution chart populates
   - Alerts panel shows detected anomalies

---

## Scenario 5: File Upload Detection

### Steps:

1. **Navigate to Detection page**

2. **Upload Sample CSV** (`sample_traffic.csv`)

3. **Select Detection Method**:
   - `combined` - Uses all methods
   - `statistical` - Z-score only
   - `isolation_forest` - ML only

4. **View Results**:
   - Anomaly count and rate
   - Per-flow classification
   - Export results as CSV

---

## Scenario 6: Rules Configuration Demo

### Steps:

1. **Navigate to Rules page**

2. **Demonstrate Configurable Thresholds**:
   - Max packets per minute
   - Port scan threshold
   - SYN flood threshold

3. **Add IP to Whitelist/Blacklist**

4. **Show Real-time Rule Effects**

---

## Demo Flow for Presentation

| Time | Activity |
|------|----------|
| 0:00-1:00 | Introduce system architecture |
| 1:00-3:00 | Show dashboard and navigate pages |
| 3:00-5:00 | Run traffic simulation |
| 5:00-7:00 | Demonstrate port scan detection (if VM available) |
| 7:00-9:00 | Show file upload analysis |
| 9:00-10:00 | Demonstrate rules configuration |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Sniffer not starting | Run backend as Administrator |
| No alerts appearing | Check if models are trained (`python -m src.train`) |
| Firewall block fails | Requires Administrator privileges |
| Frontend not loading | Check if backend is running on port 8000 |

---

## Key Points to Highlight

1. **Multi-method Detection**: Statistical + Isolation Forest + Random Forest
2. **Real-time Sniffing**: Scapy-based packet capture
3. **Active Response**: Windows Firewall integration for blocking
4. **Configurable Rules**: Adjustable thresholds for different environments
5. **Modern Dashboard**: Real-time visualization with Next.js
