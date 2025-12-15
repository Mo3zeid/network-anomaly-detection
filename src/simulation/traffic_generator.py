"""
Realistic traffic simulation for demonstration purposes.
Generates network flows that look like real traffic with proper IPs, ports, and protocols.
"""
import random
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict


# Realistic IP ranges
INTERNAL_IPS = [
    f"192.168.1.{i}" for i in range(1, 255)
] + [
    f"10.0.0.{i}" for i in range(1, 255)
]

EXTERNAL_IPS = [
    f"{random.randint(1, 223)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"
    for _ in range(500)
]

# Known malicious IPs (for simulation)
MALICIOUS_IPS = [
    "185.220.101.1", "185.220.101.2", "185.220.101.3",
    "45.155.205.1", "45.155.205.2",
    "23.129.64.1", "23.129.64.2",
    "192.42.116.1", "192.42.116.2"
]

# Common services
SERVICES = {
    80: {"name": "HTTP", "protocol": "TCP"},
    443: {"name": "HTTPS", "protocol": "TCP"},
    22: {"name": "SSH", "protocol": "TCP"},
    21: {"name": "FTP", "protocol": "TCP"},
    23: {"name": "Telnet", "protocol": "TCP"},
    25: {"name": "SMTP", "protocol": "TCP"},
    53: {"name": "DNS", "protocol": "UDP"},
    3389: {"name": "RDP", "protocol": "TCP"},
    3306: {"name": "MySQL", "protocol": "TCP"},
    5432: {"name": "PostgreSQL", "protocol": "TCP"},
    8080: {"name": "HTTP-Alt", "protocol": "TCP"},
    445: {"name": "SMB", "protocol": "TCP"},
}

# Attack type descriptions
ATTACK_DESCRIPTIONS = {
    "DDoS Attack": {
        "description": "Distributed Denial of Service - Overwhelming traffic from multiple sources designed to make a service unavailable",
        "indicators": ["High packet rate", "Multiple source IPs", "Short flow duration", "Unusual SYN flags"],
        "severity": "critical",
        "mitigation": "Enable rate limiting, use DDoS protection service, block source IPs"
    },
    "Port Scan": {
        "description": "Reconnaissance attack scanning for open ports to identify running services and vulnerabilities",
        "indicators": ["Sequential port access", "Many failed connections", "Single source IP", "Low data volume"],
        "severity": "high",
        "mitigation": "Enable port scan detection, configure firewall to block scanner IPs"
    },
    "Brute Force": {
        "description": "Repeated login attempts using different password combinations to gain unauthorized access",
        "indicators": ["Many connections to auth port", "Failed login patterns", "Consistent source IP"],
        "severity": "high",
        "mitigation": "Implement account lockout, use CAPTCHA, enable 2FA"
    },
    "Web Attack": {
        "description": "Attack targeting web applications including SQL injection, XSS, and other OWASP vulnerabilities",
        "indicators": ["Unusual HTTP requests", "Long URLs", "Special characters in requests"],
        "severity": "critical",
        "mitigation": "Use WAF, sanitize inputs, update web application"
    },
    "Botnet": {
        "description": "Traffic from compromised machines controlled by attackers, often for spam or DDoS",
        "indicators": ["Regular beacon patterns", "C2 communication", "Unusual destinations"],
        "severity": "critical",
        "mitigation": "Isolate infected hosts, block C2 servers, update antivirus"
    },
    "DoS Attack": {
        "description": "Denial of Service from single source attempting to exhaust server resources",
        "indicators": ["High volume from one IP", "Resource exhaustion patterns", "Slowloris-style attacks"],
        "severity": "high",
        "mitigation": "Rate limit connections, timeout slow connections"
    },
    "Normal": {
        "description": "Regular network traffic with no malicious indicators detected",
        "indicators": ["Normal packet sizes", "Expected protocols", "Known destinations"],
        "severity": "info",
        "mitigation": "No action required"
    }
}


def generate_realistic_ip():
    """Generate a realistic looking IP address."""
    if random.random() < 0.7:
        return random.choice(EXTERNAL_IPS)
    return random.choice(INTERNAL_IPS)


def generate_flow_id():
    """Generate a unique flow identifier."""
    return f"{random.randint(100000, 999999)}-{random.randint(1000, 9999)}"


def generate_normal_traffic() -> Dict:
    """Generate a realistic normal traffic flow."""
    src_ip = random.choice(INTERNAL_IPS)
    dst_port = random.choice(list(SERVICES.keys()))
    service = SERVICES[dst_port]
    
    # Normal traffic characteristics
    flow = {
        "flow_id": generate_flow_id(),
        "timestamp": datetime.now().isoformat(),
        "src_ip": src_ip,
        "dst_ip": generate_realistic_ip(),
        "src_port": random.randint(49152, 65535),
        "dst_port": dst_port,
        "protocol": service["protocol"],
        "service": service["name"],
        
        # Flow features (normal ranges)
        "flow_duration": random.randint(10000, 500000),
        "total_fwd_packets": random.randint(5, 50),
        "total_bwd_packets": random.randint(5, 50),
        "flow_bytes_per_sec": random.uniform(100, 5000),
        "flow_packets_per_sec": random.uniform(1, 50),
        "fwd_packet_length_mean": random.uniform(200, 800),
        "bwd_packet_length_mean": random.uniform(200, 800),
        "flow_iat_mean": random.uniform(5000, 50000),
        "fwd_iat_mean": random.uniform(5000, 30000),
        "bwd_iat_mean": random.uniform(5000, 30000),
        "fwd_psh_flags": random.randint(1, 5),
        "syn_flag_count": random.randint(1, 3),
        "ack_flag_count": random.randint(10, 50),
        "packet_length_variance": random.uniform(100, 500),
        "average_packet_size": random.uniform(300, 600),
        
        # Labels
        "is_attack": False,
        "attack_type": "Normal",
        "attack_description": ATTACK_DESCRIPTIONS["Normal"]["description"],
        "severity": "info"
    }
    return flow


def generate_ddos_traffic() -> Dict:
    """Generate DDoS attack traffic."""
    src_ip = random.choice(MALICIOUS_IPS + [generate_realistic_ip()])
    dst_port = random.choice([80, 443, 8080])
    
    flow = {
        "flow_id": generate_flow_id(),
        "timestamp": datetime.now().isoformat(),
        "src_ip": src_ip,
        "dst_ip": random.choice(INTERNAL_IPS[:10]),  # Target internal servers
        "src_port": random.randint(1024, 65535),
        "dst_port": dst_port,
        "protocol": "TCP",
        "service": "HTTP" if dst_port == 80 else "HTTPS",
        
        # DDoS characteristics: high volume, short duration
        "flow_duration": random.randint(100, 2000),
        "total_fwd_packets": random.randint(500, 5000),
        "total_bwd_packets": random.randint(0, 10),
        "flow_bytes_per_sec": random.uniform(50000, 500000),
        "flow_packets_per_sec": random.uniform(500, 5000),
        "fwd_packet_length_mean": random.uniform(40, 100),
        "bwd_packet_length_mean": random.uniform(0, 40),
        "flow_iat_mean": random.uniform(1, 50),
        "fwd_iat_mean": random.uniform(1, 20),
        "bwd_iat_mean": random.uniform(100, 1000),
        "fwd_psh_flags": random.randint(0, 2),
        "syn_flag_count": random.randint(50, 500),
        "ack_flag_count": random.randint(0, 10),
        "packet_length_variance": random.uniform(50, 200),
        "average_packet_size": random.uniform(40, 80),
        
        "is_attack": True,
        "attack_type": "DDoS Attack",
        "attack_description": ATTACK_DESCRIPTIONS["DDoS Attack"]["description"],
        "severity": "critical",
        "indicators": ATTACK_DESCRIPTIONS["DDoS Attack"]["indicators"]
    }
    return flow


def generate_port_scan_traffic() -> Dict:
    """Generate port scanning traffic."""
    src_ip = generate_realistic_ip()
    
    flow = {
        "flow_id": generate_flow_id(),
        "timestamp": datetime.now().isoformat(),
        "src_ip": src_ip,
        "dst_ip": random.choice(INTERNAL_IPS),
        "src_port": random.randint(40000, 60000),
        "dst_port": random.randint(1, 1024),
        "protocol": "TCP",
        "service": "Unknown",
        
        # Port scan characteristics: many short connections
        "flow_duration": random.randint(10, 500),
        "total_fwd_packets": random.randint(1, 5),
        "total_bwd_packets": random.randint(0, 2),
        "flow_bytes_per_sec": random.uniform(100, 2000),
        "flow_packets_per_sec": random.uniform(10, 100),
        "fwd_packet_length_mean": random.uniform(40, 80),
        "bwd_packet_length_mean": random.uniform(0, 60),
        "flow_iat_mean": random.uniform(10, 200),
        "fwd_iat_mean": random.uniform(10, 100),
        "bwd_iat_mean": random.uniform(10, 500),
        "fwd_psh_flags": 0,
        "syn_flag_count": random.randint(1, 3),
        "ack_flag_count": random.randint(0, 2),
        "packet_length_variance": random.uniform(10, 100),
        "average_packet_size": random.uniform(40, 60),
        
        "is_attack": True,
        "attack_type": "Port Scan",
        "attack_description": ATTACK_DESCRIPTIONS["Port Scan"]["description"],
        "severity": "high",
        "indicators": ATTACK_DESCRIPTIONS["Port Scan"]["indicators"]
    }
    return flow


def generate_brute_force_traffic() -> Dict:
    """Generate brute force attack traffic."""
    target_port = random.choice([22, 21, 3389, 23])
    service = SERVICES[target_port]
    
    flow = {
        "flow_id": generate_flow_id(),
        "timestamp": datetime.now().isoformat(),
        "src_ip": generate_realistic_ip(),
        "dst_ip": random.choice(INTERNAL_IPS[:20]),
        "src_port": random.randint(49152, 65535),
        "dst_port": target_port,
        "protocol": service["protocol"],
        "service": service["name"],
        
        # Brute force characteristics
        "flow_duration": random.randint(1000, 10000),
        "total_fwd_packets": random.randint(50, 200),
        "total_bwd_packets": random.randint(50, 200),
        "flow_bytes_per_sec": random.uniform(5000, 20000),
        "flow_packets_per_sec": random.uniform(50, 200),
        "fwd_packet_length_mean": random.uniform(100, 200),
        "bwd_packet_length_mean": random.uniform(50, 150),
        "flow_iat_mean": random.uniform(50, 500),
        "fwd_iat_mean": random.uniform(50, 300),
        "bwd_iat_mean": random.uniform(50, 300),
        "fwd_psh_flags": random.randint(10, 50),
        "syn_flag_count": random.randint(20, 100),
        "ack_flag_count": random.randint(20, 100),
        "packet_length_variance": random.uniform(200, 500),
        "average_packet_size": random.uniform(100, 200),
        
        "is_attack": True,
        "attack_type": "Brute Force",
        "attack_description": ATTACK_DESCRIPTIONS["Brute Force"]["description"],
        "severity": "high",
        "indicators": ATTACK_DESCRIPTIONS["Brute Force"]["indicators"]
    }
    return flow


def generate_web_attack_traffic() -> Dict:
    """Generate web attack traffic (SQL injection, XSS)."""
    flow = {
        "flow_id": generate_flow_id(),
        "timestamp": datetime.now().isoformat(),
        "src_ip": generate_realistic_ip(),
        "dst_ip": random.choice(INTERNAL_IPS[:5]),
        "src_port": random.randint(49152, 65535),
        "dst_port": random.choice([80, 443, 8080]),
        "protocol": "TCP",
        "service": "HTTP/HTTPS",
        
        # Web attack characteristics
        "flow_duration": random.randint(500, 5000),
        "total_fwd_packets": random.randint(20, 100),
        "total_bwd_packets": random.randint(10, 50),
        "flow_bytes_per_sec": random.uniform(10000, 50000),
        "flow_packets_per_sec": random.uniform(20, 100),
        "fwd_packet_length_mean": random.uniform(500, 1500),  # Large requests (SQL payloads)
        "bwd_packet_length_mean": random.uniform(200, 800),
        "flow_iat_mean": random.uniform(100, 1000),
        "fwd_iat_mean": random.uniform(50, 500),
        "bwd_iat_mean": random.uniform(100, 500),
        "fwd_psh_flags": random.randint(5, 20),
        "syn_flag_count": random.randint(1, 5),
        "ack_flag_count": random.randint(10, 50),
        "packet_length_variance": random.uniform(1000, 3000),
        "average_packet_size": random.uniform(400, 800),
        
        "is_attack": True,
        "attack_type": "Web Attack",
        "attack_description": ATTACK_DESCRIPTIONS["Web Attack"]["description"],
        "severity": "critical",
        "indicators": ATTACK_DESCRIPTIONS["Web Attack"]["indicators"]
    }
    return flow


def generate_traffic_batch(
    num_samples: int = 20,
    anomaly_ratio: float = 0.3
) -> List[Dict]:
    """
    Generate a batch of realistic network traffic.
    
    Args:
        num_samples: Number of flows to generate
        anomaly_ratio: Ratio of attack traffic
        
    Returns:
        List of traffic flow dictionaries
    """
    flows = []
    attack_generators = [
        generate_ddos_traffic,
        generate_port_scan_traffic,
        generate_brute_force_traffic,
        generate_web_attack_traffic
    ]
    
    for _ in range(num_samples):
        if random.random() < anomaly_ratio:
            # Generate attack traffic
            generator = random.choice(attack_generators)
            flow = generator()
        else:
            # Generate normal traffic
            flow = generate_normal_traffic()
        
        flows.append(flow)
    
    # Sort by timestamp for realism
    flows.sort(key=lambda x: x["timestamp"])
    
    return flows


def get_attack_info(attack_type: str) -> Dict:
    """Get detailed information about an attack type."""
    return ATTACK_DESCRIPTIONS.get(attack_type, ATTACK_DESCRIPTIONS["Normal"])
