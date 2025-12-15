"""
API routes for real-time sniffing controls.
Optimized for high-traffic scenarios with rate limiting and async ML.
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, List
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import random

from ..sniffing.sniffer import SnifferService
from ..api.routes import detect_anomalies
from ..api.models import DetectionRequest, NetworkFlow, DetectionMethod
from ..api.firewall import register_malicious_ip

router = APIRouter()

# Global sniffer instance
sniffer = SnifferService()

# Thread pool for running ML detection without blocking
ml_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ml_detect")

# Rate limiting for heavy traffic
MAX_FLOWS_PER_REQUEST = 50  # Sample if more than this many flows
ML_DETECTION_TIMEOUT = 2.0  # Maximum seconds for ML detection


@router.post("/sniffer/start")
async def start_sniffer(interface: str = None):
    """Start the packet sniffer."""
    if sniffer.is_running:
        return {"message": "Sniffer is already running"}
    
    try:
        sniffer.start(interface)
        return {"message": "Sniffer started", "interface": interface or "auto"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sniffer/stop")
async def stop_sniffer():
    """Stop the packet sniffer."""
    sniffer.stop()
    return {"message": "Sniffer stopped"}

@router.get("/sniffer/status")
async def get_sniffer_status():
    """Get current status and stats."""
    return {
        "is_running": sniffer.is_running,
        "packets_captured": sniffer.stats.get("packets_captured", 0),
        "packets_seen": sniffer.stats.get("packets_seen", 0),
        "active_flows": sniffer.stats.get("active_flows", 0),
        "packets_dropped": sniffer.stats.get("packets_dropped", 0),
        "sample_rate": sniffer.stats.get("sample_rate", 1),
        "mode": sniffer.stats.get("mode", "normal")
    }

@router.get("/sniffer/latest")
async def get_latest_traffic():
    """
    Get latest flows, classify them, and return results.
    Optimized with sampling and timeout protection for heavy traffic.
    """
    raw_flows = sniffer.get_flows(timeout_seconds=2.0)
    
    if not raw_flows:
        return {"flows": []}
    
    # Rate limit: sample flows if too many (during heavy attacks)
    original_count = len(raw_flows)
    if len(raw_flows) > MAX_FLOWS_PER_REQUEST:
        # Keep a random sample to avoid overwhelming ML
        raw_flows = random.sample(raw_flows, MAX_FLOWS_PER_REQUEST)
        
    # Convert to Pydantic models
    flow_objects = []
    
    for f in raw_flows:
        flow_objects.append(NetworkFlow(
            flow_duration=f['Flow Duration'],
            total_fwd_packets=f['Total Fwd Packets'],
            total_bwd_packets=f['Total Backward Packets'],
            flow_bytes_per_sec=f['Flow Bytes/s'],
            flow_packets_per_sec=f['Flow Packets/s'],
            fwd_packet_length_mean=f['Fwd Packet Length Mean'],
            bwd_packet_length_mean=f['Bwd Packet Length Mean'],
            flow_iat_mean=f['Flow IAT Mean'],
            fwd_iat_mean=f['Fwd IAT Mean'],
            bwd_iat_mean=f['Bwd IAT Mean'],
            fwd_psh_flags=f['Fwd PSH Flags'],
            syn_flag_count=f['SYN Flag Count'],
            ack_flag_count=f['ACK Flag Count'],
            packet_length_variance=f['Packet Length Variance'],
            average_packet_size=f['Average Packet Size']
        ))
        
    # Import rules
    from ..api.rules import check_rules, rules
    
    # Check if we're in heavy traffic mode - skip ML for speed
    is_heavy_traffic = sniffer.stats.get("mode") == "heavy_traffic"
        
    # Run detection with timeout protection
    try:
        merged_results = []
        
        # Run ML detection ONLY if:
        # 1. ML is enabled
        # 2. We have flows to process
        # 3. We are NOT in heavy traffic mode (ML is too slow during floods)
        ml_results = None
        if rules.use_ml_detection and flow_objects and not is_heavy_traffic:
            try:
                request = DetectionRequest(flows=flow_objects, method=DetectionMethod.COMBINED)
                # Use wait_for with timeout to prevent blocking
                detection_result = await asyncio.wait_for(
                    detect_anomalies(request),
                    timeout=ML_DETECTION_TIMEOUT
                )
                ml_results = detection_result.results
            except asyncio.TimeoutError:
                # ML took too long, continue with rule-based only
                pass
            except Exception as e:
                # ML failed, continue with rule-based only
                pass
        
        for i, flow_meta in enumerate(raw_flows):
            # Get ML result if available
            ml_is_anomaly = False
            ml_score = 0.0
            ml_attack_type = None
            
            if ml_results and i < len(ml_results):
                ml_result = ml_results[i]
                ml_score = ml_result.score
                ml_attack_type = ml_result.attack_type
                # Only consider it an anomaly if:
                # 1. ML says it's an anomaly
                # 2. Score exceeds threshold
                # 3. Attack type is NOT 'Normal' (classifier confirmed it's an attack)
                if ml_result.is_anomaly and ml_score > rules.anomaly_score_threshold:
                    # Only flag as malicious if attack type indicates actual attack
                    if ml_attack_type and ml_attack_type not in ['Normal', 'BENIGN', 'normal', 'benign']:
                        ml_is_anomaly = True
                    else:
                        # It's classified as Normal, so don't flag it
                        ml_is_anomaly = False
            
            # Check rule-based detection
            rule_check = check_rules(flow_meta['src_ip'], {
                "total_fwd_packets": flow_meta['Total Fwd Packets'],
                "total_bwd_packets": flow_meta['Total Backward Packets'],
                "flow_bytes_per_sec": flow_meta['Flow Bytes/s'],
                "flow_duration": flow_meta['Flow Duration'],
                "dst_port": flow_meta['dst_port'],
                "syn_flag_count": flow_meta['SYN Flag Count']
            })
            
            # Combine results: is_anomaly if EITHER ML or rules flagged it
            is_anomaly = ml_is_anomaly or rule_check["is_anomaly"]
            
            # Determine attack type and severity
            if rule_check["is_anomaly"]:
                attack_type = rule_check["reason"]
                severity = rule_check["severity"]
            elif ml_is_anomaly:
                attack_type = ml_attack_type or "ML Detected"
                severity = "critical" if ml_score > 4 else "high" if ml_score > 3 else "medium"
            else:
                attack_type = None
                severity = "none"
            
            # Register malicious IPs automatically - but ONLY for real attacks
            # Don't register if attack type is Normal/Benign
            if is_anomaly and attack_type and attack_type not in ['Normal', 'BENIGN', 'normal', 'benign', None]:
                register_malicious_ip(
                    ip=flow_meta['src_ip'],
                    score=ml_score if ml_score > 0 else 3.0,
                    attack_type=attack_type
                )
            
            merged_results.append({
                "src_ip": flow_meta['src_ip'],
                "dst_ip": flow_meta['dst_ip'],
                "src_port": flow_meta['src_port'],
                "dst_port": flow_meta['dst_port'],
                "protocol": flow_meta['protocol'],
                "is_attack": is_anomaly,
                "score": ml_score,
                "attack_type": attack_type,
                "severity": severity,
                "detection_method": "rule" if rule_check["is_anomaly"] else ("ml" if ml_is_anomaly else "none"),
                "recommendation": "Block IP" if is_anomaly else "Monitor",
                "sampled": original_count > MAX_FLOWS_PER_REQUEST  # Indicate if sampling occurred
            })
            
        return {"flows": merged_results, "sampled_from": original_count if original_count > MAX_FLOWS_PER_REQUEST else None}
        
    except Exception as e:
        return {"error": str(e), "flows": []}
