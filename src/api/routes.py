"""
Extended API routes with additional endpoints.
"""
import uuid
import io
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse

from .models import (
    DetectionRequest, DetectionResponse, AnomalyDetail,
    Alert, AlertsResponse, StatsResponse, DetectionMethod
)
from ..detection.statistical import StatisticalDetector
from ..detection.isolation_forest import IsolationForestDetector
from ..detection.classifier import AttackClassifier
from ..data.preprocessor import DataPreprocessor
from ..utils.config import MODELS_DIR, SELECTED_FEATURES
from ..utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

# Global state for models and statistics
models = {
    'preprocessor': None,
    'statistical': None,
    'isolation_forest': None,
    'classifier': None
}

stats = {
    'total_flows': 0,
    'total_anomalies': 0,
    'attack_distribution': {},
    'start_time': datetime.now()
}

alerts: List[Alert] = []


def load_models():
    """Load all trained models."""
    global models
    
    try:
        models['preprocessor'] = DataPreprocessor.load()
        logger.info("Preprocessor loaded")
    except Exception as e:
        logger.warning(f"Could not load preprocessor: {e}")
    
    try:
        import joblib
        models['statistical'] = joblib.load(MODELS_DIR / "statistical_detector.joblib")
        logger.info("Statistical detector loaded")
    except Exception as e:
        logger.warning(f"Could not load statistical detector: {e}")
    
    try:
        models['isolation_forest'] = IsolationForestDetector.load()
        logger.info("Isolation Forest loaded")
    except Exception as e:
        logger.warning(f"Could not load Isolation Forest: {e}")
    
    try:
        models['classifier'] = AttackClassifier()
        logger.info("Attack classifier loaded")
    except Exception as e:
        logger.warning(f"Could not load classifier: {e}")


def flow_to_array(flow) -> np.ndarray:
    """Convert a NetworkFlow to numpy array matching our features."""
    return np.array([
        flow.flow_duration,
        flow.total_fwd_packets,
        flow.total_bwd_packets,
        flow.flow_bytes_per_sec,
        flow.flow_packets_per_sec,
        flow.fwd_packet_length_mean,
        flow.bwd_packet_length_mean,
        flow.flow_iat_mean,
        flow.fwd_iat_mean,
        flow.bwd_iat_mean,
        flow.fwd_psh_flags,
        flow.syn_flag_count,
        flow.ack_flag_count,
        flow.packet_length_variance,
        flow.average_packet_size
    ])


@router.post("/detect", response_model=DetectionResponse)
async def detect_anomalies(request: DetectionRequest):
    """
    Detect anomalies in network flows.
    Limited to 100 flows per request for performance.
    """
    global stats, alerts
    
    # Early return for empty requests
    if not request.flows:
        return DetectionResponse(
            total=0,
            anomalies=0,
            results=[]
        )
    
    # Limit batch size to prevent blocking
    MAX_BATCH_SIZE = 100
    flows_to_process = request.flows[:MAX_BATCH_SIZE]
    
    if not any(models.values()):
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please train models first."
        )
    
    X = np.array([flow_to_array(flow) for flow in flows_to_process])
    # Sanitize Input: Replace Inf with 0.0 to prevent XGBoost crash
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    if models['preprocessor']:
        try:
            X = models['preprocessor'].transform(X)
        except Exception as e:
            logger.warning(f"Preprocessing failed: {e}")
    
    results = []
    method_used = request.method.value
    
    for i, x in enumerate(X):
        x_2d = x.reshape(1, -1)
        is_anomaly = False
        score = 0.0
        attack_type = None
        attack_confidence = None
        
        if request.method in [DetectionMethod.STATISTICAL, DetectionMethod.COMBINED]:
            if models['statistical']:
                anomaly, scores = models['statistical'].detect(x_2d, method='zscore')
                is_anomaly = is_anomaly or anomaly[0]
                score = max(score, float(scores[0]))
        
        if request.method in [DetectionMethod.ISOLATION_FOREST, DetectionMethod.COMBINED]:
            if models['isolation_forest']:
                anomaly, scores = models['isolation_forest'].detect(x_2d)
                is_anomaly = is_anomaly or anomaly[0]
                score = max(score, float(scores[0]))
        
        # If ML detected anomaly, use classifier to determine attack type
        if is_anomaly and models['classifier']:
            try:
                classification = models['classifier'].classify_single(x)
                attack_type = classification['attack_type']
                attack_confidence = classification['confidence']
                
                # CRITICAL: If classifier says Normal/BENIGN, it's NOT an anomaly
                # The classifier is the final authority - it knows attack patterns
                if attack_type in ['Normal', 'BENIGN', 'normal', 'benign']:
                    is_anomaly = False  # Override ML detector - classifier says it's normal
                    attack_type = None  # Clear attack type
                    
            except Exception as e:
                logger.warning(f"Classification failed: {e}")
        
        results.append(AnomalyDetail(
            index=i,
            is_anomaly=is_anomaly,
            score=score,
            attack_type=attack_type,
            attack_confidence=attack_confidence,
            method=method_used
        ))
        
        # Only create alerts for ACTUAL attacks (not Normal traffic)
        if is_anomaly and attack_type and attack_type not in ['Normal', 'BENIGN'] and score > 2.0:
            severity = "critical" if score > 4 else "high" if score > 3 else "medium"
            alert = Alert(
                id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                severity=severity,
                attack_type=attack_type,
                source_info={"flow_index": i},
                score=score
            )
            alerts.append(alert)
            
            if len(alerts) > 100:
                alerts = alerts[-100:]
    
    anomaly_count = sum(1 for r in results if r.is_anomaly)
    stats['total_flows'] += len(request.flows)
    stats['total_anomalies'] += anomaly_count
    
    for r in results:
        if r.attack_type:
            stats['attack_distribution'][r.attack_type] = \
                stats['attack_distribution'].get(r.attack_type, 0) + 1
    
    return DetectionResponse(
        total_flows=len(request.flows),
        anomalies_detected=anomaly_count,
        detection_rate=anomaly_count / len(request.flows) if request.flows else 0,
        method_used=method_used,
        results=results
    )


@router.post("/detect/upload")
async def detect_from_file(
    file: UploadFile = File(...),
    method: str = Form("combined")
):
    """
    Detect anomalies from an uploaded CSV file.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    
    # Map columns to expected format
    column_mapping = {
        'Flow Duration': 'flow_duration',
        'Total Fwd Packets': 'total_fwd_packets',
        'Total Backward Packets': 'total_bwd_packets',
        'Flow Bytes/s': 'flow_bytes_per_sec',
        'Flow Packets/s': 'flow_packets_per_sec',
        'Fwd Packet Length Mean': 'fwd_packet_length_mean',
        'Bwd Packet Length Mean': 'bwd_packet_length_mean',
        'Flow IAT Mean': 'flow_iat_mean',
        'Fwd IAT Mean': 'fwd_iat_mean',
        'Bwd IAT Mean': 'bwd_iat_mean',
        'Fwd PSH Flags': 'fwd_psh_flags',
        'SYN Flag Count': 'syn_flag_count',
        'ACK Flag Count': 'ack_flag_count',
        'Packet Length Variance': 'packet_length_variance',
        'Average Packet Size': 'average_packet_size'
    }
    
    # Limit to first 1000 rows
    df = df.head(1000)
    
    results = []
    for idx, row in df.iterrows():
        try:
            x = np.array([
                float(row.get('Flow Duration', 0)),
                float(row.get('Total Fwd Packets', 0)),
                float(row.get('Total Backward Packets', 0)),
                float(row.get('Flow Bytes/s', 0)),
                float(row.get('Flow Packets/s', 0)),
                float(row.get('Fwd Packet Length Mean', 0)),
                float(row.get('Bwd Packet Length Mean', 0)),
                float(row.get('Flow IAT Mean', 0)),
                float(row.get('Fwd IAT Mean', 0)),
                float(row.get('Bwd IAT Mean', 0)),
                float(row.get('Fwd PSH Flags', 0)),
                float(row.get('SYN Flag Count', 0)),
                float(row.get('ACK Flag Count', 0)),
                float(row.get('Packet Length Variance', 0)),
                float(row.get('Average Packet Size', 0))
            ])
            
            # Handle NaN/Inf
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            
            x_2d = x.reshape(1, -1)
            is_anomaly = False
            score = 0.0
            
            if models['statistical']:
                anomaly, scores = models['statistical'].detect(x_2d, method='zscore')
                is_anomaly = anomaly[0]
                score = float(scores[0])
            
            if models['isolation_forest']:
                anomaly, scores = models['isolation_forest'].detect(x_2d)
                is_anomaly = is_anomaly or anomaly[0]
                score = max(score, float(scores[0]))
            
            attack_type = None
            if is_anomaly and models['classifier']:
                try:
                    classification = models['classifier'].classify_single(x)
                    attack_type = classification['attack_type']
                except:
                    pass
            
            results.append({
                'index': idx,
                'is_anomaly': bool(is_anomaly),
                'score': score,
                'attack_type': attack_type
            })
        except Exception as e:
            logger.warning(f"Row {idx} failed: {e}")
    
    anomaly_count = sum(1 for r in results if r['is_anomaly'])
    
    return {
        'filename': file.filename,
        'total_rows': len(results),
        'anomalies_detected': anomaly_count,
        'detection_rate': anomaly_count / len(results) if results else 0,
        'results': results[:100]  # Return first 100 for preview
    }


@router.get("/stats", response_model=StatsResponse)
async def get_statistics():
    """Get detection statistics."""
    uptime = (datetime.now() - stats['start_time']).total_seconds()
    
    return StatsResponse(
        total_flows_analyzed=stats['total_flows'],
        total_anomalies_detected=stats['total_anomalies'],
        detection_rate=stats['total_anomalies'] / stats['total_flows'] if stats['total_flows'] > 0 else 0,
        attack_distribution=stats['attack_distribution'],
        model_status={
            'preprocessor': models['preprocessor'] is not None,
            'statistical': models['statistical'] is not None,
            'isolation_forest': models['isolation_forest'] is not None,
            'classifier': models['classifier'] is not None
        },
        uptime_seconds=uptime
    )


@router.get("/alerts", response_model=AlertsResponse)
async def get_alerts(limit: int = 20, unacknowledged_only: bool = False):
    """Get recent security alerts."""
    filtered = alerts
    if unacknowledged_only:
        filtered = [a for a in alerts if not a.is_acknowledged]
    
    sorted_alerts = sorted(filtered, key=lambda x: x.timestamp, reverse=True)[:limit]
    
    return AlertsResponse(
        total_alerts=len(alerts),
        unacknowledged=sum(1 for a in alerts if not a.is_acknowledged),
        alerts=sorted_alerts
    )


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str):
    """Acknowledge an alert."""
    for alert in alerts:
        if alert.id == alert_id:
            alert.is_acknowledged = True
            return {"message": "Alert acknowledged", "id": alert_id}
    
    raise HTTPException(status_code=404, detail="Alert not found")


@router.post("/reset")
async def reset_system():
    """Reset all system statistics and alerts."""
    global stats, alerts
    
    # Reset stats
    stats['total_flows'] = 0
    stats['total_anomalies'] = 0
    stats['attack_distribution'] = {}
    stats['start_time'] = datetime.now()
    
    # Clear alerts
    alerts.clear()
    
    logger.info("System data reset requested")
    return {"message": "All system data has been reset"}


@router.get("/models/info")
async def get_model_info():
    """Get detailed model information."""
    info = {
        'models': {},
        'features': SELECTED_FEATURES,
        'feature_count': len(SELECTED_FEATURES)
    }
    
    # Preprocessor info
    if models['preprocessor']:
        info['models']['preprocessor'] = {
            'status': 'loaded',
            'type': 'StandardScaler',
            'features': models['preprocessor'].features
        }
    
    # Statistical detector info
    if models['statistical']:
        info['models']['statistical'] = {
            'status': 'loaded',
            'type': 'Statistical (Z-score, IQR)',
            'zscore_threshold': models['statistical'].zscore_threshold
        }
    
    # Isolation Forest info
    if models['isolation_forest']:
        info['models']['isolation_forest'] = {
            'status': 'loaded',
            'type': 'Isolation Forest',
            'contamination': models['isolation_forest'].contamination,
            'n_estimators': models['isolation_forest'].n_estimators
        }
    
    # Classifier info
    if models['classifier']:
        info['models']['classifier'] = {
            'status': 'loaded',
            'type': 'Random Forest Classifier',
            'classes': list(models['classifier'].classes_) if models['classifier'].classes_ is not None else [],
            'n_estimators': models['classifier'].n_estimators
        }
    
    return info


@router.post("/simulate")
async def simulate_traffic(num_samples: int = 20, anomaly_ratio: float = 0.3):
    """
    Simulate realistic network traffic for demo purposes.
    Returns detailed traffic with IPs, ports, protocols, and attack information.
    Updates global stats so dashboard reflects simulated traffic.
    """
    global stats, alerts
    from ..simulation.traffic_generator import generate_traffic_batch, get_attack_info
    
    # Generate realistic traffic
    flows = generate_traffic_batch(num_samples, anomaly_ratio)
    
    # Process through detection if models are loaded
    detection_results = []
    detected_count = 0
    
    for flow in flows:
        result = {
            **flow,
            "detection_status": "pending",
            "ml_detected": False,
            "ml_score": 0.0
        }
        
        # Run through detection models
        if any(models.values()):
            try:
                import numpy as np
                x = np.array([
                    float(flow["flow_duration"]),
                    float(flow["total_fwd_packets"]),
                    float(flow["total_bwd_packets"]),
                    float(flow["flow_bytes_per_sec"]),
                    float(flow["flow_packets_per_sec"]),
                    float(flow["fwd_packet_length_mean"]),
                    float(flow["bwd_packet_length_mean"]),
                    float(flow["flow_iat_mean"]),
                    float(flow["fwd_iat_mean"]),
                    float(flow["bwd_iat_mean"]),
                    float(flow["fwd_psh_flags"]),
                    float(flow["syn_flag_count"]),
                    float(flow["ack_flag_count"]),
                    float(flow["packet_length_variance"]),
                    float(flow["average_packet_size"])
                ])
                
                x_2d = x.reshape(1, -1)
                detected_anomaly = False
                score = 0.0
                
                if models['statistical']:
                    anomaly, scores = models['statistical'].detect(x_2d, method='zscore')
                    detected_anomaly = bool(anomaly[0])
                    score = float(scores[0])
                
                if models['isolation_forest']:
                    anomaly, scores = models['isolation_forest'].detect(x_2d)
                    detected_anomaly = detected_anomaly or bool(anomaly[0])
                    score = max(score, float(scores[0]))
                
                result["ml_detected"] = bool(detected_anomaly)
                result["ml_score"] = round(float(score), 4)
                result["detection_status"] = "detected" if detected_anomaly else "normal"
                
                if detected_anomaly:
                    detected_count += 1
                
                # Get attack info if this is an attack
                if flow["is_attack"]:
                    attack_info = get_attack_info(flow["attack_type"])
                    result["attack_info"] = attack_info
                    
                    # Update attack distribution in global stats
                    attack_type = flow["attack_type"]
                    stats['attack_distribution'][attack_type] = \
                        stats['attack_distribution'].get(attack_type, 0) + 1
                    
                    # Create alert for detected attacks
                    if detected_anomaly and score > 1.5:
                        severity = "critical" if score > 4 else "high" if score > 2.5 else "medium"
                        alert = Alert(
                            id=str(uuid.uuid4()),
                            timestamp=datetime.now(),
                            severity=severity,
                            attack_type=attack_type,
                            source_info={
                                "src_ip": flow["src_ip"],
                                "dst_ip": flow["dst_ip"],
                                "src_port": flow["src_port"],
                                "dst_port": flow["dst_port"]
                            },
                            score=score
                        )
                        alerts.append(alert)
                        
                        # Keep only last 100 alerts
                        if len(alerts) > 100:
                            alerts = alerts[-100:]
                    
            except Exception as e:
                result["detection_status"] = "error"
                result["error"] = str(e)
        
        detection_results.append(result)
    
    # Update global stats
    total = len(detection_results)
    attacks_simulated = sum(1 for r in detection_results if r.get("is_attack"))
    
    stats['total_flows'] += total
    stats['total_anomalies'] += detected_count
    
    # Calculate accuracy: what % of actual attacks were detected
    # This should be <=100%
    true_positives = sum(1 for r in detection_results if r.get("is_attack") and r.get("ml_detected"))
    accuracy = round(true_positives / attacks_simulated * 100 if attacks_simulated > 0 else 0, 1)
    
    return {
        "message": f"Generated {num_samples} realistic network flows",
        "summary": {
            "total_flows": total,
            "simulated_attacks": attacks_simulated,
            "ml_detected_anomalies": detected_count,
            "true_positives": true_positives,
            "detection_accuracy": min(accuracy, 100.0)  # Cap at 100%
        },
        "flows": detection_results
    }


@router.get("/attack-types")
async def get_attack_types():
    """Get information about all supported attack types."""
    from ..simulation.traffic_generator import ATTACK_DESCRIPTIONS
    return ATTACK_DESCRIPTIONS


@router.get("/export/results")
async def export_results(format: str = "csv"):
    """Export detection results as CSV."""
    global stats, alerts
    
    # Create DataFrame from alerts
    data = []
    for alert in alerts:
        data.append({
            'Timestamp': alert.timestamp.isoformat(),
            'Severity': alert.severity,
            'Attack Type': alert.attack_type,
            'Score': alert.score,
            'Acknowledged': alert.is_acknowledged
        })
    
    df = pd.DataFrame(data)
    
    if format == "csv":
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode()),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=alerts_export.csv"}
        )
    
    return {"error": "Unsupported format"}
