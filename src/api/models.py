"""
Pydantic models for API request/response schemas.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime
from enum import Enum


class DetectionMethod(str, Enum):
    """Available detection methods."""
    STATISTICAL = "statistical"
    ISOLATION_FOREST = "isolation_forest"
    COMBINED = "combined"


class NetworkFlow(BaseModel):
    """Single network flow for analysis."""
    flow_duration: float = Field(..., description="Duration of the flow in microseconds")
    total_fwd_packets: int = Field(..., description="Total packets sent forward")
    total_bwd_packets: int = Field(..., description="Total packets sent backward")
    flow_bytes_per_sec: float = Field(..., description="Bytes per second")
    flow_packets_per_sec: float = Field(..., description="Packets per second")
    fwd_packet_length_mean: float = Field(..., description="Mean forward packet length")
    bwd_packet_length_mean: float = Field(..., description="Mean backward packet length")
    flow_iat_mean: float = Field(..., description="Mean inter-arrival time")
    fwd_iat_mean: float = Field(..., description="Forward IAT mean")
    bwd_iat_mean: float = Field(..., description="Backward IAT mean")
    fwd_psh_flags: int = Field(0, description="Forward PSH flags count")
    syn_flag_count: int = Field(0, description="SYN flag count")
    ack_flag_count: int = Field(0, description="ACK flag count")
    packet_length_variance: float = Field(0, description="Packet length variance")
    average_packet_size: float = Field(..., description="Average packet size")

    class Config:
        json_schema_extra = {
            "example": {
                "flow_duration": 120000,
                "total_fwd_packets": 10,
                "total_bwd_packets": 8,
                "flow_bytes_per_sec": 1500.5,
                "flow_packets_per_sec": 15.0,
                "fwd_packet_length_mean": 150.5,
                "bwd_packet_length_mean": 200.3,
                "flow_iat_mean": 8000.0,
                "fwd_iat_mean": 10000.0,
                "bwd_iat_mean": 12000.0,
                "fwd_psh_flags": 2,
                "syn_flag_count": 1,
                "ack_flag_count": 5,
                "packet_length_variance": 500.0,
                "average_packet_size": 175.4
            }
        }


class DetectionRequest(BaseModel):
    """Request for anomaly detection."""
    flows: List[NetworkFlow] = Field(..., description="List of network flows to analyze")
    method: DetectionMethod = Field(
        DetectionMethod.COMBINED,
        description="Detection method to use"
    )


class AnomalyDetail(BaseModel):
    """Details about a detected anomaly."""
    index: int
    is_anomaly: bool
    score: float
    attack_type: Optional[str] = None
    attack_confidence: Optional[float] = None
    method: str


class DetectionResponse(BaseModel):
    """Response from anomaly detection."""
    total_flows: int
    anomalies_detected: int
    detection_rate: float
    method_used: str
    results: List[AnomalyDetail]
    timestamp: datetime = Field(default_factory=datetime.now)


class Alert(BaseModel):
    """Security alert for detected anomaly."""
    id: str
    timestamp: datetime
    severity: str  # low, medium, high, critical
    attack_type: str
    source_info: Dict
    score: float
    is_acknowledged: bool = False


class AlertsResponse(BaseModel):
    """Response with list of alerts."""
    total_alerts: int
    unacknowledged: int
    alerts: List[Alert]


class StatsResponse(BaseModel):
    """System statistics response."""
    total_flows_analyzed: int
    total_anomalies_detected: int
    detection_rate: float
    attack_distribution: Dict[str, int]
    model_status: Dict[str, bool]
    uptime_seconds: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models_loaded: bool
    timestamp: datetime = Field(default_factory=datetime.now)
