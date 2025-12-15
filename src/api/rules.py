"""
Configurable detection rules for network anomaly detection.
Provides rule-based detection that's more controllable than ML.
"""
from typing import Set, Dict, List
from datetime import datetime, timedelta
from fastapi import APIRouter
from pydantic import BaseModel

from ..utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

# ==================== CONFIGURABLE RULES ====================

class DetectionRules(BaseModel):
    # Rate limits (per source IP, per minute) - set high to avoid false positives
    max_packets_per_minute: int = 10000  # Very high to avoid CDN false positives
    max_bytes_per_minute: int = 100_000_000  # 100 MB - streaming/downloads
    max_connections_per_minute: int = 500  # CDNs use many connections
    
    # Thresholds for anomaly scoring
    anomaly_score_threshold: float = 3.0  # Only flag if score > this
    
    # Enable/disable detection methods
    use_ml_detection: bool = True
    use_rate_detection: bool = False  # Disabled by default - too many false positives
    use_port_scan_detection: bool = True
    
    # Port scan detection (increased to avoid false positives from CDNs)
    port_scan_threshold: int = 50  # >50 different ports = port scan
    
    # SYN flood detection
    syn_flood_threshold: int = 100  # >100 SYN packets/min = flood


# Global rules configuration
rules = DetectionRules()

# Whitelist (never flag these IPs)
whitelist: Set[str] = {
    "127.0.0.1",
    "localhost",
    "0.0.0.0",
    # Common DNS servers
    "8.8.8.8",        # Google DNS
    "8.8.4.4",        # Google DNS
    "1.1.1.1",        # Cloudflare DNS
    "1.0.0.1",        # Cloudflare DNS
    "9.9.9.9",        # Quad9 DNS
    "208.67.222.222", # OpenDNS
    "208.67.220.220", # OpenDNS
}

# Common service IP prefixes to ignore (Google, Microsoft, CDNs, etc.)
WHITELIST_PREFIXES = [
    # Google
    "64.233.", "142.250.", "172.217.", "216.58.", "74.125.", "173.194.",
    "209.85.", "66.102.", "66.249.", "72.14.", "108.177.", "172.253.",
    # Microsoft
    "13.107.", "52.96.", "20.190.", "204.79.", "40.126.", "52.114.",
    # Akamai CDN (large range)
    "2.16.", "2.17.", "2.18.", "2.19.", "2.20.", "2.21.", "2.22.", "2.23.",
    "23.32.", "23.33.", "23.34.", "23.35.", "23.36.", "23.37.",
    "23.64.", "23.65.", "23.72.", "23.73.", "104.64.", "104.65.",
    # Cloudflare
    "104.16.", "104.17.", "104.18.", "104.19.", "104.20.", "104.21.",
    "172.64.", "172.65.", "172.66.", "172.67.",
    # Amazon AWS/CloudFront
    "13.32.", "13.33.", "13.35.", "54.230.", "54.239.", "99.84.",
    "52.84.", "52.85.", "52.222.", "143.204.", "205.251.",
    # Facebook/Meta
    "157.240.", "31.13.", "179.60.", "185.60.",
    # Apple
    "17.253.", "17.142.", "17.172.",
    # Netflix
    "45.57.", "108.175.", "192.173.", "198.38.", "198.45.",
]

# Blacklist (always block these IPs)
blacklist: Set[str] = set()

# Rate tracking per IP
ip_stats: Dict[str, dict] = {}


def get_or_create_ip_stats(ip: str) -> dict:
    """Get or create statistics for an IP."""
    now = datetime.now()
    
    if ip not in ip_stats:
        ip_stats[ip] = {
            "packets": 0,
            "bytes": 0,
            "connections": 0,
            "ports_accessed": set(),
            "syn_count": 0,
            "window_start": now,
            "last_seen": now
        }
    
    stats = ip_stats[ip]
    
    # Reset counters if window expired (1 minute)
    if now - stats["window_start"] > timedelta(minutes=1):
        stats["packets"] = 0
        stats["bytes"] = 0
        stats["connections"] = 0
        stats["ports_accessed"] = set()
        stats["syn_count"] = 0
        stats["window_start"] = now
    
    stats["last_seen"] = now
    return stats


def check_rules(ip: str, flow_data: dict) -> dict:
    """
    Check if traffic from this IP violates any rules.
    Returns a dict with is_anomaly, reason, and severity.
    """
    # Whitelist check
    if ip in whitelist:
        return {"is_anomaly": False, "reason": "Whitelisted", "severity": "none"}
    
    # Check against whitelist prefixes (Google, Microsoft, CDNs, etc.)
    for prefix in WHITELIST_PREFIXES:
        if ip.startswith(prefix):
            return {"is_anomaly": False, "reason": "Known service provider", "severity": "none"}
    
    # Blacklist check
    if ip in blacklist:
        return {"is_anomaly": True, "reason": "Blacklisted", "severity": "critical"}
    
    stats = get_or_create_ip_stats(ip)
    
    # Update stats from flow
    stats["packets"] += flow_data.get("total_fwd_packets", 0) + flow_data.get("total_bwd_packets", 0)
    stats["bytes"] += int(flow_data.get("flow_bytes_per_sec", 0) * flow_data.get("flow_duration", 0) / 1_000_000)
    stats["connections"] += 1
    
    if flow_data.get("dst_port"):
        stats["ports_accessed"].add(flow_data["dst_port"])
    
    stats["syn_count"] += flow_data.get("syn_flag_count", 0)
    
    violations = []
    severity = "medium"
    
    # Rate-based detection
    if rules.use_rate_detection:
        if stats["packets"] > rules.max_packets_per_minute:
            violations.append(f"High packet rate: {stats['packets']}/min")
            severity = "high"
        
        if stats["bytes"] > rules.max_bytes_per_minute:
            violations.append(f"High bandwidth: {stats['bytes']/1_000_000:.1f}MB/min")
            severity = "high"
        
        if stats["connections"] > rules.max_connections_per_minute:
            violations.append(f"Too many connections: {stats['connections']}/min")
            severity = "high"
    
    # Port scan detection
    if rules.use_port_scan_detection:
        if len(stats["ports_accessed"]) > rules.port_scan_threshold:
            violations.append(f"Port scan: {len(stats['ports_accessed'])} ports")
            severity = "critical"
    
    # SYN flood detection
    if stats["syn_count"] > rules.syn_flood_threshold:
        violations.append(f"SYN flood: {stats['syn_count']} SYN packets")
        severity = "critical"
    
    if violations:
        return {
            "is_anomaly": True,
            "reason": "; ".join(violations),
            "severity": severity
        }
    
    return {"is_anomaly": False, "reason": "Normal", "severity": "none"}


# ==================== API ENDPOINTS ====================

@router.get("/rules")
async def get_rules():
    """Get current detection rules."""
    return {
        "rules": rules.dict(),
        "whitelist": list(whitelist),
        "blacklist": list(blacklist)
    }


class RulesUpdate(BaseModel):
    max_packets_per_minute: int = None
    max_bytes_per_minute: int = None
    max_connections_per_minute: int = None
    anomaly_score_threshold: float = None
    use_ml_detection: bool = None
    use_rate_detection: bool = None
    use_port_scan_detection: bool = None
    port_scan_threshold: int = None
    syn_flood_threshold: int = None


@router.post("/rules")
async def update_rules(update: RulesUpdate):
    """Update detection rules."""
    global rules
    
    for field, value in update.dict().items():
        if value is not None:
            setattr(rules, field, value)
    
    logger.info(f"Rules updated: {update.dict()}")
    return {"message": "Rules updated", "rules": rules.dict()}


class IPListAction(BaseModel):
    ip: str


@router.post("/rules/whitelist/add")
async def add_to_whitelist(action: IPListAction):
    """Add an IP to the whitelist."""
    whitelist.add(action.ip)
    # Also remove from blacklist if present
    blacklist.discard(action.ip)
    return {"message": f"IP {action.ip} added to whitelist", "whitelist": list(whitelist)}


@router.post("/rules/whitelist/remove")
async def remove_from_whitelist(action: IPListAction):
    """Remove an IP from the whitelist."""
    whitelist.discard(action.ip)
    return {"message": f"IP {action.ip} removed from whitelist", "whitelist": list(whitelist)}


@router.post("/rules/blacklist/add")
async def add_to_blacklist(action: IPListAction):
    """Add an IP to the blacklist."""
    blacklist.add(action.ip)
    # Also remove from whitelist if present
    whitelist.discard(action.ip)
    return {"message": f"IP {action.ip} added to blacklist", "blacklist": list(blacklist)}


@router.post("/rules/blacklist/remove")
async def remove_from_blacklist(action: IPListAction):
    """Remove an IP from the blacklist."""
    blacklist.discard(action.ip)
    return {"message": f"IP {action.ip} removed from blacklist", "blacklist": list(blacklist)}


@router.get("/rules/ip-stats")
async def get_ip_statistics():
    """Get rate statistics for all tracked IPs."""
    result = []
    for ip, stats in ip_stats.items():
        result.append({
            "ip": ip,
            "packets": stats["packets"],
            "bytes": stats["bytes"],
            "connections": stats["connections"],
            "ports_accessed": len(stats["ports_accessed"]),
            "syn_count": stats["syn_count"],
            "last_seen": stats["last_seen"].isoformat()
        })
    return {"ip_stats": result}


@router.post("/rules/reset-stats")
async def reset_ip_stats():
    """Reset all IP statistics."""
    ip_stats.clear()
    return {"message": "IP statistics reset"}
