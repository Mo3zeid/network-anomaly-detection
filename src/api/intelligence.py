"""
Enhanced IP Intelligence API for detailed IP analysis.
Provides geolocation, reputation, and connection history.
"""
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import asyncio
import aiohttp

from ..utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

# Cache for IP lookups (to avoid repeated API calls)
ip_cache: Dict[str, dict] = {}
CACHE_DURATION = timedelta(hours=1)


class IPDetails(BaseModel):
    ip: str
    country: Optional[str] = None
    country_code: Optional[str] = None
    city: Optional[str] = None
    region: Optional[str] = None
    isp: Optional[str] = None
    org: Optional[str] = None
    as_number: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    is_vpn: Optional[bool] = None
    is_proxy: Optional[bool] = None
    is_tor: Optional[bool] = None
    is_datacenter: Optional[bool] = None
    threat_score: Optional[int] = None
    last_updated: Optional[str] = None


# Track IP activity history
ip_activity: Dict[str, List[dict]] = {}


def record_ip_activity(ip: str, activity_type: str, details: dict = None):
    """Record an activity for an IP address."""
    if ip not in ip_activity:
        ip_activity[ip] = []
    
    activity = {
        "timestamp": datetime.now().isoformat(),
        "type": activity_type,
        "details": details or {}
    }
    
    ip_activity[ip].append(activity)
    
    # Keep only last 100 activities per IP
    if len(ip_activity[ip]) > 100:
        ip_activity[ip] = ip_activity[ip][-100:]


async def fetch_ip_geolocation(ip: str) -> dict:
    """Fetch IP geolocation from free API (ip-api.com)."""
    # Check cache first
    if ip in ip_cache:
        cached = ip_cache[ip]
        if datetime.now() - cached.get("cached_at", datetime.min) < CACHE_DURATION:
            return cached.get("data", {})
    
    try:
        async with aiohttp.ClientSession() as session:
            # Using ip-api.com (free, no API key required)
            url = f"http://ip-api.com/json/{ip}?fields=status,message,country,countryCode,region,regionName,city,zip,lat,lon,timezone,isp,org,as,query"
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("status") == "success":
                        result = {
                            "country": data.get("country"),
                            "country_code": data.get("countryCode"),
                            "city": data.get("city"),
                            "region": data.get("regionName"),
                            "isp": data.get("isp"),
                            "org": data.get("org"),
                            "as_number": data.get("as"),
                            "latitude": data.get("lat"),
                            "longitude": data.get("lon"),
                        }
                        # Cache the result
                        ip_cache[ip] = {
                            "data": result,
                            "cached_at": datetime.now()
                        }
                        return result
    except asyncio.TimeoutError:
        logger.warning(f"Timeout fetching geolocation for {ip}")
    except Exception as e:
        logger.warning(f"Error fetching geolocation for {ip}: {e}")
    
    return {}


def calculate_threat_score(ip: str, activity_count: int = 0, is_flagged: bool = False) -> int:
    """Calculate a threat score for an IP (0-100)."""
    score = 0
    
    # Base score from activity
    if activity_count > 50:
        score += 30
    elif activity_count > 20:
        score += 15
    elif activity_count > 5:
        score += 5
    
    # If flagged as malicious
    if is_flagged:
        score += 50
    
    # Check for suspicious patterns in activity
    activities = ip_activity.get(ip, [])
    for activity in activities[-20:]:  # Check last 20 activities
        if activity.get("type") == "anomaly_detected":
            score += 10
        elif activity.get("type") == "blocked":
            score += 20
    
    return min(score, 100)


@router.get("/ip/{ip}/details")
async def get_ip_details(ip: str):
    """
    Get comprehensive details about an IP address.
    Includes geolocation, ISP, and threat assessment.
    """
    from ..api.firewall import malicious_ips, blocked_ips
    from ..api.rules import ip_stats, whitelist, blacklist
    
    # Fetch geolocation
    geo = await fetch_ip_geolocation(ip)
    
    # Get activity stats
    stats = ip_stats.get(ip, {})
    activities = ip_activity.get(ip, [])
    
    # Check status
    is_malicious = ip in malicious_ips
    is_blocked = ip in blocked_ips
    is_whitelisted = ip in whitelist
    is_blacklisted = ip in blacklist
    
    # Calculate threat score
    threat_score = calculate_threat_score(
        ip, 
        activity_count=len(activities),
        is_flagged=is_malicious or is_blacklisted
    )
    
    # Determine threat level
    if threat_score >= 70:
        threat_level = "critical"
    elif threat_score >= 50:
        threat_level = "high"
    elif threat_score >= 30:
        threat_level = "medium"
    elif threat_score >= 10:
        threat_level = "low"
    else:
        threat_level = "none"
    
    return {
        "ip": ip,
        "geolocation": {
            "country": geo.get("country", "Unknown"),
            "country_code": geo.get("country_code", ""),
            "city": geo.get("city", "Unknown"),
            "region": geo.get("region", ""),
            "latitude": geo.get("latitude"),
            "longitude": geo.get("longitude"),
        },
        "network": {
            "isp": geo.get("isp", "Unknown"),
            "org": geo.get("org", "Unknown"),
            "as_number": geo.get("as_number", ""),
        },
        "status": {
            "is_malicious": is_malicious,
            "is_blocked": is_blocked,
            "is_whitelisted": is_whitelisted,
            "is_blacklisted": is_blacklisted,
        },
        "threat_assessment": {
            "score": threat_score,
            "level": threat_level,
            "confidence": "high" if len(activities) > 10 else "medium" if len(activities) > 3 else "low"
        },
        "statistics": {
            "total_packets": stats.get("packets", 0),
            "total_bytes": stats.get("bytes", 0),
            "total_connections": stats.get("connections", 0),
            "ports_accessed": len(stats.get("ports_accessed", [])),
            "syn_count": stats.get("syn_count", 0),
            "last_seen": stats.get("last_seen", datetime.now()).isoformat() if stats.get("last_seen") else None,
        },
        "activity_count": len(activities),
        "recent_activities": activities[-10:][::-1],  # Last 10, newest first
        "timestamp": datetime.now().isoformat()
    }


@router.get("/ip/{ip}/history")
async def get_ip_history(ip: str, limit: int = 50):
    """Get activity history for an IP address."""
    activities = ip_activity.get(ip, [])
    return {
        "ip": ip,
        "total_activities": len(activities),
        "activities": activities[-limit:][::-1]  # Newest first
    }


@router.get("/live-stats")
async def get_live_stats():
    """
    Get real-time statistics for the dashboard.
    Designed to be polled frequently for live updates.
    """
    from ..api.routes import stats, alerts
    from ..api.firewall import malicious_ips, blocked_ips
    from ..api.rules import ip_stats
    from ..sniffing.sniffer import SnifferService
    
    # Get sniffer status
    try:
        from ..api.sniffer import sniffer
        sniffer_running = sniffer.is_running
        packets_captured = sniffer.stats.get("packets_captured", 0)
        active_flows = sniffer.stats.get("active_flows", 0)
    except:
        sniffer_running = False
        packets_captured = 0
        active_flows = 0
    
    # Get alert summary
    recent_alerts = sorted(alerts, key=lambda x: x.timestamp, reverse=True)[:5]
    alert_summary = [
        {
            "id": a.id,
            "severity": a.severity,
            "attack_type": a.attack_type,
            "timestamp": a.timestamp.isoformat(),
            "score": a.score
        }
        for a in recent_alerts
    ]
    
    # Get top talkers (IPs with most activity)
    top_talkers = sorted(
        [
            {"ip": ip, **{k: v for k, v in s.items() if k not in ["ports_accessed", "window_start", "last_seen"]}}
            for ip, s in ip_stats.items()
        ],
        key=lambda x: x.get("packets", 0),
        reverse=True
    )[:10]
    
    # Attack distribution
    attack_dist = stats.get("attack_distribution", {})
    
    return {
        "timestamp": datetime.now().isoformat(),
        "sniffer": {
            "is_running": sniffer_running,
            "packets_captured": packets_captured,
            "active_flows": active_flows,
        },
        "summary": {
            "total_flows": stats.get("total_flows", 0),
            "total_anomalies": stats.get("total_anomalies", 0),
            "detection_rate": round(stats.get("total_anomalies", 0) / max(stats.get("total_flows", 1), 1) * 100, 1),
            "malicious_ips": len(malicious_ips),
            "blocked_ips": len(blocked_ips),
            "tracked_ips": len(ip_stats),
        },
        "attack_distribution": attack_dist,
        "recent_alerts": alert_summary,
        "top_talkers": top_talkers,
        "uptime_seconds": (datetime.now() - stats.get("start_time", datetime.now())).total_seconds()
    }


@router.get("/threat-timeline")
async def get_threat_timeline(hours: int = 24):
    """
    Get threat activity timeline for visualization.
    Returns hourly aggregated threat data.
    """
    from ..api.routes import alerts
    
    now = datetime.now()
    start_time = now - timedelta(hours=hours)
    
    # Initialize hourly buckets
    timeline = []
    for i in range(hours):
        hour_start = now - timedelta(hours=hours - i)
        hour_end = hour_start + timedelta(hours=1)
        
        # Count alerts in this hour
        hour_alerts = [
            a for a in alerts
            if hour_start <= a.timestamp < hour_end
        ]
        
        # Categorize by severity
        critical = sum(1 for a in hour_alerts if a.severity == "critical")
        high = sum(1 for a in hour_alerts if a.severity == "high")
        medium = sum(1 for a in hour_alerts if a.severity == "medium")
        
        timeline.append({
            "hour": hour_start.strftime("%H:00"),
            "timestamp": hour_start.isoformat(),
            "total": len(hour_alerts),
            "critical": critical,
            "high": high,
            "medium": medium,
        })
    
    return {
        "hours": hours,
        "timeline": timeline,
        "generated_at": now.isoformat()
    }
