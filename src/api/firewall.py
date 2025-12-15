"""
Firewall management for blocking malicious IPs.
Uses Windows Firewall (netsh) commands.
"""
import subprocess
from typing import Set, List, Dict
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

# Store blocked IPs and detected malicious IPs
blocked_ips: Set[str] = set()
malicious_ips: Dict[str, dict] = {}  # ip -> {score, attack_type, first_seen, blocked}


class IPAction(BaseModel):
    ip: str


def add_firewall_rule(ip: str) -> bool:
    """Add Windows Firewall rules to completely block an IP."""
    try:
        rule_name = f"NetGuard_Block_{ip.replace('.', '_')}"
        
        # Block ALL inbound traffic from this IP
        cmd_in = f'netsh advfirewall firewall add rule name="{rule_name}_IN" dir=in action=block remoteip={ip} protocol=any profile=any'
        subprocess.run(cmd_in, shell=True, check=True, capture_output=True)
        
        # Block ALL outbound traffic to this IP
        cmd_out = f'netsh advfirewall firewall add rule name="{rule_name}_OUT" dir=out action=block remoteip={ip} protocol=any profile=any'
        subprocess.run(cmd_out, shell=True, check=True, capture_output=True)
        
        # Also add route to null (black hole) - more aggressive
        try:
            cmd_route = f'route add {ip} mask 255.255.255.255 0.0.0.0'
            subprocess.run(cmd_route, shell=True, capture_output=True)
        except:
            pass  # Route command is optional
        
        logger.info(f"Firewall rule added to block IP: {ip}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to add firewall rule for {ip}: {e}")
        return False


def remove_firewall_rule(ip: str) -> bool:
    """Remove a Windows Firewall rule for an IP."""
    try:
        rule_name = f"NetGuard_Block_{ip.replace('.', '_')}"
        
        # Remove inbound rule
        cmd_in = f'netsh advfirewall firewall delete rule name="{rule_name}_IN"'
        subprocess.run(cmd_in, shell=True, check=True, capture_output=True)
        
        # Remove outbound rule
        cmd_out = f'netsh advfirewall firewall delete rule name="{rule_name}_OUT"'
        subprocess.run(cmd_out, shell=True, check=True, capture_output=True)
        
        # Remove null route if it exists
        try:
            cmd_route = f'route delete {ip}'
            subprocess.run(cmd_route, shell=True, capture_output=True)
        except:
            pass
        
        logger.info(f"Firewall rule removed for IP: {ip}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to remove firewall rule for {ip}: {e}")
        return False


def register_malicious_ip(ip: str, score: float, attack_type: str = None):
    """Register an IP as potentially malicious."""
    from datetime import datetime
    
    if ip not in malicious_ips:
        malicious_ips[ip] = {
            "ip": ip,
            "score": score,
            "attack_type": attack_type or "Unknown",
            "first_seen": datetime.now().isoformat(),
            "hit_count": 1,
            "blocked": ip in blocked_ips
        }
    else:
        malicious_ips[ip]["hit_count"] += 1
        malicious_ips[ip]["score"] = max(malicious_ips[ip]["score"], score)
        if attack_type:
            malicious_ips[ip]["attack_type"] = attack_type


@router.get("/firewall/malicious")
async def get_malicious_ips():
    """Get list of detected malicious IPs."""
    # Update blocked status
    for ip in malicious_ips:
        malicious_ips[ip]["blocked"] = ip in blocked_ips
    
    return {
        "malicious_ips": list(malicious_ips.values()),
        "blocked_count": len(blocked_ips)
    }


@router.post("/firewall/block")
async def block_ip(action: IPAction):
    """Block an IP address using Windows Firewall."""
    ip = action.ip
    
    if ip in blocked_ips:
        return {"message": f"IP {ip} is already blocked", "success": True}
    
    success = add_firewall_rule(ip)
    
    if success:
        blocked_ips.add(ip)
        if ip in malicious_ips:
            malicious_ips[ip]["blocked"] = True
        return {"message": f"IP {ip} has been blocked", "success": True}
    else:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to block IP {ip}. Make sure you're running as Administrator."
        )


@router.post("/firewall/unblock")
async def unblock_ip(action: IPAction):
    """Unblock an IP address."""
    ip = action.ip
    
    if ip not in blocked_ips:
        return {"message": f"IP {ip} is not blocked", "success": True}
    
    success = remove_firewall_rule(ip)
    
    if success:
        blocked_ips.discard(ip)
        if ip in malicious_ips:
            malicious_ips[ip]["blocked"] = False
        return {"message": f"IP {ip} has been unblocked", "success": True}
    else:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to unblock IP {ip}. Make sure you're running as Administrator."
        )


@router.get("/firewall/blocked")
async def get_blocked_ips():
    """Get list of currently blocked IPs."""
    return {"blocked_ips": list(blocked_ips)}


@router.post("/firewall/clear")
async def clear_malicious_list():
    """Clear the malicious IPs list (does not unblock)."""
    malicious_ips.clear()
    return {"message": "Malicious IP list cleared"}
