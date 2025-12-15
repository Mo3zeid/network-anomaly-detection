"""
Background service for packet sniffing and flow aggregation.
Optimized for extreme traffic conditions (4000+ pps flood attacks).
"""
import threading
import time
import queue
from scapy.all import sniff, IP
from typing import List, Dict
from pathlib import Path

from .feature_extractor import FeatureExtractor
from ..utils.logger import get_logger

logger = get_logger(__name__)

# AGGRESSIVE memory limits for flood attack survival
MAX_QUEUE_SIZE = 500  # Reduced from 10000 - small queue, fast processing
MAX_ACTIVE_FLOWS = 1000  # Reduced from 5000
MAX_PACKETS_PER_FLOW = 100  # Reduced from 1000

# Adaptive sampling - when traffic is heavy, skip packets
SAMPLE_RATE_NORMAL = 1  # Process every packet when traffic is light
SAMPLE_RATE_HEAVY = 10  # Process 1 in 10 packets when under attack
HEAVY_TRAFFIC_THRESHOLD = 1000  # packets/sec threshold for heavy mode


class SnifferService:
    def __init__(self):
        self.is_running = False
        self.sniffer_thread = None
        # Use bounded queue to prevent memory exhaustion
        self.packet_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        self.feature_extractor = FeatureExtractor(max_packets_per_flow=MAX_PACKETS_PER_FLOW)
        self.stats = {
            "packets_captured": 0,
            "packets_seen": 0,
            "active_flows": 0,
            "packets_dropped": 0,
            "sample_rate": SAMPLE_RATE_NORMAL,
            "mode": "normal"
        }
        self.stop_event = threading.Event()
        self.ignored_ip = None
        
        # Packet counter for sampling
        self._packet_counter = 0
        self._last_rate_check = time.time()
        self._packets_since_check = 0

    def start(self, interface: str = None, exclude_local: bool = True):
        """Start the sniffing service."""
        if self.is_running:
            return
            
        self.is_running = True
        self.stop_event.clear()
        self._packet_counter = 0
        self._last_rate_check = time.time()
        self._packets_since_check = 0
        
        # Determine local IP to exclude
        if exclude_local:
            try:
                from scapy.all import get_if_addr, conf
                iface = interface or conf.iface
                self.ignored_ip = get_if_addr(iface)
                logger.info(f"Excluding traffic from local IP: {self.ignored_ip}")
            except Exception as e:
                logger.warning(f"Could not determine local IP: {e}")
                self.ignored_ip = None
        else:
            self.ignored_ip = None
        
        # Start packet processing thread
        threading.Thread(target=self._process_packets, daemon=True).start()
        
        # Start rate monitoring thread
        threading.Thread(target=self._monitor_rate, daemon=True).start()
        
        # Start sniffing thread
        self.sniffer_thread = threading.Thread(
            target=self._sniff_loop,
            args=(interface,),
            daemon=True
        )
        self.sniffer_thread.start()
        logger.info(f"Sniffer started on interface: {interface or 'default'}")

    def stop(self):
        """Stop the sniffing service."""
        self.is_running = False
        self.stop_event.set()
        logger.info("Sniffer stopped")
    
    def _monitor_rate(self):
        """Monitor packet rate and adjust sampling dynamically."""
        while self.is_running:
            time.sleep(1.0)  # Check every second
            
            current_time = time.time()
            elapsed = current_time - self._last_rate_check
            
            if elapsed >= 1.0:
                rate = self._packets_since_check / elapsed
                
                # Adaptive sampling based on traffic rate
                if rate > HEAVY_TRAFFIC_THRESHOLD:
                    self.stats["sample_rate"] = SAMPLE_RATE_HEAVY
                    self.stats["mode"] = "heavy_traffic"
                else:
                    self.stats["sample_rate"] = SAMPLE_RATE_NORMAL
                    self.stats["mode"] = "normal"
                
                self._last_rate_check = current_time
                self._packets_since_check = 0

    def _sniff_loop(self, interface):
        """Main Scapy sniffing loop with packet sampling."""
        try:
            def sampled_enqueue(pkt):
                """Sample packets to prevent overwhelming under flood."""
                self._packets_since_check += 1
                self._packet_counter += 1
                self.stats["packets_seen"] = self._packet_counter
                
                # Sample based on current rate
                sample_rate = self.stats["sample_rate"]
                if sample_rate > 1 and (self._packet_counter % sample_rate) != 0:
                    # Skip this packet (sampling)
                    return
                
                try:
                    self.packet_queue.put_nowait(pkt)
                except queue.Full:
                    # Queue is full, drop packet to prevent memory issues
                    self.stats["packets_dropped"] += 1
                    
            # Filter for IP traffic only
            # Try sniffing with standard socket
            try:
                sniff(
                    iface=interface,
                    prn=sampled_enqueue,
                    store=False,
                    stop_filter=lambda x: not self.is_running,
                    filter="ip",
                    monitor=False  # Disable monitor mode to avoid permission issues
                )
            except PermissionError:
                logger.error("Permission denied: Live sniffing requires root/sudo privileges.")
                # We can't fix root, but we can stop gracefully or try a user-mode socket
                self.is_running = False
            except Exception as e:
                 logger.error(f"Sniffer crashed: {e}")
                 self.is_running = False
        except Exception as e:
            logger.error(f"Sniffing error: {e}")
            self.is_running = False

    def _process_packets(self):
        """Process queued packets into flows."""
        cleanup_counter = 0
        
        while self.is_running or not self.packet_queue.empty():
            try:
                # Shorter timeout for faster response
                pkt = self.packet_queue.get(timeout=0.5)
                
                # Filter excluded IP
                if self.ignored_ip and IP in pkt:
                    if pkt[IP].src == self.ignored_ip:
                        continue
                
                # Check if we're at flow limit before adding more
                if len(self.feature_extractor.active_flows) >= MAX_ACTIVE_FLOWS:
                    # Force cleanup of old flows
                    self._cleanup_old_flows(force=True)
                        
                self.feature_extractor.process_packet(pkt)
                self.stats["packets_captured"] += 1
                self.stats["active_flows"] = len(self.feature_extractor.active_flows)
                
                # More frequent cleanup during heavy traffic
                cleanup_counter += 1
                cleanup_interval = 100 if self.stats["mode"] == "heavy_traffic" else 500
                if cleanup_counter >= cleanup_interval:
                    self._cleanup_old_flows()
                    cleanup_counter = 0
                    
            except queue.Empty:
                # Good time to do cleanup when idle
                self._cleanup_old_flows()
                continue
            except Exception as e:
                logger.error(f"Processing error: {e}")
    
    def _cleanup_old_flows(self, force: bool = False):
        """Clean up stale flows to free memory."""
        current_time = time.time()
        keys_to_remove = []
        
        for key, packets in list(self.feature_extractor.active_flows.items()):
            if not packets:
                keys_to_remove.append(key)
                continue
            
            last_time = packets[-1].time
            
            # Shorter timeouts during heavy traffic
            if force:
                timeout = 5  # Very aggressive: 5 seconds
            elif self.stats["mode"] == "heavy_traffic":
                timeout = 10  # Aggressive: 10 seconds
            else:
                timeout = 30  # Normal: 30 seconds
                
            if (current_time - last_time) > timeout:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            if key in self.feature_extractor.active_flows:
                del self.feature_extractor.active_flows[key]
            if key in self.feature_extractor.flow_start_times:
                del self.feature_extractor.flow_start_times[key]
            if key in self.feature_extractor.flow_last_packet_time:
                del self.feature_extractor.flow_last_packet_time[key]

    def get_flows(self, timeout_seconds=2.0) -> List[Dict]:
        """
        Get completed flows or flows active longer than timeout.
        Returns list of feature dictionaries ready for model.
        """
        current_time = time.time()
        ready_flows = []
        
        # Limit how many flows we return per call to prevent API blocking
        MAX_FLOWS_RETURN = 20
        
        # Convert timestamp keys to list to avoid runtime error during iteration
        keys = list(self.feature_extractor.active_flows.keys())
        
        for key in keys:
            if len(ready_flows) >= MAX_FLOWS_RETURN:
                break  # Stop early if we have enough
                
            packets = self.feature_extractor.active_flows.get(key)
            if not packets:
                continue
                
            start_time = packets[0].time
            last_time = packets[-1].time
            duration = last_time - start_time
            
            # Shorter timeout during heavy traffic
            effective_timeout = 1.0 if self.stats["mode"] == "heavy_traffic" else timeout_seconds
            
            # If flow is long enough or hasn't updated recently, process it
            if duration > effective_timeout or (current_time - last_time) > 1.0:
                features = self.feature_extractor.extract_features(key, packets)
                if features:
                    ready_flows.append(features)
                
                # Reset/Remove processed flows to prevent infinite growth
                if key in self.feature_extractor.active_flows:
                    del self.feature_extractor.active_flows[key]
                if key in self.feature_extractor.flow_start_times:
                    del self.feature_extractor.flow_start_times[key]
        
        return ready_flows
