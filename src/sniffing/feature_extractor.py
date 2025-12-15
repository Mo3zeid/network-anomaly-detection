"""
Feature extractor for converting Scapy packets into ML model features.
Matches the exact 15 features used in training.
"""
import numpy as np
from scapy.all import IP, TCP, UDP
from collections import defaultdict
from typing import Dict, List, Tuple
import time

class FeatureExtractor:
    def __init__(self, max_packets_per_flow: int = 1000):
        # Tracking active flows: flow_key -> list of packets
        self.active_flows = defaultdict(list)
        # Flow start times: flow_key -> timestamp
        self.flow_start_times = {}
        # Last packet time for IAT calculation
        self.flow_last_packet_time = {}
        # Maximum packets to keep per flow (prevents memory issues)
        self.max_packets_per_flow = max_packets_per_flow
        
    def get_flow_key(self, pkt) -> Tuple:
        """Create a unique key for the flow (5-tuple)."""
        if IP in pkt:
            src_ip = pkt[IP].src
            dst_ip = pkt[IP].dst
            proto = pkt[IP].proto
            
            src_port = 0
            dst_port = 0
            
            if TCP in pkt:
                src_port = pkt[TCP].sport
                dst_port = pkt[TCP].dport
            elif UDP in pkt:
                src_port = pkt[UDP].sport
                dst_port = pkt[UDP].dport
                
            # Sort IPs/Ports to make it bidirectional
            if src_ip < dst_ip:
                return (src_ip, dst_ip, src_port, dst_port, proto)
            else:
                return (dst_ip, src_ip, dst_port, src_port, proto)
        return None

    def process_packet(self, pkt):
        """Add packet to active flows."""
        if IP not in pkt:
            return

        key = self.get_flow_key(pkt)
        if not key:
            return

        current_time = pkt.time
        
        # Initialize flow if new
        if key not in self.flow_start_times:
            self.flow_start_times[key] = current_time
            self.flow_last_packet_time[key] = current_time
        
        # Add packet to flow
        self.active_flows[key].append(pkt)
        
        # Enforce packet limit per flow - keep only most recent packets
        # This prevents a single flow from consuming all memory during heavy traffic
        if len(self.active_flows[key]) > self.max_packets_per_flow:
            # Keep the first packet (for flow start time) and the most recent ones
            first_pkt = self.active_flows[key][0]
            recent_pkts = self.active_flows[key][-(self.max_packets_per_flow - 1):]
            self.active_flows[key] = [first_pkt] + recent_pkts

    def extract_features(self, flow_key: Tuple, packets: List) -> Dict:
        """Calculate the 15 specific features for a flow."""
        if not packets:
            return None

        start_time = self.flow_start_times.get(flow_key, packets[0].time)
        end_time = packets[-1].time
        duration = end_time - start_time
        if duration == 0:
            duration = 1e-6 # Avoid division by zero

        # Initialize counters
        fwd_pkts = 0
        bwd_pkts = 0
        fwd_len_sum = 0
        bwd_len_sum = 0
        fwd_lens = []
        bwd_lens = []
        
        # IAT (Inter-Arrival Time) lists
        flow_iats = []
        fwd_iats = []
        bwd_iats = []
        
        # Flags
        fwd_psh_flags = 0
        syn_flags = 0
        ack_flags = 0
        
        # Packet sizes for variance
        pkt_sizes = []
        
        last_flow_time = start_time
        last_fwd_time = 0
        last_bwd_time = 0
        
        # Determine direction (first packet determines forward direction)
        src_ip_fwd = packets[0][IP].src
        
        for i, pkt in enumerate(packets):
            current_time = pkt.time
            length = len(pkt)
            pkt_sizes.append(length)
            
            # Update Flow IAT
            if i > 0:
                flow_iats.append(current_time - last_flow_time)
            last_flow_time = current_time

            # Check direction
            if pkt[IP].src == src_ip_fwd:
                # Forward
                fwd_pkts += 1
                fwd_len_sum += length
                fwd_lens.append(length)
                
                if last_fwd_time > 0:
                    fwd_iats.append(current_time - last_fwd_time)
                last_fwd_time = current_time
                
                if TCP in pkt:
                    flags = pkt[TCP].flags
                    if 'P' in flags: fwd_psh_flags += 1
                    if 'S' in flags: syn_flags += 1
                    if 'A' in flags: ack_flags += 1
            else:
                # Backward
                bwd_pkts += 1
                bwd_len_sum += length
                bwd_lens.append(length)
                
                if last_bwd_time > 0:
                    bwd_iats.append(current_time - last_bwd_time)
                last_bwd_time = current_time
                
                if TCP in pkt:
                    flags = pkt[TCP].flags
                    if 'S' in flags: syn_flags += 1
                    if 'A' in flags: ack_flags += 1

        # Calculate Means
        fwd_len_mean = np.mean(fwd_lens) if fwd_lens else 0
        bwd_len_mean = np.mean(bwd_lens) if bwd_lens else 0
        
        flow_iat_mean = np.mean(flow_iats) * 1000000 if flow_iats else 0 # Convert to microseconds
        fwd_iat_mean = np.mean(fwd_iats) * 1000000 if fwd_iats else 0
        bwd_iat_mean = np.mean(bwd_iats) * 1000000 if bwd_iats else 0
        
        duration_micros = duration * 1000000
        
        total_bytes = fwd_len_sum + bwd_len_sum
        
        return {
            'Flow Duration': duration_micros,
            'Total Fwd Packets': fwd_pkts,
            'Total Backward Packets': bwd_pkts,
            'Flow Bytes/s': total_bytes / duration,
            'Flow Packets/s': len(packets) / duration,
            'Fwd Packet Length Mean': fwd_len_mean,
            'Bwd Packet Length Mean': bwd_len_mean,
            'Flow IAT Mean': flow_iat_mean,
            'Fwd IAT Mean': fwd_iat_mean,
            'Bwd IAT Mean': bwd_iat_mean,
            'Fwd PSH Flags': fwd_psh_flags,
            'SYN Flag Count': syn_flags,
            'ACK Flag Count': ack_flags,
            'Packet Length Variance': np.var(pkt_sizes) if pkt_sizes else 0,
            'Average Packet Size': np.mean(pkt_sizes) if pkt_sizes else 0,
            
            # Metadata (not for model)
            'src_ip': packets[0][IP].src,
            'dst_ip': packets[0][IP].dst,
            'src_port': packets[0][TCP].sport if TCP in packets[0] else (packets[0][UDP].sport if UDP in packets[0] else 0),
            'dst_port': packets[0][TCP].dport if TCP in packets[0] else (packets[0][UDP].dport if UDP in packets[0] else 0),
            'protocol': 'TCP' if TCP in packets[0] else ('UDP' if UDP in packets[0] else 'Other')
        }
