# Simulation module
from .traffic_generator import (
    generate_traffic_batch,
    generate_normal_traffic,
    generate_ddos_traffic,
    generate_port_scan_traffic,
    generate_brute_force_traffic,
    generate_web_attack_traffic,
    get_attack_info,
    ATTACK_DESCRIPTIONS
)
