# Core module for EasyRemote 
from .tools import BasicMonitor, quick_health_check, quick_metrics
from .nodes import Server
from .nodes import ComputeNode
from .nodes import Client
from .config import EasyRemoteConfig, get_config, create_config

# Backward compatibility - create a simple performance monitor function
def get_performance_monitor():
    """Create a basic performance monitor for backward compatibility."""
    return BasicMonitor()

__all__ = [
    "get_performance_monitor",  # Keep for backward compatibility
    "BasicMonitor",
    "quick_health_check", 
    "quick_metrics",
    "Server",
    "ComputeNode",
    "Client",
    "EasyRemoteConfig",
    "get_config",
    "create_config"
]
