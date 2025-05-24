# easyremote/__init__.py
from .core import Server
from .core import ComputeNode
from .core import Client
from .decorators import remote
from .core import get_performance_monitor
from .core.load_balancing import (
    LoadBalancer,
    LoadBalancingStrategy,
    RequestContext,
    NodeStats
)

__version__ = "0.1.3.2"
__author__ = "Silan Hu"
__email__ = "silan.hu@u.nus.edu"

__all__ = [
    "Server",
    "ComputeNode", 
    "Client",
    "remote",
    "get_performance_monitor",
    "LoadBalancer",
    "LoadBalancingStrategy", 
    "RequestContext",
    "NodeStats"
]
