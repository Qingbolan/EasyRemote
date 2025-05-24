# easyremote/__init__.py
from .core import Server
from .core import ComputeNode
from .decorators import remote
from .core import get_performance_monitor

__version__ = "0.1.3.2"
__author__ = "Silan Hu"
__email__ = "silan.hu@u.nus.edu"

__all__ = [
    "Server",
    "ComputeNode",
    "remote",
    "get_performance_monitor"
]
