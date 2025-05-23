# easyremote/types.py
from dataclasses import dataclass
from typing import Dict, Optional, Callable
from datetime import datetime

@dataclass
class FunctionInfo:
    """注册函数的信息"""
    name: str
    callable: Callable
    is_async: bool
    is_generator: bool
    node_id: Optional[str] = None

@dataclass
class NodeInfo:
    """计算节点信息"""
    node_id: str
    functions: Dict[str, FunctionInfo]
    last_heartbeat: datetime
    status: str = "connected"  # "connected", "disconnected", "reconnecting"



