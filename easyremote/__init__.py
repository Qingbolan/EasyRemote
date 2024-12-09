# easyremote/__init__.py
from .server import Server
from .compute_node import ComputeNode
from .decorators import remote
from .exceptions import (
    EasyRemoteError,
    NodeNotFoundError,
    FunctionNotFoundError,
    ConnectionError,
    SerializationError,
    RemoteExecutionError,
)

__version__ = "0.1.0.2.3"

__all__ = [
    'Server',
    'ComputeNode',
    'remote',
    'EasyRemoteError',
    'NodeNotFoundError',
    'FunctionNotFoundError',
    'ConnectionError',
    'SerializationError',
    'RemoteExecutionError',
]