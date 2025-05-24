"""
Data serialization module providing advanced serialization capabilities
with multiple backends, compression, and safety features.
"""

from .config import (
    SerializationProtocol,
    CompressionAlgorithm,
    SerializationConfig
)

from .backends import (
    SerializationBackend,
    PickleBackend,
    JSONBackend
)

from .analysis import (
    FunctionAnalysis,
    analyze_function
)

from .serialize import Serializer

# Add missing exports that are used by server and compute_node
from .data_types import (
    NodeInfo,
    FunctionInfo
)

__all__ = [
    # Configuration
    'SerializationProtocol',
    'CompressionAlgorithm', 
    'SerializationConfig',
    
    # Backends
    'SerializationBackend',
    'PickleBackend',
    'JSONBackend',
    
    # Analysis
    'FunctionAnalysis',
    'analyze_function',
    
    # Main serializer
    'Serializer',
    
    # Data types
    'NodeInfo',
    'FunctionInfo'
]