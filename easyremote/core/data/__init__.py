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

# Add all exports from data_types module including new refactored classes
from .data_types import (
    # Core data types
    NodeInfo,
    FunctionInfo,
    
    # New enum types for better type safety
    NodeStatus,
    FunctionType,
    
    # Advanced data structures
    ResourceRequirements,
    NodeHealthMetrics,
    
    # Type aliases for convenience
    NodeRegistry,
    FunctionRegistry
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
    
    # Core data types
    'NodeInfo',
    'FunctionInfo',
    
    # Enum types
    'NodeStatus',
    'FunctionType',
    
    # Advanced data structures
    'ResourceRequirements',
    'NodeHealthMetrics',
    
    # Type aliases
    'NodeRegistry',
    'FunctionRegistry'
]