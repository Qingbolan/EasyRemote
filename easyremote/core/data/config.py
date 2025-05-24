from enum import Enum
from dataclasses import dataclass
from typing import Optional


class SerializationProtocol(Enum):
    """Supported serialization protocols"""
    PICKLE_V4 = "pickle_v4"
    PICKLE_V5 = "pickle_v5"
    JSON = "json"


class CompressionAlgorithm(Enum):
    """Supported compression algorithms"""
    NONE = "none"
    ZLIB = "zlib"
    GZIP = "gzip"


@dataclass
class SerializationConfig:
    """Configuration for serialization operations"""
    protocol: SerializationProtocol = SerializationProtocol.PICKLE_V4
    compress: bool = False
    compression_algorithm: CompressionAlgorithm = CompressionAlgorithm.ZLIB
    compression_level: int = 6  # 1-9, higher = better compression but slower
    safe_mode: bool = True
    max_size: Optional[int] = None  # Maximum serialized size in bytes
    
    @classmethod
    def fast_config(cls) -> 'SerializationConfig':
        """Create a configuration optimized for speed"""
        return cls(
            protocol=SerializationProtocol.PICKLE_V5,
            compress=False,
            safe_mode=False,
            compression_level=1
        )
    
    @classmethod
    def safe_config(cls) -> 'SerializationConfig':
        """Create a configuration optimized for safety"""
        return cls(
            protocol=SerializationProtocol.PICKLE_V4,
            compress=True,
            safe_mode=True,
            max_size=100 * 1024 * 1024  # 100MB limit
        )
    
    @classmethod
    def compact_config(cls) -> 'SerializationConfig':
        """Create a configuration optimized for size"""
        return cls(
            protocol=SerializationProtocol.PICKLE_V5,
            compress=True,
            compression_level=9,
            safe_mode=True
        ) 