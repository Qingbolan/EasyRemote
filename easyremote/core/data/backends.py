import pickle
import json
import gzip
import zlib
from typing import Any, Protocol, runtime_checkable
from .config import CompressionAlgorithm
from ..utils.exceptions import SerializationError


@runtime_checkable
class SerializationBackend(Protocol):
    """Protocol defining the interface for serialization backends"""
    
    def serialize(self, obj: Any) -> bytes:
        """Serialize an object to bytes"""
        ...
    
    def deserialize(self, data: bytes) -> Any:
        """Deserialize bytes back to object"""
        ...


class PickleBackend:
    """Pickle-based serialization backend"""
    
    def __init__(self, protocol: int = 4, safe_mode: bool = True, compress: bool = False, 
                 compression_algorithm: CompressionAlgorithm = CompressionAlgorithm.ZLIB,
                 compression_level: int = 6):
        self.protocol = protocol
        self.safe_mode = safe_mode
        self.compress = compress
        self.compression_algorithm = compression_algorithm
        self.compression_level = compression_level
    
    def _compress_data(self, data: bytes) -> bytes:
        """Compress data using the configured algorithm"""
        if self.compression_algorithm == CompressionAlgorithm.ZLIB:
            return zlib.compress(data, level=self.compression_level)
        elif self.compression_algorithm == CompressionAlgorithm.GZIP:
            return gzip.compress(data, compresslevel=self.compression_level)
        else:
            return data  # No compression
    
    def _decompress_data(self, data: bytes) -> bytes:
        """Decompress data using the configured algorithm"""
        if self.compression_algorithm == CompressionAlgorithm.ZLIB:
            return zlib.decompress(data)
        elif self.compression_algorithm == CompressionAlgorithm.GZIP:
            return gzip.decompress(data)
        else:
            return data  # No compression
    
    def serialize(self, obj: Any) -> bytes:
        """Serialize object using pickle"""
        try:
            data = pickle.dumps(obj, protocol=self.protocol)
            if self.compress:
                data = self._compress_data(data)
            return data
        except (pickle.PicklingError, TypeError, AttributeError) as e:
            raise SerializationError(f"Pickle serialization failed: {e}") from e
    
    def deserialize(self, data: bytes) -> Any:
        """Deserialize bytes using pickle"""
        if not data:
            return None
        try:
            if self.compress:
                data = self._decompress_data(data)
            
            if self.safe_mode:
                # Perform safety checks before deserialization
                self._validate_pickle_data(data)
            return pickle.loads(data)
        except (pickle.UnpicklingError, EOFError, ValueError, zlib.error, OSError) as e:
            raise SerializationError(f"Pickle deserialization failed: {e}") from e
    
    def _validate_pickle_data(self, data: bytes) -> None:
        """
        Validate pickle data for potential security risks.
        
        This is a basic implementation - in production, you might want
        to use more sophisticated security checks.
        """
        # Check for minimum data size (empty pickle is 4+ bytes)
        if len(data) < 4:
            raise SerializationError("Invalid pickle data: too short")
        
        # Check for pickle protocol magic bytes
        if data[0] not in [0x80, 0x03, 0x02, 0x01, 0x00]:  # Valid pickle protocols
            raise SerializationError("Invalid pickle data: bad magic bytes")
        
        # Check for maximum reasonable size (1GB default)
        max_size = 1024 * 1024 * 1024  # 1GB
        if len(data) > max_size:
            raise SerializationError(f"Pickle data too large: {len(data)} bytes")
        
        # You could add more checks here:
        # - Scan for dangerous opcodes
        # - Check for suspicious module imports
        # - Validate against whitelist of allowed types


class JSONBackend:
    """JSON-based serialization backend with extended type support"""
    
    def __init__(self):
        self.supported_types = {
            'tuple': lambda x: {'__type__': 'tuple', 'data': list(x)},
            'set': lambda x: {'__type__': 'set', 'data': list(x)},
            'complex': lambda x: {'__type__': 'complex', 'real': x.real, 'imag': x.imag},
            'bytes': lambda x: {'__type__': 'bytes', 'data': x.decode('latin-1')},
        }
    
    def _custom_encoder(self, obj: Any) -> Any:
        """Custom encoder for non-JSON-native types"""
        if isinstance(obj, tuple):
            return self.supported_types['tuple'](obj)
        elif isinstance(obj, set):
            return self.supported_types['set'](obj)
        elif isinstance(obj, complex):
            return self.supported_types['complex'](obj)
        elif isinstance(obj, bytes):
            return self.supported_types['bytes'](obj)
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
    
    def _custom_decoder(self, obj: Any) -> Any:
        """Custom decoder for non-JSON-native types"""
        if isinstance(obj, dict) and '__type__' in obj:
            type_name = obj['__type__']
            if type_name == 'tuple':
                return tuple(obj['data'])
            elif type_name == 'set':
                return set(obj['data'])
            elif type_name == 'complex':
                return complex(obj['real'], obj['imag'])
            elif type_name == 'bytes':
                return obj['data'].encode('latin-1')
        return obj
    
    def serialize(self, obj: Any) -> bytes:
        """Serialize object using JSON with extended type support"""
        try:
            json_str = json.dumps(obj, ensure_ascii=False, default=self._custom_encoder)
            return json_str.encode('utf-8')
        except (TypeError, ValueError) as e:
            raise SerializationError(f"JSON serialization failed: {e}") from e
    
    def deserialize(self, data: bytes) -> Any:
        """Deserialize bytes using JSON with extended type support"""
        if not data:
            return None
        try:
            decoded_str = data.decode('utf-8')
            obj = json.loads(decoded_str)
            return self._decode_recursive(obj)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise SerializationError(f"JSON deserialization failed: {e}") from e
    
    def _decode_recursive(self, obj: Any) -> Any:
        """Recursively decode custom types"""
        if isinstance(obj, dict):
            # First check if it's a custom type marker
            decoded = self._custom_decoder(obj)
            if decoded is not obj:  # It was a custom type
                return decoded
            # Otherwise decode all values in the dict
            return {k: self._decode_recursive(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._decode_recursive(item) for item in obj]
        return obj 