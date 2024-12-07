# easyremote/utils.py
import pickle
import logging
from typing import Any, Tuple, Dict
import inspect
import asyncio
from .exceptions import SerializationError

logger = logging.getLogger(__name__)

def serialize_args(*args, **kwargs) -> Tuple[bytes, bytes]:
    """序列化参数"""
    try:
        args_bytes = pickle.dumps(args)
        kwargs_bytes = pickle.dumps(kwargs)
        return args_bytes, kwargs_bytes
    except Exception as e:
        raise SerializationError(f"Failed to serialize arguments: {e}")

def deserialize_args(args_bytes: bytes, kwargs_bytes: bytes) -> Tuple[tuple, dict]:
    """反序列化参数"""
    try:
        args = pickle.loads(args_bytes) if args_bytes else ()
        kwargs = pickle.loads(kwargs_bytes) if kwargs_bytes else {}
        return args, kwargs
    except Exception as e:
        raise SerializationError(f"Failed to deserialize arguments: {e}")

def serialize_result(result: Any) -> bytes:
    """序列化结果"""
    try:
        return pickle.dumps(result)
    except Exception as e:
        raise SerializationError(f"Failed to serialize result: {e}")

def deserialize_result(result_bytes: bytes) -> Any:
    """反序列化结果"""
    try:
        return pickle.loads(result_bytes)
    except Exception as e:
        raise SerializationError(f"Failed to deserialize result: {e}")

def analyze_function(func) -> Dict[str, bool]:
    """分析函数类型"""
    return {
        'is_async': asyncio.iscoroutinefunction(func),
        'is_generator': inspect.isgeneratorfunction(func),
        'is_class': inspect.isclass(func),
    }

def setup_logger(name: str, level: str = 'INFO') -> logging.Logger:
    """配置日志"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger