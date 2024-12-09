import pickle
import logging
from typing import Any, Tuple, Dict
import inspect
import asyncio
from rich.logging import RichHandler
from rich.console import Console
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

def serialize_result(result):
    """序列化结果，确保返回字节流"""
    try:
        if result is None:
            logger.debug("Serializing None result")
            return b''
        logger.debug(f"Serializing result of type {type(result)}")
        return pickle.dumps(result, protocol=4)
    except Exception as e:
        logger.error(f"Serialization failed for type {type(result)}")
        raise SerializationError(
            operation="serialize",
            message=f"Failed to serialize result of type {type(result)}",
            cause=e
        )

def deserialize_result(result_bytes: bytes):
    """反序列化结果"""
    if not result_bytes:
        return None
    try:
        return pickle.loads(result_bytes)
    except Exception as e:
        raise SerializationError(
            operation="deserialize",
            message=f"Failed to deserialize result: {str(e)}",
            cause=e
        )

def analyze_function(func) -> Dict[str, bool]:
    """分析函数类型"""
    return {
        'is_async': asyncio.iscoroutinefunction(func),
        'is_generator': inspect.isgeneratorfunction(func),
        'is_class': inspect.isclass(func),
    }

def setup_logger(name: str, level: str = 'INFO') -> logging.Logger:
    """配置日志"""
    console = Console()
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        # 创建 RichHandler
        handler = RichHandler(
            console=console,
            rich_tracebacks=True,
            markup=True,
            show_time=True,
            show_path=True
        )
        
        # 设置格式
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        
        # 设置日志级别
        handler.setLevel(level)
        
        # 添加处理器到logger
        logger.addHandler(handler)
    
    return logger


if __name__ == "__main__":
    # 使用 logger
    logger.info("这是一个 [bold green]信息[/bold green]日志")
    logger.debug("这是一个 [cyan]调试[/cyan]日志")
    logger.warning("这是一个 [yellow]警告[/yellow]日志")
    logger.error("这是一个 [red]错误[/red]日志")
    logger.critical("这是一个 [bold red]严重错误[/bold red]日志")