# easyremote/decorators.py
import asyncio
import functools
import inspect
from typing import Optional, Callable, Any, Union, TypeVar, cast
from contextlib import contextmanager
from .exceptions import RemoteExecutionError
from .server import Server
from .utils import setup_logger, analyze_function

logger = setup_logger(__name__)

# 类型变量用于保持函数签名
T = TypeVar('T', bound=Callable)

class RemoteFunction:
    """远程函数包装器，保持类型提示和文档"""
    
    def __init__(
        self,
        func: Callable,
        node_id: Optional[str] = None,
        timeout: Optional[float] = None
    ):
        self.func = func
        self.node_id = node_id
        self.timeout = timeout
        self.is_async = asyncio.iscoroutinefunction(func)
        self.is_generator = inspect.isgeneratorfunction(func)
        
        # 复制原函数的元数据
        functools.update_wrapper(self, func)
        
    async def _async_call(self, *args, **kwargs) -> Any:
        """异步调用实现"""
        server = Server.current()
        try:
            return await server.async_call(
                function_name=self.func.__name__,
                args=args,
                kwargs=kwargs,
                node_id=self.node_id,
                timeout=self.timeout
            )
        except Exception as e:
            logger.error(f"Error calling {self.func.__name__}: {e}")
            raise RemoteExecutionError(str(e))
            
    def _sync_call(self, *args, **kwargs) -> Any:
        """同步调用实现"""
        server = Server.current()
        try:
            return server.sync_call(
                function_name=self.func.__name__,
                args=args,
                kwargs=kwargs,
                node_id=self.node_id,
                timeout=self.timeout
            )
        except Exception as e:
            logger.error(f"Error calling {self.func.__name__}: {e}")
            raise RemoteExecutionError(str(e))
            
    async def _stream_call(self, *args, **kwargs):
        """流式调用实现"""
        server = Server.current()
        try:
            async for item in server.stream_call(
                function_name=self.func.__name__,
                args=args,
                kwargs=kwargs,
                node_id=self.node_id,
                timeout=self.timeout
            ):
                yield item
        except Exception as e:
            logger.error(f"Error in stream {self.func.__name__}: {e}")
            raise RemoteExecutionError(str(e))
            
    def __call__(self, *args, **kwargs):
        """根据函数类型选择调用方式"""
        if self.is_generator:
            return self._stream_call(*args, **kwargs)
        elif self.is_async:
            return self._async_call(*args, **kwargs)
        else:
            return self._sync_call(*args, **kwargs)

def remote(
    func: Optional[Callable] = None,
    *,
    node_id: Optional[str] = None,
    timeout: Optional[float] = None
) -> Union[Callable[[T], T], T]:
    """将函数标记为远程调用
    
    可以直接使用：
        @remote
        def func(): pass
        
    或者带参数：
        @remote(node_id="node-1", timeout=30)
        def func(): pass
        
    支持：
    1. 同步函数
    2. 异步函数
    3. 生成器函数
    4. 类方法（实验性）
    
    Args:
        func: 要包装的函数
        node_id: 指定执行节点的ID
        timeout: 执行超时时间（秒）
        
    Returns:
        包装后的函数，保持原始签名
    """
    def decorator(f: T) -> T:
        # 检查是否已经是远程函数
        if isinstance(f, RemoteFunction):
            return cast(T, f)
            
        # 处理类方法
        if inspect.ismethod(f):
            method = f.__func__
            
            @functools.wraps(method)
            def wrapped(self, *args, **kwargs):
                bound_method = method.__get__(self, type(self))
                remote_func = RemoteFunction(bound_method, node_id, timeout)
                return remote_func(*args, **kwargs)
                
            return cast(T, wrapped)
            
        # 正常函数
        return cast(T, RemoteFunction(f, node_id, timeout))
        
    if func is None:
        return decorator
    return decorator(func)

# 便捷装饰器
def remote_sync(
    func: Optional[Callable] = None,
    **kwargs
) -> Union[Callable[[T], T], T]:
    """明确标记为同步远程调用"""
    def decorator(f: T) -> T:
        # 确保是同步函数
        if asyncio.iscoroutinefunction(f):
            raise TypeError("Cannot use @remote_sync with async function")
        return cast(T, remote(f, **kwargs))
    
    if func is None:
        return decorator
    return decorator(func)

def remote_async(
    func: Optional[Callable] = None,
    **kwargs
) -> Union[Callable[[T], T], T]:
    """明确标记为异步远程调用"""
    def decorator(f: T) -> T:
        # 确保是异步函数
        if not asyncio.iscoroutinefunction(f):
            raise TypeError("Must use @remote_async with async function")
        return cast(T, remote(f, **kwargs))
        
    if func is None:
        return decorator
    return decorator(func)

def remote_stream(
    func: Optional[Callable] = None,
    **kwargs
) -> Union[Callable[[T], T], T]:
    """明确标记为流式远程调用"""
    def decorator(f: T) -> T:
        # 确保是生成器函数
        if not inspect.isgeneratorfunction(f):
            raise TypeError("Must use @remote_stream with generator function")
        return cast(T, remote(f, **kwargs))
        
    if func is None:
        return decorator
    return decorator(func)