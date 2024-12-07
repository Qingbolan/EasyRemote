# easyremote/decorators.py
import functools
from typing import Optional, Callable, Any, Union, TypeVar, cast
from .exceptions import RemoteExecutionError
from .server import Server
from .utils import setup_logger
import asyncio

logger = setup_logger(__name__)

T = TypeVar('T', bound=Callable)

class RemoteFunction:
    def __init__(
        self,
        func: Callable,
        node_id: Optional[str] = None,
        timeout: Optional[float] = None,
        is_stream: bool = False,
        is_async: bool = False
    ):
        self.func = func
        self.node_id = node_id
        self.timeout = timeout
        self.is_stream = is_stream
        self.is_async = is_async
        functools.update_wrapper(self, func)

    def __call__(self, *args, **kwargs) -> Any:
        server = Server.current()
        try:
            # 使用位置参数传递 node_id 和 function_name
            result = server.execute_function(
                self.node_id,
                self.func.__name__,
                *args,
                **kwargs
            )
            if self.is_stream:
                return result
            return result
        except Exception as e:
            logger.error(f"Error calling remote function {self.func.__name__}: {e}", exc_info=True)
            raise RemoteExecutionError(str(e))

    async def __call_async__(self, *args, **kwargs) -> Any:
        server = Server.current()
        try:
            loop = asyncio.get_running_loop()
            # 使用位置参数传递 node_id 和 function_name
            result = await loop.run_in_executor(
                None,
                lambda: server.execute_function(
                    self.node_id,
                    self.func.__name__,
                    *args,
                    **kwargs
                )
            )
            if self.is_stream:
                return result
            return result
        except Exception as e:
            logger.error(f"Error calling remote async function {self.func.__name__}: {e}", exc_info=True)
            raise RemoteExecutionError(str(e))

def register(
    *,
    node_id: Optional[str] = None,
    timeout: Optional[float] = None,
    stream: bool = False,
    async_func: bool = False
) -> Callable[[T], T]:
    def decorator(f: T) -> T:
        if isinstance(f, RemoteFunction):
            return cast(T, f)

        wrapped = RemoteFunction(
            f,
            node_id=node_id,
            timeout=timeout,
            is_stream=stream,
            is_async=async_func
        )

        if async_func:
            return cast(T, wrapped.__call_async__)
        else:
            return cast(T, wrapped.__call__)

    return decorator

def remote(
    func: Optional[Callable] = None,
    *,
    node_id: Optional[str] = None,
    timeout: Optional[float] = None,
    stream: bool = False,
    async_func: bool = False
) -> Union[Callable[[T], T], T]:
    if func is not None and callable(func):
        return register()(func)
    return register(node_id=node_id, timeout=timeout, stream=stream, async_func=async_func)