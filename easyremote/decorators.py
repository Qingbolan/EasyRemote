# easyremote/decorators.py
import functools
from typing import Optional, Callable, Any, Union, TypeVar, cast
from .core.utils.exceptions import RemoteExecutionError
from .core.nodes.server import Server
from .core.load_balancing.strategies import RequestContext, LoadBalancingConfig
import asyncio


T = TypeVar('T', bound=Callable)

class RemoteFunction:
    def __init__(
        self,
        func: Callable,
        node_id: Optional[str] = None,
        function_name: Optional[str] = None,
        timeout: Optional[float] = None,
        is_stream: bool = False,
        is_async: bool = False,
        load_balancing: Union[bool, str, dict] = False
    ):
        self.func = func
        self.node_id = node_id
        self.function_name = function_name or func.__name__
        self.timeout = timeout
        self.is_stream = is_stream
        self.is_async = is_async
        self.load_balancing = load_balancing
        functools.update_wrapper(self, func)

    def __call__(self, *args, **kwargs) -> Any:
        server = Server.current()
        try:
            if self.load_balancing and self.load_balancing is not False:
                # Use load balancing
                result = server.execute_function_with_load_balancing(
                    self.function_name,
                    self.load_balancing,
                    *args,
                    **kwargs
                )
            else:
                # Direct node execution
                result = server.execute_function(
                    self.node_id,
                    self.function_name,
                    *args,
                    **kwargs
                )
            if self.is_stream:
                return result
            return result
        except Exception as e:
            raise RemoteExecutionError(str(e))

    async def __call_async__(self, *args, **kwargs) -> Any:
        server = Server.current()
        try:
            loop = asyncio.get_running_loop()
            if self.load_balancing and self.load_balancing is not False:
                # Use load balancing
                result = await loop.run_in_executor(
                    None,
                    lambda: server.execute_function_with_load_balancing(
                        self.function_name,
                        self.load_balancing,
                        *args,
                        **kwargs
                    )
                )
            else:
                # Direct node execution
                result = await loop.run_in_executor(
                    None,
                    lambda: server.execute_function(
                        self.node_id,
                        self.function_name,
                        *args,
                        **kwargs
                    )
                )
            if self.is_stream:
                return result
            return result
        except Exception as e:
            raise RemoteExecutionError(str(e))

def register(
    *,
    node_id: Optional[str] = None,
    function_name: Optional[str] = None,
    timeout: Optional[float] = None,
    stream: bool = False,
    async_func: bool = False,
    load_balancing: Union[bool, str, dict] = False
) -> Callable[[T], T]:
    def decorator(f: T) -> T:
        if isinstance(f, RemoteFunction):
            return cast(T, f)

        wrapped = RemoteFunction(
            f,
            node_id=node_id,
            function_name=function_name,
            timeout=timeout,
            is_stream=stream,
            is_async=async_func,
            load_balancing=load_balancing
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
    function_name: Optional[str] = None,
    timeout: Optional[float] = None,
    stream: bool = False,
    async_func: bool = False,
    load_balancing: Union[bool, str, dict] = False
) -> Union[Callable[[T], T], T]:
    if func is not None and callable(func):
        return register()(func)
    return register(
        node_id=node_id, 
        function_name=function_name,
        timeout=timeout, 
        stream=stream, 
        async_func=async_func,
        load_balancing=load_balancing
    )