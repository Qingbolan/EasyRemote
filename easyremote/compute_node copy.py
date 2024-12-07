# easyremote/compute_node.py
import threading
import time
import asyncio
import grpc
from typing import Optional, Callable, Dict, Any
from concurrent import futures
import datetime

from .types import FunctionInfo
from .exceptions import ConnectionError, RemoteExecutionError
from .utils import (
    deserialize_args,
    serialize_result,
    analyze_function,
    setup_logger
)
from .protos import service_pb2, service_pb2_grpc

logger = setup_logger(__name__)

class ComputeNode(service_pb2_grpc.RemoteServiceServicer):
    """计算节点，负责注册和执行远程函数"""
        
    def __init__(
        self,
        vps_address: str,
        node_id: Optional[str] = None,
        reconnect_interval: int = 5,
        heartbeat_interval: int = 1,
        max_retry_attempts: int = 3
    ):
        logger.debug(f"Initializing ComputeNode with VPS address: {vps_address}")
        self.vps_address = vps_address
        self.node_id = node_id or f"node-{id(self)}"
        self.reconnect_interval = reconnect_interval
        self.heartbeat_interval = heartbeat_interval
        self.max_retry_attempts = max_retry_attempts
        
        self._functions: Dict[str, FunctionInfo] = {}
        self._channel: Optional[grpc.Channel] = None
        self._stub: Optional[service_pb2_grpc.RemoteServiceStub] = None
        self._running = False
        self._connected = threading.Event()
        self._executor = futures.ThreadPoolExecutor(max_workers=10)
        self._heartbeat_thread = None
        self._last_heartbeat_time = None
        logger.info(f"ComputeNode {self.node_id} initialized")
        
    def register(
        self,
        func: Optional[Callable] = None,
        *,
        name: Optional[str] = None
    ) -> Callable:
        """注册一个函数供远程调用
        
        可以作为装饰器使用：
        @node.register
        def func(): pass
        
        或者指定名称：
        @node.register(name="custom_name")
        def func(): pass
        """
        def decorator(f: Callable) -> Callable:
            func_name = name or f.__name__
            func_info = analyze_function(f)
            
            self._functions[func_name] = FunctionInfo(
                name=func_name,
                callable=f,
                is_async=func_info['is_async'],
                is_generator=func_info['is_generator'],
                node_id=self.node_id
            )
            
            logger.info(f"Registered function: {func_name}")
            return f
            
        return decorator(func) if func else decorator
        
    def serve(self, blocking: bool = True) -> Optional[threading.Thread]:
        """启动计算节点服务
        
        Args:
            blocking: 是否阻塞当前线程
            
        Returns:
            如果non-blocking模式，返回服务线程
        """
        self._running = True
        
        def _serve():
            while self._running:
                try:
                    self._connect()
                    self._register_with_vps()
                    self._start_heartbeat()
                    self._connected.set()
                    
                    logger.info(f"Node {self.node_id} connected to VPS")
                    
                    # 保持连接直到出错或停止
                    while self._running and self._connected.is_set():
                        time.sleep(1)
                        
                except Exception as e:
                    logger.error(f"Connection error: {e}")
                    self._connected.clear()
                    self._cleanup()
                    
                    if self._running:
                        logger.info(f"Reconnecting in {self.reconnect_interval} seconds...")
                        time.sleep(self.reconnect_interval)
                        
            self._cleanup()
            
        if blocking:
            _serve()
            return None
        else:
            thread = threading.Thread(target=_serve, daemon=True)
            thread.start()
            return thread
            
    def stop(self):
        """停止计算节点服务"""
        self._running = False
        self._cleanup()
        self._executor.shutdown(wait=True)
        logger.info("Node stopped")
        
    def _connect(self):
        """连接到VPS服务器"""
        logger.debug(f"Connecting to VPS at {self.vps_address}")
        
        for attempt in range(self.max_retry_attempts):
            try:
                if self._channel is not None:
                    logger.debug("Closing existing channel")
                    self._channel.close()
                    
                # 设置gRPC通道选项
                options = [
                    ('grpc.keepalive_time_ms', 1000),  # 每1秒发送一次keepalive ping
                    ('grpc.keepalive_timeout_ms', 500),  # 500ms超时
                    ('grpc.keepalive_permit_without_calls', True),
                    ('grpc.http2.min_time_between_pings_ms', 1000),  # 最小ping间隔
                    ('grpc.http2.max_pings_without_data', 0),  # 允许无数据的ping
                ]
                
                self._channel = grpc.insecure_channel(self.vps_address, options=options)
                self._stub = service_pb2_grpc.RemoteServiceStub(self._channel)
                
                # 测试连接是否成功
                grpc.channel_ready_future(self._channel).result(timeout=5)
                logger.debug("gRPC channel established successfully")
                return
                
            except Exception as e:
                if attempt < self.max_retry_attempts - 1:
                    wait_time = self.reconnect_interval * (attempt + 1)
                    logger.warning(f"Connection attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to connect after {self.max_retry_attempts} attempts")
                    raise ConnectionError(f"Failed to connect to VPS: {e}")
        
    def _register_with_vps(self):
        """向VPS注册节点和函数"""
        if not self._stub:
            raise ConnectionError("Not connected to VPS")
            
        request = service_pb2.RegisterRequest(
            node_id=self.node_id,
            functions=[
                service_pb2.FunctionSpec(
                    name=func.name,
                    is_async=func.is_async,
                    is_generator=func.is_generator
                )
                for func in self._functions.values()
            ]
        )
        
        response = self._stub.Register(request)
        if not response.success:
            raise ConnectionError(f"Registration failed: {response.message}")
            
    def _start_heartbeat(self):
        """启动心跳检测"""
        def _heartbeat():
            logger.debug("Starting heartbeat thread")
            failures = 0
            
            while self._running and self._connected.is_set():
                try:
                    start_time = time.time()
                    logger.debug(f"Sending heartbeat for node {self.node_id}")
                    
                    # 使用新的带超时的心跳请求
                    request = service_pb2.HeartbeatRequest(node_id=self.node_id)
                    response = self._stub.Heartbeat(
                        request,
                        timeout=min(self.heartbeat_interval, 1.0)  # 设置较短的超时时间
                    )
                    
                    if not response.accepted:
                        logger.error(f"Heartbeat rejected for node {self.node_id}")
                        failures += 1
                        if failures >= 3:
                            self._connected.clear()
                            break
                    else:
                        failures = 0
                        elapsed = time.time() - start_time
                        self._last_heartbeat_time = datetime.now()
                        logger.debug(
                            f"Heartbeat successful for node {self.node_id} "
                            f"(latency: {elapsed*1000:.2f}ms)"
                        )
                    
                    # 确保心跳间隔准确
                    elapsed = time.time() - start_time
                    sleep_time = max(0.0, self.heartbeat_interval - elapsed)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                        
                except grpc.RpcError as e:
                    status_code = e.code() if isinstance(e, grpc.Call) else None
                    logger.error(
                        f"Heartbeat RPC failed for node {self.node_id}: {e} "
                        f"(status: {status_code})"
                    )
                    failures += 1
                    if failures >= 3:
                        self._connected.clear()
                        break
                    time.sleep(1)
                except Exception as e:
                    logger.error(
                        f"Unexpected error in heartbeat for node {self.node_id}: {e}",
                        exc_info=True
                    )
                    failures += 1
                    if failures >= 3:
                        self._connected.clear()
                        break
                    time.sleep(1)
            
            logger.debug(f"Heartbeat thread stopped for node {self.node_id}")
            
        if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
            logger.debug("Stopping existing heartbeat thread")
            self._connected.clear()
            self._heartbeat_thread.join(timeout=5)
            
        self._heartbeat_thread = threading.Thread(target=_heartbeat, daemon=True)
        self._heartbeat_thread.start()
        logger.debug("New heartbeat thread started")
        
    async def _execute_function(
        self,
        func_info: FunctionInfo,
        args: tuple,
        kwargs: dict
    ) -> Any:
        """执行函数并处理不同类型的返回值"""
        func = func_info.callable
        
        try:
            if func_info.is_async:
                result = await func(*args, **kwargs)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self._executor,
                    lambda: func(*args, **kwargs)
                )
            return result
            
        except Exception as e:
            logger.error(f"Error executing {func_info.name}: {e}")
            raise RemoteExecutionError(str(e))
            
    def _cleanup(self):
        """清理资源"""
        if self._channel:
            self._channel.close()
            self._channel = None
            self._stub = None
        self._connected.clear()

# Internal handler for RPC calls
    async def _handle_call(
        self,
        function_name: str,
        args_bytes: bytes,
        kwargs_bytes: bytes
    ) -> bytes:
        """处理RPC调用"""
        if function_name not in self._functions:
            raise RemoteExecutionError(f"Function {function_name} not found")
            
        func_info = self._functions[function_name]
        args, kwargs = deserialize_args(args_bytes, kwargs_bytes)
        
        result = await self._execute_function(func_info, args, kwargs)
        return serialize_result(result)
    
    async def Call(self, request, context):
        """处理远程调用请求"""
        try:
            result = await self._handle_call(
                request.function_name,
                request.args,
                request.kwargs
            )
            return service_pb2.CallResponse(result=result)
            
        except Exception as e:
            logger.error(f"Error in Call: {e}")
            return service_pb2.CallResponse(
                has_error=True,
                error_message=str(e)
            )
            
    async def StreamingCall(self, request, context):
        """处理流式调用请求"""
        try:
            func_info = self._functions.get(request.function_name)
            if not func_info:
                raise ValueError(f"Function {request.function_name} not found")
                
            args, kwargs = deserialize_args(request.args, request.kwargs)
            
            async for chunk in self._stream_results(func_info, args, kwargs):
                yield service_pb2.StreamChunk(
                    data=serialize_result(chunk)
                )
                
        except Exception as e:
            yield service_pb2.StreamChunk(
                has_error=True,
                error_message=str(e)
            )
            
    async def _stream_results(self, func_info: FunctionInfo, args: tuple, kwargs: dict):
        """处理生成器函数的结果"""
        func = func_info.callable
        
        if asyncio.iscoroutinefunction(func):
            # 异步生成器
            async for item in func(*args, **kwargs):
                yield item
        else:
            # 同步生成器
            gen = func(*args, **kwargs)
            while True:
                try:
                    item = await self._loop.run_in_executor(
                        self._executor,
                        next,
                        gen
                    )
                    yield item
                except StopIteration:
                    break

