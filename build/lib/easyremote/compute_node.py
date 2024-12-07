import asyncio
import logging
import grpc
import time
import threading
from typing import Optional, Callable, Dict, Any
import uuid
from concurrent import futures

from .types import FunctionInfo, NodeInfo
from .exceptions import ConnectionError, RemoteExecutionError
from .utils import (
    deserialize_args,
    serialize_result,
    analyze_function,
    setup_logger
)
from .protos import service_pb2, service_pb2_grpc

logger = setup_logger(__name__)

class ComputeNode:
    """计算节点，负责注册和执行远程函数，并作为gRPC客户端连接到VPS"""
    
    def __init__(
        self,
        vps_address: str,
        node_id: Optional[str] = None,
        reconnect_interval: int = 5,
        heartbeat_interval: int = 5,
        max_retry_attempts: int = 3
    ):
        """
        初始化计算节点
        """
        logger.debug(f"Initializing ComputeNode with VPS address: {vps_address}")
        self.vps_address = vps_address
        self.node_id = node_id or f"node-{id(self)}"
        self.reconnect_interval = reconnect_interval
        self.heartbeat_interval = heartbeat_interval
        self.max_retry_attempts = max_retry_attempts
        
        self._functions: Dict[str, FunctionInfo] = {}
        self._vps_channel: Optional[grpc.aio.Channel] = None
        self._vps_stub: Optional[service_pb2_grpc.RemoteServiceStub] = None
        self._running = False
        self._connected = threading.Event()
        self._executor = futures.ThreadPoolExecutor(max_workers=10)
        self._heartbeat_task = None
        self._last_heartbeat_time = None
        self._loop = None
        
        logger.info(f"ComputeNode {self.node_id} initialized")

    def register(
        self,
        func: Optional[Callable] = None,
        *,
        name: Optional[str] = None,
        stream: bool = False,
        async_func: bool = False,
        node_id: Optional[str] = None,
        timeout: Optional[float] = None
    ) -> Callable:
        """
        注册一个远程函数。
        """
        def decorator(f: Callable) -> Callable:
            func_name = name or f.__name__
            func_info = analyze_function(f)
            
            self._functions[func_name] = FunctionInfo(
                name=func_name,
                callable=f,
                is_async=async_func or func_info['is_async'],
                is_generator=stream or func_info['is_generator'],
                node_id=node_id or self.node_id
            )
            
            logger.info(f"Registered function: {func_name} (async={self._functions[func_name].is_async}, stream={self._functions[func_name].is_generator})")
            return f
        
        if func is None:
            return decorator
        return decorator(func)

    def serve(self, blocking: bool = True):
        """
        启动计算节点服务
        """
        self._running = True
        
        def _serve():
            try:
                # Create and set event loop for this thread
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
                
                while self._running:
                    try:
                        self._loop.run_until_complete(self._connect_and_run())
                    except Exception as e:
                        logger.error(f"Connection error: {e}")
                        self._connected.clear()
                        if self._running:
                            logger.info(f"Reconnecting in {self.reconnect_interval} seconds...")
                            time.sleep(self.reconnect_interval)
            finally:
                # Clean up
                if self._loop and self._running:
                    pending = asyncio.all_tasks(self._loop)
                    for task in pending:
                        task.cancel()
                    self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                    self._loop.close()
                
        if blocking:
            _serve()
        else:
            thread = threading.Thread(target=_serve, daemon=True)
            thread.start()
            return thread

    async def _connect_and_run(self):
        """连接到VPS并处理控制流"""
        logger.debug(f"Connecting to VPS at {self.vps_address}")
        
        if self._vps_channel:
            await self._vps_channel.close()
            
        self._vps_channel = grpc.aio.insecure_channel(self.vps_address)
        self._vps_stub = service_pb2_grpc.RemoteServiceStub(self._vps_channel)
        
        # 等待通道就绪
        try:
            await self._vps_channel.channel_ready()
        except grpc.aio.AioRpcError as e:
            logger.error(f"Failed to connect to VPS: {e}")
            await self._vps_channel.close()
            raise ConnectionError("Failed to connect to VPS")
            
        logger.debug("gRPC channel to VPS established successfully")
        
        stream = self._vps_stub.ControlStream()
        
        # 发送注册请求
        await stream.write(service_pb2.ControlMessage(
            register_req=service_pb2.RegisterRequest(
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
        ))
        
        # 启动心跳任务
        self._heartbeat_task = asyncio.create_task(self._send_heartbeats(stream))
        
        try:
            async for msg in stream:
                await self._handle_message(stream, msg)
        except grpc.aio.AioRpcError as e:
            logger.error(f"Stream error: {e}")
        finally:
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
                try:
                    await self._heartbeat_task
                except asyncio.CancelledError:
                    pass
            # 当写操作完成后，如果需要可以调用 done_writing()
            await stream.done_writing()
            await self._vps_channel.close()

    async def _send_heartbeats(self, stream):
        """
        发送心跳包
        """
        try:
            while self._running:
                await asyncio.sleep(self.heartbeat_interval)
                await stream.write(service_pb2.ControlMessage(
                    heartbeat_req=service_pb2.HeartbeatRequest(
                        node_id=self.node_id
                    )
                ))
                logger.debug(f"Node {self.node_id} sent HeartbeatRequest to VPS")
        except asyncio.CancelledError:
            logger.debug("Heartbeat task cancelled")
        except Exception as e:
            logger.error(f"Heartbeat error: {e}")
            self._running = False

    async def _handle_message(self, stream, msg):
        """
        处理来自VPS的消息
        """
        if msg.HasField("register_resp"):
            if msg.register_resp.success:
                logger.info("Registered to VPS successfully")
                self._connected.set()
            else:
                logger.error(f"Registration failed: {msg.register_resp.message}")
        elif msg.HasField("heartbeat_resp"):
            if not msg.heartbeat_resp.accepted:
                logger.error("Heartbeat rejected by VPS")
        elif msg.HasField("exec_req"):
            await self._handle_execution_request(stream, msg.exec_req)

    async def _handle_execution_request(self, stream, req: service_pb2.ExecutionRequest):
        """
        处理函数执行请求
        """
        function_name = req.function_name
        call_id = req.call_id
        args, kwargs = deserialize_args(req.args, req.kwargs)
        
        if function_name not in self._functions:
            await stream.write(service_pb2.ControlMessage(exec_res=service_pb2.ExecutionResult(
                call_id=call_id,
                has_error=True,
                error_message=f"Function {function_name} not found"
            )))
            logger.warning(f"Function {function_name} not found on node {self.node_id}")
            return
        
        func_info = self._functions[function_name]
        
        try:
            if func_info.is_generator:
                # Handle generator function
                if func_info.is_async:
                    # Async generator
                    async for item in func_info.callable(*args, **kwargs):
                        await stream.write(service_pb2.ControlMessage(exec_res=service_pb2.ExecutionResult(
                            call_id=call_id,
                            has_error=False,
                            chunk=serialize_result(item)
                        )))
                else:
                    # Sync generator
                    gen = func_info.callable(*args, **kwargs)
                    while True:
                        try:
                            item = await self._loop.run_in_executor(self._executor, next, gen)
                            await stream.write(service_pb2.ControlMessage(exec_res=service_pb2.ExecutionResult(
                                call_id=call_id,
                                has_error=False,
                                chunk=serialize_result(item)
                            )))
                        except StopIteration:
                            break
                
                # Send completion message
                await stream.write(service_pb2.ControlMessage(exec_res=service_pb2.ExecutionResult(
                    call_id=call_id,
                    is_done=True
                )))
            else:
                # Handle regular function
                if func_info.is_async:
                    result = await func_info.callable(*args, **kwargs)
                else:
                    result = await self._loop.run_in_executor(
                        self._executor,
                        func_info.callable,
                        *args,
                        **kwargs
                    )
                
                await stream.write(service_pb2.ControlMessage(exec_res=service_pb2.ExecutionResult(
                    call_id=call_id,
                    has_error=False,
                    result=serialize_result(result),
                    is_done=True
                )))
                
        except Exception as e:
            await stream.write(service_pb2.ControlMessage(exec_res=service_pb2.ExecutionResult(
                call_id=call_id,
                has_error=True,
                error_message=str(e)
            )))
            logger.error(f"Error executing function {function_name} on node {self.node_id}: {e}", exc_info=True)

    def stop(self):
        """停止计算节点服务"""
        self._running = False
        logger.info("Node stopping...")
        if self._loop and not self._loop.is_closed():
            async def cleanup():
                if self._vps_channel:
                    await self._vps_channel.close()
                if self._heartbeat_task:
                    self._heartbeat_task.cancel()
                    try:
                        await self._heartbeat_task
                    except asyncio.CancelledError:
                        pass
            
            future = asyncio.run_coroutine_threadsafe(cleanup(), self._loop)
            future.result(timeout=5)
        
        self._executor.shutdown(wait=False)
        logger.info("Node stopped")

if __name__ == "__main__":
    # Example usage
    node = ComputeNode("localhost:8080", node_id="basic-compute")
    
    @node.register
    def add(x: int) -> int:
        return x * 2
    
    @node.register
    def process_data(data: dict) -> dict:
        return {k: v * 2 for k, v in data.items()}
    
    try:
        node.serve(blocking=True)
    except KeyboardInterrupt:
        node.stop()
