# easyremote/compute_node.py
import asyncio
import logging
import grpc
import time
import threading
from typing import Optional, Callable, Dict, Any, AsyncGenerator
import uuid
from concurrent import futures

from .types import FunctionInfo, NodeInfo
from .exceptions import (
    format_exception,
    FunctionNotFoundError,
    ConnectionError as EasyRemoteConnectionError,
    RemoteExecutionError,
    EasyRemoteError
)
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
        reconnect_interval: int = 3,
        heartbeat_interval: int = 2,
        max_retry_attempts: int = 3
    ):
        """
        初始化计算节点
        """
        logger.debug(f"Initializing ComputeNode with VPS address: {vps_address}")
        self.vps_address = vps_address
        self.node_id = node_id or f"node-{uuid.uuid4()}"
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
        self._send_queue = None  # 用于发送消息

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
        启动计算节点服务，支持无限重试和Ctrl+C优雅退出
        """
        self._running = True

        def _serve():
            while self._running:  # 持续运行直到被停止
                try:
                    # 创建并设置此线程的事件循环
                    self._loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(self._loop)

                    self._loop.run_until_complete(self._connect_and_run())
                except KeyboardInterrupt:
                    logger.info("Received Ctrl+C, stopping node...")
                    self._running = False
                    break
                except EasyRemoteError as e:
                    logger.error(str(e))
                    self._connected.clear()
                    if self._running:
                        logger.info(f"Reconnecting in {self.reconnect_interval} seconds...")
                        time.sleep(self.reconnect_interval)
                        logger.info("Attempting to reconnect...")
                except Exception as e:
                    # 捕获所有其他异常，避免线程崩溃
                    logger.error(f"Unexpected error: {e}", exc_info=True)
                    self._connected.clear()
                    if self._running:
                        logger.info(f"Reconnecting in {self.reconnect_interval} seconds...")
                        time.sleep(self.reconnect_interval)
                        logger.info("Attempting to reconnect...")
                finally:
                    # 清理当前循环的资源
                    if self._loop and not self._loop.is_closed():
                        try:
                            pending = asyncio.all_tasks(self._loop)
                            for task in pending:
                                task.cancel()
                            self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                            self._loop.close()
                        except Exception as e:
                            logger.error(f"Error during cleanup: {e}", exc_info=True)

            logger.info("Node service stopped")

        if blocking:
            try:
                _serve()
            except KeyboardInterrupt:
                logger.info("Received Ctrl+C, stopping node...")
                self._running = False
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
            raise EasyRemoteConnectionError("Failed to connect to VPS") from e

        logger.debug("gRPC channel to VPS established successfully")

        # 初始化发送队列
        self._send_queue = asyncio.Queue()

        # 使用异步生成器来发送控制消息
        async def control_stream_generator():
            # 发送注册请求
            register_msg = service_pb2.ControlMessage(
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
            )
            await self._send_queue.put(register_msg)
            logger.debug(f"Node {self.node_id} queued RegisterRequest to VPS")

            # 启动心跳任务
            self._heartbeat_task = asyncio.create_task(self._send_heartbeats())

            while self._running:
                try:
                    # 从发送队列中获取消息
                    msg = await self._send_queue.get()
                    yield msg
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in control_stream_generator: {e}", exc_info=True)
                    break

        # 建立双向流
        try:
            async for msg in self._vps_stub.ControlStream(control_stream_generator()):
                await self._handle_message(msg)
        except grpc.aio.AioRpcError as e:
            raise EasyRemoteConnectionError("Stream error occurred") from e
        finally:
            if self._vps_channel:
                await self._vps_channel.close()
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
                try:
                    await self._heartbeat_task
                except asyncio.CancelledError:
                    pass

    async def _handle_message(self, msg):
        """
        处理来自VPS的消息
        """
        if msg.HasField("register_resp"):
            if msg.register_resp.success:
                logger.info("Registered to VPS successfully")
                self._connected.set()
            else:
                raise EasyRemoteError(f"Registration failed: {msg.register_resp.message}")
        elif msg.HasField("heartbeat_resp"):
            if not msg.heartbeat_resp.accepted:
                raise EasyRemoteError("Heartbeat rejected by VPS")
        elif msg.HasField("exec_req"):
            await self._handle_execution_request(msg.exec_req)

    async def _handle_execution_request(self, req: service_pb2.ExecutionRequest):
        """处理函数执行请求"""
        function_name = req.function_name
        call_id = req.call_id
        try:
            # 反序列化参数
            print("handle_execution_request-deserialize_args")
            args, kwargs = deserialize_args(req.args, req.kwargs)
            print("handle_execution_request-deserialize_args-end")

            if function_name not in self._functions:
                raise FunctionNotFoundError(function_name, node_id=self.node_id)

            func_info = self._functions[function_name]

            if func_info.is_generator:
                # 处理生成器函数
                async for chunk in self._handle_generator(func_info, args, kwargs):
                    print("handle_execution_request-genrate-serialize_result")
                    serialized_chunk = serialize_result(chunk)
                    print("handle_execution_request-genrate-serialize_result-end")
                    exec_res_msg = service_pb2.ControlMessage(
                        exec_res=service_pb2.ExecutionResult(
                            call_id=call_id,
                            has_error=False,
                            chunk=serialized_chunk,
                            function_name=func_info.name,
                            node_id=self.node_id
                        )
                    )
                    await self._send_message(exec_res_msg)

                # 发送完成信号
                exec_res_done_msg = service_pb2.ControlMessage(
                    exec_res=service_pb2.ExecutionResult(
                        call_id=call_id,
                        is_done=True,
                        function_name=func_info.name,
                        node_id=self.node_id
                    )
                )
                await self._send_message(exec_res_done_msg)
            else:
                # 执行普通函数
                result = await self._execute_function(func_info, args, kwargs)
                print("handle_execution_request-serialize_result")
                result_bytes = serialize_result(result)
                print("handle_execution_request-serialize_result-end")

                exec_res_msg = service_pb2.ControlMessage(
                    exec_res=service_pb2.ExecutionResult(
                        call_id=call_id,
                        has_error=False,
                        result=result_bytes,
                        is_done=True,
                        function_name=func_info.name,
                        node_id=self.node_id
                    )
                )
                await self._send_message(exec_res_msg)

        except Exception as e:
            error_msg = format_exception(e)
            exec_res_error_msg = service_pb2.ControlMessage(
                exec_res=service_pb2.ExecutionResult(
                    call_id=call_id,
                    has_error=True,
                    error_message=error_msg,
                    function_name=function_name if 'function_name' in locals() else "unknown",
                    node_id=self.node_id
                )
            )
            await self._send_message(exec_res_error_msg)
            logger.error(f"Error executing {function_name}: {error_msg}")

    async def _execute_function(self, func_info: FunctionInfo, args: tuple, kwargs: dict) -> bytes:
        """执行普通函数"""
        try:
            if func_info.is_async:
                result = await func_info.callable(*args, **kwargs)
            else:
                result = await self._loop.run_in_executor(
                    self._executor,
                    func_info.callable,
                    *args,
                    **kwargs
                )
            # Serialize the result before returning
            return result
        except Exception as e:
            raise RemoteExecutionError(
                function_name=func_info.name,
                node_id=self.node_id,
                message=str(e),
                cause=e
            ) from e

    async def _handle_generator(self, func_info: FunctionInfo, args: tuple, kwargs: dict):
        """处理生成器函数"""
        try:
            if func_info.is_async:
                async for item in func_info.callable(*args, **kwargs):
                    yield item
            else:
                loop = asyncio.get_event_loop()
                gen = func_info.callable(*args, **kwargs)
                while True:
                    try:
                        item = await loop.run_in_executor(
                            self._executor,
                            lambda: next(gen)
                        )
                        yield item
                    except StopIteration:
                        break
        except Exception as e:
            raise RemoteExecutionError(
                function_name=func_info.name,
                node_id=self.node_id,
                message=str(e),
                cause=e
            ) from e

    async def _send_heartbeats(self):
        """发送心跳消息"""
        try:
            while self._running:
                await asyncio.sleep(self.heartbeat_interval)
                heartbeat_msg = service_pb2.ControlMessage(
                    heartbeat_req=service_pb2.HeartbeatRequest(
                        node_id=self.node_id
                    )
                )
                await self._send_queue.put(heartbeat_msg)
                logger.debug(f"Node {self.node_id} sent HeartbeatRequest to VPS")
        except asyncio.CancelledError:
            logger.debug("Heartbeat task cancelled")
        except Exception as e:
            raise EasyRemoteError("Heartbeat error occurred") from e

    async def _send_message(self, msg: service_pb2.ControlMessage):
        """发送控制消息到VPS"""
        if not self._send_queue:
            raise EasyRemoteError("Send queue not initialized")
        await self._send_queue.put(msg)

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
            try:
                future.result(timeout=5)
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")

        self._executor.shutdown(wait=False)
        logger.info("Node stopped")

if __name__ == "__main__":
    # 示例用法
    node = ComputeNode("localhost:8080", node_id="basic-compute")

    @node.register
    def add(x: int) -> int:
        return x * 2

    @node.register
    def process_data(data: dict) -> dict:
        return {k: v * 2 for k, v in data.items()}

    @node.register
    def stream_process(data_range) -> Any:
        """同步生成器示例函数"""
        for item in data_range:
            time.sleep(1)  # 模拟耗时操作
            yield item * 2

    try:
        node.serve(blocking=True)
    except KeyboardInterrupt:
        node.stop()
    except EasyRemoteError as e:
        logger.error(str(e))
