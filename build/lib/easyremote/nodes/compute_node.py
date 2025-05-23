# easyremote/nodes/compute_node.py
import asyncio
import grpc
import time
import threading
from typing import Optional, Callable, Dict, Any, Set
import uuid
from concurrent import futures
from datetime import datetime

from ..core.data.data_types import FunctionInfo
from ..core.utils.exceptions import (
    format_exception,
    FunctionNotFoundError,
    ConnectionError as EasyRemoteConnectionError,
    RemoteExecutionError,
    EasyRemoteError
)
from ..core.utils.serialize import (
    deserialize_args,
    serialize_result,
    analyze_function,
    setup_logger
)
from ..core.network.protos import service_pb2, service_pb2_grpc

logger = setup_logger(__name__)

class ComputeNode:
    """计算节点，负责注册和执行远程函数，并作为gRPC客户端连接到VPS"""

    def __init__(
        self,
        vps_address: str,
        node_id: Optional[str] = None,
        reconnect_interval: int = 3,
        heartbeat_interval: int = 2,
        max_retry_attempts: int = 3,
        max_queue_size: int = 1000,
        execution_timeout: int = 300
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
        self.max_queue_size = max_queue_size
        self.execution_timeout = execution_timeout

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
        self._active_executions: Set[str] = set()  # 跟踪活跃的执行
        self._execution_tasks: Dict[str, asyncio.Task] = {}  # 跟踪执行任务
        self._lock = asyncio.Lock()  # 异步锁保护共享状态
        self._shutdown_event = asyncio.Event()  # 关闭事件

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
                            # 清理所有待处理的任务
                            pending = asyncio.all_tasks(self._loop)
                            for task in pending:
                                if not task.done():
                                    task.cancel()
                            
                            # 等待所有任务完成或取消
                            if pending:
                                self._loop.run_until_complete(
                                    asyncio.gather(*pending, return_exceptions=True)
                                )
                            
                            # 关闭事件循环
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

        # 配置gRPC通道选项以提高稳定性
        options = [
            ('grpc.keepalive_time_ms', 30000),
            ('grpc.keepalive_timeout_ms', 5000),
            ('grpc.keepalive_permit_without_calls', True),
            ('grpc.http2.max_pings_without_data', 0),
            ('grpc.http2.min_time_between_pings_ms', 10000),
            ('grpc.max_send_message_length', 50 * 1024 * 1024),
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),
        ]
        
        self._vps_channel = grpc.aio.insecure_channel(self.vps_address, options=options)
        self._vps_stub = service_pb2_grpc.RemoteServiceStub(self._vps_channel)

        # 等待通道就绪
        try:
            await asyncio.wait_for(self._vps_channel.channel_ready(), timeout=10.0)
        except (grpc.aio.AioRpcError, asyncio.TimeoutError) as e:
            raise EasyRemoteConnectionError("Failed to connect to VPS") from e

        logger.debug("gRPC channel to VPS established successfully")

        # 初始化发送队列和其他状态
        self._send_queue = asyncio.Queue(maxsize=self.max_queue_size)
        self._shutdown_event.clear()
        
        async with self._lock:
            self._active_executions.clear()
            self._execution_tasks.clear()

        # 使用异步生成器来发送控制消息
        async def control_stream_generator():
            try:
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

                while self._running and not self._shutdown_event.is_set():
                    try:
                        # 从发送队列中获取消息，使用超时避免永久阻塞
                        msg = await asyncio.wait_for(
                            self._send_queue.get(), 
                            timeout=1.0
                        )
                        yield msg
                    except asyncio.TimeoutError:
                        # 超时是正常的，继续循环
                        continue
                    except asyncio.CancelledError:
                        break
                    except Exception as e:
                        logger.error(f"Error in control_stream_generator: {e}", exc_info=True)
                        break
            finally:
                logger.debug("Control stream generator shutting down")

        # 建立双向流
        try:
            stream_call = self._vps_stub.ControlStream(control_stream_generator())
            async for msg in stream_call:
                if self._shutdown_event.is_set():
                    break
                await self._handle_message(msg)
        except grpc.aio.AioRpcError as e:
            if e.code() != grpc.StatusCode.CANCELLED:
                raise EasyRemoteConnectionError("Stream error occurred") from e
        except Exception as e:
            logger.error(f"Unexpected error in stream processing: {e}", exc_info=True)
            raise EasyRemoteConnectionError("Unexpected stream error") from e
        finally:
            await self._cleanup_connection()

    async def _cleanup_connection(self):
        """清理连接相关资源"""
        logger.debug("Cleaning up connection resources")
        
        # 设置关闭事件
        self._shutdown_event.set()
        
        # 取消心跳任务
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

        # 取消所有活跃的执行任务
        async with self._lock:
            tasks_to_cancel = list(self._execution_tasks.values())
            self._execution_tasks.clear()
            self._active_executions.clear()

        for task in tasks_to_cancel:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.error(f"Error cancelling execution task: {e}")

        # 关闭gRPC通道
        if self._vps_channel:
            await self._vps_channel.close()
            self._vps_channel = None
            self._vps_stub = None

        # 清空发送队列
        if self._send_queue:
            while not self._send_queue.empty():
                try:
                    self._send_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

    async def _handle_message(self, msg):
        """
        处理来自VPS的消息
        """
        try:
            if msg.HasField("register_resp"):
                if msg.register_resp.success:
                    logger.info("Registered to VPS successfully")
                    self._connected.set()
                else:
                    raise EasyRemoteError(f"Registration failed: {msg.register_resp.message}")
            elif msg.HasField("heartbeat_resp"):
                if not msg.heartbeat_resp.accepted:
                    raise EasyRemoteError("Heartbeat rejected by VPS")
                self._last_heartbeat_time = datetime.now()
            elif msg.HasField("exec_req"):
                # 异步处理执行请求，避免阻塞消息处理
                asyncio.create_task(self._handle_execution_request(msg.exec_req))
        except Exception as e:
            logger.error(f"Error handling message: {e}", exc_info=True)

    async def _handle_execution_request(self, req: service_pb2.ExecutionRequest):
        """处理函数执行请求"""
        function_name = req.function_name
        call_id = req.call_id
        
        # 将执行添加到活跃列表
        async with self._lock:
            if call_id in self._active_executions:
                logger.warning(f"Duplicate execution request for call_id: {call_id}")
                return
            self._active_executions.add(call_id)

        # 创建执行任务
        execution_task = asyncio.create_task(
            self._execute_request(req)
        )
        
        async with self._lock:
            self._execution_tasks[call_id] = execution_task

        try:
            await execution_task
        except asyncio.CancelledError:
            logger.debug(f"Execution cancelled for call_id: {call_id}")
        except Exception as e:
            logger.error(f"Execution task failed for call_id: {call_id}, error: {e}")
        finally:
            # 清理执行状态
            async with self._lock:
                self._active_executions.discard(call_id)
                self._execution_tasks.pop(call_id, None)

    async def _execute_request(self, req: service_pb2.ExecutionRequest):
        """实际执行请求"""
        function_name = req.function_name
        call_id = req.call_id
        
        try:
            # 反序列化参数
            args, kwargs = deserialize_args(req.args, req.kwargs)

            if function_name not in self._functions:
                raise FunctionNotFoundError(function_name, node_id=self.node_id)

            func_info = self._functions[function_name]

            if func_info.is_generator:
                # 处理生成器函数
                await self._handle_generator_execution(func_info, args, kwargs, call_id)
            else:
                # 执行普通函数
                result = await self._execute_function(func_info, args, kwargs)
                result_bytes = serialize_result(result)

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
            await self._send_error_result(call_id, function_name, e)

    async def _handle_generator_execution(self, func_info: FunctionInfo, args: tuple, kwargs: dict, call_id: str):
        """处理生成器函数执行"""
        try:
            async for chunk in self._handle_generator(func_info, args, kwargs):
                # 检查是否应该停止
                if self._shutdown_event.is_set():
                    break
                    
                async with self._lock:
                    if call_id not in self._active_executions:
                        break

                serialized_chunk = serialize_result(chunk)
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
            
        except Exception as e:
            await self._send_error_result(call_id, func_info.name, e)

    async def _send_error_result(self, call_id: str, function_name: str, error: Exception):
        """发送错误结果"""
        error_msg = format_exception(error)
        exec_res_error_msg = service_pb2.ControlMessage(
            exec_res=service_pb2.ExecutionResult(
                call_id=call_id,
                has_error=True,
                error_message=error_msg,
                function_name=function_name,
                node_id=self.node_id
            )
        )
        await self._send_message(exec_res_error_msg)
        logger.error(f"Error executing {function_name}: {error_msg}")

    async def _execute_function(self, func_info: FunctionInfo, args: tuple, kwargs: dict) -> Any:
        """执行普通函数"""
        try:
            if func_info.is_async:
                # 为异步函数添加超时
                result = await asyncio.wait_for(
                    func_info.callable(*args, **kwargs),
                    timeout=self.execution_timeout
                )
            else:
                # 为同步函数添加超时
                result = await asyncio.wait_for(
                    self._loop.run_in_executor(
                        self._executor,
                        func_info.callable,
                        *args,
                        **kwargs
                    ),
                    timeout=self.execution_timeout
                )
            return result
        except asyncio.TimeoutError as e:
            raise RemoteExecutionError(
                function_name=func_info.name,
                node_id=self.node_id,
                message=f"Function execution timeout after {self.execution_timeout}s",
                cause=e
            ) from e
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
                # 异步生成器
                async for item in func_info.callable(*args, **kwargs):
                    yield item
            else:
                # 同步生成器，使用线程池执行
                loop = asyncio.get_event_loop()
                gen = await loop.run_in_executor(
                    self._executor, 
                    func_info.callable, 
                    *args, 
                    **kwargs
                )
                
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
            while self._running and not self._shutdown_event.is_set():
                try:
                    await asyncio.sleep(self.heartbeat_interval)
                    
                    heartbeat_msg = service_pb2.ControlMessage(
                        heartbeat_req=service_pb2.HeartbeatRequest(
                            node_id=self.node_id
                        )
                    )
                    await self._send_queue.put(heartbeat_msg)
                    logger.debug(f"Node {self.node_id} sent HeartbeatRequest to VPS")
                except asyncio.QueueFull:
                    logger.warning("Send queue is full, skipping heartbeat")
                except Exception as e:
                    if not self._shutdown_event.is_set():
                        logger.error(f"Error sending heartbeat: {e}")
                        break
        except asyncio.CancelledError:
            logger.debug("Heartbeat task cancelled")
        except Exception as e:
            if not self._shutdown_event.is_set():
                raise EasyRemoteError("Heartbeat error occurred") from e

    async def _send_message(self, msg: service_pb2.ControlMessage):
        """发送控制消息到VPS"""
        if not self._send_queue:
            raise EasyRemoteError("Send queue not initialized")
        
        try:
            # 使用非阻塞方式放入队列，避免死锁
            await asyncio.wait_for(
                self._send_queue.put(msg), 
                timeout=5.0
            )
        except asyncio.TimeoutError:
            logger.warning("Send queue timeout, message may be lost")
        except asyncio.QueueFull:
            logger.warning("Send queue is full, message may be lost")

    def stop(self):
        """停止计算节点服务"""
        self._running = False
        logger.info("Node stopping...")
        
        if self._loop and not self._loop.is_closed():
            # 在事件循环中设置关闭事件
            asyncio.run_coroutine_threadsafe(
                self._shutdown_event.set(), 
                self._loop
            )
            
            # 异步清理资源
            cleanup_future = asyncio.run_coroutine_threadsafe(
                self._cleanup_connection(), 
                self._loop
            )
            try:
                cleanup_future.result(timeout=10)
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")

        # 关闭线程池
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

    @node.register(stream=True)
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
