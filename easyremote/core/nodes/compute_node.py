# easyremote/core/nodes/compute_node.py
import asyncio
import grpc
import time
import threading
from typing import Optional, Callable, Dict, Any, Set
import uuid
from concurrent import futures
from datetime import datetime, timedelta

from ..data import FunctionInfo
from ..utils import format_exception
from ..utils.exceptions import (
    FunctionNotFoundError,
    ConnectionError as EasyRemoteConnectionError,
    RemoteExecutionError,
    EasyRemoteError
)
from ..data.serialize import serialize_result, deserialize_args, analyze_function
from ..protos import service_pb2, service_pb2_grpc
from ..utils.logger import ModernLogger

class ComputeNode(ModernLogger):
    """计算节点，负责注册和执行远程函数，并作为gRPC客户端连接到VPS"""

    def __init__(
        self,
        vps_address: str,
        node_id: Optional[str] = None,
        reconnect_interval: int = 3,
        heartbeat_interval: int = 5,
        max_retry_attempts: int = 3,
        max_queue_size: int = 1000,
        execution_timeout: int = 300,
        connection_timeout: int = 10,
        heartbeat_timeout: int = 15,
        health_check_interval: int = 30
    ):
        """
        初始化计算节点
        
        Args:
            vps_address: VPS服务器地址
            node_id: 节点ID
            reconnect_interval: 重连间隔（秒）
            heartbeat_interval: 心跳间隔（秒）
            max_retry_attempts: 最大重试次数
            max_queue_size: 最大队列大小
            execution_timeout: 执行超时时间（秒）
            connection_timeout: 连接超时时间（秒）
            heartbeat_timeout: 心跳超时时间（秒）
            health_check_interval: 健康检查间隔（秒）
        """
        super().__init__(name="ComputeNode")
        self.info(f"Initializing ComputeNode with VPS address: {vps_address}")
        self.vps_address = vps_address
        self.node_id = node_id or f"node-{uuid.uuid4()}"
        self.reconnect_interval = reconnect_interval
        self.heartbeat_interval = heartbeat_interval
        self.max_retry_attempts = max_retry_attempts
        self.max_queue_size = max_queue_size
        self.execution_timeout = execution_timeout
        self.connection_timeout = connection_timeout
        self.heartbeat_timeout = heartbeat_timeout
        self.health_check_interval = health_check_interval

        self._functions: Dict[str, FunctionInfo] = {}
        self._vps_channel: Optional[grpc.aio.Channel] = None
        self._vps_stub: Optional[service_pb2_grpc.RemoteServiceStub] = None
        self._running = False
        self._connected = threading.Event()
        self._executor = futures.ThreadPoolExecutor(max_workers=10)
        self._heartbeat_task = None
        self._health_check_task = None
        self._last_heartbeat_time = None
        self._connection_healthy = True
        self._reconnect_count = 0
        
        # 异步相关
        self._loop = None
        self._lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()
        self._send_queue = None
        self._active_executions: Set[str] = set()
        self._execution_tasks: Dict[str, asyncio.Task] = {}

        self.info(f"ComputeNode {self.node_id} initialized")

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
                is_async=async_func or func_info.is_async,
                is_generator=stream or func_info.is_generator,
                node_id=node_id or self.node_id
            )

            self.info(f"Registered function: {func_name} (async={self._functions[func_name].is_async}, stream={self._functions[func_name].is_generator})")
            return f

        if func is None:
            return decorator
        return decorator(func)

    def serve(self, blocking: bool = True):
        """
        启动计算节点服务，支持自动重连
        """
        def _serve():
            self._running = True
            retry_count = 0
            
            while self._running:
                try:
                    self.info(f"Starting node service (attempt {retry_count + 1})")
                    
                    # 重置连接状态
                    self._connected.clear()
                    self._connection_healthy = True
                    
                    # 检查是否需要新的事件循环
                    if self._loop is None or self._loop.is_closed():
                        def run_in_new_thread():
                            self._loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(self._loop)
                            try:
                                self._loop.run_until_complete(self._connect_and_run())
                            except Exception as e:
                                self.error(f"Error in event loop: {e}")
                                raise
                            finally:
                                if not self._loop.is_closed():
                                    self._loop.close()
                                self._loop = None
                        
                        thread = threading.Thread(target=run_in_new_thread)
                        thread.start()
                        thread.join()  # 等待线程完成
                    else:
                        # 创建并设置此线程的事件循环
                        self._loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(self._loop)
                        try:
                            self._loop.run_until_complete(self._connect_and_run())
                        finally:
                            if not self._loop.is_closed():
                                self._loop.close()
                            self._loop = None
                            
                    # 如果正常退出，重置重试计数
                    retry_count = 0
                    self._reconnect_count = 0
                    
                except KeyboardInterrupt:
                    self.info("Received Ctrl+C, stopping node...")
                    self._running = False
                    break
                except EasyRemoteError as e:
                    retry_count += 1
                    self._reconnect_count += 1
                    self.error(f"Connection error (attempt {retry_count}): {str(e)}")
                    self._connected.clear()
                    self._connection_healthy = False
                    
                    if retry_count >= self.max_retry_attempts:
                        self.error(f"Maximum retry attempts ({self.max_retry_attempts}) reached. Stopping.")
                        break
                        
                    if self._running:
                        wait_time = min(self.reconnect_interval * (2 ** (retry_count - 1)), 60)  # Exponential backoff
                        self.info(f"Reconnecting in {wait_time} seconds... (attempt {retry_count}/{self.max_retry_attempts})")
                        time.sleep(wait_time)
                        self.info("Attempting to reconnect...")
                except Exception as e:
                    retry_count += 1
                    self._reconnect_count += 1
                    self.error(f"Unexpected error (attempt {retry_count}): {e}", exc_info=True)
                    self._connected.clear()
                    self._connection_healthy = False
                    
                    if retry_count >= self.max_retry_attempts:
                        self.error(f"Maximum retry attempts ({self.max_retry_attempts}) reached. Stopping.")
                        break
                        
                    if self._running:
                        wait_time = min(self.reconnect_interval * (2 ** (retry_count - 1)), 60)  # Exponential backoff
                        self.info(f"Reconnecting in {wait_time} seconds... (attempt {retry_count}/{self.max_retry_attempts})")
                        time.sleep(wait_time)
                        self.info("Attempting to reconnect...")
                finally:
                    # 确保清理资源
                    if self._loop and not self._loop.is_closed():
                        try:
                            # 清理所有待处理的任务
                            self._loop.run_until_complete(self._force_cleanup())
                        except Exception as e:
                            self.debug(f"Error during cleanup: {e}")

            self.info("Node service stopped")

        if blocking:
            try:
                _serve()
            except KeyboardInterrupt:
                self.info("Received Ctrl+C, stopping node...")
                self._running = False
        else:
            thread = threading.Thread(target=_serve, daemon=True)
            thread.start()
            return thread

    async def _force_cleanup(self):
        """强制清理所有资源"""
        try:
            # 检查事件循环是否仍然活跃
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    self.debug("Event loop is closed, skipping async cleanup")
                    self._safe_sync_cleanup()
                    return
            except RuntimeError:
                self.debug("No event loop available, using sync cleanup")
                self._safe_sync_cleanup()
                return

            # 设置关闭事件
            if not self._shutdown_event.is_set():
                self._shutdown_event.set()
            
            # 安全取消心跳和健康检查任务
            await self._safe_cancel_task(self._heartbeat_task, "heartbeat")
            self._heartbeat_task = None
            
            await self._safe_cancel_task(self._health_check_task, "health_check")
            self._health_check_task = None

            # 取消所有执行任务
            tasks_to_cancel = list(self._execution_tasks.values())
            self._execution_tasks.clear()
            self._active_executions.clear()

            # 安全取消执行任务
            for task in tasks_to_cancel:
                if not task.done():
                    try:
                        task.cancel()
                    except Exception as e:
                        self.debug(f"Error cancelling execution task: {e}")

            # 等待任务完成（设置超时避免无限等待）
            if tasks_to_cancel:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*tasks_to_cancel, return_exceptions=True),
                        timeout=2.0
                    )
                except asyncio.TimeoutError:
                    self.debug("Timeout waiting for tasks to complete")

            # 关闭gRPC通道
            if self._vps_channel:
                try:
                    await asyncio.wait_for(self._vps_channel.close(), timeout=1.0)
                except asyncio.TimeoutError:
                    self.debug("Timeout closing gRPC channel")
                except Exception as e:
                    self.debug(f"Error closing gRPC channel: {e}")
                finally:
                    self._vps_channel = None
                    self._vps_stub = None

            # 清空发送队列
            self._safe_clear_queue()
                    
        except RuntimeError as e:
            if "Event loop is closed" in str(e):
                self.debug("Event loop closed during cleanup")
                self._safe_sync_cleanup()
            else:
                self.debug(f"Runtime error in force cleanup: {e}")
        except Exception as e:
            # 只在非事件循环关闭的情况下记录错误
            if "Event loop is closed" not in str(e):
                self.debug(f"Error in force cleanup: {e}")

    async def _safe_cancel_task(self, task, task_name):
        """安全取消任务"""
        if task and not task.done():
            try:
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=1.0)
                except asyncio.CancelledError:
                    self.debug(f"{task_name} task cancelled successfully")
                except asyncio.TimeoutError:
                    self.debug(f"Timeout waiting for {task_name} task to cancel")
            except RuntimeError as e:
                if "Event loop is closed" in str(e):
                    self.debug(f"Event loop closed while cancelling {task_name} task")
                else:
                    raise
            except Exception as e:
                self.debug(f"Error cancelling {task_name} task: {e}")

    def _safe_sync_cleanup(self):
        """同步方式清理资源（当事件循环关闭时使用）"""
        try:
            # 重置状态变量
            self._vps_channel = None
            self._vps_stub = None
            self._heartbeat_task = None
            self._health_check_task = None
            self._execution_tasks.clear()
            self._active_executions.clear()
            self._connection_healthy = False
            
            # 清空队列
            self._safe_clear_queue()
            
            self.debug("Sync cleanup completed")
        except Exception as e:
            self.debug(f"Error in sync cleanup: {e}")

    def _safe_clear_queue(self):
        """安全清空发送队列"""
        if self._send_queue:
            try:
                while not self._send_queue.empty():
                    try:
                        self._send_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    except Exception:
                        break
            except Exception as e:
                self.debug(f"Error clearing send queue: {e}")

    async def _connect_and_run(self):
        """连接到VPS并处理控制流"""
        self.debug(f"Connecting to VPS at {self.vps_address}")

        # 确保旧连接完全关闭
        await self._force_cleanup()

        # 配置gRPC通道选项以提高稳定性和检测能力
        options = [
            ('grpc.keepalive_time_ms', 20000),  # 减少keepalive时间
            ('grpc.keepalive_timeout_ms', 3000),  # 减少keepalive超时
            ('grpc.keepalive_permit_without_calls', True),
            ('grpc.http2.max_pings_without_data', 0),
            ('grpc.http2.min_time_between_pings_ms', 5000),
            ('grpc.max_send_message_length', 50 * 1024 * 1024),
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),
            # 添加连接检测选项
            ('grpc.so_reuseport', 1),
            ('grpc.tcp_user_timeout_ms', 10000),  # TCP用户超时
        ]
        
        self._vps_channel = grpc.aio.insecure_channel(self.vps_address, options=options)
        self._vps_stub = service_pb2_grpc.RemoteServiceStub(self._vps_channel)

        # 等待通道就绪，增加重试机制
        max_connection_attempts = 3
        for attempt in range(max_connection_attempts):
            try:
                await asyncio.wait_for(
                    self._vps_channel.channel_ready(), 
                    timeout=self.connection_timeout
                )
                break
            except (grpc.aio.AioRpcError, asyncio.TimeoutError) as e:
                if attempt == max_connection_attempts - 1:
                    raise EasyRemoteConnectionError(f"Failed to connect to VPS after {max_connection_attempts} attempts") from e
                self.warning(f"Connection attempt {attempt + 1} failed, retrying...")
                await asyncio.sleep(1)

        self.debug("gRPC channel to VPS established successfully")

        # 重新初始化状态
        self._send_queue = asyncio.Queue(maxsize=self.max_queue_size)
        self._shutdown_event.clear()
        self._connection_healthy = True
        self._last_heartbeat_time = datetime.now()
        
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
                self.debug(f"Node {self.node_id} queued RegisterRequest to VPS")

                # 启动心跳和健康检查任务
                self._heartbeat_task = asyncio.create_task(self._send_heartbeats())
                self._health_check_task = asyncio.create_task(self._connection_health_check())

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
                        self.error(f"Error in control_stream_generator: {e}", exc_info=True)
                        break
            finally:
                self.debug("Control stream generator shutting down")

        # 建立双向流
        try:
            stream_call = self._vps_stub.ControlStream(control_stream_generator())
            async for msg in stream_call:
                if self._shutdown_event.is_set():
                    break
                await self._handle_message(msg)
        except grpc.aio.AioRpcError as e:
            if e.code() == grpc.StatusCode.UNAVAILABLE:
                # 降低连接断开错误的日志级别，这是正常的重连情况
                self.debug(f"Server unavailable (normal during reconnection): {e}")
                raise EasyRemoteConnectionError("Server unavailable") from e
            elif e.code() == grpc.StatusCode.CANCELLED:
                self.debug("Stream cancelled")
            else:
                self.warning(f"Stream error: {e.code()}")
                raise EasyRemoteConnectionError(f"Stream error: {e.code()}") from e
        except Exception as e:
            self.error(f"Unexpected error in stream processing: {e}", exc_info=True)
            raise EasyRemoteConnectionError("Unexpected stream error") from e
        finally:
            await self._cleanup_connection()

    async def _connection_health_check(self):
        """定期检查连接健康状态"""
        try:
            while self._running and not self._shutdown_event.is_set():
                await asyncio.sleep(self.health_check_interval)
                
                # 检查心跳超时
                if self._last_heartbeat_time:
                    time_since_heartbeat = datetime.now() - self._last_heartbeat_time
                    if time_since_heartbeat.total_seconds() > self.heartbeat_timeout:
                        self.warning(f"Heartbeat timeout: {time_since_heartbeat.total_seconds()}s > {self.heartbeat_timeout}s")
                        self._connection_healthy = False
                        raise EasyRemoteConnectionError("Heartbeat timeout detected")
                
                # 主动检查gRPC通道状态
                if self._vps_channel:
                    try:
                        state = self._vps_channel.get_state()
                        if state in [grpc.ChannelConnectivity.TRANSIENT_FAILURE, 
                                   grpc.ChannelConnectivity.SHUTDOWN]:
                            self.warning(f"Channel in bad state: {state}")
                            self._connection_healthy = False
                            raise EasyRemoteConnectionError(f"Channel state: {state}")
                    except Exception as e:
                        self.warning(f"Error checking channel state: {e}")
                        
        except asyncio.CancelledError:
            self.debug("Health check task cancelled")
        except Exception as e:
            if not self._shutdown_event.is_set():
                self.error(f"Health check failed: {e}")
                raise EasyRemoteConnectionError("Health check failed") from e

    async def _cleanup_connection(self):
        """清理连接相关资源"""
        self.debug("Cleaning up connection resources")
        
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
                    self.error(f"Error cancelling execution task: {e}")

        # 关闭gRPC通道
        if self._vps_channel:
            try:
                await self._vps_channel.close()
            except Exception as e:
                self.error(f"Error closing gRPC channel: {e}")
            finally:
                self._vps_channel = None
                self._vps_stub = None

        # 清空发送队列
        if self._send_queue:
            try:
                while not self._send_queue.empty():
                    try:
                        self._send_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
            except Exception as e:
                self.error(f"Error clearing send queue: {e}")

    async def _handle_message(self, msg):
        """
        处理来自VPS的消息
        """
        try:
            if msg.HasField("register_resp"):
                if msg.register_resp.success:
                    self.info("Registered to VPS successfully")
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
            self.error(f"Error handling message: {e}", exc_info=True)

    async def _handle_execution_request(self, req: service_pb2.ExecutionRequest):
        """处理函数执行请求"""
        function_name = req.function_name
        call_id = req.call_id
        
        # 将执行添加到活跃列表
        async with self._lock:
            if call_id in self._active_executions:
                self.warning(f"Duplicate execution request for call_id: {call_id}")
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
            self.debug(f"Execution cancelled for call_id: {call_id}")
        except Exception as e:
            self.error(f"Execution task failed for call_id: {call_id}, error: {e}")
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
        self.error(f"Error executing {function_name}: {error_msg}")

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
                    self.debug(f"Node {self.node_id} sent HeartbeatRequest to VPS")
                except asyncio.QueueFull:
                    self.warning("Send queue is full, skipping heartbeat")
                except Exception as e:
                    if not self._shutdown_event.is_set():
                        self.error(f"Error sending heartbeat: {e}")
                        break
        except asyncio.CancelledError:
            self.debug("Heartbeat task cancelled")
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
            self.warning("Send queue timeout, message may be lost")
        except asyncio.QueueFull:
            self.warning("Send queue is full, message may be lost")

    def stop(self):
        """停止计算节点服务"""
        self._running = False
        self.info("Node stopping...")
        
        if self._loop and not self._loop.is_closed():
            try:
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
                cleanup_future.result(timeout=10)
            except Exception as e:
                self.error(f"Error during cleanup: {e}")

        # 关闭线程池
        try:
            self._executor.shutdown(wait=False)
        except Exception as e:
            self.error(f"Error shutting down executor: {e}")
        
        self.info("Node stopped")
