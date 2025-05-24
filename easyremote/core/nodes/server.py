# easyremote/nodes/server.py
import asyncio
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Set, Union, Any
from grpc import aio as grpc_aio
from concurrent import futures
import uuid

from ..utils.logger import ModernLogger

from ..data import NodeInfo, FunctionInfo
from ..utils.exceptions import (
    NodeNotFoundError,
    FunctionNotFoundError,
    SerializationError,
    RemoteExecutionError,
    EasyRemoteError
)
from ..data import Serializer
from ..protos import service_pb2, service_pb2_grpc


# 定义一个独特的哨兵对象，用于标识生成器已耗尽
_SENTINEL = object()

class StreamContext(ModernLogger):
    """流式调用上下文，用于管理流式调用的生命周期"""
    def __init__(self, call_id: str, function_name: str, node_id: str, queue: asyncio.Queue):
        super().__init__(name=__name__)
        self.call_id = call_id
        self.function_name = function_name
        self.node_id = node_id
        self.queue = queue
        self.created_at = datetime.now()
        self.is_active = True
        self._cleanup_callbacks = []

    def add_cleanup_callback(self, callback):
        """添加清理回调"""
        self._cleanup_callbacks.append(callback)

    async def cleanup(self):
        """清理资源"""
        self.is_active = False
        for callback in self._cleanup_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as e:
                self.error(f"Error in cleanup callback: {e}")

class Server(service_pb2_grpc.RemoteServiceServicer, ModernLogger):
    """使用ControlStream双向流实现的VPS服务器，支持普通和流式函数调用"""

    _instance = None  # 单例模式

    def __init__(self, port: int = 8080, heartbeat_timeout: int = 5, max_queue_size: int = 1000):
        """初始化服务器实例"""
        service_pb2_grpc.RemoteServiceServicer.__init__(self)
        ModernLogger.__init__(self, name=__name__)
        self.debug(f"Initializing Server instance on port {port} with heartbeat timeout {heartbeat_timeout}s")
        self.port = port
        self.heartbeat_timeout = heartbeat_timeout
        self.max_queue_size = max_queue_size
        
        # 使用锁保护的数据结构
        self._lock = asyncio.Lock()
        self._nodes: Dict[str, NodeInfo] = {}
        self._node_queues: Dict[str, asyncio.Queue] = {}
        self._pending_calls: Dict[str, Union[asyncio.Future, Dict[str, Any]]] = {}
        self._stream_contexts: Dict[str, StreamContext] = {}
        self._active_streams: Set[str] = set()
        
        self._running = False
        self._server = None
        self._loop = None
        self._monitor_thread = None
        self._cleanup_task = None
        
        self._serializer = Serializer()
        
        Server._instance = self
        self.debug("Server instance initialized")

    def start(self):
        """在主线程中启动服务器（阻塞模式）"""
        from ..utils.async_helpers import AsyncHelpers
        
        helpers = AsyncHelpers()
        
        try:
            if helpers.is_running_in_event_loop():
                # 如果已经在事件循环中，使用后台线程模式
                self.info("Event loop detected, starting server in background mode")
                return self.start_background()
            else:
                # 安全地运行异步服务器
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
                try:
                    self._loop.run_until_complete(self._serve())
                finally:
                    if not self._loop.is_closed():
                        self._loop.close()
        except EasyRemoteError as e:
            self.error(str(e))
        except Exception as e:
            self.error(f"Unexpected server error: {e}", exc_info=True)

    def start_background(self):
        """在后台线程中启动服务器（非阻塞模式）"""
        def run_server():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            try:
                self._loop.run_until_complete(self._serve())
            except EasyRemoteError as e:
                self.error(str(e))
            except Exception as e:
                self.error(f"Server error: {e}", exc_info=True)
            finally:
                if not self._loop.is_closed():
                    self._loop.close()

        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        time.sleep(1)  # 给服务器一些启动时间
        return server_thread

    async def _serve(self):
        """服务器主运行循环"""
        self._running = True
        self._server = grpc_aio.server(
            futures.ThreadPoolExecutor(max_workers=10),
            options=[
                ('grpc.max_send_message_length', 50 * 1024 * 1024),
                ('grpc.max_receive_message_length', 50 * 1024 * 1024),
                ('grpc.keepalive_time_ms', 30000),
                ('grpc.keepalive_timeout_ms', 5000),
                ('grpc.keepalive_permit_without_calls', True),
                ('grpc.http2.max_pings_without_data', 0),
                ('grpc.http2.min_time_between_pings_ms', 10000),
                ('grpc.http2.min_ping_interval_without_data_ms', 300000)
            ]
        )
        service_pb2_grpc.add_RemoteServiceServicer_to_server(self, self._server)

        try:
            addr = f'[::]:{self.port}'
            self._server.add_insecure_port(addr)
            await self._server.start()
            self.info(f"Server started on {addr}")

            self._start_node_monitor()
            self._start_cleanup_task()

            await self._server.wait_for_termination()

        except EasyRemoteError as e:
            self.error(str(e))
            self._running = False
        except Exception as e:
            self.error(f"Server error: {e}", exc_info=True)
            self._running = False
        finally:
            await self._cleanup_server()

    def _start_node_monitor(self):
        """启动节点监控线程"""
        def monitor():
            while self._running:
                try:
                    asyncio.run_coroutine_threadsafe(
                        self._monitor_nodes(), self._loop
                    ).result(timeout=1)
                    time.sleep(self.heartbeat_timeout / 2)
                except Exception as e:
                    self.error(f"Monitor error: {e}")
                    if not self._running:
                        break
                    time.sleep(1)

        self._monitor_thread = threading.Thread(target=monitor, daemon=True)
        self._monitor_thread.start()

    async def _monitor_nodes(self):
        """监控节点状态"""
        async with self._lock:
            now = datetime.now()
            timeout = timedelta(seconds=self.heartbeat_timeout)
            nodes_to_remove = []

            for node_id, node in self._nodes.items():
                time_since = now - node.last_heartbeat
                if time_since > timeout:
                    self.warning(f"Node {node_id} timed out, removing")
                    nodes_to_remove.append(node_id)

            for node_id in nodes_to_remove:
                await self._remove_node(node_id)

    async def _remove_node(self, node_id: str):
        """安全移除节点"""
        # 清理相关的待处理调用
        calls_to_remove = []
        for call_id, call_ctx in self._pending_calls.items():
            if isinstance(call_ctx, dict) and call_ctx.get('node_id') == node_id:
                calls_to_remove.append(call_id)
            elif hasattr(call_ctx, 'node_id') and call_ctx.node_id == node_id:
                calls_to_remove.append(call_id)

        for call_id in calls_to_remove:
            await self._cleanup_pending_call(call_id, "Node disconnected")

        # 清理与该节点相关的流上下文
        streams_to_remove = []
        for stream_id, ctx in self._stream_contexts.items():
            if hasattr(ctx, 'node_id') and ctx.node_id == node_id:
                streams_to_remove.append(stream_id)
        
        for stream_id in streams_to_remove:
            await self._cleanup_stream_context(stream_id)

        # 移除节点相关数据
        self._nodes.pop(node_id, None)
        queue = self._node_queues.pop(node_id, None)
        if queue:
            # 更完整的队列清理
            try:
                queue_size = queue.qsize()
                if queue_size > 0:
                    self.warning(f"Discarding {queue_size} unprocessed messages for node {node_id}")
                
                while not queue.empty():
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    except Exception as e:
                        self.error(f"Error clearing queue item for node {node_id}: {e}")
                        break
            except Exception as e:
                self.error(f"Error clearing queue for node {node_id}: {e}")

    def _start_cleanup_task(self):
        """启动清理任务"""
        async def cleanup_routine():
            while self._running:
                try:
                    await asyncio.sleep(60)  # 每分钟清理一次
                    await self._cleanup_stale_resources()
                except Exception as e:
                    self.error(f"Cleanup routine error: {e}")

        self._cleanup_task = asyncio.create_task(cleanup_routine())

    async def _cleanup_stale_resources(self):
        """清理过期资源"""
        async with self._lock:
            now = datetime.now()
            stale_timeout = timedelta(minutes=5)  # 5分钟超时
            
            # 清理过期的流上下文
            stale_streams = []
            for call_id, ctx in self._stream_contexts.items():
                if now - ctx.created_at > stale_timeout:
                    stale_streams.append(call_id)

            for call_id in stale_streams:
                await self._cleanup_stream_context(call_id)

            # 清理过期的待处理调用
            stale_calls = []
            for call_id, call_ctx in self._pending_calls.items():
                if hasattr(call_ctx, 'created_at'):
                    if now - call_ctx.created_at > stale_timeout:
                        stale_calls.append(call_id)

            for call_id in stale_calls:
                await self._cleanup_pending_call(call_id, "Call timeout")

    async def _cleanup_pending_call(self, call_id: str, reason: str):
        """清理待处理调用"""
        call_ctx = self._pending_calls.pop(call_id, None)
        if call_ctx:
            try:
                if isinstance(call_ctx, asyncio.Future) and not call_ctx.done():
                    call_ctx.set_exception(RemoteExecutionError(
                        function_name="unknown",
                        node_id="unknown",
                        message=f"Call cancelled: {reason}",
                        cause=None
                    ))
                elif isinstance(call_ctx, dict) and 'queue' in call_ctx:
                    try:
                        await call_ctx['queue'].put(RemoteExecutionError(
                            function_name=call_ctx.get('function_name', 'unknown'),
                            node_id=call_ctx.get('node_id', 'unknown'),
                            message=f"Stream cancelled: {reason}",
                            cause=None
                        ))
                    except Exception as e:
                        self.error(f"Error notifying stream cancellation: {e}")
            except Exception as e:
                self.error(f"Error cleaning up pending call {call_id}: {e}")

    async def _cleanup_stream_context(self, call_id: str):
        """清理流上下文"""
        ctx = self._stream_contexts.pop(call_id, None)
        if ctx:
            try:
                await ctx.cleanup()
            except Exception as e:
                self.error(f"Error cleaning up stream context {call_id}: {e}")
            finally:
                self._active_streams.discard(call_id)

    async def _cleanup_server(self):
        """清理服务器资源"""
        self._running = False
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        if self._server:
            await self._server.stop(grace=None)

        # 清理所有连接和资源
        async with self._lock:
            # 清理所有待处理调用
            for call_id in list(self._pending_calls.keys()):
                await self._cleanup_pending_call(call_id, "Server shutdown")

            # 清理所有流上下文
            for call_id in list(self._stream_contexts.keys()):
                await self._cleanup_stream_context(call_id)

            # 清理节点数据
            for node_id in list(self._node_queues.keys()):
                await self._remove_node(node_id)

        self.info("Server cleanup completed")

    async def stop(self):
        """停止服务器"""
        await self._cleanup_server()

    def stop_sync(self):
        """同步方式停止服务器"""
        if self._loop and not self._loop.is_closed():
            try:
                asyncio.run_coroutine_threadsafe(self.stop(), self._loop).result(timeout=10)
            except Exception as e:
                self.error(f"Error stopping server: {e}")

    async def ControlStream(self, request_iterator, context):
        node_id = None
        out_queue = asyncio.Queue(maxsize=self.max_queue_size)

        async def read_requests():
            nonlocal node_id
            try:
                async for msg in request_iterator:
                    if msg.HasField("register_req"):
                        async with self._lock:
                            node_id = msg.register_req.node_id
                            functions = {}
                            for f in msg.register_req.functions:
                                functions[f.name] = FunctionInfo(
                                    name=f.name,
                                    callable=None,
                                    is_async=f.is_async,
                                    is_generator=f.is_generator,
                                    node_id=node_id
                                )

                            self._nodes[node_id] = NodeInfo(
                                node_id=node_id,
                                functions=functions,
                                last_heartbeat=datetime.now()
                            )
                            self._node_queues[node_id] = out_queue
                            self.info(f"Node {node_id} registered with functions: {list(functions.keys())}")

                        await out_queue.put(service_pb2.ControlMessage(
                            register_resp=service_pb2.RegisterResponse(
                                success=True,
                                message="Registered successfully"
                            )
                        ))

                    elif msg.HasField("heartbeat_req"):
                        req = msg.heartbeat_req
                        async with self._lock:
                            if req.node_id in self._nodes:
                                self._nodes[req.node_id].last_heartbeat = datetime.now()
                                accepted = True
                            else:
                                accepted = False

                        await out_queue.put(service_pb2.ControlMessage(
                            heartbeat_resp=service_pb2.HeartbeatResponse(accepted=accepted)
                        ))

                    elif msg.HasField("exec_res"):
                        res = msg.exec_res
                        await self._handle_execution_result(res, node_id)

            except Exception as e:
                self.error(f"Error in ControlStream read_requests: {e}", exc_info=True)
                raise EasyRemoteError("Error in ControlStream") from e
            finally:
                # 当客户端断开或请求结束时进行清理
                if node_id:
                    async with self._lock:
                        if node_id in self._nodes:
                            self.info(f"Node {node_id} disconnected")
                            await self._remove_node(node_id)

                # 向输出队列发送结束信号
                try:
                    await out_queue.put(_SENTINEL)
                except Exception as e:
                    self.error(f"Error sending sentinel: {e}")

        # 后台任务：处理请求并将响应放入队列中
        reader_task = None
        try:
            reader_task = asyncio.create_task(read_requests())

            # 从队列中获取消息并yield
            while True:
                try:
                    msg = await asyncio.wait_for(out_queue.get(), timeout=1.0)
                    if msg is _SENTINEL:
                        break
                    yield msg
                except asyncio.TimeoutError:
                    # 检查连接是否仍然活跃
                    if context.cancelled():
                        break
                    continue
                except Exception as e:
                    self.error(f"Error while yielding messages: {e}", exc_info=True)
                    break

        except Exception as e:
            self.error(f"Error in ControlStream: {e}", exc_info=True)
            raise EasyRemoteError("Error in ControlStream") from e
        finally:
            # 确保后台任务结束
            if reader_task and not reader_task.done():
                reader_task.cancel()
                try:
                    await reader_task
                except asyncio.CancelledError:
                    pass

    async def _handle_execution_result(self, res, node_id):
        """处理执行结果"""
        call_id = res.call_id
        
        async with self._lock:
            call_ctx = self._pending_calls.get(call_id)
            
        if not call_ctx:
            self.warning(f"Received result for unknown call_id: {call_id}")
            return

        try:
            if isinstance(call_ctx, asyncio.Future):
                if res.has_error:
                    function_name = res.function_name if hasattr(res, 'function_name') else "unknown"
                    if not call_ctx.done():
                        call_ctx.set_exception(RemoteExecutionError(
                            function_name=function_name,
                            node_id=node_id,
                            message=res.error_message,
                            cause=None
                        ))
                else:
                    result = self._serializer.deserialize_result(res.result) if res.result else None
                    if not call_ctx.done():
                        call_ctx.set_result(result)
                
                async with self._lock:
                    self._pending_calls.pop(call_id, None)
                    
            else:  # Stream context
                q = call_ctx['queue']
                if res.has_error:
                    function_name = res.function_name if hasattr(res, 'function_name') else "unknown"
                    await q.put(RemoteExecutionError(
                        function_name=function_name,
                        node_id=node_id,
                        message=res.error_message,
                        cause=None
                    ))
                    async with self._lock:
                        self._pending_calls.pop(call_id, None)
                        await self._cleanup_stream_context(call_id)
                else:
                    if res.chunk:
                        chunk = self._serializer.deserialize_result(res.chunk)
                        await q.put(chunk)
                    if res.is_done:
                        await q.put(_SENTINEL)
                        async with self._lock:
                            self._pending_calls.pop(call_id, None)
                            await self._cleanup_stream_context(call_id)
                            
        except Exception as e:
            self.error(f"Error handling execution result: {e}", exc_info=True)
            async with self._lock:
                self._pending_calls.pop(call_id, None)
                await self._cleanup_stream_context(call_id)

    def execute_function(self, node_id: str, function_name: str, *args, **kwargs):
        """执行远程函数"""
        async def _check_and_get_function():
            async with self._lock:
                if node_id not in self._nodes:
                    raise NodeNotFoundError(node_id=node_id, message=f"Node {node_id} not found")

                node = self._nodes[node_id]
                if function_name not in node.functions:
                    raise FunctionNotFoundError(function_name=function_name, node_id=node_id, message=f"Function {function_name} not found on node {node_id}")

                return node.functions[function_name]
        
        if not self._loop or self._loop.is_closed():
            raise EasyRemoteError("Server not started or loop is closed")
            
        try:
            func_info = asyncio.run_coroutine_threadsafe(
                _check_and_get_function(), self._loop
            ).result(timeout=5)
        except Exception as e:
            if isinstance(e, (NodeNotFoundError, FunctionNotFoundError)):
                raise
            raise EasyRemoteError("Failed to check function availability") from e
        
        is_stream = func_info.is_generator
        call_id = str(uuid.uuid4())
        
        try:
            args_bytes, kwargs_bytes = self._serializer.serialize_args(*args, **kwargs)
        except Exception as e:
            raise SerializationError(
                operation="serialize", 
                message="Failed to serialize function arguments",
                cause=e
            )

        if is_stream:
            return self._execute_stream_function(node_id, call_id, function_name, args_bytes, kwargs_bytes)
        else:
            fut = asyncio.run_coroutine_threadsafe(
                self._request_execution(node_id, call_id, function_name, args_bytes, kwargs_bytes, is_stream=False),
                self._loop
            )
            try:
                result = fut.result(timeout=30)
                return result if result is not None else None
            except Exception as e:
                # 清理可能残留的调用上下文
                if self._loop and not self._loop.is_closed():
                    asyncio.run_coroutine_threadsafe(
                        self._cleanup_pending_call(call_id, "Execution failed"),
                        self._loop
                    )
                raise

    def _execute_stream_function(self, node_id: str, call_id: str, function_name: str, args_bytes: bytes, kwargs_bytes: bytes):
        """执行流式函数"""
        q = asyncio.Queue(maxsize=self.max_queue_size)
        
        # 创建流上下文
        stream_ctx = StreamContext(call_id, function_name, node_id, q)
        
        async def register_stream():
            async with self._lock:
                self._pending_calls[call_id] = {
                    'queue': q, 
                    'function_name': function_name, 
                    'node_id': node_id,
                    'created_at': datetime.now()
                }
                self._stream_contexts[call_id] = stream_ctx
                self._active_streams.add(call_id)

        if not self._loop or self._loop.is_closed():
            raise EasyRemoteError("Server not started or loop is closed")

        try:
            # 注册流上下文
            asyncio.run_coroutine_threadsafe(register_stream(), self._loop).result(timeout=5)
            
            # 发送执行请求
            asyncio.run_coroutine_threadsafe(
                self._request_execution(node_id, call_id, function_name, args_bytes, kwargs_bytes, is_stream=True),
                self._loop
            )
        except Exception as e:
            # 清理注册的流上下文
            if self._loop and not self._loop.is_closed():
                asyncio.run_coroutine_threadsafe(
                    self._cleanup_stream_context(call_id),
                    self._loop
                )
            raise RemoteExecutionError(
                function_name=function_name,
                node_id=node_id,
                message="Failed to request execution",
                cause=e
            ) from e

        async def async_generator():
            try:
                while call_id in self._active_streams:
                    try:
                        chunk = await asyncio.wait_for(q.get(), timeout=30.0)
                        if chunk is _SENTINEL:
                            break
                        if isinstance(chunk, Exception):
                            raise chunk
                        yield chunk
                    except asyncio.TimeoutError:
                        self.warning(f"Stream {call_id} timeout")
                        break
                    except Exception as e:
                        self.error(f"Error in stream generator: {e}")
                        raise
            finally:
                # 确保清理
                if self._loop and not self._loop.is_closed():
                    asyncio.run_coroutine_threadsafe(
                        self._cleanup_stream_context(call_id),
                        self._loop
                    )

        return async_generator()

    async def _request_execution(self, node_id, call_id, function_name, args_bytes, kwargs_bytes, is_stream: bool):
        """发送执行请求并处理响应"""
        async with self._lock:
            if node_id not in self._node_queues:
                raise EasyRemoteError(f"Node {node_id} not connected")
            
            node_queue = self._node_queues[node_id]

        if not is_stream:
            fut = asyncio.Future()
            async with self._lock:
                self._pending_calls[call_id] = fut

        req = service_pb2.ControlMessage(
            exec_req=service_pb2.ExecutionRequest(
                function_name=function_name,
                args=args_bytes,
                kwargs=kwargs_bytes,
                call_id=call_id
            )
        )

        try:
            await node_queue.put(req)

            if not is_stream:
                result = await fut
                return result
        except Exception as e:
            # 清理相关资源
            async with self._lock:
                self._pending_calls.pop(call_id, None)
            raise EasyRemoteError("Failed to send execution request") from e

    @staticmethod
    def current() -> 'Server':
        """获取当前服务器实例"""
        if Server._instance is None:
            raise EasyRemoteError("No Server instance available")
        return Server._instance