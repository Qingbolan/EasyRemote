import threading
import asyncio
import time
import grpc
from concurrent import futures
from typing import Dict
from datetime import datetime, timedelta
import uuid
import asyncio

_SENTINEL = object()

from .types import NodeInfo, FunctionInfo
from .exceptions import (
    NodeNotFoundError, 
    FunctionNotFoundError,
    RemoteExecutionError
)
from .utils import (
    serialize_args,
    deserialize_result,
    setup_logger
)
from .protos import service_pb2, service_pb2_grpc

logger = setup_logger(__name__)

class Server(service_pb2_grpc.RemoteServiceServicer):
    """使用ControlStream双向流实现的VPS服务器，支持普通和流式函数调用"""
    
    _instance = None  # 单例模式
    
    def __init__(self, port: int = 8080, heartbeat_timeout: int = 5):
        """初始化服务器实例"""
        logger.debug(f"Initializing Server instance on port {port} with heartbeat timeout {heartbeat_timeout}s")
        self.port = port
        self.heartbeat_timeout = heartbeat_timeout
        self._nodes: Dict[str, NodeInfo] = {}
        self._running = False
        self._node_queues: Dict[str, asyncio.Queue] = {}
        self._pending_calls = {}
        self._server = None
        self._loop = None
        self._monitor_thread = None
        Server._instance = self
        logger.debug("Server instance initialized")

    def start(self):
        """在主线程中启动服务器（阻塞模式）"""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())

    def start_background(self):
        """在后台线程中启动服务器（非阻塞模式）"""
        def run_server():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            try:
                self._loop.run_until_complete(self._serve())
            except Exception as e:
                logger.error(f"Server error: {e}")
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
        self._server = grpc.aio.server(
            futures.ThreadPoolExecutor(max_workers=10),
            options=[
                ('grpc.max_send_message_length', 50 * 1024 * 1024),
                ('grpc.max_receive_message_length', 50 * 1024 * 1024)
            ]
        )
        service_pb2_grpc.add_RemoteServiceServicer_to_server(self, self._server)
        
        try:
            addr = f'[::]:{self.port}'
            self._server.add_insecure_port(addr)
            await self._server.start()
            logger.info(f"Server started on {addr}")
            
            self._start_node_monitor()
            
            while self._running:
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Server error: {e}")
            self._running = False
        finally:
            if self._server:
                await self._server.stop(grace=None)
            logger.info("Server stopped")

    def _start_node_monitor(self):
        """启动节点监控线程"""
        def monitor():
            while self._running:
                try:
                    now = datetime.now()
                    timeout = timedelta(seconds=self.heartbeat_timeout)
                    
                    for node_id, node in list(self._nodes.items()):
                        time_since = now - node.last_heartbeat
                        if time_since > timeout:
                            logger.warning(f"Node {node_id} timed out, removing")
                            self._nodes.pop(node_id, None)
                            self._node_queues.pop(node_id, None)
                    
                    time.sleep(self.heartbeat_timeout / 2)
                except Exception as e:
                    logger.error(f"Monitor error: {e}")
                    if not self._running:
                        break
                    time.sleep(1)

        self._monitor_thread = threading.Thread(target=monitor, daemon=True)
        self._monitor_thread.start()

    async def stop(self):
        """停止服务器"""
        self._running = False
        if self._server:
            await self._server.stop(grace=None)
        
        # 清理所有连接
        for node_id in list(self._node_queues.keys()):
            self._node_queues.pop(node_id, None)
            self._nodes.pop(node_id, None)
        
        logger.info("Server stopped")

    def stop_sync(self):
        """同步方式停止服务器"""
        if self._loop and not self._loop.is_closed():
            asyncio.run_coroutine_threadsafe(self.stop(), self._loop).result(timeout=5)

    async def ControlStream(self, request_iterator, context):
        node_id = None
        out_queue = asyncio.Queue()

        async def read_requests():
            nonlocal node_id
            try:
                async for msg in request_iterator:
                    if msg.HasField("register_req"):
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
                        logger.info(f"Node {node_id} registered with functions: {list(functions.keys())}")

                        await out_queue.put(service_pb2.ControlMessage(
                            register_resp=service_pb2.RegisterResponse(
                                success=True,
                                message="Registered successfully"
                            )
                        ))

                    elif msg.HasField("heartbeat_req"):
                        req = msg.heartbeat_req
                        if req.node_id in self._nodes:
                            self._nodes[req.node_id].last_heartbeat = datetime.now()
                            await out_queue.put(service_pb2.ControlMessage(
                                heartbeat_resp=service_pb2.HeartbeatResponse(accepted=True)
                            ))
                        else:
                            await out_queue.put(service_pb2.ControlMessage(
                                heartbeat_resp=service_pb2.HeartbeatResponse(accepted=False)
                            ))

                    elif msg.HasField("exec_res"):
                        res = msg.exec_res
                        await self._handle_execution_result(res)

            except Exception as e:
                logger.error(f"Error in ControlStream: {e}", exc_info=True)
            finally:
                # 当客户端断开或请求结束时进行清理
                if node_id and node_id in self._nodes:
                    logger.info(f"Node {node_id} disconnected")
                    self._nodes.pop(node_id, None)
                    self._node_queues.pop(node_id, None)
                
                # 向输出队列发送None表示数据流结束
                await out_queue.put(None)

        # 后台任务：处理请求并将响应放入队列中
        reader_task = asyncio.create_task(read_requests())

        # 在此处直接使用 async for 或 while 循环从 out_queue 中拿消息并 yield
        try:
            while True:
                msg = await out_queue.get()
                if msg is None:
                    # None表示数据发送完成
                    break
                yield msg
        finally:
            # 确保后台任务结束
            await reader_task

    async def _handle_execution_result(self, res):
        """处理执行结果"""
        call_id = res.call_id
        if call_id in self._pending_calls:
            call_ctx = self._pending_calls[call_id]
            if isinstance(call_ctx, asyncio.Future):
                if res.has_error:
                    call_ctx.set_exception(RemoteExecutionError(res.error_message))
                else:
                    call_ctx.set_result(res.result if res.result else None)
                self._pending_calls.pop(call_id, None)
            else:
                q = call_ctx['queue']
                if res.has_error:
                    q.put_nowait(RemoteExecutionError(res.error_message))
                    self._pending_calls.pop(call_id, None)
                else:
                    if res.chunk:
                        q.put_nowait(res.chunk)
                    if res.is_done:
                        q.put_nowait(_SENTINEL)
                        self._pending_calls.pop(call_id, None)

    def execute_function(self, node_id: str, function_name: str, *args, **kwargs):
        """执行远程函数"""
        if node_id not in self._nodes:
            raise NodeNotFoundError(f"Node {node_id} not found")

        node = self._nodes[node_id]
        if function_name not in node.functions:
            raise FunctionNotFoundError(f"Function {function_name} not found on node {node_id}")

        func_info = node.functions[function_name]
        is_stream = func_info.is_generator
        call_id = str(uuid.uuid4())
        args_bytes, kwargs_bytes = serialize_args(*args, **kwargs)

        if is_stream:
            return self._execute_stream_function(node_id, call_id, function_name, args_bytes, kwargs_bytes)
        else:
            if not self._loop:
                raise RuntimeError("Server not started")
            fut = asyncio.run_coroutine_threadsafe(
                self._request_execution(node_id, call_id, function_name, args_bytes, kwargs_bytes, is_stream=False),
                self._loop
            )
            try:
                result = fut.result(timeout=30)
                return deserialize_result(result) if result is not None else None
            except Exception as e:
                logger.error(f"Error executing function {function_name}: {e}")
                raise RemoteExecutionError(str(e))

    def _execute_stream_function(self, node_id: str, call_id: str, function_name: str, args_bytes: bytes, kwargs_bytes: bytes):
        """执行流式函数"""
        q = asyncio.Queue()
        self._pending_calls[call_id] = {'queue': q}

        if not self._loop:
            raise RuntimeError("Server not started")

        asyncio.run_coroutine_threadsafe(
            self._request_execution(node_id, call_id, function_name, args_bytes, kwargs_bytes, is_stream=True),
            self._loop
        )

        async def async_generator():
            while True:
                chunk = await q.get()
                if chunk is _SENTINEL:
                    break
                if isinstance(chunk, Exception):
                    raise chunk
                yield deserialize_result(chunk)

        return async_generator()

    async def _request_execution(self, node_id, call_id, function_name, args_bytes, kwargs_bytes, is_stream: bool):
        """发送执行请求"""
        if node_id not in self._node_queues:
            raise ConnectionError(f"Node {node_id} not connected")

        if not is_stream:
            fut = asyncio.Future()
            self._pending_calls[call_id] = fut

        req = service_pb2.ControlMessage(
            exec_req=service_pb2.ExecutionRequest(
                function_name=function_name,
                args=args_bytes,
                kwargs=kwargs_bytes,
                call_id=call_id
            )
        )
        await self._node_queues[node_id].put(req)

        if not is_stream:
            return await fut

    @staticmethod
    def current() -> 'Server':
        """获取当前服务器实例"""
        if Server._instance is None:
            raise RuntimeError("No Server instance available")
        return Server._instance

if __name__ == "__main__":
    # 使用示例
    server = Server(port=8080)
    try:
        server.start()  # 阻塞模式
    except KeyboardInterrupt:
        server.stop_sync()
