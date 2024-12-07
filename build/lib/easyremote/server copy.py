import threading
import asyncio
import time
import grpc
from concurrent import futures
import logging
from typing import Dict, Optional, Any, Set
from datetime import datetime, timedelta

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
    """VPS服务器，负责管理计算节点和路由请求"""
    
    _instance = None  # 单例模式
    
    def __init__(self, port: int = 8080, heartbeat_timeout: int = 5):
        logger.debug(f"Initializing Server instance on port {port} with heartbeat timeout {heartbeat_timeout}s")
        self.port = port
        self.heartbeat_timeout = heartbeat_timeout
        self._nodes: Dict[str, NodeInfo] = {}
        self._server = None
        self._background_thread = None
        self._running = False
        self._active_calls: Set[str] = set()
        self._loop = None
        Server._instance = self
        logger.debug("Server instance initialized")
        
    @staticmethod
    def current() -> 'Server':
        if Server._instance is None:
            raise RuntimeError("No Server instance available")
        return Server._instance
        
    def start_background(self):
        """在后台启动服务器"""
        def run_server():
            self._running = True
            
            # 创建gRPC服务器
            self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
            service_pb2_grpc.add_RemoteServiceServicer_to_server(self, self._server)
            self._server.add_insecure_port(f'[::]:{self.port}')
            self._server.start()
            logger.info(f"Server started on port {self.port}")
            
            # 创建事件循环，支持异步调用节点
            self._loop = asyncio.new_event_loop()
            threading.Thread(target=self._run_loop, daemon=True).start()
            
            # 启动节点监控
            self._start_node_monitor()
            
            # 保持运行
            try:
                while self._running:
                    time.sleep(1)
            except KeyboardInterrupt:
                self.stop()
                
        self._background_thread = threading.Thread(target=run_server, daemon=True)
        self._background_thread.start()
        
        # 等待服务器启动
        time.sleep(1)
        
    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()
        
    def stop(self):
        """停止服务器"""
        self._running = False
        if self._server:
            self._server.stop(0)
        logger.info("Server stopped")
        
    def Call(self, request, context):
        """处理同步调用请求"""
        try:
            node_id = self._get_node_id(request.node_id, request.function_name)
            # 同步调用远程函数
            result = self.sync_call(
                request.function_name,
                args=(),
                kwargs={}, 
                node_id=node_id
            )
            return service_pb2.CallResponse(result=result)
        except Exception as e:
            logger.error(f"Error in Call: {e}", exc_info=True)
            return service_pb2.CallResponse(
                has_error=True,
                error_message=str(e)
            )
            
    def StreamingCall(self, request, context):
        """处理流式调用请求"""
        try:
            node_id = self._get_node_id(request.node_id, request.function_name)
            for chunk in self._stream_from_node(node_id, request.function_name, (), {}):
                yield service_pb2.StreamChunk(data=chunk)
        except Exception as e:
            logger.error(f"Error in StreamingCall: {e}", exc_info=True)
            yield service_pb2.StreamChunk(
                has_error=True,
                error_message=str(e)
            )
            
    async def async_call(
        self,
        function_name: str,
        args: tuple = (),
        kwargs: dict = None,
        node_id: Optional[str] = None,
        timeout: Optional[float] = None
    ) -> Any:
        """异步调用远程函数"""
        node_id = self._get_node_id(node_id, function_name)
        kwargs = kwargs or {}
        
        args_bytes, kwargs_bytes = serialize_args(*args, **kwargs)
        call_id = f"{node_id}:{function_name}:{id(args)}"
        
        try:
            self._active_calls.add(call_id)
            result = await self._execute_on_node_async(node_id, function_name, args_bytes, kwargs_bytes, timeout)
            return deserialize_result(result)
        finally:
            self._active_calls.remove(call_id)
            
    def sync_call(
        self,
        function_name: str,
        args: tuple = (),
        kwargs: dict = None,
        node_id: Optional[str] = None,
        timeout: Optional[float] = None
    ) -> Any:
        """同步调用远程函数"""
        future = asyncio.run_coroutine_threadsafe(
            self.async_call(function_name, args, kwargs, node_id, timeout),
            self._loop
        )
        return future.result(timeout=timeout)
        
    def _stream_from_node(
        self,
        node_id: str,
        function_name: str,
        args: tuple = (),
        kwargs: dict = None,
        timeout: Optional[float] = None
    ):
        """同步流式调用"""
        if kwargs is None:
            kwargs = {}
        args_bytes, kwargs_bytes = serialize_args(*args, **kwargs)
        request = service_pb2.StreamRequest(
            node_id=node_id,
            function_name=function_name,
            args=args_bytes,
            kwargs=kwargs_bytes
        )
        
        stub = self._nodes[node_id].stub
        call = stub.StreamingCall(request, timeout=timeout or 60)
        for chunk in call:
            if chunk.has_error:
                raise RemoteExecutionError(chunk.error_message)
            yield chunk.data
        
    def _get_node_id(self, node_id: Optional[str], function_name: str) -> str:
        """获取要使用的节点ID"""
        if node_id:
            if node_id not in self._nodes:
                raise NodeNotFoundError(f"Node {node_id} not found")
            if function_name not in self._nodes[node_id].functions:
                raise FunctionNotFoundError(
                    f"Function {function_name} not found on node {node_id}"
                )
            return node_id
            
        # 查找提供该函数的节点
        available_nodes = [
            nid for nid, node in self._nodes.items()
            if function_name in node.functions
        ]
        
        if not available_nodes:
            raise FunctionNotFoundError(f"No node provides function {function_name}")
            
        return available_nodes[0]
        
    def Register(self, request, context):
        """处理节点注册请求"""
        node_id = request.node_id
        node_address = request.node_address  # Node在注册时提供自己的服务地址

        logger.debug(f"Processing registration request for node {request.node_id}")
        functions = {}
        
        for func_spec in request.functions:
            logger.debug(f"Registering function {func_spec.name} for node {node_id}")
            functions[func_spec.name] = FunctionInfo(
                name=func_spec.name,
                callable=None,
                is_async=func_spec.is_async,
                is_generator=func_spec.is_generator,
                node_id=node_id
            )
            
        now = datetime.now()
        
        # 创建到Node的连接stub
        channel = grpc.insecure_channel(node_address)
        stub = service_pb2_grpc.RemoteServiceStub(channel)
        
        self._nodes[node_id] = NodeInfo(
            node_id=node_id,
            functions=functions,
            last_heartbeat=now,
            address=node_address,
            stub=stub
        )
        
        logger.info(f"Node {node_id} registered with functions: {list(functions.keys())}")
        return service_pb2.RegisterResponse(success=True)

    def _start_node_monitor(self):
        """启动节点监控"""
        logger.debug("Starting node monitor thread")
        
        def monitor():
            logger.debug("Node monitor started")
            time.sleep(self.heartbeat_timeout)
            
            while self._running:
                try:
                    now = datetime.now()
                    timeout = timedelta(seconds=self.heartbeat_timeout)
                    
                    for node_id, node in list(self._nodes.items()):
                        time_since_last_heartbeat = now - node.last_heartbeat
                        grace_period = timeout * 1.2
                        
                        if time_since_last_heartbeat > grace_period:
                            logger.warning(
                                f"Node {node_id} timed out: {time_since_last_heartbeat.total_seconds():.2f}s "
                                f"exceeds timeout of {grace_period.total_seconds():.2f}s"
                            )
                            del self._nodes[node_id]
                        else:
                            logger.debug(
                                f"Node {node_id} healthy: {time_since_last_heartbeat.total_seconds():.2f}s "
                                f"within timeout of {grace_period.total_seconds():.2f}s"
                            )
                    time.sleep(timeout.total_seconds() / 2)
                except Exception as e:
                    logger.error(f"Error in node monitor: {e}", exc_info=True)
                    time.sleep(1)
                    
        threading.Thread(target=monitor, daemon=True).start()
        logger.debug(f"Node monitor thread started with timeout {self.heartbeat_timeout}s")
        
    def Heartbeat(self, request, context):
        """处理心跳请求"""
        node_id = request.node_id
        logger.debug(f"Received heartbeat from node {node_id}")
        
        if node_id in self._nodes:
            now = datetime.now()
            last_heartbeat = self._nodes[node_id].last_heartbeat
            time_since_last = now - last_heartbeat
            
            self._nodes[node_id].last_heartbeat = now
            logger.debug(
                f"Heartbeat accepted for node {node_id} "
                f"(interval: {time_since_last.total_seconds():.2f}s)"
            )
            return service_pb2.HeartbeatResponse(accepted=True)
        
        logger.warning(f"Heartbeat rejected - unknown node {node_id}")
        return service_pb2.HeartbeatResponse(accepted=False)
        
    async def _execute_on_node_async(
        self,
        node_id: str,
        function_name: str,
        args_bytes: bytes,
        kwargs_bytes: bytes,
        timeout: Optional[float] = None
    ) -> bytes:
        """执行远程函数调用"""
        if node_id not in self._nodes:
            raise NodeNotFoundError(f"Node {node_id} not found")
        
        request = service_pb2.CallRequest(
            node_id=node_id,
            function_name=function_name,
            args=args_bytes,
            kwargs=kwargs_bytes
        )
        
        stub = self._nodes[node_id].stub
        try:
            timeout = timeout or 60
            # 使用线程池异步调用
            response = await self._loop.run_in_executor(
                None,
                lambda: stub.Call(request, timeout=timeout)
            )
            
            if response.has_error:
                raise RemoteExecutionError(response.error_message)
                
            return response.result
            
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                raise TimeoutError(f"Call to {function_name} timed out")
            raise ConnectionError(f"RPC failed: {e}")
