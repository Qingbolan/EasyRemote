# easyremote/core/nodes/client.py
import grpc
import time
from typing import Optional, Dict, Any, List
import uuid

from ..utils.logger import ModernLogger
from ..utils.exceptions import (
    ConnectionError as EasyRemoteConnectionError,
    RemoteExecutionError,
    EasyRemoteError,
    NoAvailableNodesError
)
from ..data.serialize import serialize_args, deserialize_result
from ..protos import service_pb2, service_pb2_grpc
from ..load_balancing.strategies import RequestContext


class Client(ModernLogger):
    """
    EasyRemote客户端，负责发起远程计算请求
    
    这是三角色架构中的第三个角色：
    1. ComputeNode - 计算节点（提供计算能力）
    2. Server - VPS网关（负载均衡和路由）
    3. Client - 客户端（发起计算请求）
    """
    
    def __init__(
        self, 
        gateway_address: str,
        client_id: Optional[str] = None,
        connection_timeout: int = 10,
        request_timeout: int = 300,
        retry_attempts: int = 3,
        enable_load_balancing: bool = True
    ):
        """
        初始化客户端
        
        Args:
            gateway_address: VPS网关地址
            client_id: 客户端ID
            connection_timeout: 连接超时时间
            request_timeout: 请求超时时间
            retry_attempts: 重试次数
            enable_load_balancing: 是否启用负载均衡
        """
        super().__init__(name="EasyRemoteClient")
        self.gateway_address = gateway_address
        self.client_id = client_id or f"client-{uuid.uuid4()}"
        self.connection_timeout = connection_timeout
        self.request_timeout = request_timeout
        self.retry_attempts = retry_attempts
        self.enable_load_balancing = enable_load_balancing
        
        self._gateway_channel: Optional[grpc.Channel] = None
        self._gateway_stub: Optional[service_pb2_grpc.RemoteServiceStub] = None
        self._connected = False
        
        self.info(f"Client {self.client_id} initialized, target gateway: {gateway_address}")
    
    def connect(self):
        """连接到VPS网关"""
        try:
            # 配置gRPC通道
            options = [
                ('grpc.keepalive_time_ms', 30000),
                ('grpc.keepalive_timeout_ms', 5000),
                ('grpc.keepalive_permit_without_calls', True),
                ('grpc.max_send_message_length', 50 * 1024 * 1024),
                ('grpc.max_receive_message_length', 50 * 1024 * 1024),
            ]
            
            self._gateway_channel = grpc.insecure_channel(self.gateway_address, options=options)
            self._gateway_stub = service_pb2_grpc.RemoteServiceStub(self._gateway_channel)
            
            # 测试连接
            try:
                # 发送ping请求测试连接
                future = self._gateway_channel.channel_ready()
                grpc.channel_ready_future = future
                future.result(timeout=self.connection_timeout)
                self._connected = True
                self.info(f"Connected to gateway at {self.gateway_address}")
                
            except grpc.FutureTimeoutError:
                raise EasyRemoteConnectionError(
                    f"Connection timeout to gateway {self.gateway_address}",
                    address=self.gateway_address,
                    timeout=self.connection_timeout
                )
                
        except Exception as e:
            self.error(f"Failed to connect to gateway: {e}")
            raise EasyRemoteConnectionError(
                f"Failed to connect to gateway {self.gateway_address}",
                cause=e
            )
    
    def disconnect(self):
        """断开与网关的连接"""
        if self._gateway_channel:
            try:
                self._gateway_channel.close()
                self._connected = False
                self.info("Disconnected from gateway")
            except Exception as e:
                self.warning(f"Error during disconnect: {e}")
    
    def call(
        self, 
        function_name: str, 
        *args, 
        **kwargs
    ) -> Any:
        """
        调用远程函数（不指定节点，使用负载均衡）
        
        Args:
            function_name: 函数名称
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            函数执行结果
        """
        return self.call_with_config(function_name, {}, *args, **kwargs)
    
    def call_with_config(
        self,
        function_name: str,
        config: Dict[str, Any],
        *args,
        **kwargs
    ) -> Any:
        """
        使用指定配置调用远程函数
        
        Args:
            function_name: 函数名称
            config: 调用配置（负载均衡策略、要求等）
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            函数执行结果
        """
        if not self._connected:
            self.connect()
        
        # 创建请求上下文
        request_context = RequestContext(
            function_name=function_name,
            data_size=len(str(args)) + len(str(kwargs)),
            complexity_score=config.get("complexity_score", 1.0),
            requirements=config.get("requirements"),
            client_location=config.get("client_location"),
            priority=config.get("priority", "normal"),
            timeout=config.get("timeout", self.request_timeout),
            cost_limit=config.get("cost_limit")
        )
        
        # 序列化参数
        try:
            args_bytes, kwargs_bytes = serialize_args(*args, **kwargs)
        except Exception as e:
            raise EasyRemoteError(f"Failed to serialize arguments: {e}", cause=e)
        
        # 创建调用ID
        call_id = str(uuid.uuid4())
        
        # 发起远程调用
        for attempt in range(self.retry_attempts):
            try:
                if self.enable_load_balancing:
                    result = self._call_with_load_balancing(
                        call_id, function_name, args_bytes, kwargs_bytes, config
                    )
                else:
                    # 直接调用（需要指定node_id）
                    node_id = config.get("node_id")
                    if not node_id:
                        raise EasyRemoteError("node_id required when load_balancing is disabled")
                    result = self._call_direct(
                        call_id, node_id, function_name, args_bytes, kwargs_bytes
                    )
                
                self.debug(f"Function {function_name} executed successfully")
                return result
                
            except Exception as e:
                self.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == self.retry_attempts - 1:
                    raise
                time.sleep(2 ** attempt)  # 指数退避
    
    def call_node(
        self,
        node_id: str,
        function_name: str,
        *args,
        **kwargs
    ) -> Any:
        """
        直接调用指定节点的函数
        
        Args:
            node_id: 目标节点ID
            function_name: 函数名称
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            函数执行结果
        """
        config = {"node_id": node_id}
        return self.call_with_config(function_name, config, *args, **kwargs)
    
    def call_stream(
        self,
        function_name: str,
        *args,
        **kwargs
    ):
        """
        调用流式函数
        
        Args:
            function_name: 流式函数名称
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            异步生成器
        """
        if not self._connected:
            self.connect()
        
        # 序列化参数
        try:
            args_bytes, kwargs_bytes = serialize_args(*args, **kwargs)
        except Exception as e:
            raise EasyRemoteError(f"Failed to serialize arguments: {e}", cause=e)
        
        call_id = str(uuid.uuid4())
        return self._call_stream_function(call_id, function_name, args_bytes, kwargs_bytes)
    
    def _call_with_load_balancing(
        self,
        call_id: str,
        function_name: str,
        args_bytes: bytes,
        kwargs_bytes: bytes,
        config: Dict[str, Any]
    ) -> Any:
        """使用负载均衡调用函数"""
        # 创建负载均衡请求
        request = service_pb2.LoadBalancedCallRequest(
            call_id=call_id,
            function_name=function_name,
            args=args_bytes,
            kwargs=kwargs_bytes,
            strategy=config.get("strategy", "resource_aware"),
            requirements=str(config.get("requirements", {})),
            timeout=config.get("timeout", self.request_timeout)
        )
        
        try:
            # 发送请求
            response = self._gateway_stub.CallWithLoadBalancing(
                request, 
                timeout=self.request_timeout
            )
            
            if response.has_error:
                raise RemoteExecutionError(
                    function_name=function_name,
                    node_id=response.selected_node_id,
                    message=response.error_message
                )
            
            # 反序列化结果
            result = deserialize_result(response.result)
            
            self.info(f"Load-balanced call successful: {function_name} -> {response.selected_node_id}")
            return result
            
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                raise NoAvailableNodesError(
                    f"No nodes available for function '{function_name}'",
                    function_name=function_name
                )
            raise EasyRemoteConnectionError(f"gRPC error: {e}", cause=e)
    
    def _call_direct(
        self,
        call_id: str,
        node_id: str,
        function_name: str,
        args_bytes: bytes,
        kwargs_bytes: bytes
    ) -> Any:
        """直接调用指定节点"""
        request = service_pb2.DirectCallRequest(
            call_id=call_id,
            node_id=node_id,
            function_name=function_name,
            args=args_bytes,
            kwargs=kwargs_bytes
        )
        
        try:
            response = self._gateway_stub.CallDirect(request, timeout=self.request_timeout)
            
            if response.has_error:
                raise RemoteExecutionError(
                    function_name=function_name,
                    node_id=node_id,
                    message=response.error_message
                )
            
            result = deserialize_result(response.result)
            return result
            
        except grpc.RpcError as e:
            raise EasyRemoteConnectionError(f"Direct call failed: {e}", cause=e)
    
    def _call_stream_function(
        self,
        call_id: str,
        function_name: str,
        args_bytes: bytes,
        kwargs_bytes: bytes
    ):
        """调用流式函数"""
        request = service_pb2.StreamCallRequest(
            call_id=call_id,
            function_name=function_name,
            args=args_bytes,
            kwargs=kwargs_bytes
        )
        
        try:
            response_stream = self._gateway_stub.CallStream(request)
            
            async def async_generator():
                try:
                    for response in response_stream:
                        if response.has_error:
                            raise RemoteExecutionError(
                                function_name=function_name,
                                message=response.error_message
                            )
                        
                        if response.is_done:
                            break
                            
                        if response.chunk:
                            chunk = deserialize_result(response.chunk)
                            yield chunk
                            
                except grpc.RpcError as e:
                    raise EasyRemoteConnectionError(f"Stream call failed: {e}", cause=e)
            
            return async_generator()
            
        except grpc.RpcError as e:
            raise EasyRemoteConnectionError(f"Stream call failed: {e}", cause=e)
    
    def list_nodes(self) -> List[Dict[str, Any]]:
        """获取可用节点列表"""
        if not self._connected:
            self.connect()
        
        try:
            request = service_pb2.ListNodesRequest(client_id=self.client_id)
            response = self._gateway_stub.ListNodes(request)
            
            nodes = []
            for node_info in response.nodes:
                nodes.append({
                    "node_id": node_info.node_id,
                    "functions": list(node_info.functions),
                    "status": node_info.status,
                    "last_heartbeat": node_info.last_heartbeat,
                    "load": node_info.current_load
                })
            
            return nodes
            
        except grpc.RpcError as e:
            raise EasyRemoteConnectionError(f"Failed to list nodes: {e}", cause=e)
    
    def get_node_status(self, node_id: str) -> Dict[str, Any]:
        """获取指定节点的状态"""
        if not self._connected:
            self.connect()
        
        try:
            request = service_pb2.NodeStatusRequest(
                client_id=self.client_id,
                node_id=node_id
            )
            response = self._gateway_stub.GetNodeStatus(request)
            
            return {
                "node_id": response.node_id,
                "status": response.status,
                "cpu_usage": response.cpu_usage,
                "memory_usage": response.memory_usage,
                "gpu_usage": response.gpu_usage,
                "current_load": response.current_load,
                "functions": list(response.functions)
            }
            
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                raise NoAvailableNodesError(f"Node '{node_id}' not found")
            raise EasyRemoteConnectionError(f"Failed to get node status: {e}", cause=e)
    
    def __enter__(self):
        """支持上下文管理器"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文管理器时断开连接"""
        self.disconnect()


# 便捷的函数式接口
_default_client: Optional[Client] = None

def set_default_gateway(gateway_address: str, **kwargs):
    """设置默认网关地址"""
    global _default_client
    _default_client = Client(gateway_address, **kwargs)
    _default_client.connect()

def call(function_name: str, *args, **kwargs):
    """使用默认客户端调用函数"""
    if _default_client is None:
        raise EasyRemoteError("No default gateway set. Call set_default_gateway() first.")
    return _default_client.call(function_name, *args, **kwargs)

def call_node(node_id: str, function_name: str, *args, **kwargs):
    """使用默认客户端调用指定节点"""
    if _default_client is None:
        raise EasyRemoteError("No default gateway set. Call set_default_gateway() first.")
    return _default_client.call_node(node_id, function_name, *args, **kwargs)

def list_nodes():
    """使用默认客户端列出节点"""
    if _default_client is None:
        raise EasyRemoteError("No default gateway set. Call set_default_gateway() first.")
    return _default_client.list_nodes() 