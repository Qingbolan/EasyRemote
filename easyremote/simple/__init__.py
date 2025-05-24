#!/usr/bin/env python3
"""
EasyRemote 简化版 - 极简API设计
目标：将API复杂度从5种方式简化到2种核心方式

设计原则：
1. 零配置 - 所有参数都有智能默认值
2. 最小API - 只保留最核心的方法
3. 自动优化 - 内置最佳实践
"""

# 导入完整版本的核心组件
from ..core.nodes.server import Server as _FullServer
from ..core.nodes.compute_node import ComputeNode as _FullComputeNode
from ..core.nodes.client import Client as _FullClient
from ..decorators import remote as _full_remote


class Server:
    """简化版Server - 零配置启动"""
    
    def __init__(self, port: int = 8080):
        self._server = _FullServer(port=port)
        
    def start(self):
        """启动服务器 - 自动优化配置"""
        return self._server.start()
    
    def start_background(self):
        """后台启动 - 最常用方式"""
        return self._server.start_background()


class ComputeNode:
    """简化版ComputeNode - 智能默认配置"""
    
    def __init__(self, gateway_address: str = "localhost:8080", node_id: str = None):
        # 自动生成node_id
        if node_id is None:
            import uuid
            node_id = f"node-{str(uuid.uuid4())[:8]}"
        
        self._node = _FullComputeNode(gateway_address, node_id)
    
    def register(self, func=None):
        """注册函数 - 默认启用负载均衡"""
        if func is None:
            # 装饰器模式
            def decorator(f):
                return self._node.register(load_balancing=True)(f)
            return decorator
        else:
            # 直接调用模式
            return self._node.register(load_balancing=True)(func)
    
    def serve(self, blocking: bool = False):
        """启动节点服务"""
        return self._node.serve(blocking=blocking)


class Client:
    """简化版Client - 只保留核心方法"""
    
    def __init__(self, gateway_address: str = "localhost:8080"):
        self._client = _FullClient(gateway_address)
    
    def call(self, function_name: str, *args, **kwargs):
        """调用远程函数 - 自动负载均衡，智能重试"""
        return self._client.call(function_name, *args, **kwargs)
    
    def connect(self):
        """连接到网关"""
        return self._client.connect()
    
    def disconnect(self):
        """断开连接"""
        return self._client.disconnect()
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


def remote(func):
    """简化版@remote装饰器 - 默认启用负载均衡"""
    return _full_remote(load_balancing=True)(func)


# 便捷函数 - 进一步简化使用
def quick_start(port: int = 8080):
    """一键启动 - 极简使用"""
    server = Server(port)
    return server.start_background()


def quick_node(gateway: str = "localhost:8080", node_id: str = None):
    """快速创建节点"""
    return ComputeNode(gateway, node_id)


def quick_client(gateway: str = "localhost:8080"):
    """快速创建客户端"""
    return Client(gateway)


# 导出简化API
__all__ = [
    "Server", 
    "ComputeNode", 
    "Client", 
    "remote",
    "quick_start",
    "quick_node", 
    "quick_client"
]

# 版本信息
__version__ = "0.1.0-simple" 