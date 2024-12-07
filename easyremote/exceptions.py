# easyremote/exceptions.py
# easyremote/exceptions.py
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class EasyRemoteError(Exception):
    """EasyRemote 基础异常类"""
    
    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(message)
        self.cause = cause
        logger.debug(f"Creating {self.__class__.__name__}: {message}")
        if cause:
            logger.debug(f"Caused by: {cause.__class__.__name__}: {str(cause)}")

    def __str__(self) -> str:
        msg = super().__str__()
        if self.cause:
            return f"{msg} (caused by {self.cause.__class__.__name__}: {str(self.cause)})"
        return msg

class NodeNotFoundError(EasyRemoteError):
    """找不到指定节点时抛出此异常
    
    属性:
        node_id: 找不到的节点ID
        message: 错误描述
    """
    
    def __init__(self, node_id: str, message: Optional[str] = None):
        self.node_id = node_id
        msg = message or f"Node not found: {node_id}"
        super().__init__(msg)
        logger.debug(f"Node not found error for node_id: {node_id}")

class FunctionNotFoundError(EasyRemoteError):
    """找不到指定函数时抛出此异常
    
    属性:
        function_name: 找不到的函数名
        node_id: 相关的节点ID（如果有）
        message: 错误描述
    """
    
    def __init__(self, function_name: str, node_id: Optional[str] = None, message: Optional[str] = None):
        self.function_name = function_name
        self.node_id = node_id
        msg = message or f"Function not found: {function_name}" + (f" on node {node_id}" if node_id else "")
        super().__init__(msg)
        logger.debug(f"Function not found error: {function_name} (node: {node_id})")

class ConnectionError(EasyRemoteError):
    """连接相关错误时抛出此异常
    
    属性:
        address: 连接地址
        message: 错误描述
        cause: 原始异常（如果有）
    """
    
    def __init__(self, address: str, message: Optional[str] = None, cause: Optional[Exception] = None):
        self.address = address
        msg = message or f"Connection error to {address}"
        super().__init__(msg, cause)
        logger.debug(f"Connection error for address: {address}")

class SerializationError(EasyRemoteError):
    """序列化或反序列化错误时抛出此异常
    
    属性:
        operation: 'serialize' 或 'deserialize'
        data_type: 数据类型描述
        message: 错误描述
        cause: 原始异常（如果有）
    """
    
    def __init__(
        self, 
        operation: str,
        data_type: str,
        message: Optional[str] = None,
        cause: Optional[Exception] = None
    ):
        self.operation = operation
        self.data_type = data_type
        msg = message or f"{operation.capitalize()} error for {data_type}"
        super().__init__(msg, cause)
        logger.debug(f"Serialization error: {operation} failed for {data_type}")

class RemoteExecutionError(EasyRemoteError):
    """远程执行错误时抛出此异常
    
    属性:
        function_name: 执行失败的函数名
        node_id: 相关的节点ID（如果有）
        message: 错误描述
        cause: 原始异常（如果有）
    """
    
    def __init__(
        self,
        function_name: str,
        node_id: Optional[str] = None,
        message: Optional[str] = None,
        cause: Optional[Exception] = None
    ):
        self.function_name = function_name
        self.node_id = node_id
        msg = message or f"Remote execution failed for {function_name}" + (f" on node {node_id}" if node_id else "")
        super().__init__(msg, cause)
        logger.debug(f"Remote execution error: {function_name} (node: {node_id})")

def format_exception(e: Exception) -> str:
    """格式化异常信息，用于日志记录和错误报告
    
    Args:
        e: 异常对象
        
    Returns:
        格式化后的异常信息字符串
    """
    if isinstance(e, EasyRemoteError):
        return str(e)
    return f"{e.__class__.__name__}: {str(e)}"