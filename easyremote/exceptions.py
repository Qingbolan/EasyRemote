# easyremote/exceptions.py
import logging
from typing import Optional
from rich.logging import RichHandler
from rich.console import Console
from rich.theme import Theme
import time

# 简化的颜色主题
custom_theme = Theme({
   "critical": "red bold",    # 严重错误
   "error": "red",           # 一般错误
   "warning": "yellow",      # 警告信息
})

# 错误类型编号
ERROR_CODES = {
   'NodeNotFoundError': 'E001',
   'FunctionNotFoundError': 'E002',
   'ConnectionError': 'E003',
   'SerializationError': 'E004',
   'RemoteExecutionError': 'E005'
}

logger = logging.getLogger(__name__)
level = logging.DEBUG

if not logger.handlers:
   console = Console(theme=custom_theme)
   handler = RichHandler(
       console=console,
       rich_tracebacks=True,
       markup=True,
       show_time=True,
       show_path=True
   )
   formatter = logging.Formatter("%(message)s")
   handler.setFormatter(formatter)
   handler.setLevel(level)
   logger.addHandler(handler)
   logger.setLevel(level)

error_counter = 0

def get_error_id():
   """生成错误ID"""
   global error_counter
   error_counter += 1
   timestamp = time.strftime("%Y%m%d")
   return f"ERR-{timestamp}-{error_counter:04d}"

class EasyRemoteError(Exception):
   """EasyRemote 基础异常类"""
   
   def __init__(self, message: str, cause: Optional[Exception] = None):
       super().__init__(message)
       self.cause = cause
       self.error_id = get_error_id()
       self.error_code = ERROR_CODES.get(self.__class__.__name__, 'E999')
       
       # 主错误信息
       logger.critical(f"⛔ [critical]{self.error_code} {self.error_id} - {self.__class__.__name__}[/critical]: {message}")
       
       # 原因信息（如果存在）
       if cause:
           logger.error(f"    ↳ [error]Caused by: {cause.__class__.__name__}[/error]: {str(cause)}")

   def __str__(self) -> str:
       error_str = f"{self.error_code} {self.error_id} - {super().__str__()}"
       if self.cause:
           error_str += f"\n    ↳ Caused by: {self.cause.__class__.__name__}: {str(self.cause)}"
       return error_str

class NodeNotFoundError(EasyRemoteError):
   """找不到指定节点时抛出此异常"""
   
   def __init__(self, node_id: str, message: Optional[str] = None):
       msg = f"Node '{node_id}' not found ❌"
       super().__init__(msg)
       logger.warning(f"    ↳ [warning]Node ID not found in system[/warning]")

class FunctionNotFoundError(EasyRemoteError):
   """找不到指定函数时抛出此异常"""
   
   def __init__(self, function_name: str, node_id: Optional[str] = None, message: Optional[str] = None):
       msg = f"Function '{function_name}' not found" + (f" on node '{node_id}' ❌" if node_id else " ❌")
       super().__init__(msg)
       logger.warning(f"    ↳ [warning]Function not available in system[/warning]")

class ConnectionError(EasyRemoteError):
   """连接相关错误时抛出此异常"""
   
   def __init__(self, address: str, message: Optional[str] = None, cause: Optional[Exception] = None):
       msg = f"Failed to connect to {address} ❌"
       super().__init__(msg, cause)
       logger.warning(f"    ↳ [warning]Connection attempt failed[/warning]")

class SerializationError(EasyRemoteError):
   """序列化或反序列化错误时抛出此异常"""
   
   def __init__(
       self, 
       operation: str,
       message: Optional[str] = None,
       cause: Optional[Exception] = None
   ):
       msg = f"Failed to {operation} data" + (f": {message}" if message else "")
       super().__init__(msg, cause)
       logger.warning(f"    ↳ [warning]{operation.capitalize()} operation failed[/warning]")

class RemoteExecutionError(EasyRemoteError):
   """远程执行错误时抛出此异常"""
   
   def __init__(
       self,
       function_name: str,
       node_id: Optional[str] = None,
       message: Optional[str] = None,
       cause: Optional[Exception] = None
   ):
       msg = f"Failed to execute '{function_name}'" + (f" on node '{node_id}' ❌" if node_id else " ❌")
       super().__init__(msg, cause)
       logger.warning(f"    ↳ [warning]Remote execution failed[/warning]")

def format_exception(e: Exception) -> str:
   """格式化异常信息，用于日志记录和错误报告"""
   if isinstance(e, EasyRemoteError):
       return str(e)
   return f"{e.__class__.__name__}: {str(e)}"

def main():
   """测试各种异常情况"""
   logger.warning("⚡ Starting exception tests...")
   
   try:
       raise NodeNotFoundError("node123")
   except EasyRemoteError as e:
       logger.error(str(e))
   
   try:
       raise FunctionNotFoundError("test_func", "node123")
   except EasyRemoteError as e:
       logger.error(str(e))

   try:
       raise ConnectionError("localhost:8080", cause=Exception("Connection refused"))
   except EasyRemoteError as e:
       logger.error(str(e))

   try:
       raise SerializationError(
           operation="serialize",
           cause=TypeError("Object is not serializable")
       )
   except EasyRemoteError as e:
       logger.error(str(e))

   try:
       raise RemoteExecutionError(
           function_name="remote_process",
           node_id="node123",
           cause=Exception("Remote process crashed")
       )
   except EasyRemoteError as e:
       logger.error(str(e))

if __name__ == "__main__":
   main()