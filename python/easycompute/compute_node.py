import json
import time
import uuid
from typing import Any, Callable, Optional
import grpc

from .resource_manager import Task

class EasyComputeClient:
    """EasyCompute客户端"""
    
    def __init__(self, vps_host: str = "localhost", vps_port: int = 8888):
        self._channel = grpc.insecure_channel(f"{vps_host}:{vps_port}")
        self._stub = self._create_stub()
        
    def submit(self, func: Callable, *args, **kwargs) -> str:
        """提交任务"""
        task = Task(
            task_id=str(uuid.uuid4()),
            function_name=func.__name__,
            args_json=json.dumps({
                "args": args,
                "kwargs": kwargs
            }),
            priority=0,
            status="pending",
            create_time=int(time.time())
        )
        
        return self._stub.SubmitTask(task)
        
    def get_result(self, task_id: str, 
                  timeout: Optional[float] = None) -> Any:
        """获取任务结果"""
        start_time = time.time()
        while True:
            task = self._stub.GetTaskStatus(task_id)
            if task.status == "completed":
                return json.loads(task.result_json)
            elif task.status == "failed":
                raise Exception(f"Task failed: {task.error}")
                
            if timeout and time.time() - start_time > timeout:
                raise TimeoutError("Task timeout")
                
            time.sleep(1)
            
    def list_nodes(self):
        """获取所有计算节点"""
        return self._stub.ListNodes()