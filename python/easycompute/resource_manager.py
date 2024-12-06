from dataclasses import dataclass
from typing import Dict, List, Optional
import threading
import time
import uuid

@dataclass
class NodeStatus:
    """计算节点状态"""
    node_id: str
    address: str
    cpu_usage: float
    memory_usage: float
    active_tasks: int

@dataclass
class Task:
    """任务信息"""
    task_id: str
    function_name: str
    args_json: str
    priority: int
    status: str
    create_time: int
    assigned_node: Optional[str] = None
    result_json: Optional[str] = None
    error: Optional[str] = None

class ResourceManager:
    """资源管理器"""
    def __init__(self):
        self._nodes: Dict[str, NodeStatus] = {}
        self._tasks: Dict[str, Task] = {}
        self._lock = threading.Lock()
        
        # 启动健康检查
        self._start_health_check()
        
    def register_node(self, node: NodeStatus) -> None:
        """注册计算节点"""
        with self._lock:
            self._nodes[node.node_id] = node
            
    def update_node_status(self, node: NodeStatus) -> None:
        """更新节点状态"""
        with self._lock:
            if node.node_id in self._nodes:
                self._nodes[node.node_id] = node
                
    def get_nodes(self) -> List[NodeStatus]:
        """获取所有节点"""
        with self._lock:
            return list(self._nodes.values())
            
    def submit_task(self, task: Task) -> str:
        """提交任务"""
        with self._lock:
            self._tasks[task.task_id] = task
            assigned_node = self._select_node(task)
            if assigned_node:
                task.assigned_node = assigned_node
                task.status = "running"
            return task.task_id
            
    def get_task(self, task_id: str) -> Optional[Task]:
        """获取任务信息"""
        return self._tasks.get(task_id)
        
    def get_tasks(self) -> List[Task]:
        """获取所有任务"""
        return list(self._tasks.values())
        
    def _select_node(self, task: Task) -> Optional[str]:
        """选择计算节点"""
        best_node = None
        min_load = float('inf')
        
        for node_id, node in self._nodes.items():
            load = node.cpu_usage * 0.6 + node.memory_usage * 0.4
            if load < min_load:
                min_load = load
                best_node = node_id
                
        return best_node
        
    def _start_health_check(self):
        """启动节点健康检查"""
        def check():
            while True:
                with self._lock:
                    for task in self._tasks.values():
                        if task.status == "running":
                            if task.assigned_node not in self._nodes:
                                # 节点已死,重新分配
                                task.status = "pending"
                                task.assigned_node = None
                                new_node = self._select_node(task)
                                if new_node:
                                    task.assigned_node = new_node
                                    task.status = "running"
                time.sleep(10)
                
        thread = threading.Thread(target=check, daemon=True)
        thread.start()