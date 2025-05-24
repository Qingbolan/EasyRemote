# easyremote/core/utils/performance.py
import asyncio
import threading
import psutil
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
from ..utils.logger import ModernLogger
@dataclass
class StreamMetrics:
    """流式处理指标"""
    stream_id: str
    function_name: str
    node_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    chunks_sent: int = 0
    bytes_sent: int = 0
    errors: int = 0
    last_activity: datetime = field(default_factory=datetime.now)
    
    @property
    def duration(self) -> Optional[timedelta]:
        if self.end_time:
            return self.end_time - self.start_time
        return datetime.now() - self.start_time
    
    @property
    def throughput(self) -> float:
        """计算吞吐量 (chunks/second)"""
        duration = self.duration
        if duration and duration.total_seconds() > 0:
            return self.chunks_sent / duration.total_seconds()
        return 0.0

@dataclass
class SystemMetrics:
    """系统资源指标"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_rss: int
    memory_vms: int
    active_threads: int
    async_tasks: int
    
class PerformanceMonitor(ModernLogger):
    """性能监控器"""
    def __init__(self, collection_interval: float = 5.0, max_history: int = 1000):
        super().__init__(name="PerformanceMonitor")
        self.collection_interval = collection_interval
        self.max_history = max_history
        
        # 指标存储
        self._stream_metrics: Dict[str, StreamMetrics] = {}
        self._system_metrics: deque = deque(maxlen=max_history)
        self._function_stats: Dict[str, Dict] = defaultdict(lambda: {
            'total_calls': 0,
            'total_duration': 0.0,
            'errors': 0,
            'avg_duration': 0.0
        })
        
        # 控制
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        
        # 回调
        self._alert_callbacks: List[Callable] = []
        
        # 阈值
        self.cpu_threshold = 80.0
        self.memory_threshold = 80.0
        self.error_rate_threshold = 0.1
        
    async def start(self):
        """启动性能监控"""
        if self._running:
            return
            
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        self.info("Performance monitor started")
    
    async def stop(self):
        """停止性能监控"""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        self.info("Performance monitor stopped")
    
    async def _monitor_loop(self):
        """监控循环"""
        try:
            while self._running:
                await self._collect_system_metrics()
                await self._check_alerts()
                await asyncio.sleep(self.collection_interval)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.error(f"Error in monitor loop: {e}", exc_info=True)
    
    async def _collect_system_metrics(self):
        """收集系统指标"""
        try:
            process = psutil.Process()
            cpu_percent = process.cpu_percent()
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            
            # 获取异步任务数量
            try:
                async_tasks = len(asyncio.all_tasks())
            except RuntimeError:
                async_tasks = 0
            
            metrics = SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_rss=memory_info.rss,
                memory_vms=memory_info.vms,
                active_threads=threading.active_count(),
                async_tasks=async_tasks
            )
            
            async with self._lock:
                self._system_metrics.append(metrics)
                
        except Exception as e:
            self.error(f"Error collecting system metrics: {e}")
    
    async def start_stream_tracking(self, stream_id: str, function_name: str, node_id: str):
        """开始跟踪流"""
        metrics = StreamMetrics(
            stream_id=stream_id,
            function_name=function_name,
            node_id=node_id,
            start_time=datetime.now()
        )
        
        async with self._lock:
            self._stream_metrics[stream_id] = metrics
            
        self.debug(f"Started tracking stream {stream_id}")
    
    async def update_stream_metrics(self, stream_id: str, chunks_sent: int = 0, bytes_sent: int = 0, errors: int = 0):
        """更新流指标"""
        async with self._lock:
            if stream_id in self._stream_metrics:
                metrics = self._stream_metrics[stream_id]
                metrics.chunks_sent += chunks_sent
                metrics.bytes_sent += bytes_sent
                metrics.errors += errors
                metrics.last_activity = datetime.now()
    
    async def end_stream_tracking(self, stream_id: str):
        """结束流跟踪"""
        async with self._lock:
            if stream_id in self._stream_metrics:
                metrics = self._stream_metrics[stream_id]
                metrics.end_time = datetime.now()
                
                # 更新函数统计
                stats = self._function_stats[metrics.function_name]
                stats['total_calls'] += 1
                if metrics.duration:
                    duration = metrics.duration.total_seconds()
                    stats['total_duration'] += duration
                    stats['avg_duration'] = stats['total_duration'] / stats['total_calls']
                stats['errors'] += metrics.errors
                
                # 移除完成的流
                del self._stream_metrics[stream_id]
                
        self.debug(f"Ended tracking stream {stream_id}")
    
    async def _check_alerts(self):
        """检查性能告警"""
        try:
            async with self._lock:
                if not self._system_metrics:
                    return
                
                latest_metrics = self._system_metrics[-1]
                
                # CPU告警
                if latest_metrics.cpu_percent > self.cpu_threshold:
                    message = f"High CPU usage: {latest_metrics.cpu_percent:.1f}%"
                    self.warning(message)
                    await self._trigger_alert("cpu_high", message, latest_metrics)
                
                # 内存告警
                if latest_metrics.memory_percent > self.memory_threshold:
                    message = f"High memory usage: {latest_metrics.memory_percent:.1f}%"
                    self.warning(message)
                    await self._trigger_alert("memory_high", message, latest_metrics)
                
                # 检查流错误率
                for stream_id, metrics in self._stream_metrics.items():
                    if metrics.chunks_sent > 0:
                        error_rate = metrics.errors / metrics.chunks_sent
                        if error_rate > self.error_rate_threshold:
                            message = f"High error rate in stream {stream_id}: {error_rate:.2%}"
                            self.warning(message)
                            await self._trigger_alert("error_rate_high", message, metrics)
                            
        except Exception as e:
            self.error(f"Error checking alerts: {e}")
    
    async def _trigger_alert(self, alert_type: str, message: str, metrics_data):
        """触发告警"""
        try:
            for callback in self._alert_callbacks:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert_type, message, metrics_data)
                else:
                    callback(alert_type, message, metrics_data)
        except Exception as e:
            self.error(f"Error triggering alert callback: {e}")
    
    def add_alert_callback(self, callback: Callable):
        """添加告警回调"""
        self._alert_callbacks.append(callback)
    
    def remove_alert_callback(self, callback: Callable):
        """移除告警回调"""
        if callback in self._alert_callbacks:
            self._alert_callbacks.remove(callback)
    
    async def get_function_stats(self) -> Dict[str, Dict]:
        """获取函数统计信息"""
        async with self._lock:
            return dict(self._function_stats)
    
    async def get_stream_metrics(self) -> Dict[str, StreamMetrics]:
        """获取流指标"""
        async with self._lock:
            return dict(self._stream_metrics)
    
    async def get_system_metrics(self, limit: int = 100) -> List[SystemMetrics]:
        """获取系统指标"""
        async with self._lock:
            return list(self._system_metrics)[-limit:]
    
    def get_current_stats(self) -> Dict:
        """获取当前统计信息（同步方法）"""
        try:
            # 获取最新的系统指标
            latest_system = None
            if self._system_metrics:
                latest_system = self._system_metrics[-1]
            
            # 活跃流数量
            active_streams = len(self._stream_metrics)
            
            # 总函数调用数
            total_calls = sum(stats['total_calls'] for stats in self._function_stats.values())
            
            return {
                'active_streams': active_streams,
                'total_function_calls': total_calls,
                'functions_registered': len(self._function_stats),
                'system_metrics': {
                    'cpu_percent': latest_system.cpu_percent if latest_system else 0,
                    'memory_percent': latest_system.memory_percent if latest_system else 0,
                    'active_threads': latest_system.active_threads if latest_system else 0,
                    'async_tasks': latest_system.async_tasks if latest_system else 0
                } if latest_system else {}
            }
        except Exception as e:
            self.error(f"Error getting current stats: {e}")
            return {}

# 全局实例
performance_monitor = PerformanceMonitor()

def get_performance_monitor() -> PerformanceMonitor:
    """获取性能监控器实例"""
    return performance_monitor 