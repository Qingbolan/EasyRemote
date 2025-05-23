# easyremote/core/utils/performance.py
import asyncio
import threading
import psutil
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque

from .serialize import setup_logger

logger = setup_logger(__name__)

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
    
class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, collection_interval: float = 1.0, max_history: int = 1000):
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
        logger.info("Performance monitor started")
    
    async def stop(self):
        """停止性能监控"""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Performance monitor stopped")
    
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
            logger.error(f"Error in monitor loop: {e}", exc_info=True)
    
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
            logger.error(f"Error collecting system metrics: {e}")
    
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
            
        logger.debug(f"Started tracking stream {stream_id}")
    
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
                
        logger.debug(f"Ended tracking stream {stream_id}")

# 全局实例
performance_monitor = PerformanceMonitor()

def get_performance_monitor() -> PerformanceMonitor:
    """获取性能监控器实例"""
    return performance_monitor 