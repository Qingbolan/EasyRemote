# Performance metrics collection for EasyRemote load balancing
import time
from typing import Dict, List, Optional
from collections import defaultdict, deque
from dataclasses import dataclass

from ..utils.logger import ModernLogger


@dataclass
class RequestMetrics:
    """Metrics for a single request"""
    node_id: str
    function_name: str
    execution_time: float
    success: bool
    timestamp: float
    data_size: int = 0
    error_message: Optional[str] = None


@dataclass
class NodePerformanceStats:
    """Aggregated performance statistics for a node"""
    node_id: str
    total_requests: int
    successful_requests: int
    average_execution_time: float
    success_rate: float
    requests_per_minute: float
    error_count: int
    last_updated: float


class PerformanceCollector(ModernLogger):
    """Collect and analyze performance metrics for load balancing decisions"""
    
    def __init__(self, max_history_size: int = 1000, analysis_window: int = 3600):
        super().__init__(name="PerformanceCollector")
        self.max_history_size = max_history_size
        self.analysis_window = analysis_window  # seconds
        
        # Store metrics in memory (in production, use proper database)
        self.request_history: deque = deque(maxlen=max_history_size)
        self.node_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history_size))
        self.function_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history_size))
        
        # Cached statistics
        self._stats_cache: Dict[str, NodePerformanceStats] = {}
        self._cache_expiry: Dict[str, float] = {}
        self._cache_duration = 60  # Cache for 60 seconds
        
    async def record_request_metrics(self, 
                                   node_id: str, 
                                   function_name: str,
                                   execution_time: float, 
                                   success: bool,
                                   data_size: int = 0,
                                   error_message: Optional[str] = None):
        """Record metrics for a completed request"""
        
        metrics = RequestMetrics(
            node_id=node_id,
            function_name=function_name,
            execution_time=execution_time,
            success=success,
            timestamp=time.time(),
            data_size=data_size,
            error_message=error_message
        )
        
        # Store in various collections for different analysis
        self.request_history.append(metrics)
        self.node_metrics[node_id].append(metrics)
        self.function_metrics[function_name].append(metrics)
        
        # Invalidate cache for this node
        if node_id in self._cache_expiry:
            del self._cache_expiry[node_id]
            
        self.debug(f"Recorded metrics for {function_name} on {node_id}: "
                  f"{execution_time:.3f}s, success={success}")
    
    async def get_node_performance_stats(self, 
                                       node_id: str, 
                                       time_window: Optional[int] = None) -> NodePerformanceStats:
        """Get aggregated performance statistics for a node"""
        
        # Check cache first
        cache_key = f"{node_id}_{time_window or self.analysis_window}"
        if (cache_key in self._stats_cache and 
            cache_key in self._cache_expiry and 
            time.time() < self._cache_expiry[cache_key]):
            return self._stats_cache[cache_key]
        
        # Calculate statistics
        time_window = time_window or self.analysis_window
        cutoff_time = time.time() - time_window
        
        node_requests = [
            req for req in self.node_metrics[node_id] 
            if req.timestamp > cutoff_time
        ]
        
        if not node_requests:
            stats = NodePerformanceStats(
                node_id=node_id,
                total_requests=0,
                successful_requests=0,
                average_execution_time=0.0,
                success_rate=0.0,
                requests_per_minute=0.0,
                error_count=0,
                last_updated=time.time()
            )
        else:
            total_requests = len(node_requests)
            successful_requests = sum(1 for req in node_requests if req.success)
            total_execution_time = sum(req.execution_time for req in node_requests)
            error_count = total_requests - successful_requests
            
            # Calculate time span for rate calculation
            if total_requests > 1:
                time_span = max(node_requests, key=lambda x: x.timestamp).timestamp - \
                           min(node_requests, key=lambda x: x.timestamp).timestamp
                requests_per_minute = (total_requests / max(time_span, 1)) * 60
            else:
                requests_per_minute = 0.0
            
            stats = NodePerformanceStats(
                node_id=node_id,
                total_requests=total_requests,
                successful_requests=successful_requests,
                average_execution_time=total_execution_time / total_requests if total_requests > 0 else 0.0,
                success_rate=successful_requests / total_requests if total_requests > 0 else 0.0,
                requests_per_minute=requests_per_minute,
                error_count=error_count,
                last_updated=time.time()
            )
        
        # Cache the result
        self._stats_cache[cache_key] = stats
        self._cache_expiry[cache_key] = time.time() + self._cache_duration
        
        return stats
    
    async def get_function_performance_stats(self, function_name: str) -> Dict[str, any]:
        """Get performance statistics for a specific function across all nodes"""
        
        cutoff_time = time.time() - self.analysis_window
        function_requests = [
            req for req in self.function_metrics[function_name]
            if req.timestamp > cutoff_time
        ]
        
        if not function_requests:
            return {
                "function_name": function_name,
                "total_requests": 0,
                "average_execution_time": 0.0,
                "success_rate": 0.0,
                "node_distribution": {}
            }
        
        # Analyze by node
        node_distribution = defaultdict(int)
        for req in function_requests:
            node_distribution[req.node_id] += 1
        
        total_requests = len(function_requests)
        successful_requests = sum(1 for req in function_requests if req.success)
        total_execution_time = sum(req.execution_time for req in function_requests)
        
        return {
            "function_name": function_name,
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "average_execution_time": total_execution_time / total_requests if total_requests > 0 else 0.0,
            "success_rate": successful_requests / total_requests if total_requests > 0 else 0.0,
            "node_distribution": dict(node_distribution)
        }
    
    async def get_overall_system_stats(self) -> Dict[str, any]:
        """Get overall system performance statistics"""
        
        cutoff_time = time.time() - self.analysis_window
        recent_requests = [
            req for req in self.request_history
            if req.timestamp > cutoff_time
        ]
        
        if not recent_requests:
            return {
                "total_requests": 0,
                "system_success_rate": 0.0,
                "average_response_time": 0.0,
                "requests_per_minute": 0.0,
                "active_nodes": 0,
                "load_distribution_variance": 0.0
            }
        
        # Calculate system-wide metrics
        total_requests = len(recent_requests)
        successful_requests = sum(1 for req in recent_requests if req.success)
        total_execution_time = sum(req.execution_time for req in recent_requests)
        
        # Calculate requests per minute
        if total_requests > 1:
            time_span = max(recent_requests, key=lambda x: x.timestamp).timestamp - \
                       min(recent_requests, key=lambda x: x.timestamp).timestamp
            requests_per_minute = (total_requests / max(time_span, 1)) * 60
        else:
            requests_per_minute = 0.0
        
        # Analyze load distribution
        node_loads = defaultdict(int)
        for req in recent_requests:
            node_loads[req.node_id] += 1
        
        if node_loads:
            loads = list(node_loads.values())
            mean_load = sum(loads) / len(loads)
            variance = sum((load - mean_load) ** 2 for load in loads) / len(loads)
            load_distribution_variance = variance / (mean_load ** 2) if mean_load > 0 else 0
        else:
            load_distribution_variance = 0.0
        
        return {
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "system_success_rate": successful_requests / total_requests if total_requests > 0 else 0.0,
            "average_response_time": total_execution_time / total_requests if total_requests > 0 else 0.0,
            "requests_per_minute": requests_per_minute,
            "active_nodes": len(node_loads),
            "load_distribution_variance": load_distribution_variance,
            "analysis_window_hours": self.analysis_window / 3600
        }
    
    async def get_node_comparison(self, node_ids: List[str]) -> Dict[str, NodePerformanceStats]:
        """Compare performance statistics across multiple nodes"""
        
        results = {}
        for node_id in node_ids:
            results[node_id] = await self.get_node_performance_stats(node_id)
        
        return results
    
    def get_historical_data(self, 
                          node_id: Optional[str] = None, 
                          function_name: Optional[str] = None,
                          limit: int = 100) -> List[RequestMetrics]:
        """Get historical request data for analysis"""
        
        if node_id and function_name:
            # Filter by both node and function
            data = [req for req in self.request_history 
                   if req.node_id == node_id and req.function_name == function_name]
        elif node_id:
            # Filter by node only
            data = list(self.node_metrics[node_id])
        elif function_name:
            # Filter by function only
            data = list(self.function_metrics[function_name])
        else:
            # All data
            data = list(self.request_history)
        
        # Return most recent data up to limit
        return sorted(data, key=lambda x: x.timestamp, reverse=True)[:limit]
    
    async def cleanup_old_data(self, max_age_hours: int = 24):
        """Clean up old performance data to prevent memory bloat"""
        
        cutoff_time = time.time() - (max_age_hours * 3600)
        
        # Clean main history
        while self.request_history and self.request_history[0].timestamp < cutoff_time:
            self.request_history.popleft()
        
        # Clean node metrics
        for node_id in list(self.node_metrics.keys()):
            node_queue = self.node_metrics[node_id]
            while node_queue and node_queue[0].timestamp < cutoff_time:
                node_queue.popleft()
            
            # Remove empty queues
            if not node_queue:
                del self.node_metrics[node_id]
        
        # Clean function metrics
        for function_name in list(self.function_metrics.keys()):
            function_queue = self.function_metrics[function_name]
            while function_queue and function_queue[0].timestamp < cutoff_time:
                function_queue.popleft()
            
            # Remove empty queues
            if not function_queue:
                del self.function_metrics[function_name]
        
        # Clear expired cache entries
        current_time = time.time()
        expired_keys = [
            key for key, expiry in self._cache_expiry.items()
            if current_time > expiry
        ]
        
        for key in expired_keys:
            self._stats_cache.pop(key, None)
            self._cache_expiry.pop(key, None)
        
        self.debug(f"Cleaned up performance data older than {max_age_hours} hours") 