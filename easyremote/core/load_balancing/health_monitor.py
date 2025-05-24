# Node health monitoring for EasyRemote load balancing
import asyncio
import time
from typing import Dict, Optional
from dataclasses import dataclass

from ..utils.logger import ModernLogger
from .strategies import NodeStats


@dataclass
class NodeHealthStatus:
    """Node health status information"""
    node_id: str
    is_healthy: bool
    is_available: bool
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: float = 0.0
    current_load: float = 0.0
    response_time: float = 0.0
    error: Optional[str] = None
    timestamp: float = 0.0
    
    
class NodeHealthMonitor(ModernLogger):
    """Monitor node health and availability"""
    
    def __init__(self, monitoring_interval: int = 10, timeout: float = 5.0):
        super().__init__(name="NodeHealthMonitor")
        self.monitoring_interval = monitoring_interval
        self.timeout = timeout
        self.health_cache: Dict[str, NodeHealthStatus] = {}
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False
        
    async def start_monitoring(self, gateway_instance):
        """Start continuous health monitoring"""
        self._running = True
        self._gateway = gateway_instance
        self._monitoring_task = asyncio.create_task(self._monitor_loop())
        self.info("Node health monitoring started")
        
    async def stop_monitoring(self):
        """Stop health monitoring"""
        self._running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        self.info("Node health monitoring stopped")
        
    async def _monitor_loop(self):
        """Main monitoring loop"""
        while self._running:
            try:
                await self._check_all_nodes()
                await asyncio.sleep(self.monitoring_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.monitoring_interval)
                
    async def _check_all_nodes(self):
        """Check health of all registered nodes"""
        if not hasattr(self, '_gateway') or not self._gateway:
            return
            
        try:
            # Get all registered nodes from gateway
            async with self._gateway._lock:
                node_ids = list(self._gateway._nodes.keys())
                
            # Check each node in parallel
            tasks = [self.check_node_health(node_id) for node_id in node_ids]
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            self.error(f"Error checking all nodes: {e}")
    
    async def check_node_health(self, node_id: str) -> NodeHealthStatus:
        """Check health status of a specific node"""
        # Check cache first
        if node_id in self.health_cache:
            cached_health = self.health_cache[node_id]
            if time.time() - cached_health.timestamp < self.monitoring_interval:
                return cached_health
        
        try:
            # Send health check request to node
            health_response = await self._send_health_check(node_id)
            
            health_status = NodeHealthStatus(
                node_id=node_id,
                is_healthy=True,
                is_available=health_response.get('current_load', 0) < 0.9,
                cpu_usage=health_response.get('cpu_usage', 0),
                memory_usage=health_response.get('memory_usage', 0),
                gpu_usage=health_response.get('gpu_usage', 0),
                current_load=health_response.get('current_load', 0),
                response_time=health_response.get('response_time', 0),
                timestamp=time.time()
            )
            
        except Exception as e:
            health_status = NodeHealthStatus(
                node_id=node_id,
                is_healthy=False,
                is_available=False,
                error=str(e),
                timestamp=time.time()
            )
        
        # Update cache
        self.health_cache[node_id] = health_status
        return health_status
    
    async def _send_health_check(self, node_id: str) -> Dict:
        """Send health check request to node"""
        try:
            # Simulate health check - in real implementation, 
            # this would send a gRPC health check request
            await asyncio.sleep(0.1)  # Simulate network delay
            
            # Mock response with simulated metrics
            import random
            response = {
                'cpu_usage': random.uniform(10, 80),
                'memory_usage': random.uniform(20, 70),
                'gpu_usage': random.uniform(0, 90),
                'current_load': random.uniform(0.1, 0.8),
                'response_time': random.uniform(10, 100)
            }
            
            return response
            
        except Exception as e:
            self.error(f"Health check failed for node {node_id}: {e}")
            raise
    
    def get_node_health(self, node_id: str) -> Optional[NodeHealthStatus]:
        """Get cached health status for a node"""
        return self.health_cache.get(node_id)
    
    def is_node_healthy(self, node_id: str) -> bool:
        """Check if a node is healthy"""
        health = self.get_node_health(node_id)
        return health is not None and health.is_healthy
    
    def is_node_available(self, node_id: str) -> bool:
        """Check if a node is available for new requests"""
        health = self.get_node_health(node_id)
        return health is not None and health.is_healthy and health.is_available
    
    def get_node_stats(self, node_id: str) -> Optional[NodeStats]:
        """Convert health status to node stats"""
        health = self.get_node_health(node_id)
        if not health:
            return None
            
        return NodeStats(
            node_id=node_id,
            cpu_usage=health.cpu_usage,
            memory_usage=health.memory_usage,
            gpu_usage=health.gpu_usage,
            current_load=health.current_load,
            response_time=health.response_time,
            has_gpu=health.gpu_usage > 0,
            last_updated=health.timestamp
        ) 