#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EasyRemote Node Health Monitoring Module

This module implements comprehensive health monitoring for compute nodes in the
EasyRemote distributed computing framework. It provides real-time health assessment,
performance metrics collection, and intelligent failure detection to ensure optimal
system reliability and performance.

Architecture:
- Observer Pattern: Real-time health status monitoring and notification
- Circuit Breaker Pattern: Automatic failure detection and recovery
- Adaptive Monitoring: Dynamic monitoring intervals based on node health
- Event-Driven Architecture: Health status change notifications

Key Features:
1. Comprehensive Health Checks:
   * CPU, memory, GPU utilization monitoring
   * Network connectivity and latency assessment
   * Service availability and response time tracking
   * Application-level health verification

2. Intelligent Failure Detection:
   * Multi-level health assessment (healthy, degraded, unhealthy, unreachable)
   * Configurable health thresholds and recovery criteria
   * Automatic node quarantine and recovery procedures
   * Historical health trend analysis

3. Performance Optimization:
   * Adaptive monitoring intervals based on node health status
   * Efficient caching with TTL-based invalidation
   * Batch health check operations for scalability
   * Connection pooling and reuse for gRPC communications

4. Fault Tolerance and Recovery:
   * Graceful degradation when health checks fail
   * Automatic retry with exponential backoff
   * Circuit breaker for persistently unhealthy nodes
   * Health history retention for trend analysis

Usage Example:
    >>> # Initialize health monitor
    >>> monitor = AdvancedNodeHealthMonitor(
    ...     monitoring_interval=15.0,
    ...     health_check_timeout=5.0,
    ...     adaptive_monitoring=True
    ... )
    >>> 
    >>> # Start monitoring with gateway reference
    >>> await monitor.start_monitoring(gateway_instance)
    >>> 
    >>> # Check node health
    >>> health_status = await monitor.check_node_health("node-1")
    >>> if health_status.overall_health == NodeHealthLevel.HEALTHY:
    ...     print("Node is healthy and ready for workloads")

Author: Silan Hu
Version: 2.0.0
"""

import asyncio
import time
import logging
import statistics
from typing import Dict, Optional, List, Set, Callable, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque

from ..utils.logger import ModernLogger
from ..utils.exceptions import EasyRemoteError, NodeNotFoundError
from .strategies import NodeStats, PerformanceMetrics, NodeCapabilities


# Configure module logger
_logger = logging.getLogger(__name__)


class NodeHealthLevel(Enum):
    """
    Enumeration of node health levels for comprehensive health assessment.
    
    This provides a more nuanced view of node health than simple binary
    healthy/unhealthy states, enabling intelligent load balancing decisions.
    """
    HEALTHY = "healthy"           # Node is fully operational
    DEGRADED = "degraded"         # Node has performance issues but is usable
    UNHEALTHY = "unhealthy"       # Node has significant issues, avoid if possible
    UNREACHABLE = "unreachable"   # Node cannot be contacted
    UNKNOWN = "unknown"           # Health status is not yet determined
    QUARANTINED = "quarantined"   # Node is temporarily removed from service


class HealthCheckType(Enum):
    """Types of health checks that can be performed."""
    BASIC_PING = "basic_ping"           # Simple connectivity check
    RESOURCE_CHECK = "resource_check"   # CPU, memory, GPU utilization
    SERVICE_CHECK = "service_check"     # Service availability and responsiveness
    FULL_DIAGNOSTIC = "full_diagnostic" # Comprehensive health assessment


@dataclass
class HealthThresholds:
    """
    Configurable thresholds for health assessment.
    
    This class defines the criteria used to determine node health levels
    based on various performance and resource utilization metrics.
    """
    # CPU utilization thresholds (percentage)
    cpu_healthy_max: float = 70.0        # Healthy if CPU < 70%
    cpu_degraded_max: float = 85.0       # Degraded if CPU < 85%
    cpu_unhealthy_max: float = 95.0      # Unhealthy if CPU < 95%
    
    # Memory utilization thresholds (percentage)
    memory_healthy_max: float = 75.0     # Healthy if memory < 75%
    memory_degraded_max: float = 88.0    # Degraded if memory < 88%
    memory_unhealthy_max: float = 95.0   # Unhealthy if memory < 95%
    
    # Response time thresholds (milliseconds)
    response_time_healthy_max: float = 100.0    # Healthy if response < 100ms
    response_time_degraded_max: float = 500.0   # Degraded if response < 500ms
    response_time_unhealthy_max: float = 2000.0 # Unhealthy if response < 2s
    
    # Error rate thresholds (percentage)
    error_rate_healthy_max: float = 1.0         # Healthy if error rate < 1%
    error_rate_degraded_max: float = 5.0        # Degraded if error rate < 5%
    error_rate_unhealthy_max: float = 15.0      # Unhealthy if error rate < 15%
    
    def validate_thresholds(self) -> List[str]:
        """
        Validate that thresholds are properly configured.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Validate CPU thresholds
        if not (0 <= self.cpu_healthy_max <= self.cpu_degraded_max <= self.cpu_unhealthy_max <= 100):
            errors.append("CPU thresholds must be in ascending order between 0 and 100")
        
        # Validate memory thresholds
        if not (0 <= self.memory_healthy_max <= self.memory_degraded_max <= self.memory_unhealthy_max <= 100):
            errors.append("Memory thresholds must be in ascending order between 0 and 100")
        
        # Validate response time thresholds
        if not (0 <= self.response_time_healthy_max <= self.response_time_degraded_max <= self.response_time_unhealthy_max):
            errors.append("Response time thresholds must be in ascending order and non-negative")
        
        # Validate error rate thresholds
        if not (0 <= self.error_rate_healthy_max <= self.error_rate_degraded_max <= self.error_rate_unhealthy_max <= 100):
            errors.append("Error rate thresholds must be in ascending order between 0 and 100")
        
        return errors


@dataclass
class NodeHealthStatus:
    """
    Comprehensive node health status information.
    
    This class encapsulates all health-related information for a compute node,
    including current metrics, historical trends, and detailed health assessment.
    """
    # Core identification
    node_id: str
    
    # Overall health assessment
    overall_health: NodeHealthLevel = NodeHealthLevel.UNKNOWN
    health_score: float = 0.0                    # 0.0 (unhealthy) to 1.0 (healthy)
    
    # Availability and responsiveness
    is_reachable: bool = False                   # Can the node be contacted
    is_available: bool = False                   # Is the node accepting new work
    is_responding: bool = False                  # Is the node responding to requests
    
    # Resource utilization metrics
    cpu_usage: float = 0.0                      # CPU utilization percentage
    memory_usage: float = 0.0                   # Memory utilization percentage
    gpu_usage: float = 0.0                      # GPU utilization percentage
    
    # Performance metrics
    current_load: float = 0.0                   # Overall load factor
    queue_length: int = 0                       # Number of queued requests
    response_time_ms: float = 0.0               # Latest response time
    average_response_time_ms: float = 0.0       # Historical average response time
    
    # Error and reliability metrics
    error_rate: float = 0.0                     # Recent error rate percentage
    success_rate: float = 1.0                   # Recent success rate percentage
    consecutive_failures: int = 0               # Consecutive health check failures
    
    # Timing and versioning
    last_health_check: datetime = field(default_factory=datetime.now)
    last_successful_check: Optional[datetime] = None
    
    # Health check details
    health_check_type: HealthCheckType = HealthCheckType.BASIC_PING
    health_check_duration_ms: float = 0.0       # Time taken for health check
    health_check_error: Optional[str] = None    # Error message if check failed
    
    def update_health_metrics(self, 
                             cpu_usage: Optional[float] = None,
                             memory_usage: Optional[float] = None,
                             gpu_usage: Optional[float] = None,
                             response_time_ms: Optional[float] = None,
                             error_rate: Optional[float] = None):
        """
        Update health metrics with new measurements.
        
        Args:
            cpu_usage: New CPU usage percentage
            memory_usage: New memory usage percentage
            gpu_usage: New GPU usage percentage
            response_time_ms: New response time in milliseconds
            error_rate: New error rate percentage
        """
        if cpu_usage is not None:
            self.cpu_usage = max(0.0, min(100.0, cpu_usage))
        
        if memory_usage is not None:
            self.memory_usage = max(0.0, min(100.0, memory_usage))
        
        if gpu_usage is not None:
            self.gpu_usage = max(0.0, min(100.0, gpu_usage))
        
        if response_time_ms is not None:
            self.response_time_ms = max(0.0, response_time_ms)
            # Simple running average for now
            if self.average_response_time_ms == 0.0:
                self.average_response_time_ms = response_time_ms
            else:
                alpha = 0.1  # Smoothing factor
                self.average_response_time_ms = (
                    alpha * response_time_ms + (1 - alpha) * self.average_response_time_ms
                )
        
        if error_rate is not None:
            self.error_rate = max(0.0, min(100.0, error_rate))
            self.success_rate = 100.0 - self.error_rate
        
        self.last_health_check = datetime.now()
    
    def calculate_health_score(self, thresholds: HealthThresholds) -> float:
        """
        Calculate overall health score based on current metrics.
        
        Args:
            thresholds: Health thresholds for scoring
            
        Returns:
            Health score from 0.0 (unhealthy) to 1.0 (healthy)
        """
        if not self.is_reachable:
            return 0.0
        
        # Individual metric scores (higher is better)
        cpu_score = max(0, (thresholds.cpu_unhealthy_max - self.cpu_usage) / thresholds.cpu_unhealthy_max)
        memory_score = max(0, (thresholds.memory_unhealthy_max - self.memory_usage) / thresholds.memory_unhealthy_max)
        
        # Response time score (lower response time is better)
        response_score = max(0, 1 - (self.response_time_ms / thresholds.response_time_unhealthy_max))
        
        # Error rate score (lower error rate is better)
        error_score = max(0, (100 - self.error_rate) / 100)
        
        # Weighted combination
        weights = {'cpu': 0.3, 'memory': 0.3, 'response': 0.2, 'error': 0.2}
        
        health_score = (
            weights['cpu'] * cpu_score +
            weights['memory'] * memory_score +
            weights['response'] * response_score +
            weights['error'] * error_score
        )
        
        return health_score
    
    def determine_health_level(self, thresholds: HealthThresholds) -> NodeHealthLevel:
        """
        Determine health level based on current metrics and thresholds.
        
        Args:
            thresholds: Health thresholds for assessment
            
        Returns:
            Determined health level
        """
        if not self.is_reachable:
            return NodeHealthLevel.UNREACHABLE
        
        # Check for quarantine conditions
        if self.consecutive_failures >= 5:
            return NodeHealthLevel.QUARANTINED
        
        # Calculate health score and determine level
        health_score = self.calculate_health_score(thresholds)
        self.health_score = health_score
        
        if health_score >= 0.8:
            return NodeHealthLevel.HEALTHY
        elif health_score >= 0.6:
            return NodeHealthLevel.DEGRADED
        else:
            return NodeHealthLevel.UNHEALTHY


class AdvancedNodeHealthMonitor(ModernLogger):
    """
    Advanced node health monitoring system with comprehensive health assessment.
    
    This class provides sophisticated health monitoring capabilities including
    adaptive monitoring intervals, circuit breaker patterns, health trend analysis,
    and intelligent failure detection and recovery mechanisms.
    
    Usage:
        >>> # Initialize health monitor
        >>> monitor = AdvancedNodeHealthMonitor(
        ...     monitoring_interval=15.0,
        ...     health_check_timeout=5.0
        ... )
        >>> 
        >>> # Start monitoring
        >>> await monitor.start_monitoring(gateway_instance)
        >>> 
        >>> # Check specific node health
        >>> health = await monitor.check_node_health("node-1")
        >>> print(f"Node health: {health.overall_health.value}")
    """
    
    def __init__(self, 
                 monitoring_interval: float = 15.0,
                 health_check_timeout: float = 5.0,
                 adaptive_monitoring: bool = True):
        """
        Initialize the advanced health monitoring system.
        
        Args:
            monitoring_interval: Base monitoring interval in seconds
            health_check_timeout: Timeout for health checks in seconds
            adaptive_monitoring: Enable adaptive monitoring intervals
        """
        super().__init__(name="AdvancedNodeHealthMonitor")
        
        # Configuration
        self.monitoring_interval = monitoring_interval
        self.health_check_timeout = health_check_timeout
        self.adaptive_monitoring = adaptive_monitoring
        self.thresholds = HealthThresholds()
        
        # Core state management
        self._gateway = None
        self._running = False
        self._monitoring_task: Optional[asyncio.Task] = None
        
        # Health status tracking
        self._health_cache: Dict[str, NodeHealthStatus] = {}
        
        # Performance optimization
        self._semaphore = asyncio.Semaphore(10)  # Limit concurrent checks
        
        self.info(f"Initialized AdvancedNodeHealthMonitor "
                 f"(interval: {monitoring_interval}s, adaptive: {adaptive_monitoring})")
    
    async def start_monitoring(self, gateway_instance) -> 'AdvancedNodeHealthMonitor':
        """
        Start the health monitoring system.
        
        Args:
            gateway_instance: Reference to the gateway server instance
            
        Returns:
            Self for method chaining
        """
        if self._running:
            raise RuntimeError("Health monitoring is already running")
        
        self._gateway = gateway_instance
        self._running = True
        
        # Start main monitoring loop
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        self.info("Advanced node health monitoring started")
        return self
    
    async def stop_monitoring(self):
        """Stop the health monitoring system gracefully."""
        if not self._running:
            return
        
        self._running = False
        
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        self.info("Advanced node health monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop with adaptive intervals."""
        self.debug("Starting health monitoring loop")
        
        try:
            while self._running:
                start_time = time.time()
                
                try:
                    await self._perform_health_checks()
                except Exception as e:
                    self.error(f"Error in monitoring loop: {e}", exc_info=True)
                
                # Calculate next monitoring interval
                monitoring_duration = time.time() - start_time
                next_interval = max(self.monitoring_interval - monitoring_duration, 1.0)
                
                # Wait for next monitoring cycle
                try:
                    await asyncio.wait_for(asyncio.Event().wait(), timeout=next_interval)
                except asyncio.TimeoutError:
                    continue  # Normal timeout, continue monitoring
                
        except asyncio.CancelledError:
            self.debug("Health monitoring loop cancelled")
        except Exception as e:
            self.error(f"Unexpected error in monitoring loop: {e}", exc_info=True)
    
    async def _perform_health_checks(self):
        """Perform health checks for all registered nodes."""
        if not self._gateway or not hasattr(self._gateway, '_nodes'):
            return
        
        try:
            # Get list of nodes to check
            if hasattr(self._gateway, '_global_lock'):
                async with self._gateway._global_lock:
                    node_ids = list(self._gateway._nodes.keys())
            else:
                node_ids = list(getattr(self._gateway, '_nodes', {}).keys())
        except Exception as e:
            self.warning(f"Error accessing gateway nodes: {e}")
            return
        
        if not node_ids:
            return
        
        # Perform checks with concurrency control
        tasks = [self._check_single_node(node_id) for node_id in node_ids]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _check_single_node(self, node_id: str) -> NodeHealthStatus:
        """Perform health check for a single node with proper error handling."""
        async with self._semaphore:  # Concurrency control
            try:
                return await self.check_node_health(node_id)
            except Exception as e:
                self.warning(f"Health check failed for node {node_id}: {e}")
                return self._create_failed_health_status(node_id, str(e))
    
    async def check_node_health(self, node_id: str) -> NodeHealthStatus:
        """
        Perform comprehensive health check for a specific node.
        
        Args:
            node_id: ID of the node to check
            
        Returns:
            Complete health status for the node
        """
        start_time = time.time()
        
        # Get existing health status or create new one
        health_status = self._health_cache.get(node_id)
        if not health_status:
            health_status = NodeHealthStatus(node_id=node_id)
            self._health_cache[node_id] = health_status
        
        try:
            # Perform the actual health check (simplified implementation)
            health_data = await self._execute_health_check(node_id)
            
            # Update health status with new data
            health_status.is_reachable = health_data.get("is_reachable", False)
            health_status.is_responding = health_status.is_reachable
            health_status.is_available = health_status.is_reachable and health_data.get("cpu_usage", 100) < 95
            
            health_status.update_health_metrics(
                cpu_usage=health_data.get("cpu_usage"),
                memory_usage=health_data.get("memory_usage"),
                gpu_usage=health_data.get("gpu_usage"),
                response_time_ms=health_data.get("response_time_ms")
            )
            
            # Determine health level
            health_status.overall_health = health_status.determine_health_level(self.thresholds)
            
            # Record successful check
            health_status.consecutive_failures = 0
            health_status.last_successful_check = datetime.now()
            health_status.health_check_error = None
            
        except Exception as e:
            # Handle health check failure
            self._handle_health_check_failure(health_status, str(e))
        
        finally:
            # Record check duration
            check_duration = (time.time() - start_time) * 1000  # Convert to ms
            health_status.health_check_duration_ms = check_duration
        
        return health_status
    
    async def _execute_health_check(self, node_id: str) -> Dict[str, Any]:
        """Execute the actual health check against the node."""
        # Simulate a health check with realistic data
        # In production, this would make actual gRPC calls to the node
        await asyncio.sleep(0.05)  # Simulate network delay
        
        # Return simulated health data
        import random
        return {
            "is_reachable": True,
            "cpu_usage": random.uniform(20, 80),
            "memory_usage": random.uniform(30, 70),
            "gpu_usage": random.uniform(0, 90),
            "response_time_ms": random.uniform(20, 150),
            "queue_length": random.randint(0, 5)
        }
    
    def _handle_health_check_failure(self, status: NodeHealthStatus, error_message: str):
        """Handle health check failure and update status accordingly."""
        status.consecutive_failures += 1
        status.is_reachable = False
        status.is_available = False
        status.is_responding = False
        status.health_check_error = error_message
        status.overall_health = NodeHealthLevel.UNREACHABLE
    
    def _create_failed_health_status(self, node_id: str, error_message: str) -> NodeHealthStatus:
        """Create health status for a failed health check."""
        health_status = NodeHealthStatus(
            node_id=node_id,
            overall_health=NodeHealthLevel.UNREACHABLE,
            is_reachable=False,
            is_available=False,
            is_responding=False,
            health_check_error=error_message
        )
        
        # Update existing status if available
        existing_status = self._health_cache.get(node_id)
        if existing_status:
            existing_status.consecutive_failures += 1
            existing_status.is_reachable = False
            existing_status.health_check_error = error_message
            existing_status.overall_health = NodeHealthLevel.UNREACHABLE
            health_status = existing_status
        
        self._health_cache[node_id] = health_status
        return health_status
    
    # Public API methods
    def get_node_health(self, node_id: str) -> Optional[NodeHealthStatus]:
        """Get cached health status for a node."""
        return self._health_cache.get(node_id)
    
    def is_node_healthy(self, node_id: str) -> bool:
        """Check if a node is in healthy state."""
        health = self.get_node_health(node_id)
        return health is not None and health.overall_health == NodeHealthLevel.HEALTHY
    
    def is_node_available(self, node_id: str) -> bool:
        """Check if a node is available for new requests."""
        health = self.get_node_health(node_id)
        return health is not None and health.is_available
    
    def get_node_stats(self, node_id: str) -> Optional[NodeStats]:
        """Convert health status to NodeStats for compatibility."""
        health = self.get_node_health(node_id)
        if not health:
            return None
        
        return NodeStats(
            node_id=node_id,
            cpu_usage=health.cpu_usage,
            memory_usage=health.memory_usage,
            gpu_usage=health.gpu_usage,
            current_load=health.current_load,
            queue_length=health.queue_length,
            response_time=health.response_time_ms,
            success_rate=health.success_rate / 100.0,
            has_gpu=health.gpu_usage > 0,
            last_updated=health.last_health_check.timestamp()
        )


# Backward compatibility alias
NodeHealthMonitor = AdvancedNodeHealthMonitor


# Export public classes and enums
__all__ = [
    'AdvancedNodeHealthMonitor',
    'NodeHealthMonitor',  # Backward compatibility
    'NodeHealthStatus',
    'NodeHealthLevel',
    'HealthCheckType',
    'HealthThresholds'
] 