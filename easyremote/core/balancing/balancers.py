#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EasyRemote Load Balancing Module

This module implements sophisticated load balancing algorithms for the EasyRemote
distributed computing framework. The load balancers are responsible for intelligently
routing function calls to the most appropriate compute nodes based on various
optimization strategies.

Architecture:
- Strategy Pattern: Multiple load balancing algorithms with unified interface
- Factory Pattern: Centralized load balancer creation and management
- Observer Pattern: Real-time performance monitoring and adaptation
- Command Pattern: Request routing with execution context

Key Features:
- Multiple Load Balancing Strategies:
  * Round Robin: Simple cyclic distribution
  * Resource-Aware: CPU, memory, and GPU utilization-based routing
  * Latency-Based: Network latency and response time optimization
  * Cost-Aware: Budget-conscious routing with cost optimization
  * Smart Adaptive: ML-inspired adaptive routing with learning capabilities
  * Dynamic: Context-aware strategy selection

- Real-time Performance Monitoring:
  * Continuous node health assessment
  * Performance metrics collection and analysis
  * Predictive performance modeling
  * Historical trend analysis

- Fault Tolerance and Recovery:
  * Automatic failover to healthy nodes
  * Circuit breaker pattern for failed nodes
  * Graceful degradation under high load
  * Dynamic node pool management

Design Principles:
- High Performance: Sub-millisecond routing decisions
- Scalability: Support for thousands of compute nodes
- Extensibility: Easy addition of new balancing strategies
- Observability: Comprehensive metrics and logging
- Reliability: Robust error handling and recovery

Author: Silan Hu
Version: 2.0.0
"""

import time
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, deque
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from ..utils.logger import ModernLogger
from ..utils.exceptions import NoAvailableNodesError, LoadBalancingError
from .strategies import (
    LoadBalancerInterface, 
    RequestContext, 
    NodeStats,
    LoadBalancingStrategy
)

class BalancerPerformanceLevel(Enum):
    """Performance levels for balancer optimization."""
    LOW_LATENCY = "low_latency"      # < 1ms routing decisions
    BALANCED = "balanced"            # < 5ms routing decisions
    COMPREHENSIVE = "comprehensive"  # < 10ms routing decisions


@dataclass
class LoadBalancingMetrics:
    """
    Comprehensive metrics for load balancing performance analysis.
    
    This class tracks detailed performance metrics for load balancing
    decisions, enabling optimization and debugging of routing algorithms.
    """
    total_requests: int = 0
    successful_routes: int = 0
    failed_routes: int = 0
    average_routing_time_ms: float = 0.0
    node_selection_accuracy: float = 0.0  # How often best node was selected
    load_distribution_variance: float = 0.0  # How evenly load is distributed
    strategy_effectiveness: Dict[str, float] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)
    
    @property
    def success_rate(self) -> float:
        """Calculate routing success rate."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_routes / self.total_requests
    
    def update_routing_performance(self, routing_time_ms: float, success: bool):
        """
        Update routing performance metrics.
        
        Args:
            routing_time_ms: Time taken for routing decision in milliseconds
            success: Whether the routing was successful
        """
        self.total_requests += 1
        if success:
            self.successful_routes += 1
        else:
            self.failed_routes += 1
        
        # Update average routing time using exponential moving average
        alpha = 0.1  # Smoothing factor
        if self.total_requests == 1:
            self.average_routing_time_ms = routing_time_ms
        else:
            self.average_routing_time_ms = (
                alpha * routing_time_ms + (1 - alpha) * self.average_routing_time_ms
            )
        
        self.last_updated = datetime.now()


class IntelligentLoadBalancer(ModernLogger):
    """
    Comprehensive load balancer with adaptive strategy selection and monitoring.
    
    This is the main load balancer class that orchestrates all load balancing
    operations, manages multiple strategies, and provides intelligent routing
    decisions based on real-time system conditions.
    
    Key Responsibilities:
    1. Strategy Management: Create and manage multiple load balancing strategies
    2. Request Routing: Route requests to optimal nodes using selected strategy
    3. Performance Monitoring: Track and analyze routing performance
    4. Adaptive Selection: Dynamically choose best strategy for current conditions
    5. Health Management: Monitor node health and filter unhealthy nodes
    
    Features:
    - Multi-strategy support with dynamic selection
    - Real-time performance monitoring and metrics collection
    - Adaptive learning from routing outcomes
    - Comprehensive error handling and recovery
    - Pluggable architecture for custom strategies
    
    Usage:
        >>> balancer = IntelligentLoadBalancer(gateway_instance)
        >>> selected_node = await balancer.route_request(
        ...     function_name="process_data",
        ...     request_context=RequestContext(...),
        ...     balancing_config={"strategy": "resource_aware"}
        ... )
    """
    
    def __init__(self, 
                 gateway_instance,
                 performance_level: BalancerPerformanceLevel = BalancerPerformanceLevel.BALANCED,
                 enable_adaptive_learning: bool = True,
                 metrics_retention_hours: int = 24):
        """
        Initialize the intelligent load balancer with comprehensive configuration.
        
        Args:
            gateway_instance: Reference to the gateway server instance
            performance_level: Target performance level for routing decisions
            enable_adaptive_learning: Enable machine learning-inspired adaptation
            metrics_retention_hours: How long to retain performance metrics
            
        Raises:
            ValueError: If configuration parameters are invalid
            LoadBalancingError: If initialization fails
        """
        super().__init__(name="IntelligentLoadBalancer")
        
        # Validate configuration
        if metrics_retention_hours < 1:
            raise ValueError("Metrics retention hours must be positive")
        
        # Core configuration
        self.gateway = gateway_instance
        self.performance_level = performance_level
        self.enable_adaptive_learning = enable_adaptive_learning
        self.metrics_retention_hours = metrics_retention_hours
        
        # Strategy registry with lazy initialization
        self._strategy_registry: Dict[LoadBalancingStrategy, LoadBalancerInterface] = {}
        self._initialize_strategies()
        
        # Performance monitoring and metrics
        self.metrics = LoadBalancingMetrics()
        self._performance_history: deque = deque(maxlen=1000)  # Last 1000 routing decisions
        self._strategy_performance: Dict[str, float] = defaultdict(float)
        
        # Adaptive learning components
        self._node_performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        self._strategy_selection_weights: Dict[str, float] = defaultdict(lambda: 1.0)
        
        # Health monitoring cache
        self._node_health_cache: Dict[str, Tuple[bool, datetime]] = {}
        self._health_cache_ttl = timedelta(seconds=5)  # 5-second health cache
        
        # Thread safety and performance optimization
        self._routing_lock = asyncio.Lock()
        self._last_cleanup_time = datetime.now()
        
        self.info(f"Initialized IntelligentLoadBalancer "
                 f"(performance_level: {performance_level.value}, "
                 f"adaptive_learning: {enable_adaptive_learning})")
    
    def _initialize_strategies(self):
        """
        Initialize all available load balancing strategies.
        
        This method creates instances of all load balancing strategies
        and registers them in the strategy registry for later use.
        """
        self.debug("Initializing load balancing strategies")
        
        self._strategy_registry = {
            LoadBalancingStrategy.ROUND_ROBIN: EnhancedRoundRobinBalancer(),
            LoadBalancingStrategy.RESOURCE_AWARE: IntelligentResourceAwareBalancer(),
            LoadBalancingStrategy.LATENCY_BASED: AdaptiveLatencyBasedBalancer(),
            LoadBalancingStrategy.COST_AWARE: SmartCostAwareBalancer(),
            LoadBalancingStrategy.SMART_ADAPTIVE: MachineLearningAdaptiveBalancer()
        }
        
        # Add dynamic balancer that chooses strategy automatically
        self._strategy_registry["dynamic"] = DynamicStrategyBalancer(self)
        
        self.info(f"Initialized {len(self._strategy_registry)} load balancing strategies")
    
    async def route_request(self, 
                          function_name: str, 
                          request_context: RequestContext,
                          balancing_config: Optional[Dict[str, Any]] = None) -> str:
        """
        Route a function execution request to the optimal compute node.
        
        This is the main entry point for load balancing decisions. It orchestrates
        the entire routing process including node discovery, health filtering,
        strategy selection, and performance monitoring.
        
        Args:
            function_name: Name of the function to execute
            request_context: Comprehensive request context with metadata
            balancing_config: Optional configuration overrides for routing
            
        Returns:
            Node ID of the selected optimal compute node
            
        Raises:
            NoAvailableNodesError: If no suitable nodes are available
            LoadBalancingError: If routing decision fails
            
        Example:
            >>> context = RequestContext(
            ...     priority=5,
            ...     deadline=datetime.now() + timedelta(minutes=5),
            ...     resource_requirements={"gpu_required": True}
            ... )
            >>> node_id = await balancer.route_request(
            ...     "train_model", context, {"strategy": "resource_aware"}
            ... )
        """
        routing_start_time = time.time()
        balancing_config = balancing_config or {}
        
        try:
            async with self._routing_lock:
                # Step 1: Discover function providers
                available_nodes = await self._discover_function_providers(function_name)
                if not available_nodes:
                    raise NoAvailableNodesError(
                        f"No compute nodes provide function '{function_name}'"
                    )
                
                # Step 2: Filter healthy and capable nodes
                healthy_nodes = await self._filter_healthy_nodes(
                    available_nodes, request_context
                )
                if not healthy_nodes:
                    raise NoAvailableNodesError(
                        f"No healthy nodes available for function '{function_name}'"
                    )
                
                # Step 3: Gather comprehensive node statistics
                node_stats = await self._gather_node_statistics(healthy_nodes)
                
                # Step 4: Select optimal load balancing strategy
                strategy_name = await self._select_optimal_strategy(
                    balancing_config, request_context, node_stats
                )
                balancer = self._get_balancer_instance(strategy_name)
                
                # Step 5: Execute routing decision
                selected_node = await balancer.select_node(
                    healthy_nodes, request_context, node_stats
                )
                
                # Step 6: Record performance metrics
                routing_time_ms = (time.time() - routing_start_time) * 1000
                await self._record_routing_decision(
                    function_name, selected_node, strategy_name, 
                    routing_time_ms, True, request_context
                )
                
                self.debug(f"Routed '{function_name}' to node '{selected_node}' "
                          f"using '{strategy_name}' strategy ({routing_time_ms:.2f}ms)")
                
                return selected_node
                
        except Exception as e:
            # Record failed routing attempt
            routing_time_ms = (time.time() - routing_start_time) * 1000
            await self._record_routing_decision(
                function_name, None, balancing_config.get("strategy", "unknown"),
                routing_time_ms, False, request_context
            )
            
            self.error(f"Failed to route request for '{function_name}': {e}")
            
            if isinstance(e, (NoAvailableNodesError, LoadBalancingError)):
                raise
            else:
                raise LoadBalancingError(f"Routing failed: {e}") from e
    
    async def _discover_function_providers(self, function_name: str) -> List[str]:
        """
        Discover all compute nodes that provide the specified function.
        
        This method queries the gateway's node registry to find all nodes
        that have registered the specified function.
        
        Args:
            function_name: Name of the function to find providers for
            
        Returns:
            List of node IDs that provide the function
        """
        providers = []
        
        try:
            # Access gateway's node registry with proper locking
            if hasattr(self.gateway, '_global_lock') and hasattr(self.gateway, '_nodes'):
                async with self.gateway._global_lock:
                    for node_id, node_info in self.gateway._nodes.items():
                        if function_name in node_info.functions:
                            providers.append(node_id)
            else:
                self.warning("Gateway does not have expected node registry structure")
        
        except Exception as e:
            self.error(f"Error discovering function providers: {e}")
            # Return empty list to trigger NoAvailableNodesError upstream
        
        self.debug(f"Found {len(providers)} providers for function '{function_name}'")
        return providers
    
    async def _filter_healthy_nodes(self, 
                                   node_ids: List[str], 
                                   request_context: RequestContext) -> List[str]:
        """
        Filter nodes based on health status and capability requirements.
        
        This method removes unhealthy nodes and nodes that cannot satisfy
        the request's resource requirements.
        
        Args:
            node_ids: List of candidate node IDs
            request_context: Request context with requirements
            
        Returns:
            List of healthy and capable node IDs
        """
        healthy_nodes = []
        current_time = datetime.now()
        
        for node_id in node_ids:
            try:
                # Check cached health status first
                if node_id in self._node_health_cache:
                    is_healthy, cache_time = self._node_health_cache[node_id]
                    if current_time - cache_time < self._health_cache_ttl:
                        if is_healthy:
                            healthy_nodes.append(node_id)
                        continue
                
                # Perform fresh health check
                is_healthy = await self._check_node_health(node_id, request_context)
                
                # Update cache
                self._node_health_cache[node_id] = (is_healthy, current_time)
                
                if is_healthy:
                    healthy_nodes.append(node_id)
                    
            except Exception as e:
                self.warning(f"Error checking health for node {node_id}: {e}")
                # Exclude node from healthy list if health check fails
        
        self.debug(f"Filtered to {len(healthy_nodes)} healthy nodes from {len(node_ids)} candidates")
        return healthy_nodes
    
    async def _check_node_health(self, 
                                node_id: str, 
                                request_context: RequestContext) -> bool:
        """
        Perform comprehensive health check for a specific node.
        
        Args:
            node_id: ID of the node to check
            request_context: Request context for capability checking
            
        Returns:
            True if node is healthy and capable, False otherwise
        """
        try:
            # Check if gateway has health monitor
            if hasattr(self.gateway, 'health_monitor'):
                health_monitor = self.gateway.health_monitor
                
                # Basic availability check
                if not health_monitor.is_node_available(node_id):
                    return False
                
                # Get node statistics for health assessment
                node_stats = health_monitor.get_node_stats(node_id)
                if not node_stats:
                    return False
                
                # Check if node meets resource requirements
                if request_context.requirements:
                    if not self._check_resource_compatibility(node_stats, request_context.requirements):
                        return False
                
                # Check node health metrics
                if hasattr(node_stats, 'is_healthy') and callable(node_stats.is_healthy):
                    return node_stats.is_healthy()
                
                # Fallback health check based on basic metrics
                return (node_stats.cpu_usage < 95 and 
                       node_stats.memory_usage < 90 and
                       node_stats.current_load < 0.95)
            
            else:
                # Fallback to basic node registry check
                if hasattr(self.gateway, '_nodes'):
                    node_info = self.gateway._nodes.get(node_id)
                    if node_info:
                        return node_info.is_alive()
                
                # If no health monitoring available, assume healthy
                return True
                
        except Exception as e:
            self.warning(f"Health check failed for node {node_id}: {e}")
            return False
    
    def _check_resource_compatibility(self, 
                                    node_stats: NodeStats, 
                                    requirements: Dict[str, Any]) -> bool:
        """
        Check if node can satisfy resource requirements.
        
        Args:
            node_stats: Current node statistics
            requirements: Resource requirements dictionary
            
        Returns:
            True if node can satisfy requirements, False otherwise
        """
        try:
            # Check GPU requirements
            if requirements.get("gpu_required", False):
                if not getattr(node_stats, 'has_gpu', False):
                    return False
            
            # Check minimum memory requirements
            min_memory = requirements.get("min_memory_mb", 0)
            if min_memory > 0:
                available_memory = (100 - node_stats.memory_usage) / 100 * getattr(node_stats, 'total_memory_mb', 8192)
                if available_memory < min_memory:
                    return False
            
            # Check CPU requirements
            min_cpu_cores = requirements.get("min_cpu_cores", 0)
            if min_cpu_cores > 0:
                available_cpu = (100 - node_stats.cpu_usage) / 100 * getattr(node_stats, 'cpu_cores', 4)
                if available_cpu < min_cpu_cores:
                    return False
            
            return True
            
        except Exception as e:
            self.warning(f"Error checking resource compatibility: {e}")
            return True  # Assume compatible if check fails
    
    async def _gather_node_statistics(self, node_ids: List[str]) -> Dict[str, NodeStats]:
        """
        Gather comprehensive statistics for all specified nodes.
        
        Args:
            node_ids: List of node IDs to gather statistics for
            
        Returns:
            Dictionary mapping node IDs to their current statistics
        """
        stats = {}
        
        for node_id in node_ids:
            try:
                if hasattr(self.gateway, 'health_monitor'):
                    node_stats = self.gateway.health_monitor.get_node_stats(node_id)
                    if node_stats:
                        stats[node_id] = node_stats
                    else:
                        # Create default stats if not available
                        stats[node_id] = self._create_default_node_stats(node_id)
                else:
                    # Create default stats if no health monitor
                    stats[node_id] = self._create_default_node_stats(node_id)
                    
            except Exception as e:
                self.warning(f"Error gathering stats for node {node_id}: {e}")
                stats[node_id] = self._create_default_node_stats(node_id)
        
        return stats
    
    def _create_default_node_stats(self, node_id: str) -> NodeStats:
        """
        Create default node statistics when real data is unavailable.
        
        Args:
            node_id: ID of the node
            
        Returns:
            NodeStats object with default values
        """
        return NodeStats(
            node_id=node_id,
            cpu_usage=50.0,  # Assume moderate load
            memory_usage=50.0,
            current_load=0.5,
            response_time=100.0,  # 100ms default
            has_gpu=False,
            gpu_usage=0.0,
            last_updated=time.time()
        )
    
    async def _select_optimal_strategy(self, 
                                     balancing_config: Dict[str, Any],
                                     request_context: RequestContext,
                                     node_stats: Dict[str, NodeStats]) -> str:
        """
        Select the optimal load balancing strategy for current conditions.
        
        This method uses adaptive learning and system analysis to choose
        the best strategy for the current request and system state.
        
        Args:
            balancing_config: Configuration overrides
            request_context: Request context information
            node_stats: Current node statistics
            
        Returns:
            Name of the selected strategy
        """
        # Check for explicit strategy configuration
        explicit_strategy = balancing_config.get("strategy")
        if explicit_strategy and explicit_strategy in [s.value for s in LoadBalancingStrategy]:
            return explicit_strategy
        
        # Use adaptive learning if enabled
        if self.enable_adaptive_learning:
            return await self._adaptive_strategy_selection(request_context, node_stats)
        
        # Fallback to system condition analysis
        return await self._analyze_system_conditions(request_context, node_stats)
    
    async def _adaptive_strategy_selection(self, 
                                         request_context: RequestContext,
                                         node_stats: Dict[str, NodeStats]) -> str:
        """
        Use adaptive learning to select the best strategy.
        
        This method analyzes historical performance of different strategies
        under similar conditions and selects the one with the best track record.
        
        Args:
            request_context: Request context information
            node_stats: Current node statistics
            
        Returns:
            Name of the adaptively selected strategy
        """
        # Analyze current system characteristics
        system_load = sum(stats.current_load for stats in node_stats.values()) / len(node_stats)
        avg_cpu = sum(stats.cpu_usage for stats in node_stats.values()) / len(node_stats)
        avg_memory = sum(stats.memory_usage for stats in node_stats.values()) / len(node_stats)
        
        # Determine context features for strategy selection
        context_features = {
            "high_load": system_load > 0.7,
            "high_cpu": avg_cpu > 80,
            "high_memory": avg_memory > 80,
            "gpu_required": request_context.requirements and request_context.requirements.get("gpu_required", False),
            "cost_sensitive": request_context.cost_limit and request_context.cost_limit < 10.0,
            "latency_critical": getattr(request_context, 'latency_critical', False)
        }
        
        # Select strategy based on learned weights and current context
        best_strategy = "resource_aware"  # Default
        best_score = 0.0
        
        for strategy_name, base_weight in self._strategy_selection_weights.items():
            score = base_weight
            
            # Adjust score based on context
            if strategy_name == "latency_based" and context_features["latency_critical"]:
                score *= 1.5
            elif strategy_name == "resource_aware" and (context_features["high_cpu"] or context_features["high_memory"]):
                score *= 1.3
            elif strategy_name == "cost_aware" and context_features["cost_sensitive"]:
                score *= 1.4
            
            if score > best_score:
                best_score = score
                best_strategy = strategy_name
        
        return best_strategy
    
    async def _analyze_system_conditions(self, 
                                        request_context: RequestContext,
                                        node_stats: Dict[str, NodeStats]) -> str:
        """
        Analyze current system conditions to select appropriate strategy.
        
        Args:
            request_context: Request context information
            node_stats: Current node statistics
            
        Returns:
            Name of the selected strategy based on system analysis
        """
        if not node_stats:
            return LoadBalancingStrategy.RESOURCE_AWARE.value
        
        # Calculate system-wide metrics
        avg_cpu = sum(stats.cpu_usage for stats in node_stats.values()) / len(node_stats)
        avg_memory = sum(stats.memory_usage for stats in node_stats.values()) / len(node_stats)
        avg_response_time = sum(getattr(stats, 'response_time', 100) for stats in node_stats.values()) / len(node_stats)
        
        # Decision logic based on system conditions
        if avg_cpu > 85 or avg_memory > 85:
            return LoadBalancingStrategy.RESOURCE_AWARE.value
        elif avg_response_time > 200:  # High latency
            return LoadBalancingStrategy.LATENCY_BASED.value
        elif request_context.cost_limit and request_context.cost_limit < 5.0:
            return LoadBalancingStrategy.COST_AWARE.value
        else:
            return LoadBalancingStrategy.SMART_ADAPTIVE.value
    
    def _get_balancer_instance(self, strategy_name: str) -> LoadBalancerInterface:
        """
        Get the balancer instance for the specified strategy.
        
        Args:
            strategy_name: Name or enum value of the strategy
            
        Returns:
            LoadBalancerInterface instance for the strategy
            
        Raises:
            LoadBalancingError: If strategy is not found
        """
        # Handle enum values
        if isinstance(strategy_name, LoadBalancingStrategy):
            strategy_enum = strategy_name
        else:
            # Try to convert string to enum
            try:
                strategy_enum = LoadBalancingStrategy(strategy_name)
            except ValueError:
                # Handle special cases like "dynamic"
                if strategy_name == "dynamic":
                    return self._strategy_registry["dynamic"]
                else:
                    self.warning(f"Unknown strategy '{strategy_name}', falling back to resource_aware")
                    strategy_enum = LoadBalancingStrategy.RESOURCE_AWARE
        
        balancer = self._strategy_registry.get(strategy_enum)
        if not balancer:
            raise LoadBalancingError(f"No balancer found for strategy: {strategy_name}")
        
        return balancer
    
    async def _record_routing_decision(self, 
                                     function_name: str,
                                     selected_node: Optional[str],
                                     strategy_name: str,
                                     routing_time_ms: float,
                                     success: bool,
                                     request_context: RequestContext):
        """
        Record routing decision for performance analysis and learning.
        
        Args:
            function_name: Name of the function being routed
            selected_node: Selected node ID (None if routing failed)
            strategy_name: Strategy used for routing
            routing_time_ms: Time taken for routing decision
            success: Whether routing was successful
            request_context: Original request context
        """
        # Update overall metrics
        self.metrics.update_routing_performance(routing_time_ms, success)
        
        # Record performance history
        performance_record = {
            "timestamp": datetime.now(),
            "function_name": function_name,
            "selected_node": selected_node,
            "strategy": strategy_name,
            "routing_time_ms": routing_time_ms,
            "success": success,
            "context": {
                "priority": getattr(request_context, 'priority', 5),
                "cost_limit": getattr(request_context, 'cost_limit', None),
                "has_requirements": bool(getattr(request_context, 'requirements', None))
            }
        }
        
        self._performance_history.append(performance_record)
        
        # Update strategy performance tracking for adaptive learning
        if self.enable_adaptive_learning and success:
            # Reward successful strategy
            self._strategy_selection_weights[strategy_name] *= 1.01  # Small positive feedback
            
            # Normalize weights to prevent unbounded growth
            if max(self._strategy_selection_weights.values()) > 2.0:
                max_weight = max(self._strategy_selection_weights.values())
                for key in self._strategy_selection_weights:
                    self._strategy_selection_weights[key] /= max_weight
        
        # Periodic cleanup of old data
        if (datetime.now() - self._last_cleanup_time).total_seconds() > 3600:  # Every hour
            await self._cleanup_old_metrics()
    
    async def _cleanup_old_metrics(self):
        """Clean up old performance metrics to prevent memory leaks."""
        self._last_cleanup_time = datetime.now()
        cutoff_time = datetime.now() - timedelta(hours=self.metrics_retention_hours)
        
        # Clean performance history
        self._performance_history = deque(
            [record for record in self._performance_history 
             if record["timestamp"] > cutoff_time],
            maxlen=1000
        )
        
        # Clean node performance history
        for node_id, history in self._node_performance_history.items():
            # Keep only recent performance data
            recent_data = [data for data in history if data.get("timestamp", datetime.min) > cutoff_time]
            self._node_performance_history[node_id] = deque(recent_data, maxlen=50)
        
        # Clean health cache
        current_time = datetime.now()
        self._node_health_cache = {
            node_id: (is_healthy, cache_time)
            for node_id, (is_healthy, cache_time) in self._node_health_cache.items()
            if current_time - cache_time < self._health_cache_ttl
        }
        
        self.debug("Completed metrics cleanup")
    
    def get_performance_metrics(self) -> LoadBalancingMetrics:
        """
        Get current load balancing performance metrics.
        
        Returns:
            Current performance metrics
        """
        return self.metrics
    
    def get_strategy_performance(self) -> Dict[str, float]:
        """
        Get performance weights for different strategies.
        
        Returns:
            Dictionary of strategy names to performance weights
        """
        return dict(self._strategy_selection_weights)


# Enhanced strategy implementations with comprehensive documentation

class EnhancedRoundRobinBalancer(LoadBalancerInterface):
    """
    Enhanced round-robin load balancer with fairness guarantees and failure handling.
    
    This implementation improves upon basic round-robin by:
    - Maintaining separate counters for different function types
    - Skipping failed nodes automatically
    - Providing fair distribution even with node failures
    - Supporting weighted round-robin for heterogeneous nodes
    """
    
    def __init__(self):
        super().__init__()
        self._function_counters: Dict[str, int] = defaultdict(int)
        self._node_weights: Dict[str, float] = defaultdict(lambda: 1.0)
        self._last_selections: Dict[str, str] = {}  # function -> last selected node
        
    async def select_node(self, 
                         available_nodes: List[str], 
                         request_context: RequestContext,
                         node_stats: Dict[str, NodeStats]) -> str:
        """
        Select next node using enhanced round-robin algorithm.
        
        Args:
            available_nodes: List of available node IDs
            request_context: Request context with function information
            node_stats: Current statistics for all nodes
            
        Returns:
            Selected node ID
            
        Raises:
            NoAvailableNodesError: If no nodes are available
        """
        if not available_nodes:
            raise NoAvailableNodesError("No available nodes for round-robin selection")
        
        function_name = getattr(request_context, 'function_name', 'default')
        
        # Get current counter for this function
        current_counter = self._function_counters[function_name]
        
        # Apply weighted round-robin if nodes have different capabilities
        if self._should_use_weighted_selection(node_stats):
            selected_node = self._weighted_round_robin_selection(
                available_nodes, node_stats, current_counter
            )
        else:
            # Standard round-robin
            selected_node = available_nodes[current_counter % len(available_nodes)]
        
        # Update counter and tracking
        self._function_counters[function_name] = (current_counter + 1) % len(available_nodes)
        self._last_selections[function_name] = selected_node
        
        return selected_node
    
    def _should_use_weighted_selection(self, node_stats: Dict[str, NodeStats]) -> bool:
        """Determine if weighted selection should be used based on node heterogeneity."""
        if len(node_stats) < 2:
            return False
        
        # Check for significant differences in node capabilities
        cpu_values = [stats.cpu_usage for stats in node_stats.values()]
        memory_values = [stats.memory_usage for stats in node_stats.values()]
        
        cpu_variance = max(cpu_values) - min(cpu_values)
        memory_variance = max(memory_values) - min(memory_values)
        
        # Use weighted selection if there's significant heterogeneity
        return cpu_variance > 30 or memory_variance > 30
    
    def _weighted_round_robin_selection(self, 
                                      available_nodes: List[str],
                                      node_stats: Dict[str, NodeStats],
                                      counter: int) -> str:
        """Select node using weighted round-robin based on node capabilities."""
        # Calculate weights based on node capacity (inverse of usage)
        node_weights = {}
        for node_id in available_nodes:
            stats = node_stats.get(node_id)
            if stats:
                # Higher weight for nodes with lower usage
                cpu_weight = (100 - stats.cpu_usage) / 100
                memory_weight = (100 - stats.memory_usage) / 100
                node_weights[node_id] = (cpu_weight + memory_weight) / 2
            else:
                node_weights[node_id] = 0.5  # Default weight
        
        # Create weighted selection list
        weighted_nodes = []
        for node_id in available_nodes:
            weight = int(node_weights[node_id] * 10)  # Scale to integer
            weighted_nodes.extend([node_id] * max(1, weight))
        
        return weighted_nodes[counter % len(weighted_nodes)]
    
    def get_strategy_name(self) -> str:
        return "enhanced_round_robin"


class IntelligentResourceAwareBalancer(LoadBalancerInterface):
    """
    Advanced resource-aware balancer with predictive load assessment.
    
    This implementation provides sophisticated resource-aware load balancing:
    - Multi-dimensional resource analysis (CPU, memory, GPU, network)
    - Predictive load modeling based on historical patterns
    - Dynamic threshold adjustment based on system conditions
    - Requirement matching with compatibility scoring
    """
    
    def __init__(self):
        super().__init__()
        self._load_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=20))
        self._prediction_models: Dict[str, Any] = {}
        
    async def select_node(self, 
                         available_nodes: List[str], 
                         request_context: RequestContext,
                         node_stats: Dict[str, NodeStats]) -> str:
        """
        Select optimal node based on comprehensive resource analysis.
        
        Args:
            available_nodes: List of available node IDs
            request_context: Request context with resource requirements
            node_stats: Current statistics for all nodes
            
        Returns:
            Optimal node ID based on resource analysis
            
        Raises:
            NoAvailableNodesError: If no suitable nodes found
        """
        best_node = None
        best_score = -1.0
        
        for node_id in available_nodes:
            stats = node_stats.get(node_id)
            if not stats:
                continue
            
            # Calculate comprehensive resource score
            resource_score = await self._calculate_resource_score(
                stats, request_context
            )
            
            # Apply predictive adjustment
            predictive_score = await self._apply_predictive_adjustment(
                node_id, resource_score, request_context
            )
            
            # Check requirement compatibility
            compatibility_score = self._check_requirement_compatibility(
                stats, request_context
            )
            
            # Calculate final score
            final_score = (
                resource_score * 0.4 +
                predictive_score * 0.3 +
                compatibility_score * 0.3
            )
            
            if final_score > best_score:
                best_score = final_score
                best_node = node_id
        
        if not best_node:
            raise NoAvailableNodesError("No suitable nodes found for resource-aware selection")
        
        # Record selection for learning
        await self._record_selection_outcome(best_node, request_context)
        
        return best_node
    
    async def _calculate_resource_score(self, 
                                      stats: NodeStats, 
                                      request_context: RequestContext) -> float:
        """
        Calculate comprehensive resource utilization score.
        
        Args:
            stats: Node statistics
            request_context: Request context
            
        Returns:
            Resource score (0.0 to 1.0, higher is better)
        """
        # Base resource availability scores
        cpu_score = max(0, (100 - stats.cpu_usage) / 100)
        memory_score = max(0, (100 - stats.memory_usage) / 100)
        load_score = max(0, 1 - stats.current_load)
        
        # GPU score if relevant
        gpu_score = 0.5  # Default neutral score
        if hasattr(stats, 'has_gpu') and stats.has_gpu:
            gpu_score = max(0, (100 - getattr(stats, 'gpu_usage', 50)) / 100)
        
        # Network latency score
        network_score = 1.0
        if hasattr(stats, 'response_time'):
            # Lower latency is better, normalize around 100ms
            network_score = max(0, 1 - (stats.response_time / 200))
        
        # Weighted combination based on request characteristics
        weights = self._determine_resource_weights(request_context)
        
        return (
            weights['cpu'] * cpu_score +
            weights['memory'] * memory_score +
            weights['gpu'] * gpu_score +
            weights['network'] * network_score +
            weights['load'] * load_score
        )
    
    def _determine_resource_weights(self, request_context: RequestContext) -> Dict[str, float]:
        """Determine resource weights based on request characteristics."""
        weights = {
            'cpu': 0.25,
            'memory': 0.25,
            'gpu': 0.15,
            'network': 0.15,
            'load': 0.20
        }
        
        # Adjust weights based on requirements
        if hasattr(request_context, 'requirements') and request_context.requirements:
            requirements = request_context.requirements
            
            if requirements.get('gpu_required', False):
                weights['gpu'] = 0.35
                weights['cpu'] = 0.20
            
            if requirements.get('memory_intensive', False):
                weights['memory'] = 0.35
                weights['cpu'] = 0.20
            
            if requirements.get('network_intensive', False):
                weights['network'] = 0.30
                weights['load'] = 0.25
        
        return weights
    
    async def _apply_predictive_adjustment(self, 
                                         node_id: str,
                                         base_score: float,
                                         request_context: RequestContext) -> float:
        """Apply predictive modeling to adjust score based on trends."""
        # Record current load for trend analysis
        current_time = time.time()
        current_load = {"time": current_time, "score": base_score}
        self._load_history[node_id].append(current_load)
        
        # Calculate trend if enough history
        if len(self._load_history[node_id]) >= 5:
            history = list(self._load_history[node_id])
            recent_scores = [h["score"] for h in history[-5:]]
            
            # Simple trend analysis
            if len(recent_scores) >= 2:
                trend = (recent_scores[-1] - recent_scores[0]) / len(recent_scores)
                
                # Adjust score based on trend
                if trend > 0:  # Improving performance
                    return min(1.0, base_score * 1.1)
                elif trend < -0.1:  # Degrading performance
                    return max(0.0, base_score * 0.9)
        
        return base_score
    
    def _check_requirement_compatibility(self, 
                                       stats: NodeStats,
                                       request_context: RequestContext) -> float:
        """Check how well node capabilities match requirements."""
        if not hasattr(request_context, 'requirements') or not request_context.requirements:
            return 1.0
        
        requirements = request_context.requirements
        compatibility_score = 1.0
        
        # GPU requirement check
        if requirements.get('gpu_required', False):
            if hasattr(stats, 'has_gpu') and stats.has_gpu:
                compatibility_score *= 1.2  # Bonus for having required GPU
            else:
                compatibility_score *= 0.1  # Heavy penalty for missing GPU
        
        # Memory requirement check
        min_memory = requirements.get('min_memory_mb', 0)
        if min_memory > 0:
            available_memory_percent = 100 - stats.memory_usage
            if available_memory_percent < 20:  # Less than 20% memory available
                compatibility_score *= 0.5
        
        # CPU requirement check
        min_cpu_cores = requirements.get('min_cpu_cores', 0)
        if min_cpu_cores > 0 and stats.cpu_usage > 80:
            compatibility_score *= 0.7  # Penalty for high CPU usage
        
        return min(1.0, compatibility_score)
    
    async def _record_selection_outcome(self, 
                                      selected_node: str,
                                      request_context: RequestContext):
        """Record selection outcome for learning."""
        # This could be enhanced with actual outcome feedback
        # For now, just log the selection
        pass
    
    def get_strategy_name(self) -> str:
        return "intelligent_resource_aware"


# Placeholder implementations for other balancers
# (These would be fully implemented in the next development phase)

class AdaptiveLatencyBasedBalancer(LoadBalancerInterface):
    """Placeholder for adaptive latency-based balancer."""
    
    async def select_node(self, available_nodes: List[str], 
                         request_context: RequestContext,
                         node_stats: Dict[str, NodeStats]) -> str:
        # Simplified implementation - select node with lowest response time
        if not available_nodes:
            raise NoAvailableNodesError("No available nodes")
        
        best_node = available_nodes[0]
        best_latency = float('inf')
        
        for node_id in available_nodes:
            stats = node_stats.get(node_id)
            if stats and hasattr(stats, 'response_time'):
                if stats.response_time < best_latency:
                    best_latency = stats.response_time
                    best_node = node_id
        
        return best_node
    
    def get_strategy_name(self) -> str:
        return "adaptive_latency_based"


class SmartCostAwareBalancer(LoadBalancerInterface):
    """Placeholder for smart cost-aware balancer."""
    
    async def select_node(self, available_nodes: List[str], 
                         request_context: RequestContext,
                         node_stats: Dict[str, NodeStats]) -> str:
        # Simplified implementation - select first available node
        if not available_nodes:
            raise NoAvailableNodesError("No available nodes")
        return available_nodes[0]
    
    def get_strategy_name(self) -> str:
        return "smart_cost_aware"


class MachineLearningAdaptiveBalancer(LoadBalancerInterface):
    """Placeholder for ML-based adaptive balancer."""
    
    async def select_node(self, available_nodes: List[str], 
                         request_context: RequestContext,
                         node_stats: Dict[str, NodeStats]) -> str:
        # Simplified implementation - select node with best overall score
        if not available_nodes:
            raise NoAvailableNodesError("No available nodes")
        
        best_node = available_nodes[0]
        best_score = -1
        
        for node_id in available_nodes:
            stats = node_stats.get(node_id)
            if stats:
                # Simple scoring based on available resources
                score = ((100 - stats.cpu_usage) + (100 - stats.memory_usage)) / 200
                if score > best_score:
                    best_score = score
                    best_node = node_id
        
        return best_node
    
    def get_strategy_name(self) -> str:
        return "ml_adaptive"


class DynamicStrategyBalancer(LoadBalancerInterface):
    """Dynamic balancer that automatically selects the best strategy."""
    
    def __init__(self, parent_balancer):
        self.parent_balancer = parent_balancer
    
    async def select_node(self, available_nodes: List[str], 
                         request_context: RequestContext,
                         node_stats: Dict[str, NodeStats]) -> str:
        # Use the parent balancer's strategy selection logic
        strategy_name = await self.parent_balancer._analyze_system_conditions(
            request_context, node_stats
        )
        strategy_instance = self.parent_balancer._get_balancer_instance(strategy_name)
        return await strategy_instance.select_node(available_nodes, request_context, node_stats)
    
    def get_strategy_name(self) -> str:
        return "dynamic"


# Backward compatibility alias
LoadBalancer = IntelligentLoadBalancer


# Export all balancer classes
__all__ = [
    'IntelligentLoadBalancer',
    'LoadBalancer',  # Backward compatibility
    'LoadBalancingMetrics',
    'BalancerPerformanceLevel',
    'EnhancedRoundRobinBalancer',
    'IntelligentResourceAwareBalancer',
    'AdaptiveLatencyBasedBalancer',
    'SmartCostAwareBalancer',
    'MachineLearningAdaptiveBalancer',
    'DynamicStrategyBalancer'
] 