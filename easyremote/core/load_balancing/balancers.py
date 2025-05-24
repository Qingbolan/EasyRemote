# Load balancing algorithms for EasyRemote
import time
from typing import Dict, List
from collections import defaultdict

from ..utils.logger import ModernLogger
from ..utils.exceptions import NoAvailableNodesError
from .strategies import (
    LoadBalancerInterface, 
    RequestContext, 
    NodeStats,
    LoadBalancingStrategy
)


class LoadBalancer(ModernLogger):
    """Main load balancer that routes requests to optimal nodes"""
    
    def __init__(self, gateway_instance):
        super().__init__(name="LoadBalancer")
        self.gateway = gateway_instance
        self.balancers = {
            LoadBalancingStrategy.ROUND_ROBIN: RoundRobinBalancer(),
            LoadBalancingStrategy.RESOURCE_AWARE: ResourceAwareBalancer(),
            LoadBalancingStrategy.LATENCY_BASED: LatencyBasedBalancer(),
            LoadBalancingStrategy.COST_AWARE: CostAwareBalancer(),
            LoadBalancingStrategy.SMART_ADAPTIVE: SmartAdaptiveBalancer()
        }
        
    async def route_request(self, 
                          function_name: str, 
                          request_context: RequestContext,
                          balancing_config: Dict) -> str:
        """Route request to the optimal node"""
        
        # 1. Find all nodes that provide this function
        available_nodes = await self._find_function_providers(function_name)
        
        if not available_nodes:
            raise NoAvailableNodesError(f"No nodes provide function '{function_name}'")
        
        # 2. Filter healthy nodes
        healthy_nodes = await self._filter_healthy_nodes(available_nodes)
        
        if not healthy_nodes:
            raise NoAvailableNodesError(f"No healthy nodes available for '{function_name}'")
        
        # 3. Get node statistics
        node_stats = await self._get_node_statistics(healthy_nodes)
        
        # 4. Select load balancing strategy
        strategy = LoadBalancingStrategy(balancing_config.get("strategy", "resource_aware"))
        balancer = self.balancers.get(strategy)
        
        if not balancer:
            self.warning(f"Unknown strategy {strategy}, falling back to resource_aware")
            balancer = self.balancers[LoadBalancingStrategy.RESOURCE_AWARE]
        
        # 5. Select optimal node
        selected_node = await balancer.select_node(healthy_nodes, request_context, node_stats)
        
        self.debug(f"Selected node {selected_node} for function {function_name} using {strategy.value}")
        return selected_node
    
    async def _find_function_providers(self, function_name: str) -> List[str]:
        """Find all nodes that provide the specified function"""
        providers = []
        
        async with self.gateway._lock:
            for node_id, node_info in self.gateway._nodes.items():
                if function_name in node_info.functions:
                    providers.append(node_id)
        
        return providers
    
    async def _filter_healthy_nodes(self, node_ids: List[str]) -> List[str]:
        """Filter out unhealthy nodes"""
        if not hasattr(self.gateway, 'health_monitor'):
            # If no health monitor, assume all nodes are healthy
            return node_ids
        
        healthy_nodes = []
        for node_id in node_ids:
            if self.gateway.health_monitor.is_node_available(node_id):
                healthy_nodes.append(node_id)
        
        return healthy_nodes
    
    async def _get_node_statistics(self, node_ids: List[str]) -> Dict[str, NodeStats]:
        """Get current statistics for the specified nodes"""
        stats = {}
        
        for node_id in node_ids:
            if hasattr(self.gateway, 'health_monitor'):
                node_stats = self.gateway.health_monitor.get_node_stats(node_id)
                if node_stats:
                    stats[node_id] = node_stats
                else:
                    # Create default stats if not available
                    stats[node_id] = NodeStats(
                        node_id=node_id,
                        cpu_usage=50.0,  # Default values
                        memory_usage=50.0,
                        current_load=0.5,
                        last_updated=time.time()
                    )
            else:
                # Create default stats
                stats[node_id] = NodeStats(
                    node_id=node_id,
                    cpu_usage=50.0,
                    memory_usage=50.0,
                    current_load=0.5,
                    last_updated=time.time()
                )
        
        return stats


class RoundRobinBalancer(LoadBalancerInterface):
    """Round-robin load balancing strategy"""
    
    def __init__(self):
        self.current_index = 0
        
    async def select_node(self, 
                         available_nodes: List[str], 
                         request_context: RequestContext,
                         node_stats: Dict[str, NodeStats]) -> str:
        """Select next node in round-robin fashion"""
        
        if not available_nodes:
            raise NoAvailableNodesError("No available nodes for round-robin selection")
        
        selected_node = available_nodes[self.current_index % len(available_nodes)]
        self.current_index += 1
        
        return selected_node
    
    def get_strategy_name(self) -> str:
        return "round_robin"


class ResourceAwareBalancer(LoadBalancerInterface):
    """Resource-aware load balancing strategy"""
    
    async def select_node(self, 
                         available_nodes: List[str], 
                         request_context: RequestContext,
                         node_stats: Dict[str, NodeStats]) -> str:
        """Select node based on resource utilization"""
        
        best_node = None
        best_score = -1
        
        for node_id in available_nodes:
            stats = node_stats.get(node_id)
            if not stats:
                continue
                
            # Calculate resource score (lower is better)
            cpu_score = (100 - stats.cpu_usage) / 100
            memory_score = (100 - stats.memory_usage) / 100
            gpu_score = (100 - stats.gpu_usage) / 100 if stats.has_gpu else 0.5
            load_score = (1 - stats.current_load)
            
            # Check requirements matching
            requirement_score = self._check_requirements_match(stats, request_context.requirements or {})
            
            # Combined score
            total_score = (cpu_score + memory_score + gpu_score + load_score + requirement_score) / 5
            
            if total_score > best_score:
                best_score = total_score
                best_node = node_id
        
        if not best_node:
            raise NoAvailableNodesError("No suitable nodes found for resource-aware selection")
        
        return best_node
    
    def _check_requirements_match(self, stats: NodeStats, requirements: Dict) -> float:
        """Check how well node capabilities match requirements"""
        if not requirements:
            return 1.0
        
        # Simple requirements matching - can be extended
        score = 1.0
        
        if requirements.get("gpu_required") and not stats.has_gpu:
            score *= 0.1  # Heavy penalty for missing GPU
        
        if "min_memory" in requirements:
            available_memory = 100 - stats.memory_usage
            required_memory = requirements["min_memory"]
            if available_memory < required_memory:
                score *= 0.5
        
        return score
    
    def get_strategy_name(self) -> str:
        return "resource_aware"


class LatencyBasedBalancer(LoadBalancerInterface):
    """Latency-based load balancing strategy"""
    
    def __init__(self):
        self.latency_history: Dict[str, List[float]] = defaultdict(list)
        self.max_history = 10
        
    async def select_node(self, 
                         available_nodes: List[str], 
                         request_context: RequestContext,
                         node_stats: Dict[str, NodeStats]) -> str:
        """Select node with lowest average latency"""
        
        best_node = None
        lowest_latency = float('inf')
        
        for node_id in available_nodes:
            stats = node_stats.get(node_id)
            if not stats:
                continue
                
            # Get average latency from history
            avg_latency = self._get_average_latency(node_id)
            
            # Adjust for current load
            current_load_penalty = stats.current_load * 50  # Add 50ms per load unit
            adjusted_latency = avg_latency + current_load_penalty
            
            if adjusted_latency < lowest_latency:
                lowest_latency = adjusted_latency
                best_node = node_id
        
        if not best_node:
            # Fallback to first available node
            best_node = available_nodes[0]
        
        return best_node
    
    def _get_average_latency(self, node_id: str) -> float:
        """Get average latency for a node"""
        history = self.latency_history[node_id]
        if not history:
            return 100.0  # Default latency assumption
        
        return sum(history) / len(history)
    
    def record_latency(self, node_id: str, latency: float):
        """Record latency measurement for a node"""
        history = self.latency_history[node_id]
        history.append(latency)
        
        # Keep only recent history
        if len(history) > self.max_history:
            history.pop(0)
    
    def get_strategy_name(self) -> str:
        return "latency_based"


class CostAwareBalancer(LoadBalancerInterface):
    """Cost-aware load balancing strategy"""
    
    async def select_node(self, 
                         available_nodes: List[str], 
                         request_context: RequestContext,
                         node_stats: Dict[str, NodeStats]) -> str:
        """Select most cost-effective node"""
        
        budget_limit = request_context.cost_limit
        eligible_nodes = []
        
        for node_id in available_nodes:
            stats = node_stats.get(node_id)
            if not stats or not stats.capabilities:
                continue
                
            # Estimate cost for this request
            cost_per_hour = getattr(stats.capabilities, 'cost_per_hour', 0.0)
            estimated_duration = self._estimate_execution_time(node_id, request_context)
            total_cost = cost_per_hour * (estimated_duration / 3600)  # Convert to hours
            
            # Check budget constraints
            if budget_limit and total_cost > budget_limit:
                continue
                
            # Calculate performance-to-cost ratio
            performance_score = self._calculate_performance_score(stats)
            cost_efficiency = performance_score / max(total_cost, 0.001)  # Avoid division by zero
            
            eligible_nodes.append({
                "node_id": node_id,
                "cost": total_cost,
                "efficiency": cost_efficiency
            })
        
        if not eligible_nodes:
            raise NoAvailableNodesError("No nodes within budget constraints")
        
        # Select node with best cost efficiency
        best_node = max(eligible_nodes, key=lambda x: x["efficiency"])
        return best_node["node_id"]
    
    def _estimate_execution_time(self, node_id: str, request_context: RequestContext) -> float:
        """Estimate execution time for the request"""
        # Simple estimation based on complexity and data size
        base_time = 60  # 1 minute base
        complexity_factor = request_context.complexity_score
        size_factor = request_context.data_size / 1000000  # Per MB
        
        return base_time * complexity_factor * (1 + size_factor)
    
    def _calculate_performance_score(self, stats: NodeStats) -> float:
        """Calculate performance score for a node"""
        # Higher score for better performance
        cpu_score = (100 - stats.cpu_usage) / 100
        memory_score = (100 - stats.memory_usage) / 100
        load_score = 1 - stats.current_load
        
        return (cpu_score + memory_score + load_score) / 3
    
    def get_strategy_name(self) -> str:
        return "cost_aware"


class SmartAdaptiveBalancer(LoadBalancerInterface):
    """Smart adaptive load balancing using ML-like predictions"""
    
    def __init__(self):
        self.performance_history: Dict[str, List[float]] = defaultdict(list)
        self.prediction_weights = {
            'cpu_usage': 0.3,
            'memory_usage': 0.2,
            'current_load': 0.3,
            'response_time': 0.2
        }
        
    async def select_node(self, 
                         available_nodes: List[str], 
                         request_context: RequestContext,
                         node_stats: Dict[str, NodeStats]) -> str:
        """Select node using adaptive prediction algorithm"""
        
        best_node = None
        best_predicted_performance = -1
        
        for node_id in available_nodes:
            stats = node_stats.get(node_id)
            if not stats:
                continue
                
            # Predict performance for this node
            predicted_performance = self._predict_performance(node_id, stats, request_context)
            
            if predicted_performance > best_predicted_performance:
                best_predicted_performance = predicted_performance
                best_node = node_id
        
        if not best_node:
            raise NoAvailableNodesError("No suitable nodes found for adaptive selection")
        
        return best_node
    
    def _predict_performance(self, 
                           node_id: str, 
                           stats: NodeStats, 
                           request_context: RequestContext) -> float:
        """Predict performance score for a node"""
        
        # Base performance calculation
        cpu_factor = (100 - stats.cpu_usage) / 100
        memory_factor = (100 - stats.memory_usage) / 100
        load_factor = 1 - stats.current_load
        response_factor = max(0, (200 - stats.response_time) / 200)  # Normalize response time
        
        # Weighted performance score
        base_score = (
            cpu_factor * self.prediction_weights['cpu_usage'] +
            memory_factor * self.prediction_weights['memory_usage'] +
            load_factor * self.prediction_weights['current_load'] +
            response_factor * self.prediction_weights['response_time']
        )
        
        # Apply historical performance adjustment
        historical_adjustment = self._get_historical_performance_factor(node_id)
        
        # Apply request-specific adjustments
        request_adjustment = self._get_request_specific_factor(stats, request_context)
        
        final_score = base_score * historical_adjustment * request_adjustment
        
        return final_score
    
    def _get_historical_performance_factor(self, node_id: str) -> float:
        """Get historical performance factor for a node"""
        history = self.performance_history[node_id]
        if not history:
            return 1.0  # Neutral factor for new nodes
        
        # Calculate trend from recent history
        if len(history) >= 2:
            recent_avg = sum(history[-3:]) / min(3, len(history))
            overall_avg = sum(history) / len(history)
            trend_factor = recent_avg / max(overall_avg, 0.001)
            return min(max(trend_factor, 0.5), 2.0)  # Clamp between 0.5 and 2.0
        
        return history[0] if history else 1.0
    
    def _get_request_specific_factor(self, stats: NodeStats, request_context: RequestContext) -> float:
        """Get request-specific performance factor"""
        factor = 1.0
        
        # GPU-intensive requests
        if request_context.requirements and request_context.requirements.get("gpu_required"):
            if stats.has_gpu:
                factor *= 1.2  # Bonus for having GPU
            else:
                factor *= 0.3  # Penalty for missing GPU
        
        # High complexity requests
        if request_context.complexity_score > 2.0:
            # Prefer nodes with lower current load for complex tasks
            factor *= (2 - stats.current_load)
        
        # Large data requests
        if request_context.data_size > 10000000:  # > 10MB
            # Prefer nodes with more available memory
            memory_available = (100 - stats.memory_usage) / 100
            factor *= (0.5 + memory_available)
        
        return factor
    
    def record_performance(self, node_id: str, performance_score: float):
        """Record actual performance for learning"""
        history = self.performance_history[node_id]
        history.append(performance_score)
        
        # Keep only recent history for adaptation
        max_history = 20
        if len(history) > max_history:
            history.pop(0)
    
    def get_strategy_name(self) -> str:
        return "smart_adaptive"


class DynamicLoadBalancer(LoadBalancerInterface):
    """Dynamic balancer that adapts strategy based on system state"""
    
    def __init__(self, gateway_instance):
        self.gateway = gateway_instance
        self.sub_balancers = {
            "resource_aware": ResourceAwareBalancer(),
            "latency_based": LatencyBasedBalancer(),
            "cost_aware": CostAwareBalancer(),
            "smart_adaptive": SmartAdaptiveBalancer()
        }
        
    async def select_node(self, 
                         available_nodes: List[str], 
                         request_context: RequestContext,
                         node_stats: Dict[str, NodeStats]) -> str:
        """Dynamically select strategy and node"""
        
        # Analyze current system state
        strategy = await self._adapt_strategy(node_stats, request_context)
        
        # Use the selected strategy
        balancer = self.sub_balancers[strategy]
        return await balancer.select_node(available_nodes, request_context, node_stats)
    
    async def _adapt_strategy(self, node_stats: Dict[str, NodeStats], request_context: RequestContext) -> str:
        """Dynamically choose the best strategy based on current conditions"""
        
        if not node_stats:
            return "resource_aware"  # Default fallback
        
        # Calculate average system metrics
        avg_cpu = sum(stats.cpu_usage for stats in node_stats.values()) / len(node_stats)
        avg_memory = sum(stats.memory_usage for stats in node_stats.values()) / len(node_stats)
        avg_load = sum(stats.current_load for stats in node_stats.values()) / len(node_stats)
        
        # High resource utilization - prioritize resource awareness
        if avg_cpu > 80 or avg_memory > 80:
            return "resource_aware"
        
        # High latency concerns - prioritize latency optimization
        avg_response_time = sum(stats.response_time for stats in node_stats.values()) / len(node_stats)
        if avg_response_time > 200:  # > 200ms
            return "latency_based"
        
        # Cost-sensitive requests
        if request_context.cost_limit and request_context.cost_limit < 5.0:
            return "cost_aware"
        
        # Default to smart adaptive for complex scenarios
        return "smart_adaptive"
    
    def get_strategy_name(self) -> str:
        return "dynamic" 