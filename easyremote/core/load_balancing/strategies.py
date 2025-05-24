# Load balancing strategies and types for EasyRemote
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


class LoadBalancingStrategy(Enum):
    """Load balancing strategy enumeration"""
    ROUND_ROBIN = "round_robin"
    RESOURCE_AWARE = "resource_aware"
    LATENCY_BASED = "latency_based"
    COST_AWARE = "cost_aware"
    SMART_ADAPTIVE = "smart_adaptive"
    DYNAMIC = "dynamic"


@dataclass
class NodeCapabilities:
    """Node capability information"""
    gpu: Optional[Dict[str, Any]] = None
    cpu_cores: Optional[int] = None
    memory_gb: Optional[float] = None
    max_concurrent: int = 1
    priority: str = "medium"
    cost_per_hour: Optional[float] = None
    performance_tier: str = "standard"
    specialization: Optional[str] = None
    location: Optional[Dict[str, Any]] = None


@dataclass
class LoadBalancingConfig:
    """Configuration for load balancing"""
    strategy: LoadBalancingStrategy = LoadBalancingStrategy.RESOURCE_AWARE
    fallback_strategy: Optional[LoadBalancingStrategy] = None
    requirements: Optional[Dict[str, Any]] = None
    preferences: Optional[Dict[str, Any]] = None
    health_check: Optional[Dict[str, Any]] = None
    scaling: Optional[Dict[str, Any]] = None
    budget: Optional[Dict[str, Any]] = None
    timeout: float = 30.0
    max_retries: int = 3


@dataclass
class NodeStats:
    """Real-time node statistics"""
    node_id: str
    cpu_usage: float
    memory_usage: float
    gpu_usage: float = 0.0
    current_load: float = 0.0
    queue_length: int = 0
    response_time: float = 0.0
    success_rate: float = 1.0
    has_gpu: bool = False
    capabilities: Optional[NodeCapabilities] = None
    last_updated: float = 0.0


@dataclass
class RequestContext:
    """Context information for a request"""
    function_name: str
    data_size: int = 0
    complexity_score: float = 1.0
    requirements: Optional[Dict[str, Any]] = None
    client_location: Optional[Dict[str, Any]] = None
    priority: str = "normal"
    timeout: Optional[float] = None
    cost_limit: Optional[float] = None


class LoadBalancerInterface(ABC):
    """Abstract base class for load balancers"""
    
    @abstractmethod
    async def select_node(self, 
                         available_nodes: List[str], 
                         request_context: RequestContext,
                         node_stats: Dict[str, NodeStats]) -> str:
        """
        Select the best node for the given request
        
        Args:
            available_nodes: List of available node IDs
            request_context: Context information about the request
            node_stats: Real-time statistics for all nodes
            
        Returns:
            Selected node ID
            
        Raises:
            NoAvailableNodesError: When no suitable nodes are available
        """
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get the name of this load balancing strategy"""
        pass 