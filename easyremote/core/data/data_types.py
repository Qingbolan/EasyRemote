#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EasyRemote Core Data Types Module

This module defines the fundamental data structures used throughout the EasyRemote
distributed computing framework. These data types provide the foundation for
communication between clients, servers, and compute nodes.

Key Components:
- FunctionInfo: Metadata and configuration for registered remote functions
- NodeInfo: Comprehensive information about compute nodes including health status
- Additional utility classes for performance monitoring and load balancing

Author: EasyRemote Team
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Callable, Any, List, Set
from datetime import datetime
from enum import Enum
import uuid


class NodeStatus(Enum):
    """
    Enumeration of possible node states in the distributed system.
    
    This enum provides type safety and clear semantics for node status tracking.
    """
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    RECONNECTING = "reconnecting"
    INITIALIZING = "initializing"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    ONLINE = "online"
    OFFLINE = "offline"


class FunctionType(Enum):
    """
    Enumeration of supported function types for enhanced type checking.
    """
    SYNC = "synchronous"
    ASYNC = "asynchronous"
    GENERATOR = "generator"
    ASYNC_GENERATOR = "async_generator"
    STREAM = "stream"  # For streaming functions (generators)
    ASYNC_STREAM = "async_stream"  # For async streaming functions (async generators)


@dataclass(frozen=True)
class ResourceRequirements:
    """
    Defines resource requirements for function execution.
    
    This immutable dataclass helps with load balancing decisions by specifying
    the computational resources needed for optimal function execution.
    
    Attributes:
        min_cpu_cores: Minimum CPU cores required
        min_memory_mb: Minimum memory in megabytes
        gpu_required: Whether GPU acceleration is needed
        gpu_memory_mb: Minimum GPU memory if GPU is required
        max_execution_time: Maximum allowed execution time in seconds
        priority: Execution priority (1-10, where 10 is highest)
    """
    min_cpu_cores: int = 1
    min_memory_mb: int = 512
    gpu_required: bool = False
    gpu_memory_mb: int = 0
    max_execution_time: int = 300
    priority: int = 5
    
    def __post_init__(self):
        """Validate resource requirements after initialization."""
        if not (1 <= self.priority <= 10):
            raise ValueError("Priority must be between 1 and 10")
        if self.min_cpu_cores < 1:
            raise ValueError("Minimum CPU cores must be at least 1")
        if self.min_memory_mb < 0:
            raise ValueError("Memory requirement cannot be negative")


@dataclass
class FunctionInfo:
    """
    Comprehensive metadata for registered remote functions.
    
    This class encapsulates all information needed to properly route, execute,
    and manage remote function calls within the distributed system.
    
    Attributes:
        name: Unique function identifier
        callable: The actual function object (None for remote references)
        function_type: Type of function (sync, async, generator, etc.)
        node_id: ID of the node where this function is registered
        resource_requirements: Computational resource needs
        load_balancing_enabled: Whether this function participates in load balancing
        max_concurrent_calls: Maximum simultaneous executions allowed
        tags: Metadata tags for categorization and filtering
        created_at: Timestamp when function was registered
        last_called: Timestamp of most recent invocation
        call_count: Total number of times function has been called
        average_execution_time: Running average of execution time in seconds
        context_data: Additional metadata and configuration data
    """
    name: str
    callable: Optional[Callable] = None
    function_type: FunctionType = FunctionType.SYNC
    node_id: Optional[str] = None
    resource_requirements: ResourceRequirements = field(default_factory=ResourceRequirements)
    load_balancing_enabled: bool = False
    max_concurrent_calls: int = 1
    tags: Set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.now)
    last_called: Optional[datetime] = None
    call_count: int = 0
    average_execution_time: float = 0.0
    context_data: Dict[str, Any] = field(default_factory=dict)
    
    # Legacy compatibility properties
    @property
    def is_async(self) -> bool:
        """Check if function is asynchronous (legacy compatibility)."""
        return self.function_type in (FunctionType.ASYNC, FunctionType.ASYNC_GENERATOR, FunctionType.ASYNC_STREAM)
    
    @property
    def is_generator(self) -> bool:
        """Check if function is a generator (legacy compatibility)."""
        return self.function_type in (FunctionType.GENERATOR, FunctionType.ASYNC_GENERATOR, FunctionType.STREAM, FunctionType.ASYNC_STREAM)
    
    @property
    def is_streaming(self) -> bool:
        """Check if function is a streaming function."""
        return self.function_type in (FunctionType.STREAM, FunctionType.ASYNC_STREAM, FunctionType.GENERATOR, FunctionType.ASYNC_GENERATOR)
    
    def update_call_statistics(self, execution_time: float):
        """
        Update function call statistics after execution.
        
        Args:
            execution_time: Time taken for the function execution in seconds
        """
        self.call_count += 1
        self.last_called = datetime.now()
        
        # Update running average using incremental formula
        if self.call_count == 1:
            self.average_execution_time = execution_time
        else:
            alpha = 2.0 / (self.call_count + 1)  # Exponential moving average
            self.average_execution_time = (
                alpha * execution_time + (1 - alpha) * self.average_execution_time
            )
    
    def is_compatible_with_requirements(self, requirements: Dict[str, Any]) -> bool:
        """
        Check if this function is compatible with given execution requirements.
        
        Args:
            requirements: Dictionary of requirements to check against
            
        Returns:
            True if function can satisfy the requirements, False otherwise
        """
        if "tags" in requirements:
            required_tags = set(requirements["tags"])
            if not required_tags.issubset(self.tags):
                return False
        
        if "max_execution_time" in requirements:
            if self.average_execution_time > requirements["max_execution_time"]:
                return False
        
        return True
    
    def set_context_data(self, key: str, value: Any):
        """
        Set additional context data for this function.
        
        Args:
            key: Context data key
            value: Context data value
        """
        self.context_data[key] = value
    
    def get_context_data(self, key: str, default: Any = None) -> Any:
        """
        Get context data by key.
        
        Args:
            key: Context data key
            default: Default value if key not found
            
        Returns:
            Context data value or default
        """
        return self.context_data.get(key, default)


@dataclass
class NodeHealthMetrics:
    """
    Real-time health and performance metrics for compute nodes.
    
    This class provides detailed monitoring data used for load balancing
    decisions and system health assessment.
    """
    cpu_usage_percent: float = 0.0
    memory_usage_percent: float = 0.0
    memory_total_mb: int = 0
    memory_available_mb: int = 0
    gpu_usage_percent: float = 0.0
    gpu_memory_usage_percent: float = 0.0
    gpu_temperature_celsius: float = 0.0
    disk_usage_percent: float = 0.0
    network_latency_ms: float = 0.0
    active_connections: int = 0
    concurrent_executions: int = 0
    last_updated: datetime = field(default_factory=datetime.now)
    
    def get_overall_load_score(self) -> float:
        """
        Calculate an overall load score for load balancing decisions.
        
        Returns:
            A score from 0.0 (no load) to 1.0 (maximum load)
        """
        # Weighted combination of different metrics
        weights = {
            "cpu": 0.3,
            "memory": 0.25,
            "gpu": 0.2,
            "concurrent": 0.15,
            "network": 0.1
        }
        
        cpu_score = min(self.cpu_usage_percent / 100.0, 1.0)
        memory_score = min(self.memory_usage_percent / 100.0, 1.0)
        gpu_score = min(self.gpu_usage_percent / 100.0, 1.0)
        
        # Normalize concurrent executions (assuming max 10 concurrent)
        concurrent_score = min(self.concurrent_executions / 10.0, 1.0)
        
        # Normalize network latency (assuming 100ms as high latency)
        network_score = min(self.network_latency_ms / 100.0, 1.0)
        
        return (
            weights["cpu"] * cpu_score +
            weights["memory"] * memory_score +
            weights["gpu"] * gpu_score +
            weights["concurrent"] * concurrent_score +
            weights["network"] * network_score
        )
    
    def is_healthy(self) -> bool:
        """
        Determine if the node is in a healthy state.
        
        Returns:
            True if node is healthy, False if it's overloaded or has issues
        """
        # Define health thresholds
        max_cpu = 90.0
        max_memory = 85.0
        max_gpu_temp = 85.0
        max_latency = 1000.0  # 1 second
        
        return (
            self.cpu_usage_percent < max_cpu and
            self.memory_usage_percent < max_memory and
            self.gpu_temperature_celsius < max_gpu_temp and
            self.network_latency_ms < max_latency
        )
    
    def update_metrics(self, cpu_usage: float = None, memory_usage: float = None, 
                      gpu_usage: float = None, active_connections: int = None):
        """
        Update health metrics with new values.
        
        Args:
            cpu_usage: CPU usage percentage
            memory_usage: Memory usage percentage
            gpu_usage: GPU usage percentage
            active_connections: Number of active connections
        """
        if cpu_usage is not None:
            self.cpu_usage_percent = cpu_usage
        if memory_usage is not None:
            self.memory_usage_percent = memory_usage
        if gpu_usage is not None:
            self.gpu_usage_percent = gpu_usage
        if active_connections is not None:
            self.active_connections = active_connections
        
        self.last_updated = datetime.now()


@dataclass
class NodeInfo:
    """
    Comprehensive information about compute nodes in the distributed system.
    
    This class maintains complete state information for each compute node,
    including registered functions, health metrics, and connection status.
    
    Attributes:
        node_id: Unique identifier for the compute node
        functions: Dictionary of registered functions by name
        last_heartbeat: Timestamp of most recent heartbeat
        status: Current operational status of the node
        health_metrics: Real-time performance and resource usage data
        capabilities: Set of node capabilities (e.g., "gpu", "high_memory")
        location: Optional geographical or network location identifier
        version: Software version running on the node
        startup_time: When the node was started
        total_requests_handled: Cumulative count of processed requests
        error_count: Number of errors encountered since startup
    """
    node_id: str
    functions: Dict[str, FunctionInfo] = field(default_factory=dict)
    last_heartbeat: datetime = field(default_factory=datetime.now)
    status: NodeStatus = NodeStatus.INITIALIZING
    health_metrics: NodeHealthMetrics = field(default_factory=NodeHealthMetrics)
    capabilities: Set[str] = field(default_factory=set)
    location: Optional[str] = None
    version: str = "1.0.0"
    startup_time: datetime = field(default_factory=datetime.now)
    total_requests_handled: int = 0
    error_count: int = 0
    
    def __post_init__(self):
        """Initialize derived attributes after object creation."""
        if not self.node_id:
            self.node_id = f"node-{uuid.uuid4().hex[:8]}"
    
    def add_function(self, func_info: FunctionInfo):
        """
        Register a new function on this node.
        
        Args:
            func_info: Complete function information to register
        """
        func_info.node_id = self.node_id
        self.functions[func_info.name] = func_info
    
    def remove_function(self, function_name: str) -> bool:
        """
        Unregister a function from this node.
        
        Args:
            function_name: Name of function to remove
            
        Returns:
            True if function was removed, False if it didn't exist
        """
        return self.functions.pop(function_name, None) is not None
    
    def get_function(self, function_name: str) -> Optional[FunctionInfo]:
        """
        Retrieve function information by name.
        
        Args:
            function_name: Name of the function to retrieve
            
        Returns:
            FunctionInfo if found, None otherwise
        """
        return self.functions.get(function_name)
    
    def get_available_functions(self, requirements: Optional[Dict[str, Any]] = None) -> List[FunctionInfo]:
        """
        Get list of functions that can satisfy given requirements.
        
        Args:
            requirements: Optional requirements dictionary
            
        Returns:
            List of compatible function information objects
        """
        if requirements is None:
            return list(self.functions.values())
        
        return [
            func for func in self.functions.values()
            if func.is_compatible_with_requirements(requirements)
        ]
    
    def update_heartbeat(self):
        """Update the last heartbeat timestamp and set status to connected."""
        self.last_heartbeat = datetime.now()
        if self.status in (NodeStatus.INITIALIZING, NodeStatus.RECONNECTING):
            self.status = NodeStatus.CONNECTED
    
    def is_alive(self, timeout_seconds: int = 30) -> bool:
        """
        Check if node is considered alive based on heartbeat timing.
        
        Args:
            timeout_seconds: Maximum allowed time since last heartbeat
            
        Returns:
            True if node is considered alive, False otherwise
        """
        if self.status == NodeStatus.DISCONNECTED:
            return False
        
        time_since_heartbeat = (datetime.now() - self.last_heartbeat).total_seconds()
        return time_since_heartbeat <= timeout_seconds
    
    def increment_request_count(self):
        """Increment the total requests handled counter."""
        self.total_requests_handled += 1
    
    def increment_error_count(self):
        """Increment the error count and potentially update status."""
        self.error_count += 1
        
        # If error rate becomes too high, mark node as having issues
        if self.total_requests_handled > 0:
            error_rate = self.error_count / self.total_requests_handled
            if error_rate > 0.1:  # More than 10% error rate
                self.status = NodeStatus.ERROR
    
    def get_load_score(self) -> float:
        """
        Get the current load score for this node.
        
        Returns:
            Load score from 0.0 (no load) to 1.0 (maximum load)
        """
        return self.health_metrics.get_overall_load_score()
    
    def can_handle_function(self, function_name: str, requirements: Optional[Dict[str, Any]] = None) -> bool:
        """
        Check if this node can handle a specific function call.
        
        Args:
            function_name: Name of the function to check
            requirements: Optional execution requirements
            
        Returns:
            True if node can handle the function, False otherwise
        """
        # Check if node is alive and healthy
        if not self.is_alive() or not self.health_metrics.is_healthy():
            return False
        
        # Check if function exists
        func_info = self.get_function(function_name)
        if func_info is None:
            return False
        
        # Check if function can satisfy requirements
        if requirements and not func_info.is_compatible_with_requirements(requirements):
            return False
        
        # Check if node has capacity for more concurrent calls
        if func_info.max_concurrent_calls > 0:
            current_concurrent = self.health_metrics.concurrent_executions
            if current_concurrent >= func_info.max_concurrent_calls:
                return False
        
        return True


# Type aliases for convenience
NodeRegistry = Dict[str, NodeInfo]
FunctionRegistry = Dict[str, List[FunctionInfo]]



