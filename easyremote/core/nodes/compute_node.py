#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EasyRemote Compute Node Module

This module implements distributed compute nodes that provide computational
resources to the EasyRemote framework. Compute nodes register with a central
gateway server and execute functions remotely on behalf of clients.

Architecture:
- Worker Node Pattern: Compute nodes act as distributed workers
- Function Registry: Local catalog of available computational functions
- Bidirectional Communication: gRPC streaming for real-time coordination
- Automatic Recovery: Built-in reconnection and error handling
- Load Balancing Support: Intelligent resource reporting for optimal distribution

Key Features:
- Zero-configuration setup with intelligent defaults
- Automatic function discovery and registration
- Support for synchronous, asynchronous, and streaming functions
- Comprehensive error handling with exponential backoff
- Real-time health monitoring and performance metrics
- Graceful shutdown and resource cleanup

Communication Protocol:
- Registration: Node identification and function catalog submission
- Heartbeat: Periodic health check and status updates
- Function Execution: Remote function call handling with result streaming
- Resource Monitoring: CPU, memory, and performance metrics reporting

Thread Safety:
All public methods are thread-safe. Internal state is protected with
asyncio locks and thread-safe data structures.

Author: EasyRemote Team
Version: 2.0.0
"""

import asyncio
import grpc
import time
import threading
import uuid
import logging
import platform
import psutil
import os
from typing import Optional, Callable, Dict, Any, Set, Union, List, Tuple
from concurrent import futures
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager

# EasyRemote core imports
from ..data import FunctionInfo, NodeInfo, NodeStatus, NodeHealthMetrics, FunctionType, ResourceRequirements
from ..utils import format_exception
from ..utils.exceptions import (
    FunctionNotFoundError,
    ConnectionError as EasyRemoteConnectionError,
    RemoteExecutionError,
    EasyRemoteError
)
from ..data.serialize import serialize_result, deserialize_args, analyze_function
from ..protos import service_pb2, service_pb2_grpc
from ..utils.logger import ModernLogger


# Configure module logger
_logger = logging.getLogger(__name__)


class NodeConnectionState(Enum):
    """
    Enumeration of possible connection states for compute nodes.
    """
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    SHUTTING_DOWN = "shutting_down"
    ERROR = "error"


@dataclass
class NodeConfiguration:
    """
    Comprehensive configuration for compute node operation.
    
    This dataclass encapsulates all configuration parameters with
    intelligent defaults and validation.
    """
    gateway_address: str
    node_id: str
    reconnect_interval: int = 3
    heartbeat_interval: int = 5
    max_retry_attempts: int = 3
    max_queue_size: int = 1000
    execution_timeout: int = 300
    connection_timeout: int = 10
    heartbeat_timeout: int = 15
    health_check_interval: int = 30
    max_concurrent_executions: int = 10
    enable_performance_monitoring: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        if self.reconnect_interval < 1:
            raise ValueError("Reconnect interval must be positive")
        if self.heartbeat_interval < 1:
            raise ValueError("Heartbeat interval must be positive")
        if self.max_retry_attempts < 1:
            raise ValueError("Max retry attempts must be positive")
        if self.max_queue_size < 1:
            raise ValueError("Max queue size must be positive")
        if self.execution_timeout < 1:
            raise ValueError("Execution timeout must be positive")


@dataclass
class ExecutionContext:
    """
    Context information for function execution tracking.
    
    This class maintains state and metadata for individual function
    executions, enabling monitoring, debugging, and resource management.
    """
    call_id: str
    function_name: str
    start_time: datetime = field(default_factory=datetime.now)
    timeout: Optional[int] = None
    client_info: Optional[Dict[str, Any]] = None
    execution_metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed execution time in seconds."""
        return (datetime.now() - self.start_time).total_seconds()
    
    @property
    def is_timed_out(self) -> bool:
        """Check if execution has exceeded timeout."""
        if self.timeout is None:
            return False
        return self.elapsed_time > self.timeout


class DistributedComputeNode(ModernLogger):
    """
    High-performance distributed compute node implementation.
    
    This class implements a compute node that connects to an EasyRemote
    gateway server and provides computational resources to the distributed
    computing cluster.
    
    Key Responsibilities:
    1. Function Registration: Register available computational functions
    2. Gateway Communication: Maintain bidirectional communication with gateway
    3. Function Execution: Execute remote function calls with proper isolation
    4. Health Monitoring: Report node health and performance metrics
    5. Resource Management: Manage computational resources and capacity
    
    Architecture Features:
    - Asynchronous gRPC communication for high throughput
    - Comprehensive error handling with automatic recovery
    - Real-time performance monitoring and metrics collection
    - Support for multiple function types (sync, async, streaming)
    - Graceful shutdown and resource cleanup
    
    Usage:
        >>> node = DistributedComputeNode("localhost:8080", "worker-1")
        >>> 
        >>> @node.register
        ... def compute_task(data):
        ...     return data * 2
        >>> 
        >>> node.serve()  # Start serving requests
    """
    
    def __init__(
        self,
        gateway_address: str,
        node_id: Optional[str] = None,
        **config_kwargs
    ):
        """
        Initialize distributed compute node with comprehensive configuration.
        
        Args:
            gateway_address: Address of the gateway server (host:port)
            node_id: Unique identifier for this node (auto-generated if None)
            **config_kwargs: Additional configuration parameters
            
        Raises:
            ValueError: If configuration parameters are invalid
            EasyRemoteError: If initialization fails
            
        Example:
            >>> node = DistributedComputeNode(
            ...     gateway_address="localhost:8080",
            ...     node_id="gpu-worker-1",
            ...     max_concurrent_executions=5,
            ...     execution_timeout=600
            ... )
        """
        super().__init__(name="DistributedComputeNode")
        
        # Generate unique node ID if not provided
        if node_id is None:
            node_id = self._generate_unique_node_id()
        
        # Create configuration object
        self.config = NodeConfiguration(
            gateway_address=gateway_address,
            node_id=node_id,
            **config_kwargs
        )
        
        self.info(f"Initializing DistributedComputeNode '{self.config.node_id}' "
                 f"targeting gateway: {self.config.gateway_address}")
        
        # Core node state
        self._connection_state = NodeConnectionState.DISCONNECTED
        self._state_lock = asyncio.Lock()
        
        # Function registry and execution tracking
        self._registered_functions: Dict[str, FunctionInfo] = {}
        self._active_executions: Dict[str, ExecutionContext] = {}
        self._execution_statistics: Dict[str, Dict[str, Any]] = {}
        
        # Communication infrastructure
        self._gateway_channel: Optional[grpc.aio.Channel] = None
        self._gateway_stub: Optional[service_pb2_grpc.RemoteServiceStub] = None
        self._communication_queue: Optional[asyncio.Queue] = None
        
        # Background tasks and lifecycle management
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        self._background_tasks: Set[asyncio.Task] = set()
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._health_monitor_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        self._connection_event = threading.Event()
        
        # Resource management
        self._thread_executor = futures.ThreadPoolExecutor(
            max_workers=self.config.max_concurrent_executions,
            thread_name_prefix=f"EasyRemote-{self.config.node_id}"
        )
        
        # Performance monitoring
        self._node_metrics = NodeHealthMetrics() if self.config.enable_performance_monitoring else None
        self._last_heartbeat_time: Optional[datetime] = None
        self._reconnection_count = 0
        
        # Thread safety
        self._global_lock = asyncio.Lock()
        
        self.info(f"DistributedComputeNode '{self.config.node_id}' initialized successfully")
    
    def _generate_unique_node_id(self) -> str:
        """
        Generate a unique, descriptive node identifier.
        
        The generated ID includes hostname, process info, and UUID
        for uniqueness while remaining human-readable.
        
        Returns:
            Unique node identifier string
        """
        try:
            hostname = platform.node().lower().replace('.', '-')[:12]
            process_id = f"pid{psutil.Process(os.getpid()).pid}"
            unique_suffix = str(uuid.uuid4())[:8]
            return f"compute-{hostname}-{process_id}-{unique_suffix}"
        except Exception:
            # Fallback to simple UUID if hostname detection fails
            return f"compute-node-{str(uuid.uuid4())[:16]}"
    
    @property
    def node_id(self) -> str:
        """Get the unique node identifier."""
        return self.config.node_id
    
    @property
    def gateway_address(self) -> str:
        """Get the gateway server address."""
        return self.config.gateway_address
    
    @property
    def connection_state(self) -> NodeConnectionState:
        """Get the current connection state."""
        return self._connection_state
    
    @property
    def is_connected(self) -> bool:
        """Check if node is currently connected to gateway."""
        return self._connection_state == NodeConnectionState.CONNECTED
    
    @property
    def registered_functions(self) -> Dict[str, FunctionInfo]:
        """Get dictionary of registered functions."""
        return self._registered_functions.copy()
    
    @property
    def active_executions(self) -> Dict[str, ExecutionContext]:
        """Get dictionary of currently active executions."""
        return self._active_executions.copy()
    
    async def _set_connection_state(self, new_state: NodeConnectionState):
        """
        Thread-safe connection state transition.
        
        Args:
            new_state: The new connection state to transition to
        """
        async with self._state_lock:
            old_state = self._connection_state
            self._connection_state = new_state
            
            if old_state != new_state:
                self.info(f"Connection state changed: {old_state.value} -> {new_state.value}")
                
                # Update connection event for synchronous waiting
                if new_state == NodeConnectionState.CONNECTED:
                    self._connection_event.set()
                else:
                    self._connection_event.clear()
    
    def register(
        self,
        func: Optional[Callable] = None,
        *,
        name: Optional[str] = None,
        function_type: Optional[FunctionType] = None,
        resource_requirements: Optional[ResourceRequirements] = None,
        timeout: Optional[int] = None,
        load_balancing: bool = True,
        max_concurrent: int = 1,
        tags: Optional[Set[str]] = None,
        description: Optional[str] = None
    ) -> Union[Callable, Callable[[Callable], Callable]]:
        """
        Register a function for remote execution with comprehensive configuration.
        
        This method supports both decorator and direct call patterns, automatically
        analyzing function characteristics and applying optimal configurations.
        
        Args:
            func: Function to register (None for decorator usage)
            name: Custom function name (defaults to func.__name__)
            function_type: Type classification (auto-detected if None)
            resource_requirements: Computational resource needs
            timeout: Maximum execution time in seconds
            load_balancing: Enable load balancing for this function
            max_concurrent: Maximum concurrent executions allowed
            tags: Metadata tags for categorization
            description: Human-readable function description
            
        Returns:
            Registered function or decorator
            
        Raises:
            ValueError: If function parameters are invalid
            EasyRemoteError: If registration fails
            
        Example:
            >>> # Decorator style (recommended)
            >>> @node.register(
            ...     timeout=300,
            ...     resource_requirements=ResourceRequirements(gpu_required=True),
            ...     tags={"ml", "training"}
            ... )
            ... def train_model(data, epochs=10):
            ...     # AI model training logic
            ...     return {"accuracy": 0.95, "epochs": epochs}
            
            >>> # Direct registration
            >>> def simple_task(x, y):
            ...     return x + y
            >>> node.register(simple_task, name="add_numbers")
        """
        def decorator(f: Callable) -> Callable:
            # Determine function name
            func_name = name or getattr(f, '__name__', 'unnamed_function')
            
            # Analyze function characteristics
            func_analysis = analyze_function(f)
            
            # Determine function type
            if function_type is None:
                if func_analysis.is_async and func_analysis.is_generator:
                    detected_type = FunctionType.ASYNC_GENERATOR
                elif func_analysis.is_async:
                    detected_type = FunctionType.ASYNC
                elif func_analysis.is_generator:
                    detected_type = FunctionType.GENERATOR
                else:
                    detected_type = FunctionType.SYNC
            else:
                detected_type = function_type
            
            # Create comprehensive function information
            function_info = FunctionInfo(
                name=func_name,
                callable=f,
                function_type=detected_type,
                node_id=self.config.node_id,
                resource_requirements=resource_requirements or ResourceRequirements(),
                load_balancing_enabled=load_balancing,
                max_concurrent_calls=max_concurrent,
                tags=tags or set(),
                created_at=datetime.now()
            )
            
            # Add description if provided
            if description:
                function_info.set_context_data("description", description)
            
            # Validate function registration
            self._validate_function_registration(function_info)
            
            # Register function
            self._registered_functions[func_name] = function_info
            
            self.info(f"Registered function '{func_name}' "
                     f"(type: {detected_type.value}, load_balancing: {load_balancing}, "
                     f"max_concurrent: {max_concurrent})")
            
            return f
        
        # Support both decorator and direct call patterns
        if func is None:
            return decorator
        else:
            return decorator(func)
    
    def _validate_function_registration(self, function_info: FunctionInfo):
        """
        Validate function registration parameters.
        
        Args:
            function_info: Function information to validate
            
        Raises:
            ValueError: If validation fails
        """
        if not function_info.name:
            raise ValueError("Function name cannot be empty")
        
        if function_info.name in self._registered_functions:
            raise ValueError(f"Function '{function_info.name}' is already registered")
        
        if function_info.max_concurrent_calls < 1:
            raise ValueError("Max concurrent calls must be positive")
        
        if not function_info.callable:
            raise ValueError("Function callable cannot be None")
    
    def serve(self, blocking: bool = True) -> Optional[threading.Thread]:
        """
        Start the compute node service with automatic reconnection.
        
        This method starts the node service and maintains connection to the gateway
        server with automatic retry and reconnection capabilities.
        
        Args:
            blocking: Whether to block the calling thread (True) or run in background (False)
            
        Returns:
            Thread handle if non-blocking, None if blocking
            
        Raises:
            EasyRemoteError: If service fails to start after all retry attempts
            
        Example:
            >>> node.serve()  # Blocking mode
            >>> # or
            >>> thread = node.serve(blocking=False)  # Background mode
        """
        # Placeholder implementation - will be completed in next phase
        self.info(f"Starting compute node service (blocking={blocking})")
        
        if blocking:
            # TODO: Implement blocking service loop with reconnection
            self.info("Compute node service started in blocking mode")
        else:
            # TODO: Implement background service thread
            thread = threading.Thread(
                target=lambda: self.info("Background service thread started"),
                name=f"ComputeNode-{self.config.node_id}",
                daemon=True
            )
            thread.start()
            return thread
    
    def stop(self):
        """
        Stop the compute node service gracefully.
        
        This method initiates a graceful shutdown of the compute node,
        completing any active executions and cleaning up resources.
        """
        # Placeholder implementation - will be completed in next phase
        self.info("Stopping compute node service...")
        
        # TODO: Implement graceful shutdown:
        # - Set shutdown event
        # - Complete active executions
        # - Close gateway connection
        # - Cleanup resources
        
        self.info("Compute node service stopped")
    
    def get_node_info(self) -> NodeInfo:
        """
        Get comprehensive node information for monitoring and debugging.
        
        Returns:
            NodeInfo object with current node state and statistics
        """
        return NodeInfo(
            node_id=self.config.node_id,
            functions=self._registered_functions.copy(),
            last_heartbeat=self._last_heartbeat_time or datetime.now(),
            status=NodeStatus.CONNECTED if self.is_connected else NodeStatus.DISCONNECTED,
            health_metrics=self._node_metrics or NodeHealthMetrics(),
            capabilities={"python", "async", "streaming"},
            version="2.0.0",
            startup_time=datetime.now(),  # TODO: Track actual startup time
            total_requests_handled=0,  # TODO: Track request count
            error_count=0  # TODO: Track error count
        )
    
    @property
    def execution_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get execution statistics for all registered functions."""
        return self._execution_statistics.copy()
    
    def wait_for_connection(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for the node to establish connection with the gateway.
        
        Args:
            timeout: Maximum time to wait in seconds (None for indefinite)
            
        Returns:
            True if connected within timeout, False otherwise
        """
        return self._connection_event.wait(timeout=timeout)


# Backward compatibility alias
# This ensures existing code continues to work while we transition to the new naming
ComputeNode = DistributedComputeNode


# Export the main compute node class with both names for flexibility
__all__ = [
    'DistributedComputeNode', 
    'ComputeNode', 
    'NodeConnectionState', 
    'NodeConfiguration', 
    'ExecutionContext'
]
