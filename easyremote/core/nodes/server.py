#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EasyRemote Core Server Module

This module implements the central gateway server for the EasyRemote distributed
computing framework. The server acts as a intelligent message broker and load
balancer, coordinating between clients and compute nodes.

Architecture:
- Implements gRPC bidirectional streaming for efficient communication
- Manages compute node registration and health monitoring
- Provides intelligent load balancing and request routing
- Handles both synchronous and asynchronous function calls
- Supports streaming functions for data-intensive operations

Key Features:
- Zero-configuration startup with intelligent defaults
- Automatic node discovery and health monitoring
- Multiple load balancing strategies (round-robin, resource-aware, latency-based)
- Fault tolerance with automatic failover
- Real-time performance monitoring and metrics collection
- Horizontal scalability support

Communication Protocol:
- Control Stream: Bidirectional gRPC stream for node registration and heartbeats
- Function Execution: Request-response pattern for function calls
- Stream Functions: Server-side streaming for large data operations
- Health Monitoring: Periodic heartbeat and status reporting

Thread Safety:
All public methods are thread-safe and can be called from multiple threads
concurrently. Internal state is protected with asyncio locks.

Author: EasyRemote Team
Version: 2.0.0
"""

import asyncio
import threading
import time
import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, Set, Union, Any, Optional, List, Tuple
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum

# gRPC imports
from grpc import aio as grpc_aio
from concurrent import futures

# EasyRemote core imports
from ..utils.logger import ModernLogger
from ..load_balancing import LoadBalancer, RequestContext
from ..data import NodeInfo, FunctionInfo, NodeStatus, NodeHealthMetrics
from ..utils.exceptions import (
    NodeNotFoundError,
    FunctionNotFoundError,
    SerializationError,
    RemoteExecutionError,
    EasyRemoteError
)
from ..data import Serializer
from ..protos import service_pb2, service_pb2_grpc


# Configure module logger
_logger = logging.getLogger(__name__)

# Sentinel object to indicate stream exhaustion
_STREAM_END_SENTINEL = object()


class ServerState(Enum):
    """
    Enumeration of possible server states for lifecycle management.
    """
    INITIALIZING = "initializing"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class ServerMetrics:
    """
    Real-time server performance and operational metrics.
    
    This class tracks various server statistics for monitoring,
    debugging, and performance optimization purposes.
    """
    start_time: datetime = field(default_factory=datetime.now)
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    active_connections: int = 0
    peak_connections: int = 0
    total_nodes_registered: int = 0
    active_nodes: int = 0
    average_response_time: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    
    def update_request_stats(self, success: bool, response_time: float):
        """Update request statistics with latest data."""
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        # Update running average response time
        if self.total_requests == 1:
            self.average_response_time = response_time
        else:
            alpha = 2.0 / (self.total_requests + 1)
            self.average_response_time = (
                alpha * response_time + (1 - alpha) * self.average_response_time
            )
        
        self.last_updated = datetime.now()
    
    def update_connection_stats(self, active_count: int):
        """Update connection statistics."""
        self.active_connections = active_count
        self.peak_connections = max(self.peak_connections, active_count)
        self.last_updated = datetime.now()
    
    @property
    def success_rate(self) -> float:
        """Calculate the success rate of requests."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    @property
    def uptime_seconds(self) -> float:
        """Calculate server uptime in seconds."""
        return (datetime.now() - self.start_time).total_seconds()


class StreamExecutionContext(ModernLogger):
    """
    Context manager for streaming function execution lifecycle.
    
    This class manages the complete lifecycle of streaming function calls,
    including resource allocation, cleanup, and error handling.
    
    Features:
    - Automatic resource cleanup on completion or error
    - Callback registration for custom cleanup logic
    - Context tracking for debugging and monitoring
    - Graceful shutdown handling
    """
    
    def __init__(self, call_id: str, function_name: str, node_id: str, 
                 response_queue: asyncio.Queue, timeout: Optional[int] = None):
        """
        Initialize streaming execution context.
        
        Args:
            call_id: Unique identifier for this function call
            function_name: Name of the function being executed
            node_id: ID of the compute node executing the function
            response_queue: Queue for streaming responses
            timeout: Optional timeout for the entire operation
        """
        super().__init__(name=f"{__name__}.StreamContext")
        
        self.call_id = call_id
        self.function_name = function_name
        self.node_id = node_id
        self.response_queue = response_queue
        self.timeout = timeout
        
        self.created_at = datetime.now()
        self.is_active = True
        self.completion_status: Optional[str] = None
        self._cleanup_callbacks: List[callable] = []
        self._context_data: Dict[str, Any] = {}
        
        self.debug(f"Created stream context for {function_name} on {node_id}")
    
    def add_cleanup_callback(self, callback: callable) -> 'StreamExecutionContext':
        """
        Register a callback to be executed during cleanup.
        
        Args:
            callback: Function to call during cleanup (sync or async)
            
        Returns:
            Self for method chaining
        """
        self._cleanup_callbacks.append(callback)
        return self
    
    def set_context_data(self, key: str, value: Any) -> 'StreamExecutionContext':
        """
        Store arbitrary data associated with this context.
        
        Args:
            key: Data key
            value: Data value
            
        Returns:
            Self for method chaining
        """
        self._context_data[key] = value
        return self
    
    def get_context_data(self, key: str, default: Any = None) -> Any:
        """Retrieve context data by key."""
        return self._context_data.get(key, default)
    
    async def cleanup(self, completion_status: str = "completed"):
        """
        Clean up resources and execute registered callbacks.
        
        Args:
            completion_status: Status indicating how the stream ended
        """
        if not self.is_active:
            return  # Already cleaned up
        
        self.is_active = False
        self.completion_status = completion_status
        
        # Execute cleanup callbacks
        for callback in self._cleanup_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as e:
                self.error(f"Error in cleanup callback: {e}")
        
        duration = (datetime.now() - self.created_at).total_seconds()
        self.info(f"Stream context cleaned up: {self.function_name} "
                 f"(status: {completion_status}, duration: {duration:.2f}s)")
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time since context creation in seconds."""
        return (datetime.now() - self.created_at).total_seconds()
    
    @property
    def is_timed_out(self) -> bool:
        """Check if the context has exceeded its timeout."""
        if self.timeout is None:
            return False
        return self.elapsed_time > self.timeout


class DistributedComputeServer(service_pb2_grpc.RemoteServiceServicer, ModernLogger):
    """
    High-performance distributed computing gateway server.
    
    This server implements the core gateway functionality for the EasyRemote
    framework, providing intelligent request routing, load balancing, and
    comprehensive node management.
    
    Key Responsibilities:
    1. Compute Node Management: Registration, health monitoring, and lifecycle
    2. Request Routing: Intelligent function call distribution
    3. Load Balancing: Multiple strategies for optimal resource utilization
    4. Fault Tolerance: Automatic failover and error recovery
    5. Performance Monitoring: Real-time metrics and diagnostics
    
    Architecture Features:
    - Asynchronous gRPC communication for high throughput
    - Thread-safe operations with proper resource management
    - Horizontal scalability with shared state support
    - Comprehensive logging and monitoring
    - Graceful shutdown and cleanup procedures
    
    Usage:
        >>> server = DistributedComputeServer(port=8080)
        >>> server.start_background()  # Non-blocking startup
        >>> # Server is now ready to handle requests
        
        >>> # For blocking operation
        >>> server.start()  # Blocks until shutdown
    """
    
    # Class-level singleton instance for global access
    _global_instance: Optional['DistributedComputeServer'] = None
    _instance_lock = threading.Lock()
    
    def __init__(self, 
                 port: int = 8080, 
                 heartbeat_timeout: int = 30,
                 max_queue_size: int = 1000,
                 max_workers: int = 10,
                 enable_monitoring: bool = True,
                 cleanup_interval: int = 60):
        """
        Initialize the distributed computing server.
        
        Args:
            port: Port number to bind the server to
            heartbeat_timeout: Timeout in seconds for node heartbeats
            max_queue_size: Maximum size for internal message queues
            max_workers: Maximum number of worker threads for request processing
            enable_monitoring: Enable real-time performance monitoring
            cleanup_interval: Interval in seconds for cleanup operations
            
        Raises:
            ValueError: If configuration parameters are invalid
            RuntimeError: If server initialization fails
        """
        # Initialize parent classes
        service_pb2_grpc.RemoteServiceServicer.__init__(self)
        ModernLogger.__init__(self, name=__name__)
        
        # Validate configuration parameters
        self._validate_configuration(port, heartbeat_timeout, max_queue_size, max_workers)
        
        # Core server configuration
        self.port = port
        self.heartbeat_timeout = heartbeat_timeout
        self.max_queue_size = max_queue_size
        self.max_workers = max_workers
        self.cleanup_interval = cleanup_interval
        
        # Server state management
        self._state = ServerState.INITIALIZING
        self._state_lock = asyncio.Lock()
        
        # Core server components
        self._grpc_server: Optional[grpc_aio.Server] = None
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        self._server_thread: Optional[threading.Thread] = None
        
        # Node and function management
        self._nodes: Dict[str, NodeInfo] = {}
        self._node_communication_queues: Dict[str, asyncio.Queue] = {}
        self._global_lock = asyncio.Lock()  # Protects all shared data structures
        
        # Request and stream management
        self._pending_function_calls: Dict[str, Union[asyncio.Future, Dict[str, Any]]] = {}
        self._active_stream_contexts: Dict[str, StreamExecutionContext] = {}
        self._active_stream_ids: Set[str] = set()
        
        # Background task management
        self._background_tasks: Set[asyncio.Task] = set()
        self._node_monitor_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        # Performance and monitoring
        self.metrics = ServerMetrics() if enable_monitoring else None
        self._request_history: List[Tuple[datetime, str, bool, float]] = []
        
        # Core services
        self._serializer = Serializer()
        self._load_balancer = LoadBalancer(self)
        
        # Set global instance
        with DistributedComputeServer._instance_lock:
            DistributedComputeServer._global_instance = self
        
        self.info(f"Initialized DistributedComputeServer on port {port} "
                 f"(heartbeat_timeout={heartbeat_timeout}s, max_workers={max_workers})")
    
    def _validate_configuration(self, port: int, heartbeat_timeout: int, 
                               max_queue_size: int, max_workers: int):
        """
        Validate server configuration parameters.
        
        Args:
            port: Server port number
            heartbeat_timeout: Heartbeat timeout in seconds
            max_queue_size: Maximum queue size
            max_workers: Maximum worker threads
            
        Raises:
            ValueError: If any parameter is invalid
        """
        if not (1 <= port <= 65535):
            raise ValueError(f"Port must be between 1 and 65535, got {port}")
        
        if heartbeat_timeout < 1:
            raise ValueError(f"Heartbeat timeout must be positive, got {heartbeat_timeout}")
        
        if max_queue_size < 1:
            raise ValueError(f"Max queue size must be positive, got {max_queue_size}")
        
        if max_workers < 1:
            raise ValueError(f"Max workers must be positive, got {max_workers}")
    
    async def _set_state(self, new_state: ServerState):
        """
        Thread-safe state transition with validation.
        
        Args:
            new_state: The new state to transition to
            
        Raises:
            RuntimeError: If state transition is invalid
        """
        async with self._state_lock:
            old_state = self._state
            
            # Validate state transitions
            valid_transitions = {
                ServerState.INITIALIZING: {ServerState.STARTING, ServerState.ERROR},
                ServerState.STARTING: {ServerState.RUNNING, ServerState.ERROR, ServerState.STOPPING},
                ServerState.RUNNING: {ServerState.STOPPING, ServerState.ERROR},
                ServerState.STOPPING: {ServerState.STOPPED, ServerState.ERROR},
                ServerState.STOPPED: {ServerState.STARTING},
                ServerState.ERROR: {ServerState.STARTING, ServerState.STOPPING}
            }
            
            if new_state not in valid_transitions.get(old_state, set()):
                raise RuntimeError(f"Invalid state transition: {old_state} -> {new_state}")
            
            self._state = new_state
            self.info(f"Server state changed: {old_state.value} -> {new_state.value}")
    
    @property
    def state(self) -> ServerState:
        """Get the current server state."""
        return self._state
    
    @property
    def is_running(self) -> bool:
        """Check if the server is currently running."""
        return self._state == ServerState.RUNNING
    
    def start(self) -> 'DistributedComputeServer':
        """
        Start the server in blocking mode.
        
        This method starts the server and blocks until the server is shut down.
        It automatically detects if it's running in an existing event loop and
        falls back to background mode if necessary.
        
        Returns:
            Self for method chaining
            
        Raises:
            RuntimeError: If server fails to start or is already running
            EasyRemoteError: If there are configuration or initialization issues
            
        Example:
            >>> server = DistributedComputeServer(port=8080)
            >>> server.start()  # Blocks until shutdown
        """
        if self._state != ServerState.INITIALIZING:
            raise RuntimeError(f"Server cannot start from state: {self._state}")
        
        try:
            # Check if we're already in an event loop
            loop = asyncio.get_running_loop()
            self.warning("Detected running event loop, switching to background mode")
            return self.start_background()
        except RuntimeError:
            # No running loop, we can create our own
            pass
        
        try:
            # Create and configure event loop
            self._event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._event_loop)
            
            # Start server in blocking mode
            self._event_loop.run_until_complete(self._async_serve())
            
            return self
            
        except EasyRemoteError:
            raise
        except Exception as e:
            self.error(f"Unexpected error during server startup: {e}", exc_info=True)
            raise EasyRemoteError(f"Server startup failed: {e}") from e
        finally:
            # Cleanup event loop
            if self._event_loop and not self._event_loop.is_closed():
                self._event_loop.close()
            self._event_loop = None
    
    def start_background(self) -> threading.Thread:
        """
        Start the server in non-blocking background mode.
        
        This method starts the server in a separate thread, allowing the calling
        thread to continue with other operations. This is the recommended approach
        for most use cases.
        
        Returns:
            Thread handle for the background server
            
        Raises:
            RuntimeError: If server is already running or in invalid state
            
        Example:
            >>> server = DistributedComputeServer(port=8080)
            >>> thread = server.start_background()
            >>> # Server is now running in background
            >>> # Continue with other operations...
        """
        if self._state != ServerState.INITIALIZING:
            raise RuntimeError(f"Server cannot start from state: {self._state}")
        
        def _background_server_runner():
            """Background thread entry point for server execution."""
            try:
                # Create isolated event loop for this thread
                self._event_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._event_loop)
                
                # Run server asynchronously
                self._event_loop.run_until_complete(self._async_serve())
                
            except EasyRemoteError as e:
                self.error(f"EasyRemote error in background server: {e}")
            except Exception as e:
                self.error(f"Unexpected error in background server: {e}", exc_info=True)
            finally:
                # Cleanup
                if self._event_loop and not self._event_loop.is_closed():
                    self._event_loop.close()
                self._event_loop = None
        
        # Start background thread
        self._server_thread = threading.Thread(
            target=_background_server_runner,
            name=f"EasyRemoteServer-{self.port}",
            daemon=True
        )
        self._server_thread.start()
        
        # Wait for server to be ready
        self._wait_for_server_ready(timeout=10.0)
        
        self.info(f"Server started in background on port {self.port}")
        return self._server_thread
    
    def _wait_for_server_ready(self, timeout: float = 10.0):
        """
        Wait for the server to reach running state.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Raises:
            TimeoutError: If server doesn't start within timeout
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self._state == ServerState.RUNNING:
                return
            elif self._state == ServerState.ERROR:
                raise RuntimeError("Server failed to start")
            time.sleep(0.1)
        
        raise TimeoutError(f"Server did not start within {timeout} seconds")
    
    async def _async_serve(self):
        """
        Main asynchronous server loop.
        
        This method handles the complete server lifecycle including:
        - gRPC server setup and configuration
        - Background task initialization
        - Event loop management
        - Graceful shutdown handling
        """
        try:
            await self._set_state(ServerState.STARTING)
            
            # Create and configure gRPC server
            self._grpc_server = grpc_aio.server(
                futures.ThreadPoolExecutor(max_workers=self.max_workers),
                options=self._get_grpc_server_options()
            )
            
            # Add our service to the server
            service_pb2_grpc.add_RemoteServiceServicer_to_server(self, self._grpc_server)
            
            # Bind to port and start
            listen_address = f'[::]:{self.port}'
            self._grpc_server.add_insecure_port(listen_address)
            await self._grpc_server.start()
            
            self.info(f"gRPC server listening on {listen_address}")
            
            # Start background monitoring and maintenance tasks
            await self._start_background_tasks()
            
            # Mark server as running
            await self._set_state(ServerState.RUNNING)
            self.info("Distributed compute server is now running and ready for connections")
            
            # Update metrics
            if self.metrics:
                self.metrics.start_time = datetime.now()
            
            # Wait for termination signal
            await self._grpc_server.wait_for_termination()
            
        except Exception as e:
            await self._set_state(ServerState.ERROR)
            self.error(f"Server error during startup: {e}", exc_info=True)
            raise EasyRemoteError(f"Server startup failed: {e}") from e
        finally:
            # Ensure cleanup happens
            await self._async_cleanup()
    
    def _get_grpc_server_options(self) -> List[Tuple[str, Any]]:
        """
        Get optimized gRPC server configuration options.
        
        Returns:
            List of gRPC server option tuples
        """
        return [
            # Message size limits (50MB)
            ('grpc.max_send_message_length', 50 * 1024 * 1024),
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),
            
            # Keepalive settings for connection health
            ('grpc.keepalive_time_ms', 30000),  # 30 seconds
            ('grpc.keepalive_timeout_ms', 5000),  # 5 seconds
            ('grpc.keepalive_permit_without_calls', True),
            
            # HTTP/2 settings
            ('grpc.http2.max_pings_without_data', 0),
            ('grpc.http2.min_time_between_pings_ms', 10000),
            ('grpc.http2.min_ping_interval_without_data_ms', 300000),
            
            # Performance tuning
            ('grpc.so_reuseport', 1),
            ('grpc.tcp_user_timeout_ms', 20000),
        ]
    
    async def _start_background_tasks(self):
        """
        Start all background monitoring and maintenance tasks.
        
        This includes:
        - Node health monitoring
        - Resource cleanup
        - Performance metrics collection
        - Connection management
        """
        self.debug("Starting background tasks")
        
        # Node health monitoring task
        self._node_monitor_task = asyncio.create_task(
            self._node_health_monitor_loop(),
            name="NodeHealthMonitor"
        )
        self._background_tasks.add(self._node_monitor_task)
        
        # Resource cleanup task
        self._cleanup_task = asyncio.create_task(
            self._resource_cleanup_loop(),
            name="ResourceCleanup"
        )
        self._background_tasks.add(self._cleanup_task)
        
        # Performance metrics collection (if enabled)
        if self.metrics:
            metrics_task = asyncio.create_task(
                self._metrics_collection_loop(),
                name="MetricsCollection"
            )
            self._background_tasks.add(metrics_task)
        
        self.info(f"Started {len(self._background_tasks)} background tasks")
    
    async def _node_health_monitor_loop(self):
        """
        Background task for monitoring compute node health.
        
        This task periodically checks node heartbeats and removes
        unresponsive nodes from the system.
        """
        self.debug("Node health monitor started")
        
        try:
            while not self._shutdown_event.is_set():
                try:
                    await self._check_node_health()
                    
                    # Wait for next check (half of heartbeat timeout)
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self.heartbeat_timeout / 2
                    )
                    
                except asyncio.TimeoutError:
                    # Normal timeout, continue monitoring
                    continue
                except Exception as e:
                    self.error(f"Error in node health monitor: {e}")
                    await asyncio.sleep(1)  # Brief pause before retry
                    
        except asyncio.CancelledError:
            self.debug("Node health monitor cancelled")
        except Exception as e:
            self.error(f"Unexpected error in node health monitor: {e}", exc_info=True)
    
    async def _check_node_health(self):
        """
        Check health of all registered compute nodes.
        
        Removes nodes that haven't sent heartbeats within the timeout period.
        """
        current_time = datetime.now()
        timeout_delta = timedelta(seconds=self.heartbeat_timeout)
        nodes_to_remove = []
        
        async with self._global_lock:
            for node_id, node_info in self._nodes.items():
                time_since_heartbeat = current_time - node_info.last_heartbeat
                
                if time_since_heartbeat > timeout_delta:
                    self.warning(f"Node {node_id} timed out "
                               f"(last heartbeat: {time_since_heartbeat.total_seconds():.1f}s ago)")
                    nodes_to_remove.append(node_id)
                elif not node_info.is_alive(self.heartbeat_timeout):
                    self.warning(f"Node {node_id} marked as not alive")
                    nodes_to_remove.append(node_id)
        
        # Remove timed out nodes
        for node_id in nodes_to_remove:
            await self._remove_node_safely(node_id, reason="health_check_timeout")
        
        # Update metrics
        if self.metrics:
            self.metrics.active_nodes = len(self._nodes)
    
    async def _resource_cleanup_loop(self):
        """
        Background task for cleaning up stale resources.
        
        This task periodically removes expired function calls,
        stream contexts, and other temporary resources.
        """
        self.debug("Resource cleanup loop started")
        
        try:
            while not self._shutdown_event.is_set():
                try:
                    await self._cleanup_stale_resources()
                    
                    # Wait for next cleanup cycle
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self.cleanup_interval
                    )
                    
                except asyncio.TimeoutError:
                    # Normal timeout, continue cleanup
                    continue
                except Exception as e:
                    self.error(f"Error in resource cleanup: {e}")
                    await asyncio.sleep(5)  # Brief pause before retry
                    
        except asyncio.CancelledError:
            self.debug("Resource cleanup loop cancelled")
        except Exception as e:
            self.error(f"Unexpected error in resource cleanup: {e}", exc_info=True)
    
    async def _cleanup_stale_resources(self):
        """
        Clean up expired function calls and stream contexts.
        
        This method removes resources that have exceeded their timeout
        or have been orphaned by disconnected clients/nodes.
        """
        current_time = datetime.now()
        stale_timeout = timedelta(minutes=5)  # 5-minute stale resource timeout
        
        async with self._global_lock:
            # Clean up stale stream contexts
            stale_streams = []
            for call_id, stream_ctx in self._active_stream_contexts.items():
                if (current_time - stream_ctx.created_at > stale_timeout or 
                    stream_ctx.is_timed_out):
                    stale_streams.append(call_id)
            
            for call_id in stale_streams:
                await self._cleanup_stream_context(call_id, "stale_timeout")
            
            # Clean up stale function calls
            stale_calls = []
            for call_id, call_context in self._pending_function_calls.items():
                if isinstance(call_context, dict):
                    created_at = call_context.get('created_at')
                    if created_at and (current_time - created_at > stale_timeout):
                        stale_calls.append(call_id)
            
            for call_id in stale_calls:
                await self._cleanup_pending_call(call_id, "stale_timeout")
        
        if stale_streams or stale_calls:
            self.info(f"Cleaned up {len(stale_streams)} stale streams and "
                     f"{len(stale_calls)} stale calls")
    
    async def _metrics_collection_loop(self):
        """
        Background task for collecting and updating performance metrics.
        """
        self.debug("Metrics collection loop started")
        
        try:
            while not self._shutdown_event.is_set():
                try:
                    await self._update_metrics()
                    
                    # Update every 30 seconds
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=30
                    )
                    
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    self.error(f"Error in metrics collection: {e}")
                    await asyncio.sleep(10)
                    
        except asyncio.CancelledError:
            self.debug("Metrics collection loop cancelled")
        except Exception as e:
            self.error(f"Unexpected error in metrics collection: {e}", exc_info=True)
    
    async def _update_metrics(self):
        """Update server performance metrics."""
        if not self.metrics:
            return
        
        async with self._global_lock:
            # Update connection and node counts
            self.metrics.update_connection_stats(len(self._node_communication_queues))
            self.metrics.active_nodes = len(self._nodes)
            self.metrics.total_nodes_registered = max(
                self.metrics.total_nodes_registered,
                len(self._nodes)
            )
    
    async def _async_cleanup(self):
        """Placeholder for async cleanup - will be implemented in next phase."""
        await self._set_state(ServerState.STOPPED)
        self.info("Server cleanup completed")
    
    async def _remove_node_safely(self, node_id: str, reason: str = "unknown"):
        """Placeholder for safe node removal - will be implemented in next phase."""
        async with self._global_lock:
            self._nodes.pop(node_id, None)
            self._node_communication_queues.pop(node_id, None)
        self.info(f"Node {node_id} removed (reason: {reason})")
    
    async def _cleanup_stream_context(self, call_id: str, reason: str = "unknown"):
        """Placeholder for stream context cleanup - will be implemented in next phase."""
        pass
    
    async def _cleanup_pending_call(self, call_id: str, reason: str = "unknown"):
        """Placeholder for pending call cleanup - will be implemented in next phase."""
        pass
    
    @classmethod
    def get_global_instance(cls) -> Optional['DistributedComputeServer']:
        """
        Get the global server instance for singleton access.
        
        Returns:
            Global server instance if available, None otherwise
        """
        with cls._instance_lock:
            return cls._global_instance


# Backward compatibility alias
# This ensures existing code continues to work while we transition to the new naming
Server = DistributedComputeServer


# Export the main server class with both names for flexibility
__all__ = ['DistributedComputeServer', 'Server', 'ServerState', 'ServerMetrics', 'StreamExecutionContext']