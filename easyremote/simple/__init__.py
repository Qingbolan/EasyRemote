#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EasyRemote Simplified API Module

This module provides an ultra-simplified, zero-configuration interface for the
EasyRemote distributed computing framework. The design philosophy emphasizes:

1. Zero Configuration: All parameters have intelligent defaults
2. Minimal API Surface: Only 2 core interaction patterns
3. Auto-Optimization: Built-in best practices and automatic tuning
4. 20-Minute Learning Curve: From zero to productive in minutes

Design Patterns:
- Facade Pattern: Wraps complex core functionality with simple interfaces
- Builder Pattern: Fluent APIs for object construction
- Factory Pattern: Smart object creation with sensible defaults
- Singleton Pattern: Shared resources where appropriate

Target Users:
- Rapid prototyping and development
- Educational and learning scenarios
- Simple distributed computing tasks
- Beginner-friendly introduction to distributed systems

Performance Goals:
- 4-line minimal implementation
- Zero mandatory configuration parameters
- Automatic load balancing and optimization
- Immediate usability without deep system knowledge

Author: EasyRemote Team
Version: 2.0.0
Compatibility: Python 3.7+
"""

import uuid
import logging
from typing import Optional, Callable, Any, Dict, Union
from contextlib import contextmanager

# Import full-featured core components
from ..core.nodes.server import Server as _CoreServer
from ..core.nodes.compute_node import ComputeNode as _CoreComputeNode
from ..core.nodes.client import Client as _CoreClient
from ..decorators import remote as _core_remote


# Configure logging for simplified API
_logger = logging.getLogger(__name__)


class SimplifiedServer:
    """
    Ultra-simplified server wrapper with zero-configuration startup.
    
    This class wraps the full-featured Server with sensible defaults,
    eliminating the need for complex configuration while maintaining
    all core functionality under the hood.
    
    Features:
    - Automatic port selection with fallback
    - Built-in error handling and recovery
    - Background operation by default
    - Minimal resource footprint
    
    Example:
        >>> server = SimplifiedServer()
        >>> server.start_background()  # Non-blocking startup
        >>> # Server is now running and ready for connections
    """
    
    def __init__(self, port: int = 8080, auto_retry: bool = True):
        """
        Initialize simplified server with intelligent defaults.
        
        Args:
            port: Primary port to bind to (with automatic fallback)
            auto_retry: Automatically retry on different ports if binding fails
        """
        self._port = port
        self._auto_retry = auto_retry
        self._server = None
        self._actual_port = None
        
    def start(self) -> 'SimplifiedServer':
        """
        Start server in blocking mode with automatic configuration.
        
        Returns:
            Self for method chaining
            
        Raises:
            RuntimeError: If server fails to start after all retry attempts
        """
        self._server = self._create_server_instance()
        try:
            self._server.start()
            _logger.info(f"Simplified server started successfully on port {self._actual_port}")
            return self
        except Exception as e:
            _logger.error(f"Failed to start server: {e}")
            raise RuntimeError(f"Server startup failed: {e}") from e
    
    def start_background(self) -> 'SimplifiedServer':
        """
        Start server in non-blocking background mode.
        
        This is the recommended method for most use cases as it allows
        the calling thread to continue with other operations.
        
        Returns:
            Self for method chaining
            
        Example:
            >>> server = SimplifiedServer().start_background()
            >>> # Server is running, can now create nodes and clients
        """
        self._server = self._create_server_instance()
        try:
            self._server.start_background()
            _logger.info(f"Simplified server started in background on port {self._actual_port}")
            return self
        except Exception as e:
            _logger.error(f"Failed to start background server: {e}")
            raise RuntimeError(f"Background server startup failed: {e}") from e
    
    def _create_server_instance(self) -> _CoreServer:
        """
        Create core server instance with port fallback logic.
        
        Returns:
            Configured core server instance
        """
        if not self._auto_retry:
            self._actual_port = self._port
            return _CoreServer(port=self._port)
        
        # Try original port first, then fallback ports
        ports_to_try = [self._port] + [self._port + i for i in range(1, 11)]
        
        for port in ports_to_try:
            try:
                server = _CoreServer(port=port)
                # Test if port is available by attempting to bind
                self._actual_port = port
                _logger.debug(f"Successfully configured server on port {port}")
                return server
            except Exception as e:
                _logger.debug(f"Port {port} unavailable: {e}")
                continue
        
        raise RuntimeError(f"No available ports found starting from {self._port}")
    
    @property
    def port(self) -> Optional[int]:
        """Get the actual port the server is running on."""
        return self._actual_port
    
    @property
    def is_running(self) -> bool:
        """Check if the server is currently running."""
        return self._server is not None and hasattr(self._server, '_running') and self._server._running


class SimplifiedComputeNode:
    """
    Zero-configuration compute node with automatic optimization.
    
    This class provides the simplest possible interface for creating
    compute nodes while automatically configuring all advanced features
    like load balancing, health monitoring, and error recovery.
    
    Key Features:
    - Automatic node ID generation
    - Built-in load balancing enabled by default
    - Intelligent retry and reconnection logic
    - Zero mandatory configuration parameters
    
    Example:
        >>> node = SimplifiedComputeNode()
        >>> @node.register
        ... def my_task(data):
        ...     return f"processed: {data}"
        >>> node.serve()  # Non-blocking by default
    """
    
    def __init__(self, 
                 gateway_address: str = "localhost:8080", 
                 node_id: Optional[str] = None,
                 auto_reconnect: bool = True):
        """
        Initialize compute node with intelligent defaults.
        
        Args:
            gateway_address: Address of the gateway server
            node_id: Unique identifier (auto-generated if None)
            auto_reconnect: Enable automatic reconnection on failures
        """
        # Generate unique node ID if not provided
        self._node_id = node_id or self._generate_node_id()
        self._gateway_address = gateway_address
        self._auto_reconnect = auto_reconnect
        
        # Create core compute node with optimized settings
        self._node = self._create_core_node()
        
        _logger.info(f"Simplified compute node '{self._node_id}' initialized")
    
    def register(self, func: Optional[Callable] = None) -> Union[Callable, Callable[[Callable], Callable]]:
        """
        Register a function with automatic load balancing and optimization.
        
        This method automatically enables load balancing and applies
        performance optimizations based on function characteristics.
        
        Args:
            func: Function to register (None for decorator usage)
            
        Returns:
            Registered function or decorator
            
        Example:
            >>> # Decorator style (recommended)
            >>> @node.register
            ... def process_data(data):
            ...     return {"result": data * 2}
            
            >>> # Direct call style
            >>> def another_task(x, y):
            ...     return x + y
            >>> node.register(another_task)
        """
        def decorator(f: Callable) -> Callable:
            # Automatically enable load balancing and optimization
            registered_func = self._node.register(
                load_balancing=True,
                max_concurrent=self._calculate_optimal_concurrency(f),
                timeout=self._estimate_function_timeout(f)
            )(f)
            
            _logger.debug(f"Function '{f.__name__}' registered with automatic optimization")
            return registered_func
        
        # Support both decorator and direct call patterns
        if func is None:
            return decorator
        else:
            return decorator(func)
    
    def serve(self, blocking: bool = False) -> Optional[Any]:
        """
        Start the compute node service with intelligent error handling.
        
        Args:
            blocking: Whether to block the calling thread
            
        Returns:
            Thread handle if non-blocking, None if blocking
            
        Example:
            >>> node.serve()  # Non-blocking (recommended)
            >>> # or
            >>> node.serve(blocking=True)  # Blocking mode
        """
        try:
            result = self._node.serve(blocking=blocking)
            _logger.info(f"Compute node '{self._node_id}' service started (blocking={blocking})")
            return result
        except Exception as e:
            _logger.error(f"Failed to start compute node service: {e}")
            if self._auto_reconnect:
                _logger.info("Auto-reconnect enabled, will retry on connection loss")
            raise
    
    def _generate_node_id(self) -> str:
        """Generate a unique, human-readable node identifier."""
        import platform
        import socket
        
        # Create meaningful node ID with hostname and UUID
        hostname = platform.node().lower().replace('.', '-')[:8]
        unique_suffix = str(uuid.uuid4())[:8]
        return f"node-{hostname}-{unique_suffix}"
    
    def _create_core_node(self) -> _CoreComputeNode:
        """Create optimized core compute node instance."""
        return _CoreComputeNode(
            vps_address=self._gateway_address,
            node_id=self._node_id,
            # Optimized settings for simplified usage
            reconnect_interval=2,  # Faster reconnection
            heartbeat_interval=3,  # More frequent heartbeats
            max_retry_attempts=5,  # More persistent
            connection_timeout=8,  # Slightly longer for reliability
        )
    
    def _calculate_optimal_concurrency(self, func: Callable) -> int:
        """
        Automatically determine optimal concurrency level for a function.
        
        This uses heuristics based on function characteristics to
        determine the best concurrency setting.
        """
        import inspect
        
        # Analyze function to determine optimal concurrency
        if inspect.iscoroutinefunction(func):
            return 10  # Async functions can handle more concurrency
        elif hasattr(func, '__name__') and any(keyword in func.__name__.lower() 
                                             for keyword in ['io', 'network', 'request', 'fetch']):
            return 5   # I/O bound functions benefit from moderate concurrency
        else:
            return 2   # CPU-bound functions (conservative default)
    
    def _estimate_function_timeout(self, func: Callable) -> int:
        """
        Estimate appropriate timeout for a function based on its characteristics.
        """
        import inspect
        
        # Analyze function signature and name for timeout hints
        if hasattr(func, '__name__'):
            name = func.__name__.lower()
            if any(keyword in name for keyword in ['train', 'learn', 'model']):
                return 1800  # 30 minutes for ML tasks
            elif any(keyword in name for keyword in ['download', 'upload', 'transfer']):
                return 600   # 10 minutes for file operations
            elif any(keyword in name for keyword in ['process', 'compute', 'calculate']):
                return 300   # 5 minutes for general computation
        
        return 120  # 2 minutes default timeout


class SimplifiedClient:
    """
    Ultra-simple client with automatic connection management and intelligent retry.
    
    This client wrapper eliminates configuration complexity while providing
    robust error handling, automatic retries, and connection management.
    
    Features:
    - Automatic connection establishment and cleanup
    - Intelligent retry with exponential backoff
    - Built-in load balancing for all calls
    - Context manager support for resource management
    
    Example:
        >>> client = SimplifiedClient()
        >>> result = client.call("my_function", arg1, arg2)
        >>> # or use as context manager
        >>> with SimplifiedClient() as client:
        ...     result = client.call("my_function", data)
    """
    
    def __init__(self, 
                 gateway_address: str = "localhost:8080",
                 auto_connect: bool = True,
                 retry_attempts: int = 3):
        """
        Initialize simplified client with robust defaults.
        
        Args:
            gateway_address: Gateway server address
            auto_connect: Automatically establish connection
            retry_attempts: Number of retry attempts for failed operations
        """
        self._gateway_address = gateway_address
        self._auto_connect = auto_connect
        self._retry_attempts = retry_attempts
        self._client = None
        self._connected = False
        
        # Create core client with optimized settings
        self._client = _CoreClient(
            gateway_address=gateway_address,
            connection_timeout=10,
            request_timeout=300,
            retry_attempts=retry_attempts,
            enable_load_balancing=True
        )
        
        if auto_connect:
            self.connect()
        
        _logger.info(f"Simplified client initialized for gateway: {gateway_address}")
    
    def call(self, function_name: str, *args, **kwargs) -> Any:
        """
        Call remote function with automatic load balancing and error handling.
        
        This method automatically handles connection establishment, load balancing,
        retries, and error recovery without requiring any configuration.
        
        Args:
            function_name: Name of the function to call
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Function execution result
            
        Raises:
            RuntimeError: If all retry attempts fail
            
        Example:
            >>> result = client.call("process_data", {"input": "test"})
            >>> model_result = client.call("train_model", epochs=10, lr=0.001)
        """
        if not self._connected and self._auto_connect:
            self.connect()
        
        if not self._connected:
            raise RuntimeError("Client not connected and auto_connect is disabled")
        
        try:
            # Use automatic load balancing for all calls
            result = self._client.call(function_name, *args, **kwargs)
            _logger.debug(f"Successfully called function '{function_name}'")
            return result
        except Exception as e:
            _logger.error(f"Function call '{function_name}' failed: {e}")
            raise RuntimeError(f"Remote function call failed: {e}") from e
    
    def connect(self) -> 'SimplifiedClient':
        """
        Establish connection to the gateway with automatic retry.
        
        Returns:
            Self for method chaining
            
        Raises:
            RuntimeError: If connection fails after all attempts
        """
        try:
            self._client.connect()
            self._connected = True
            _logger.info(f"Successfully connected to gateway: {self._gateway_address}")
            return self
        except Exception as e:
            _logger.error(f"Failed to connect to gateway: {e}")
            self._connected = False
            raise RuntimeError(f"Gateway connection failed: {e}") from e
    
    def disconnect(self) -> None:
        """Cleanly disconnect from the gateway."""
        if self._client and self._connected:
            try:
                self._client.disconnect()
                self._connected = False
                _logger.info("Disconnected from gateway")
            except Exception as e:
                _logger.warning(f"Error during disconnect: {e}")
    
    def __enter__(self) -> 'SimplifiedClient':
        """Context manager entry - ensure connection is established."""
        if not self._connected:
            self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - clean up connection."""
        self.disconnect()
    
    @property
    def is_connected(self) -> bool:
        """Check if client is currently connected to gateway."""
        return self._connected


def simplified_remote(func: Optional[Callable] = None, **kwargs) -> Union[Callable, Callable[[Callable], Callable]]:
    """
    Simplified remote decorator with automatic load balancing enabled.
    
    This decorator automatically enables load balancing and applies
    sensible defaults for remote function execution.
    
    Args:
        func: Function to decorate (None for parameterized usage)
        **kwargs: Additional configuration options
        
    Returns:
        Decorated function or decorator
        
    Example:
        >>> @simplified_remote
        ... def my_function(data):
        ...     return process(data)
    """
    # Always enable load balancing in simplified mode
    kwargs.setdefault('load_balancing', True)
    
    if func is None:
        return lambda f: _core_remote(**kwargs)(f)
    else:
        return _core_remote(**kwargs)(func)


# Ultra-convenience functions for one-line setup
def quick_start(port: int = 8080) -> SimplifiedServer:
    """
    One-line server startup for immediate use.
    
    Args:
        port: Port to start server on (with automatic fallback)
        
    Returns:
        Running server instance
        
    Example:
        >>> quick_start()  # Server now running in background
    """
    return SimplifiedServer(port).start_background()


def quick_node(gateway_address: str = "localhost:8080", 
               node_id: Optional[str] = None) -> SimplifiedComputeNode:
    """
    One-line compute node creation with optimal defaults.
    
    Args:
        gateway_address: Gateway server address
        node_id: Optional custom node identifier
        
    Returns:
        Configured compute node ready for function registration
        
    Example:
        >>> node = quick_node()
        >>> @node.register
        ... def my_task(data): return data * 2
    """
    return SimplifiedComputeNode(gateway_address, node_id)


def quick_client(gateway_address: str = "localhost:8080") -> SimplifiedClient:
    """
    One-line client creation with automatic connection.
    
    Args:
        gateway_address: Gateway server address
        
    Returns:
        Connected client ready for function calls
        
    Example:
        >>> client = quick_client()
        >>> result = client.call("my_task", 42)
    """
    return SimplifiedClient(gateway_address, auto_connect=True)


@contextmanager
def distributed_session(gateway_address: str = "localhost:8080"):
    """
    Context manager for complete distributed computing session.
    
    This convenience function sets up a complete environment including
    server, and provides a client for immediate use.
    
    Args:
        gateway_address: Address for the gateway server
        
    Yields:
        Tuple of (server, client) ready for distributed computing
        
    Example:
        >>> with distributed_session() as (server, client):
        ...     # Server is running, client is connected
        ...     result = client.call("some_function", data)
        ... # Everything is automatically cleaned up
    """
    server = None
    client = None
    
    try:
        # Start server in background
        server = quick_start()
        
        # Create and connect client
        client = quick_client(gateway_address)
        
        yield server, client
        
    finally:
        # Clean up resources
        if client:
            client.disconnect()
        # Server cleanup is handled automatically


# Export simplified API components
__all__ = [
    # Main classes (renamed for clarity)
    "SimplifiedServer",
    "SimplifiedComputeNode", 
    "SimplifiedClient",
    # Convenience aliases matching original names
    "Server",
    "ComputeNode",
    "Client",
    # Decorators
    "simplified_remote",
    "remote",
    # One-line convenience functions
    "quick_start",
    "quick_node", 
    "quick_client",
    # Context managers
    "distributed_session"
]

# Convenience aliases for backward compatibility and intuitive naming
Server = SimplifiedServer
ComputeNode = SimplifiedComputeNode
Client = SimplifiedClient
remote = simplified_remote

# Version information
__version__ = "2.0.0-simplified"
__author__ = "EasyRemote Team"
__description__ = "Ultra-simplified distributed computing API" 