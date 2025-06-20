#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EasyNet Function Accelerator

Advanced function acceleration system that provides decorators and automatic
interception for seamless distributed computing acceleration.
"""

import functools
import inspect
import threading
import time
import sys
import types
from typing import Dict, List, Any, Optional, Callable, Set, Union
import importlib.util
import ast

from ..core.nodes.client import DistributedComputingClient
from ..core.utils.logger import ModernLogger
from .config import DEFAULT_EASYNET_URI, get_gateway_with_fallback


class AcceleratedFunction:
    """
    Wrapper for functions that can be accelerated with EasyRemote.
    """
    
    def __init__(self, func: Callable, 
                 gateway_address: str = None,
                 force_remote: bool = False,
                 fallback_local: bool = True,
                 profile: bool = False):
        """
        Initialize an accelerated function wrapper.
        
        Args:
            func: The function to accelerate
            gateway_address: EasyRemote gateway address
            force_remote: Whether to force remote execution
            fallback_local: Whether to fallback to local execution on failure
            profile: Whether to enable profiling for this function
        """
        self.func = func
        self.gateway_address = get_gateway_with_fallback(gateway_address)
        self.force_remote = force_remote
        self.fallback_local = fallback_local
        self.profile = profile
        
        # Function metadata
        self.func_name = func.__name__
        self.module_name = getattr(func, '__module__', 'unknown')
        
        # Performance tracking
        self.call_count = 0
        self.remote_calls = 0
        self.local_calls = 0
        self.total_remote_time = 0.0
        self.total_local_time = 0.0
        self.last_execution_time = 0.0
        self.last_execution_mode = "unknown"
        
        # Client management
        self._client: Optional[DistributedComputingClient] = None
        self._client_lock = threading.Lock()
        
        # Logger
        self.logger = ModernLogger(f"AcceleratedFunction.{self.func_name}", level="debug" if profile else "info")
        
        # Copy function metadata
        functools.update_wrapper(self, func)
        
        self.logger.debug(f"🚀 Accelerated function created: {self.func_name}")
    
    def _get_client(self) -> Optional[DistributedComputingClient]:
        """Get or create the distributed computing client."""
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    try:
                        self._client = DistributedComputingClient(self.gateway_address)
                        self.logger.debug(f"✅ Connected to gateway: {self.gateway_address}")
                    except Exception as e:
                        self.logger.warning(f"⚠️ Failed to connect to gateway: {e}")
                        if not self.fallback_local:
                            raise
        return self._client
    
    def _should_use_remote(self, args: tuple, kwargs: dict) -> bool:
        """
        Determine whether to use remote execution for this call.
        
        Args:
            args: Function arguments
            kwargs: Function keyword arguments
            
        Returns:
            True if should use remote execution
        """
        if self.force_remote:
            return True
        
        # Heuristics for when remote execution is beneficial
        
        # 1. Large data arguments
        for arg in args:
            if hasattr(arg, '__len__') and len(arg) > 1000:
                return True
            if hasattr(arg, 'shape') and any(dim > 100 for dim in arg.shape):
                return True
        
        for value in kwargs.values():
            if hasattr(value, '__len__') and len(value) > 1000:
                return True
            if hasattr(value, 'shape') and any(dim > 100 for dim in value.shape):
                return True
        
        # 2. Function complexity (based on source code analysis)
        try:
            source = inspect.getsource(self.func)
            
            # Look for compute-intensive patterns
            compute_indicators = [
                'for ' in source, 'while ' in source,
                'numpy' in source, 'np.' in source,
                'pandas' in source, 'pd.' in source,
                'torch' in source, 'tensorflow' in source,
                'sklearn' in source, 'scipy' in source,
                len(source.split('\n')) > 10  # Multi-line functions
            ]
            
            if sum(compute_indicators) >= 2:
                return True
                
        except (OSError, TypeError):
            pass
        
        # 3. Historical performance
        if self.call_count > 5:  # Have some history
            avg_remote = self.total_remote_time / max(self.remote_calls, 1)
            avg_local = self.total_local_time / max(self.local_calls, 1)
            
            # Use remote if it's been faster on average
            if avg_remote < avg_local * 0.8:  # 20% threshold
                return True
        
        return False
    
    def __call__(self, *args, **kwargs):
        """Execute the function with potential acceleration."""
        self.call_count += 1
        start_time = time.time()
        
        # Decide execution mode
        use_remote = self._should_use_remote(args, kwargs)
        
        if use_remote:
            client = self._get_client()
            if client:
                try:
                    self.logger.debug(f"🚀 Executing {self.func_name} remotely")
                    
                    # Try remote execution
                    result = client.execute(self.func_name, *args, **kwargs)
                    
                    execution_time = time.time() - start_time
                    self.remote_calls += 1
                    self.total_remote_time += execution_time
                    self.last_execution_time = execution_time
                    self.last_execution_mode = "remote"
                    
                    if self.profile:
                        self.logger.info(f"✅ {self.func_name} completed remotely in {execution_time:.3f}s")
                    
                    return result
                    
                except Exception as e:
                    self.logger.debug(f"⚠️ Remote execution failed: {e}")
                    if not self.fallback_local:
                        raise
                    # Fall through to local execution
        
        # Local execution
        self.logger.debug(f"🔄 Executing {self.func_name} locally")
        result = self.func(*args, **kwargs)
        
        execution_time = time.time() - start_time
        self.local_calls += 1
        self.total_local_time += execution_time
        self.last_execution_time = execution_time
        self.last_execution_mode = "local"
        
        if self.profile:
            self.logger.info(f"✅ {self.func_name} completed locally in {execution_time:.3f}s")
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics for this function."""
        avg_remote = self.total_remote_time / max(self.remote_calls, 1)
        avg_local = self.total_local_time / max(self.local_calls, 1)
        
        return {
            "function_name": self.func_name,
            "module": self.module_name,
            "total_calls": self.call_count,
            "remote_calls": self.remote_calls,
            "local_calls": self.local_calls,
            "remote_percentage": f"{(self.remote_calls / max(self.call_count, 1)) * 100:.1f}%",
            "total_remote_time": f"{self.total_remote_time:.3f}s",
            "total_local_time": f"{self.total_local_time:.3f}s",
            "avg_remote_time": f"{avg_remote:.3f}s",
            "avg_local_time": f"{avg_local:.3f}s",
            "last_execution_time": f"{self.last_execution_time:.3f}s",
            "last_execution_mode": self.last_execution_mode,
            "speedup_ratio": f"{avg_local / avg_remote:.2f}x" if avg_remote > 0 else "N/A"
        }


def remote(gateway_address: str = "easynet.run:8617",
            force_remote: bool = True,
            fallback_local: bool = True,
            profile: bool = False):
    """
    Decorator to execute a function remotely with EasyRemote.
    
    Args:
        gateway_address: EasyRemote gateway address
        force_remote: Whether to force remote execution (default: True)
        fallback_local: Whether to fallback to local execution on failure
        profile: Whether to enable profiling for this function
        
    Returns:
        Decorated function with remote execution capabilities
        
    Example:
        @remote()
        def compute_heavy_task(data):
            # Computationally intensive work executed remotely
            return process_data(data)
        
        @remote(profile=True)
        def ml_training(dataset, epochs=100):
            # Machine learning training executed remotely
            return train_model(dataset, epochs)
    """
    def decorator(func: Callable) -> AcceleratedFunction:
        return AcceleratedFunction(
            func=func,
            gateway_address=gateway_address,
            force_remote=force_remote,
            fallback_local=fallback_local,
            profile=profile
        )
    
    return decorator


def auto_accelerate(gateway_address: str = "easynet.run:8617",
                   profile: bool = False):
    """
    Decorator to automatically enable remote execution for all functions in a module or class.
    
    Args:
        gateway_address: EasyRemote gateway address
        profile: Whether to enable profiling
        
    Returns:
        Decorator that enables remote execution for all eligible functions
        
    Example:
        @auto_accelerate()
        class MLPipeline:
            def preprocess(self, data):
                return clean_data(data)
            
            def train(self, data):
                return train_model(data)
            
            def predict(self, model, data):
                return model.predict(data)
    """
    def decorator(target):
        if inspect.isclass(target):
            # Store original methods and create accelerated versions
            original_methods = {}
            for name, method in inspect.getmembers(target, inspect.isfunction):
                if not name.startswith('_'):  # Skip private methods
                    original_methods[name] = method
                    
                    # Create remote wrapper for the method
                    @remote(gateway_address=gateway_address, profile=profile)
                    def accelerated_method(*args, **kwargs):
                        # Get the original method and call it
                        return original_methods[name](*args, **kwargs)
                    
                    # Preserve method metadata
                    accelerated_method.__name__ = method.__name__
                    accelerated_method.__doc__ = method.__doc__
                    
                    setattr(target, name, accelerated_method)
            
            return target
        
        elif inspect.ismodule(target):
            # Accelerate all functions in the module
            for name, func in inspect.getmembers(target, inspect.isfunction):
                if func.__module__ == target.__name__ and not name.startswith('_'):
                    accelerated = AcceleratedFunction(
                        func=func,
                        gateway_address=gateway_address,
                        profile=profile
                    )
                    setattr(target, name, accelerated)
            return target
        
        else:
            # Treat as a single function
            return AcceleratedFunction(
                func=target,
                gateway_address=gateway_address,
                profile=profile
            )
    
    return decorator


class EasyNetAcceleratorManager:
    """
    Global manager for EasyNet function acceleration.
    """
    
    def __init__(self):
        """Initialize the accelerator manager."""
        self.accelerated_functions: Dict[str, AcceleratedFunction] = {}
        self.logger = ModernLogger("EasyNetAcceleratorManager", level="info")
        self._lock = threading.Lock()
    
    def register_function(self, func: AcceleratedFunction):
        """Register an accelerated function."""
        with self._lock:
            key = f"{func.module_name}.{func.func_name}"
            self.accelerated_functions[key] = func
            self.logger.debug(f"📝 Registered accelerated function: {key}")
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get global acceleration statistics."""
        with self._lock:
            total_calls = sum(f.call_count for f in self.accelerated_functions.values())
            total_remote_calls = sum(f.remote_calls for f in self.accelerated_functions.values())
            total_local_calls = sum(f.local_calls for f in self.accelerated_functions.values())
            total_remote_time = sum(f.total_remote_time for f in self.accelerated_functions.values())
            total_local_time = sum(f.total_local_time for f in self.accelerated_functions.values())
            
            return {
                "summary": {
                    "total_functions": len(self.accelerated_functions),
                    "total_calls": total_calls,
                    "remote_calls": total_remote_calls,
                    "local_calls": total_local_calls,
                    "acceleration_ratio": f"{(total_remote_calls / max(total_calls, 1)) * 100:.1f}%",
                    "total_remote_time": f"{total_remote_time:.3f}s",
                    "total_local_time": f"{total_local_time:.3f}s",
                    "total_time_saved": f"{max(0, total_local_time - total_remote_time):.3f}s"
                },
                "functions": {
                    key: func.get_stats() 
                    for key, func in self.accelerated_functions.items()
                }
            }
    
    def print_report(self):
        """Print a comprehensive acceleration report."""
        stats = self.get_global_stats()
        summary = stats["summary"]
        
        print("\n" + "="*80)
        print("🚀 EasyNet Acceleration Report")
        print("="*80)
        
        print(f"📊 Summary:")
        print(f"  Total accelerated functions: {summary['total_functions']}")
        print(f"  Total function calls: {summary['total_calls']}")
        print(f"  Remote executions: {summary['remote_calls']}")
        print(f"  Local executions: {summary['local_calls']}")
        print(f"  Acceleration ratio: {summary['acceleration_ratio']}")
        print(f"  Total remote time: {summary['total_remote_time']}")
        print(f"  Total local time: {summary['total_local_time']}")
        print(f"  Time saved: {summary['total_time_saved']}")
        
        if stats["functions"]:
            print(f"\n📋 Function Details:")
            for func_key, func_stats in stats["functions"].items():
                print(f"  {func_key}:")
                print(f"    Calls: {func_stats['total_calls']} "
                      f"(Remote: {func_stats['remote_calls']}, Local: {func_stats['local_calls']})")
                print(f"    Remote %: {func_stats['remote_percentage']}")
                print(f"    Avg times: Remote {func_stats['avg_remote_time']}, Local {func_stats['avg_local_time']}")
                print(f"    Speedup: {func_stats['speedup_ratio']}")
                print(f"    Last: {func_stats['last_execution_mode']} in {func_stats['last_execution_time']}")
        
        print("="*80)


# Global accelerator manager instance
_accelerator_manager = EasyNetAcceleratorManager()


def get_acceleration_stats() -> Dict[str, Any]:
    """Get global acceleration statistics."""
    return _accelerator_manager.get_global_stats()


def print_acceleration_report():
    """Print a comprehensive acceleration report."""
    _accelerator_manager.print_report()


# Convenience functions for common use cases
def accelerate(gateway_address: str = "easynet.run:8617", 
               force_remote: bool = False,
               fallback_local: bool = True,
               profile: bool = False):
    """
    Decorator for intelligent acceleration based on function characteristics.
    
    Example:
        @accelerate()
        def adaptive_function(data):
            # Will automatically decide local vs remote based on data size and complexity
            return process_data(data)
    """
    return remote(
        gateway_address=gateway_address,
        force_remote=force_remote,
        fallback_local=fallback_local,
        profile=profile
    )


def smart_accelerate(gateway_address: str = "easynet.run:8617", profile: bool = False):
    """
    Decorator for intelligent acceleration based on function characteristics.
    
    Example:
        @smart_accelerate()
        def adaptive_function(data):
            # Will automatically decide local vs remote based on data size and complexity
            return process_data(data)
    """
    return remote(
        gateway_address=gateway_address,
        force_remote=False,
        fallback_local=True,
        profile=profile
    ) 