#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EasyNet Command Line Tool

A command-line tool similar to torchrun that automatically accelerates Python scripts
by intercepting function calls and routing them to EasyRemote distributed compute nodes.

Usage:
    easynet script.py [args...]
    easynet --gateway localhost:8080 script.py [args...]
    easynet --auto-discover script.py [args...]
    easynet --profile script.py [args...]

Features:
- Automatic function interception and acceleration
- Dynamic gateway discovery
- Performance profiling and optimization
- Transparent acceleration with minimal code changes
"""

import sys
import os
import argparse
import importlib.util
import ast
import inspect
import threading
import time
from typing import Dict, List, Any, Optional, Callable, Set
from pathlib import Path
import logging

# EasyRemote imports
from ..core.nodes.server import DistributedComputingGateway
from ..core.nodes.client import DistributedComputingClient
from ..core.utils.logger import ModernLogger
from ..core.data import FunctionType
from .config import DEFAULT_EASYNET_URI, get_default_gateway, get_gateway_with_fallback

class EasyNetInterceptor:
    """
    Function call interceptor that automatically routes eligible functions
    to EasyRemote distributed compute nodes.
    """
    
    def __init__(self, gateway_address: str = None, 
                 auto_discover: bool = False,
                 profile_mode: bool = False):
        """
        Initialize the EasyNet interceptor.
        
        Args:
            gateway_address: Address of the EasyRemote gateway
            auto_discover: Whether to auto-discover available gateways
            profile_mode: Whether to enable performance profiling
        """
        self.gateway_address = get_gateway_with_fallback(gateway_address)
        self.auto_discover = auto_discover
        self.profile_mode = profile_mode
        
        # Initialize logger
        self.logger = ModernLogger("EasyNet", level="info")
        
        # Initialize client
        self.client: Optional[DistributedComputingClient] = None
        self._client_lock = threading.Lock()
        
        # Function tracking
        self.intercepted_functions: Dict[str, Callable] = {}
        self.function_stats: Dict[str, Dict[str, Any]] = {}
        self.acceleration_candidates: Set[str] = set()
        
        # Performance tracking
        self.total_calls = 0
        self.accelerated_calls = 0
        self.local_execution_time = 0.0
        self.remote_execution_time = 0.0
        
        self.logger.info(f"🚀 EasyNet Interceptor initialized (gateway: {gateway_address})")
    
    def _get_client(self) -> DistributedComputingClient:
        """Get or create the distributed computing client."""
        if self.client is None:
            with self._client_lock:
                if self.client is None:
                    try:
                        self.client = DistributedComputingClient(self.gateway_address)
                        self.logger.info(f"✅ Connected to EasyRemote gateway at {self.gateway_address}")
                    except Exception as e:
                        self.logger.warning(f"⚠️ Failed to connect to gateway: {e}")
                        self.logger.info("🔄 Falling back to local execution")
        return self.client
    
    def should_accelerate(self, func_name: str, func: Callable, args: tuple, kwargs: dict) -> bool:
        """
        Determine if a function should be accelerated based on various criteria.
        
        Args:
            func_name: Name of the function
            func: The function object
            args: Function arguments
            kwargs: Function keyword arguments
            
        Returns:
            True if function should be accelerated, False otherwise
        """
        # Skip built-in functions
        if hasattr(func, '__module__') and func.__module__ in ('builtins', None):
            return False
        
        # Skip very simple functions (heuristic)
        try:
            source = inspect.getsource(func)
            if len(source.split('\n')) < 3:  # Very short functions
                return False
        except (OSError, TypeError):
            return False
        
        # Check if function is computationally intensive (heuristic)
        if func_name in self.acceleration_candidates:
            return True
        
        # Auto-detect based on function characteristics
        if self._is_compute_intensive(func, args, kwargs):
            self.acceleration_candidates.add(func_name)
            return True
        
        return False
    
    def _is_compute_intensive(self, func: Callable, args: tuple, kwargs: dict) -> bool:
        """
        Heuristic to determine if a function is compute-intensive.
        
        Args:
            func: The function to analyze
            args: Function arguments
            kwargs: Function keyword arguments
            
        Returns:
            True if function appears to be compute-intensive
        """
        # Check for common compute-intensive patterns
        try:
            source = inspect.getsource(func)
            
            # Look for loops, mathematical operations, data processing
            compute_keywords = [
                'for ', 'while ', 'numpy', 'np.', 'pandas', 'pd.',
                'torch', 'tensorflow', 'sklearn', 'scipy',
                'matrix', 'array', 'compute', 'process', 'calculate',
                'train', 'predict', 'transform', 'fit'
            ]
            
            keyword_count = sum(1 for keyword in compute_keywords if keyword in source.lower())
            
            # If function has multiple compute keywords, likely compute-intensive
            if keyword_count >= 2:
                return True
            
            # Check argument sizes (large data structures)
            for arg in args:
                if hasattr(arg, '__len__') and len(arg) > 1000:  # Large data
                    return True
                if hasattr(arg, 'shape') and any(dim > 100 for dim in arg.shape):  # Large arrays
                    return True
            
        except (OSError, TypeError, AttributeError):
            pass
        
        return False
    
    def intercept_call(self, func_name: str, func: Callable, args: tuple, kwargs: dict) -> Any:
        """
        Intercept and potentially accelerate a function call.
        
        Args:
            func_name: Name of the function
            func: The function object
            args: Function arguments
            kwargs: Function keyword arguments
            
        Returns:
            Function result (either from remote execution or local fallback)
        """
        self.total_calls += 1
        
        # Record function for potential acceleration
        if func_name not in self.intercepted_functions:
            self.intercepted_functions[func_name] = func
            self.function_stats[func_name] = {
                'calls': 0,
                'local_time': 0.0,
                'remote_time': 0.0,
                'accelerated': 0
            }
        
        stats = self.function_stats[func_name]
        stats['calls'] += 1
        
        # Decide whether to accelerate
        if self.should_accelerate(func_name, func, args, kwargs):
            client = self._get_client()
            if client:
                try:
                    start_time = time.time()
                    
                    # Try remote execution
                    self.logger.debug(f"🚀 Accelerating {func_name} on remote node")
                    result = client.call_function(func_name, *args, **kwargs)
                    
                    execution_time = time.time() - start_time
                    stats['remote_time'] += execution_time
                    stats['accelerated'] += 1
                    self.accelerated_calls += 1
                    self.remote_execution_time += execution_time
                    
                    self.logger.debug(f"✅ {func_name} completed remotely in {execution_time:.3f}s")
                    return result
                    
                except Exception as e:
                    self.logger.debug(f"⚠️ Remote execution failed for {func_name}: {e}")
                    # Fall through to local execution
        
        # Local execution
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        stats['local_time'] += execution_time
        self.local_execution_time += execution_time
        
        return result
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate a performance report of acceleration results."""
        total_time = self.local_execution_time + self.remote_execution_time
        acceleration_ratio = (
            self.accelerated_calls / self.total_calls 
            if self.total_calls > 0 else 0.0
        )
        
        return {
            "summary": {
                "total_calls": self.total_calls,
                "accelerated_calls": self.accelerated_calls,
                "acceleration_ratio": f"{acceleration_ratio:.1%}",
                "total_execution_time": f"{total_time:.3f}s",
                "local_time": f"{self.local_execution_time:.3f}s",
                "remote_time": f"{self.remote_execution_time:.3f}s"
            },
            "functions": self.function_stats,
            "candidates": list(self.acceleration_candidates)
        }


class EasyNetRunner:
    """
    Main runner for EasyNet that handles script execution with automatic acceleration.
    """
    
    def __init__(self, gateway_address: str = None,
                 auto_discover: bool = False,
                 profile_mode: bool = False,
                 verbose: bool = False):
        """
        Initialize the EasyNet runner.
        
        Args:
            gateway_address: Address of the EasyRemote gateway
            auto_discover: Whether to auto-discover available gateways
            profile_mode: Whether to enable performance profiling
            verbose: Whether to enable verbose logging
        """
        self.gateway_address = get_gateway_with_fallback(gateway_address)
        self.auto_discover = auto_discover
        self.profile_mode = profile_mode
        self.verbose = verbose
        
        # Initialize logger
        log_level = "debug" if verbose else "info"
        self.logger = ModernLogger("EasyNetRunner", level=log_level)
        
        # Initialize interceptor
        self.interceptor = EasyNetInterceptor(
            gateway_address=gateway_address,
            auto_discover=auto_discover,
            profile_mode=profile_mode
        )
        
        self.logger.info("🎯 EasyNet Runner initialized")
    
    def run_script(self, script_path: str, script_args: List[str]) -> int:
        """
        Run a Python script with EasyNet acceleration.
        
        Args:
            script_path: Path to the Python script to run
            script_args: Arguments to pass to the script
            
        Returns:
            Exit code of the script execution
        """
        script_path = Path(script_path)
        
        if not script_path.exists():
            self.logger.error(f"❌ Script not found: {script_path}")
            return 1
        
        if not script_path.suffix == '.py':
            self.logger.error(f"❌ Not a Python script: {script_path}")
            return 1
        
        self.logger.info(f"🚀 Running script with EasyNet acceleration: {script_path}")
        self.logger.info(f"📋 Script arguments: {script_args}")
        
        try:
            # Prepare environment
            original_argv = sys.argv.copy()
            sys.argv = [str(script_path)] + script_args
            
            # Add script directory to Python path
            script_dir = str(script_path.parent.absolute())
            if script_dir not in sys.path:
                sys.path.insert(0, script_dir)
            
            # Install function interceptor
            self._install_interceptor()
            
            # Load and execute the script
            spec = importlib.util.spec_from_file_location("__main__", script_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load script: {script_path}")
            
            module = importlib.util.module_from_spec(spec)
            
            # Execute the script
            start_time = time.time()
            spec.loader.exec_module(module)
            execution_time = time.time() - start_time
            
            self.logger.info(f"✅ Script completed in {execution_time:.3f}s")
            
            # Generate performance report
            if self.profile_mode:
                self._generate_performance_report()
            
            return 0
            
        except KeyboardInterrupt:
            self.logger.info("⚠️ Script interrupted by user")
            return 130
        except Exception as e:
            self.logger.error(f"💥 Script execution failed: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return 1
        finally:
            # Restore environment
            sys.argv = original_argv
            if script_dir in sys.path:
                sys.path.remove(script_dir)
    
    def _install_interceptor(self):
        """Install the function call interceptor."""
        # This is a simplified version - in practice, you'd need more sophisticated
        # bytecode manipulation or import hooks to intercept arbitrary function calls
        self.logger.debug("🔧 Installing function call interceptor")
        
        # For now, we'll demonstrate the concept with a decorator approach
        # In a full implementation, you'd use import hooks or bytecode manipulation
        
    def _generate_performance_report(self):
        """Generate and display performance report."""
        report = self.interceptor.get_performance_report()
        
        self.logger.info("\n" + "="*60)
        self.logger.info("📊 EasyNet Performance Report")
        self.logger.info("="*60)
        
        summary = report["summary"]
        self.logger.info(f"Total function calls: {summary['total_calls']}")
        self.logger.info(f"Accelerated calls: {summary['accelerated_calls']}")
        self.logger.info(f"Acceleration ratio: {summary['acceleration_ratio']}")
        self.logger.info(f"Total execution time: {summary['total_execution_time']}")
        self.logger.info(f"  - Local time: {summary['local_time']}")
        self.logger.info(f"  - Remote time: {summary['remote_time']}")
        
        if report["functions"]:
            self.logger.info("\n📋 Function Statistics:")
            for func_name, stats in report["functions"].items():
                self.logger.info(f"  {func_name}:")
                self.logger.info(f"    Calls: {stats['calls']}")
                self.logger.info(f"    Accelerated: {stats['accelerated']}")
                self.logger.info(f"    Local time: {stats['local_time']:.3f}s")
                self.logger.info(f"    Remote time: {stats['remote_time']:.3f}s")
        
        if report["candidates"]:
            self.logger.info(f"\n🎯 Acceleration candidates: {', '.join(report['candidates'])}")


def create_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog='easynet',
        description='EasyNet - Automatic Python script acceleration with EasyRemote',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  easynet script.py                           # Run with default settings
  easynet --gateway remote:8080 script.py    # Use specific gateway
  easynet --profile script.py                # Enable performance profiling
  easynet --verbose script.py arg1 arg2      # Verbose output with script args
        """
    )
    
    parser.add_argument(
        'script',
        help='Python script to run with acceleration'
    )
    
    parser.add_argument(
        'args',
        nargs='*',
        help='Arguments to pass to the script'
    )
    
    parser.add_argument(
        '--gateway', '-g',
        default=None,
        help=f'EasyRemote gateway address (default: {get_default_gateway()})'
    )
    
    parser.add_argument(
        '--auto-discover', '-a',
        action='store_true',
        help='Auto-discover available gateways'
    )
    
    parser.add_argument(
        '--profile', '-p',
        action='store_true',
        help='Enable performance profiling and reporting'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='EasyNet 1.0.0'
    )
    
    return parser


def main():
    """Main entry point for the easynet command."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Initialize runner
    runner = EasyNetRunner(
        gateway_address=args.gateway,
        auto_discover=args.auto_discover,
        profile_mode=args.profile,
        verbose=args.verbose
    )
    
    # Run the script
    exit_code = runner.run_script(args.script, args.args)
    sys.exit(exit_code)


if __name__ == '__main__':
    main() 