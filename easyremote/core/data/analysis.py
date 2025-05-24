import inspect
import asyncio
from typing import Optional
from dataclasses import dataclass


@dataclass
class FunctionAnalysis:
    """Analysis result for a function"""
    is_async: bool
    is_generator: bool
    is_async_generator: bool
    is_class: bool
    is_method: bool
    is_builtin: bool
    signature: Optional[inspect.Signature] = None


def analyze_function(func, cache: dict = None) -> FunctionAnalysis:
    """
    Analyze function characteristics for execution planning.
    
    Args:
        func: Function or callable to analyze
        cache: Optional cache dictionary for storing results
        
    Returns:
        FunctionAnalysis object with function characteristics
    """
    # Create cache key
    cache_key = f"{func.__module__}.{func.__name__}" if hasattr(func, '__module__') else str(func)
    
    if cache and cache_key in cache:
        return cache[cache_key]
    
    try:
        analysis = FunctionAnalysis(
            is_async=asyncio.iscoroutinefunction(func),
            is_generator=inspect.isgeneratorfunction(func),
            is_async_generator=inspect.isasyncgenfunction(func),
            is_class=inspect.isclass(func),
            is_method=inspect.ismethod(func),
            is_builtin=inspect.isbuiltin(func),
            signature=inspect.signature(func) if not inspect.isbuiltin(func) else None
        )
        
        if cache:
            cache[cache_key] = analysis
        
        return analysis
        
    except Exception:
        # Return a basic analysis on failure
        return FunctionAnalysis(
            is_async=False,
            is_generator=False,
            is_async_generator=False,
            is_class=False,
            is_method=False,
            is_builtin=False
        ) 