"""
EasyRemote Command Line Interface Module

This module provides command-line tools for EasyRemote distributed computing,
including the main 'easynet' command for automatic function acceleration.
"""

from .easynet import main as easynet_main
from .config import (
    DEFAULT_EASYNET_PORT,
    DEFAULT_EASYNET_HOST,
    DEFAULT_EASYNET_URI,
    get_default_gateway,
    get_gateway_with_fallback
)

__all__ = [
    'easynet_main',
    'DEFAULT_EASYNET_PORT',
    'DEFAULT_EASYNET_HOST', 
    'DEFAULT_EASYNET_URI',
    'get_default_gateway',
    'get_gateway_with_fallback'
] 