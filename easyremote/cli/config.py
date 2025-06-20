#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EasyNet Configuration

This module contains configuration constants and settings for EasyNet.
"""

# Default EasyNet server configuration
DEFAULT_EASYNET_PORT = 8617
DEFAULT_EASYNET_HOST = "easynet.run"
DEFAULT_EASYNET_URI = f"{DEFAULT_EASYNET_HOST}:{DEFAULT_EASYNET_PORT}"

# Fallback configuration for local development
FALLBACK_HOST = "localhost"
FALLBACK_PORT = 8080
FALLBACK_URI = f"{FALLBACK_HOST}:{FALLBACK_PORT}"

# Configuration priority order
# 1. Command line arguments
# 2. Environment variables
# 3. Configuration file
# 4. Default EasyNet server
# 5. Local fallback

def get_default_gateway() -> str:
    """
    Get the default gateway address with fallback logic.
    
    Returns:
        Default gateway URI string
    """
    import os
    
    # Check environment variable first
    env_gateway = os.getenv('EASYNET_GATEWAY')
    if env_gateway:
        return env_gateway
    
    # Use default EasyNet server
    return DEFAULT_EASYNET_URI

def get_gateway_with_fallback(gateway: str = None) -> str:
    """
    Get gateway address with intelligent fallback.
    
    Args:
        gateway: Specified gateway address
        
    Returns:
        Gateway address with fallback logic applied
    """
    if gateway:
        return gateway
    
    return get_default_gateway()

# Export configuration constants
__all__ = [
    'DEFAULT_EASYNET_PORT',
    'DEFAULT_EASYNET_HOST', 
    'DEFAULT_EASYNET_URI',
    'FALLBACK_HOST',
    'FALLBACK_PORT',
    'FALLBACK_URI',
    'get_default_gateway',
    'get_gateway_with_fallback'
] 