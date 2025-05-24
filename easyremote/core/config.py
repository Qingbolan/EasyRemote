# easyremote/core/config.py
"""
Configuration management for EasyRemote
"""
from dataclasses import dataclass

@dataclass
class ServerConfig:
    """Server configuration settings"""
    port: int = 8080
    heartbeat_timeout: int = 10
    max_queue_size: int = 1000
    max_workers: int = 10
    grpc_max_message_size: int = 50 * 1024 * 1024
    cleanup_interval: int = 60  # seconds
    stale_timeout: int = 300  # seconds

@dataclass
class NodeConfig:
    """Compute node configuration settings"""
    reconnect_interval: int = 5
    heartbeat_interval: int = 5
    max_retry_attempts: int = 3
    max_queue_size: int = 1000
    execution_timeout: int = 300
    connection_timeout: int = 10

@dataclass
class PerformanceConfig:
    """Performance monitoring configuration"""
    collection_interval: float = 5.0
    max_history: int = 1000
    cpu_threshold: float = 80.0
    memory_threshold: float = 80.0
    error_rate_threshold: float = 0.1

@dataclass
class EasyRemoteConfig:
    """Main configuration class"""
    server: ServerConfig = None
    node: NodeConfig = None
    performance: PerformanceConfig = None
    
    def __post_init__(self):
        if self.server is None:
            self.server = ServerConfig()
        if self.node is None:
            self.node = NodeConfig()
        if self.performance is None:
            self.performance = PerformanceConfig()

# Global default configuration
default_config = EasyRemoteConfig()

def get_config() -> EasyRemoteConfig:
    """Get the default configuration"""
    return default_config

def update_config(config: EasyRemoteConfig):
    """Update the global configuration"""
    global default_config
    default_config = config 