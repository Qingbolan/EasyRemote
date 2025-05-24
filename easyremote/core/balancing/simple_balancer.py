#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EasyRemote Simple Load Balancing Module

This module provides basic load balancing functionality for the EasyRemote
distributed computing framework. It focuses on simplicity and core functionality
needed for basic distributed computing operations.

Author: Silan Hu
Version: 1.0.0 (Simplified)
"""

import random
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class BalancingStrategy(Enum):
    """Simple load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    RANDOM = "random" 
    RESOURCE_AWARE = "resource_aware"


@dataclass
class SimpleNodeStats:
    """Basic node statistics."""
    node_id: str
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    active_tasks: int = 0
    response_time_ms: float = 0.0
    
    @property
    def load_score(self) -> float:
        """Calculate simple load score (0-100, lower is better)."""
        return (self.cpu_usage + self.memory_usage) / 2 + (self.active_tasks * 5)


class SimpleLoadBalancer:
    """Simple load balancer for EasyRemote."""
    
    def __init__(self, strategy: BalancingStrategy = BalancingStrategy.ROUND_ROBIN):
        self.strategy = strategy
        self._round_robin_counter = 0
    
    def select_node(self, 
                   available_nodes: List[str],
                   node_stats: Optional[Dict[str, SimpleNodeStats]] = None) -> str:
        """
        Select optimal node using configured strategy.
        
        Args:
            available_nodes: List of available node IDs
            node_stats: Optional node statistics
            
        Returns:
            Selected node ID
        """
        if not available_nodes:
            raise ValueError("No available nodes")
        
        if self.strategy == BalancingStrategy.ROUND_ROBIN:
            return self._round_robin_select(available_nodes)
        elif self.strategy == BalancingStrategy.RANDOM:
            return random.choice(available_nodes)
        elif self.strategy == BalancingStrategy.RESOURCE_AWARE:
            return self._resource_aware_select(available_nodes, node_stats)
        else:
            return available_nodes[0]  # Fallback
    
    def _round_robin_select(self, nodes: List[str]) -> str:
        """Round robin node selection."""
        selected = nodes[self._round_robin_counter % len(nodes)]
        self._round_robin_counter += 1
        return selected
    
    def _resource_aware_select(self, 
                              nodes: List[str], 
                              node_stats: Optional[Dict[str, SimpleNodeStats]]) -> str:
        """Resource-aware node selection."""
        if not node_stats:
            return self._round_robin_select(nodes)
        
        # Select node with lowest load
        best_node = None
        best_score = float('inf')
        
        for node in nodes:
            stats = node_stats.get(node)
            if stats:
                score = stats.load_score
                if score < best_score:
                    best_score = score
                    best_node = node
        
        return best_node or nodes[0]


def create_balancer(strategy: str = "round_robin") -> SimpleLoadBalancer:
    """Create a simple load balancer with specified strategy."""
    strategy_map = {
        "round_robin": BalancingStrategy.ROUND_ROBIN,
        "random": BalancingStrategy.RANDOM,
        "resource_aware": BalancingStrategy.RESOURCE_AWARE
    }
    
    return SimpleLoadBalancer(strategy_map.get(strategy, BalancingStrategy.ROUND_ROBIN)) 