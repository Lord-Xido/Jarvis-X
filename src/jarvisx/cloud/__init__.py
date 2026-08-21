"""Jarvis-X cloud-native sparse virtual LLM runtime.

This package deliberately separates symbolic logical extent from physically
materialized state. Nothing in this package allocates a dense model matching
the declared virtual parameter extent.
"""

from .extent import HierarchicalAddress, SymbolicParameterExtent
from .multimodal import MediaEnvelope, MediaKind
from .routing import ShardCoordinate, SpatialShardRouter
from .runtime import HyperCloudRuntime, SparseParameterStore

__all__ = [
    "HierarchicalAddress",
    "HyperCloudRuntime",
    "MediaEnvelope",
    "MediaKind",
    "ShardCoordinate",
    "SparseParameterStore",
    "SpatialShardRouter",
    "SymbolicParameterExtent",
]
