"""Jarvis-X cloud-native sparse virtual LLM runtime.

This package deliberately separates symbolic logical extent from physically
materialized state. Nothing in this package allocates a dense model matching
the declared virtual parameter extent.
"""

from .codec import LatentPacket, LosslessMultimodalCodec
from .extent import HierarchicalAddress, SymbolicParameterExtent
from .multimodal import MediaEnvelope, MediaKind
from .operational import OperationalHyperCloud
from .persistence import JobRecord, SQLiteStateStore
from .routing import ShardCoordinate, SpatialShardRouter
from .runtime import HyperCloudRuntime, SparseParameterStore
from .worker import HyperCloudWorker

__all__ = [
    "HierarchicalAddress",
    "HyperCloudRuntime",
    "HyperCloudWorker",
    "JobRecord",
    "LatentPacket",
    "LosslessMultimodalCodec",
    "MediaEnvelope",
    "MediaKind",
    "OperationalHyperCloud",
    "ShardCoordinate",
    "SparseParameterStore",
    "SpatialShardRouter",
    "SQLiteStateStore",
    "SymbolicParameterExtent",
]
