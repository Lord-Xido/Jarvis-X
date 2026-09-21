"""Sparse 3D virtual-computer primitives."""

from .codec import EncodedBlock, SparseBlock, ZlibSparseBlockCodec
from .geometry import BlockAddress, Coordinate3D, VolumeGeometry
from .volume import (
    OperationalMode,
    OptimizationCandidate,
    OptimizationReport,
    Virtual3DComputer,
    VolumeStatistics,
)

__all__ = [
    "BlockAddress",
    "Coordinate3D",
    "EncodedBlock",
    "OperationalMode",
    "OptimizationCandidate",
    "OptimizationReport",
    "SparseBlock",
    "Virtual3DComputer",
    "VolumeGeometry",
    "VolumeStatistics",
    "ZlibSparseBlockCodec",
]
