"""Deterministic 3D shard routing for sparse cloud state."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b

from .extent import HierarchicalAddress


@dataclass(frozen=True)
class ShardCoordinate:
    x: int
    y: int
    z: int


@dataclass(frozen=True)
class SpatialShardRouter:
    """Map logical addresses onto a finite 3D deployment lattice.

    The lattice describes deployed workers/shards, not the full symbolic model
    extent. Rendezvous-style rebalancing can be added later; this reference
    router prioritizes deterministic reproducibility and bounded arithmetic.
    """

    shape: tuple[int, int, int] = (16, 16, 16)

    def __post_init__(self) -> None:
        if len(self.shape) != 3 or any(axis <= 0 for axis in self.shape):
            raise ValueError("shape must contain three positive dimensions")

    @property
    def shard_count(self) -> int:
        x, y, z = self.shape
        return x * y * z

    def route(
        self,
        *,
        namespace: str,
        modality: str,
        address: HierarchicalAddress,
    ) -> ShardCoordinate:
        if not namespace:
            raise ValueError("namespace must not be empty")
        payload = f"{namespace}|{modality}|{address.canonical()}".encode("utf-8")
        digest = blake2b(payload, digest_size=24, person=b"jarvisx-cloud").digest()
        a = int.from_bytes(digest[0:8], "big")
        b = int.from_bytes(digest[8:16], "big")
        c = int.from_bytes(digest[16:24], "big")
        return ShardCoordinate(a % self.shape[0], b % self.shape[1], c % self.shape[2])
