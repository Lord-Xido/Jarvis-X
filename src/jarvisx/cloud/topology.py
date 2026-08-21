"""Topology primitives for leased 3D HyperCloud worker placement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .routing import ShardCoordinate


@dataclass(frozen=True)
class WorkerDescriptor:
    """One live execution worker registered in the 3D fabric."""

    worker_id: str
    coordinate: ShardCoordinate
    capabilities: tuple[str, ...]
    backend: str
    load: float
    last_seen: float

    def supports(self, capability: str) -> bool:
        return capability in self.capabilities


@dataclass(frozen=True)
class PlacementDecision:
    worker_id: str
    coordinate: ShardCoordinate
    distance: int
    score: float


def manhattan_distance(a: ShardCoordinate, b: ShardCoordinate) -> int:
    """Return deterministic 3D Manhattan distance between two cells."""

    return abs(a.x - b.x) + abs(a.y - b.y) + abs(a.z - b.z)


class TopologyScheduler:
    """Choose the nearest capable worker while accounting for reported load.

    The reference score is intentionally transparent rather than adaptive magic:

        score = distance + load_weight * clamped_load

    Lower is better. Stable worker-id ordering makes ties deterministic.
    """

    def __init__(self, *, load_weight: float = 4.0) -> None:
        if load_weight < 0:
            raise ValueError("load_weight must be non-negative")
        self.load_weight = float(load_weight)

    def choose(
        self,
        *,
        target: ShardCoordinate,
        workers: Iterable[WorkerDescriptor],
        capability: str,
    ) -> PlacementDecision | None:
        candidates: list[PlacementDecision] = []
        for worker in workers:
            if not worker.supports(capability):
                continue
            distance = manhattan_distance(target, worker.coordinate)
            load = min(1.0, max(0.0, float(worker.load)))
            candidates.append(
                PlacementDecision(
                    worker_id=worker.worker_id,
                    coordinate=worker.coordinate,
                    distance=distance,
                    score=float(distance) + self.load_weight * load,
                )
            )
        if not candidates:
            return None
        return min(candidates, key=lambda item: (item.score, item.distance, item.worker_id))
