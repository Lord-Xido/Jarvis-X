"""Deterministic cloud control plane for Jarvis-X distributed 3D runtimes.

The module turns the canonical cloud-organ routing model into a bounded,
candidate-first execution protocol:

    task -> route -> shadow execute -> CTR verify -> commit | reject -> receipt

It intentionally does not provide a network transport or claim that a logical
1024^3 body is physically materialized. Backends can map WorkerState entries to
local processes, containers, GPU workers, cluster jobs, or remote services.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Mapping

from jarvisx.kinetic_runtime import canonical_hash


@dataclass(frozen=True)
class RoutingWeights:
    """Weights for the deterministic worker-cost function."""

    latency: float = 0.25
    queue: float = 0.25
    memory: float = 0.15
    transfer: float = 0.30
    kind: float = 0.05

    def __post_init__(self) -> None:
        values = (self.latency, self.queue, self.memory, self.transfer, self.kind)
        if any(value < 0 for value in values):
            raise ValueError("routing weights must be non-negative")
        if sum(values) <= 0:
            raise ValueError("at least one routing weight must be positive")


@dataclass(frozen=True)
class WorkerState:
    """Schedulable cloud-worker state observed by the control plane."""

    worker_id: str
    region: str
    kind: str
    capacity_units: float
    free_units: float
    queue_depth: int
    latency_ms: float
    resident_tiles: frozenset[str] = frozenset()
    healthy: bool = True

    def __post_init__(self) -> None:
        if not self.worker_id:
            raise ValueError("worker_id must not be empty")
        if self.capacity_units <= 0:
            raise ValueError("capacity_units must be positive")
        if not 0 <= self.free_units <= self.capacity_units:
            raise ValueError("free_units must be inside [0, capacity_units]")
        if self.queue_depth < 0:
            raise ValueError("queue_depth must be non-negative")
        if self.latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")


@dataclass(frozen=True)
class TileTask:
    """One versioned unit of work over a logical spatial tile."""

    task_id: str
    tile_id: str
    operation: str
    parent_version: int
    bytes_estimate: int
    preferred_kind: str = "gpu"
    priority: float = 1.0

    def __post_init__(self) -> None:
        if not self.task_id or not self.tile_id or not self.operation:
            raise ValueError("task_id, tile_id, and operation must not be empty")
        if self.parent_version < 0:
            raise ValueError("parent_version must be non-negative")
        if self.bytes_estimate < 0:
            raise ValueError("bytes_estimate must be non-negative")
        if self.priority <= 0:
            raise ValueError("priority must be positive")


@dataclass(frozen=True)
class RouteDecision:
    """Deterministic placement result and its cost components."""

    worker_id: str
    score: float
    latency_cost: float
    queue_cost: float
    memory_cost: float
    transfer_cost: float
    kind_cost: float


@dataclass(frozen=True)
class TileState:
    """Authoritative state for one logical tile."""

    tile_id: str
    version: int
    value: object
    value_hash: str


@dataclass(frozen=True)
class CommitReceipt:
    """Auditable result of one candidate-first cloud transaction."""

    schema_version: str
    task_id: str
    tile_id: str
    worker_id: str
    operation: str
    parent_version: int
    resulting_version: int
    decision: str
    reason: str
    parent_state_hash: str
    candidate_hash: str
    resulting_state_hash: str
    route_score: float
    receipt_hash: str


@dataclass(frozen=True)
class DispatchResult:
    """Route, candidate result, and commit receipt returned to the caller."""

    route: RouteDecision
    receipt: CommitReceipt
    replayed: bool


Executor = Callable[[TileTask, str, TileState], object]
Verifier = Callable[[TileTask, TileState, object], bool]


class CloudScheduler:
    """Choose a worker using bounded locality/load/latency costs."""

    def __init__(
        self,
        workers: Mapping[str, WorkerState],
        weights: RoutingWeights | None = None,
    ) -> None:
        self._workers = dict(workers)
        self.weights = weights or RoutingWeights()

    @property
    def workers(self) -> Mapping[str, WorkerState]:
        return dict(self._workers)

    def update_worker(self, worker: WorkerState) -> None:
        self._workers[worker.worker_id] = worker

    def choose(self, task: TileTask) -> RouteDecision:
        candidates: list[tuple[float, str, RouteDecision]] = []
        for worker in self._workers.values():
            if not worker.healthy or worker.free_units <= 0:
                continue

            latency_cost = worker.latency_ms / 100.0
            queue_cost = worker.queue_depth / worker.capacity_units
            memory_cost = 1.0 - (worker.free_units / worker.capacity_units)
            transfer_cost = (
                0.0 if task.tile_id in worker.resident_tiles else task.bytes_estimate / 1e9
            )
            kind_cost = (
                0.0 if not task.preferred_kind or worker.kind == task.preferred_kind else 1.0
            )
            score = (
                self.weights.latency * latency_cost
                + self.weights.queue * queue_cost
                + self.weights.memory * memory_cost
                + self.weights.transfer * transfer_cost
                + self.weights.kind * kind_cost
            )
            decision = RouteDecision(
                worker_id=worker.worker_id,
                score=score,
                latency_cost=latency_cost,
                queue_cost=queue_cost,
                memory_cost=memory_cost,
                transfer_cost=transfer_cost,
                kind_cost=kind_cost,
            )
            candidates.append((score, worker.worker_id, decision))

        if not candidates:
            raise RuntimeError("no healthy cloud worker has available capacity")

        candidates.sort(key=lambda item: (item[0], item[1]))
        return candidates[0][2]


class CloudControlPlane:
    """Versioned distributed execution with CTR-gated authoritative commit.

    Execution is candidate-first: the executor receives an immutable TileState
    snapshot and returns a proposed value. The proposal becomes authoritative
    only when the verifier accepts it and the parent version is still current.

    Successfully committed task IDs are idempotent. Replaying the same task ID
    returns the original receipt without re-executing or incrementing state.
    """

    RECEIPT_SCHEMA = "jarvisx.cloud-control-receipt.v1"

    def __init__(
        self,
        workers: Mapping[str, WorkerState],
        weights: RoutingWeights | None = None,
    ) -> None:
        self.scheduler = CloudScheduler(workers, weights)
        self._tiles: dict[str, TileState] = {}
        self._committed: dict[str, DispatchResult] = {}

    def register_tile(self, tile_id: str, value: object) -> TileState:
        if not tile_id:
            raise ValueError("tile_id must not be empty")
        if tile_id in self._tiles:
            raise ValueError(f"tile already registered: {tile_id}")
        state = TileState(
            tile_id=tile_id,
            version=0,
            value=value,
            value_hash=canonical_hash(value),
        )
        self._tiles[tile_id] = state
        return state

    def tile_state(self, tile_id: str) -> TileState:
        try:
            return self._tiles[tile_id]
        except KeyError as exc:
            raise KeyError(f"unknown tile: {tile_id}") from exc

    def update_worker(self, worker: WorkerState) -> None:
        self.scheduler.update_worker(worker)

    def route(self, task: TileTask) -> RouteDecision:
        if task.tile_id not in self._tiles:
            raise KeyError(f"unknown tile: {task.tile_id}")
        return self.scheduler.choose(task)

    def dispatch(
        self,
        task: TileTask,
        *,
        executor: Executor,
        verifier: Verifier,
    ) -> DispatchResult:
        replay = self._committed.get(task.task_id)
        if replay is not None:
            return replace(replay, replayed=True)

        route = self.route(task)
        parent = self.tile_state(task.tile_id)
        parent_state_hash = self._state_hash(parent)

        if task.parent_version != parent.version:
            receipt = self._receipt(
                task=task,
                route=route,
                parent=parent,
                resulting=parent,
                candidate_hash=canonical_hash(
                    {
                        "task_id": task.task_id,
                        "skipped": "stale-parent",
                        "requested_parent_version": task.parent_version,
                    }
                ),
                decision="reject",
                reason="stale-parent-version",
            )
            return DispatchResult(route=route, receipt=receipt, replayed=False)

        candidate = executor(task, route.worker_id, parent)
        candidate_hash = canonical_hash(candidate)

        if not verifier(task, parent, candidate):
            receipt = self._receipt(
                task=task,
                route=route,
                parent=parent,
                resulting=parent,
                candidate_hash=candidate_hash,
                decision="reject",
                reason="ctr-verification-failed",
            )
            return DispatchResult(route=route, receipt=receipt, replayed=False)

        current = self.tile_state(task.tile_id)
        if current.version != parent.version or current.value_hash != parent.value_hash:
            receipt = self._receipt(
                task=task,
                route=route,
                parent=parent,
                resulting=current,
                candidate_hash=candidate_hash,
                decision="reject",
                reason="concurrent-state-change",
            )
            return DispatchResult(route=route, receipt=receipt, replayed=False)

        committed = TileState(
            tile_id=parent.tile_id,
            version=parent.version + 1,
            value=candidate,
            value_hash=candidate_hash,
        )
        self._tiles[parent.tile_id] = committed
        receipt = self._receipt(
            task=task,
            route=route,
            parent=parent,
            resulting=committed,
            candidate_hash=candidate_hash,
            decision="commit",
            reason="verified",
        )
        result = DispatchResult(route=route, receipt=receipt, replayed=False)
        self._committed[task.task_id] = result
        return result

    @staticmethod
    def _state_hash(state: TileState) -> str:
        return str(
            canonical_hash(
                {
                    "tile_id": state.tile_id,
                    "version": state.version,
                    "value_hash": state.value_hash,
                }
            )
        )

    def _receipt(
        self,
        *,
        task: TileTask,
        route: RouteDecision,
        parent: TileState,
        resulting: TileState,
        candidate_hash: str,
        decision: str,
        reason: str,
    ) -> CommitReceipt:
        parent_hash = self._state_hash(parent)
        result_hash = self._state_hash(resulting)
        body = {
            "schema_version": self.RECEIPT_SCHEMA,
            "task_id": task.task_id,
            "tile_id": task.tile_id,
            "worker_id": route.worker_id,
            "operation": task.operation,
            "parent_version": task.parent_version,
            "resulting_version": resulting.version,
            "decision": decision,
            "reason": reason,
            "parent_state_hash": parent_hash,
            "candidate_hash": candidate_hash,
            "resulting_state_hash": result_hash,
            "route_score": route.score,
        }
        return CommitReceipt(
            schema_version=self.RECEIPT_SCHEMA,
            task_id=task.task_id,
            tile_id=task.tile_id,
            worker_id=route.worker_id,
            operation=task.operation,
            parent_version=task.parent_version,
            resulting_version=resulting.version,
            decision=decision,
            reason=reason,
            parent_state_hash=parent_hash,
            candidate_hash=candidate_hash,
            resulting_state_hash=result_hash,
            route_score=route.score,
            receipt_hash=canonical_hash(body),
        )
