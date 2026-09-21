"""Deterministic GUI control/observation contract for Jarvis-X.

The GUI is deliberately non-authoritative. It reads immutable engine snapshots,
projects them into panel views, and emits bounded command proposals. Runtime
state changes remain the responsibility of the existing CTR / Pi_Lambda
verification and commit path.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, Iterable, Mapping


Scalar = bool | int | float | str


class GUIControlError(ValueError):
    """Base class for deterministic GUI control-plane validation failures."""


class StaleSnapshotError(GUIControlError):
    """Raised when a command targets an engine epoch that is no longer current."""


class QueueFullError(GUIControlError):
    """Raised when the bounded command queue has reached its configured limit."""


class DuplicateCommandError(GUIControlError):
    """Raised when a command identifier is submitted more than once."""


class CommandKind(str, Enum):
    SET_VIEW = "set_view"
    SET_LOD = "set_lod"
    EXECUTE_STEP = "execute_step"
    RUN_BATCH = "run_batch"
    META_OPTIMIZE = "meta_optimize"
    SHUTDOWN = "shutdown"


class PanelVisibility(str, Enum):
    FOREGROUND = "foreground"
    VISIBLE = "visible"
    INACTIVE = "inactive"
    OFFSCREEN = "offscreen"


class LOD(str, Enum):
    REGION = "region"
    TILE = "tile"
    VOXEL = "voxel"
    BIT = "bit"


@dataclass(frozen=True, slots=True)
class EngineSnapshot:
    """Immutable GUI-facing projection of authoritative runtime state."""

    epoch: int
    state_hash: str
    logical_cells: int
    active_cells: int
    residual_norm: float
    fixed_point_delta: float
    lifecycle: str
    changed_bits: int = 0
    logical_bits: int = 0

    def __post_init__(self) -> None:
        if self.epoch < 0:
            raise GUIControlError("epoch must be non-negative")
        if not self.state_hash:
            raise GUIControlError("state_hash must be non-empty")
        if self.logical_cells < 0 or self.active_cells < 0:
            raise GUIControlError("cell counts must be non-negative")
        if self.active_cells > self.logical_cells:
            raise GUIControlError("active_cells cannot exceed logical_cells")
        if self.logical_bits < 0 or self.changed_bits < 0:
            raise GUIControlError("bit counts must be non-negative")
        if self.logical_bits and self.changed_bits > self.logical_bits:
            raise GUIControlError("changed_bits cannot exceed logical_bits")

    @property
    def active_fraction(self) -> float:
        if self.logical_cells == 0:
            return 0.0
        return self.active_cells / self.logical_cells

    @property
    def changed_fraction(self) -> float:
        if self.logical_bits == 0:
            return 0.0
        return self.changed_bits / self.logical_bits


@dataclass(frozen=True, slots=True)
class GUICommand:
    """A user/UI intent targeted at a specific immutable engine epoch."""

    command_id: str
    kind: CommandKind
    expected_epoch: int
    payload: tuple[tuple[str, Scalar], ...] = ()

    def __post_init__(self) -> None:
        if not self.command_id:
            raise GUIControlError("command_id must be non-empty")
        if self.expected_epoch < 0:
            raise GUIControlError("expected_epoch must be non-negative")

    @classmethod
    def from_mapping(
        cls,
        command_id: str,
        kind: CommandKind,
        expected_epoch: int,
        payload: Mapping[str, Scalar] | None = None,
    ) -> "GUICommand":
        items = tuple(sorted((payload or {}).items()))
        return cls(command_id, kind, expected_epoch, items)

    def payload_dict(self) -> dict[str, Scalar]:
        return dict(self.payload)


@dataclass(frozen=True, slots=True)
class CommandProposal:
    """A validated proposal. It is still not authoritative runtime state."""

    command_id: str
    kind: CommandKind
    expected_epoch: int
    render_only: bool
    requires_pi_lambda: bool
    required_capability: str | None
    payload: tuple[tuple[str, Scalar], ...]


@dataclass(frozen=True, slots=True)
class PanelContext:
    panel_id: str
    visibility: PanelVisibility
    camera_distance: float
    last_render_ms: int | None = None

    def __post_init__(self) -> None:
        if not self.panel_id:
            raise GUIControlError("panel_id must be non-empty")
        if self.camera_distance < 0:
            raise GUIControlError("camera_distance must be non-negative")
        if self.last_render_ms is not None and self.last_render_ms < 0:
            raise GUIControlError("last_render_ms must be non-negative")


@dataclass(frozen=True, slots=True)
class PanelProjection:
    panel_id: str
    epoch: int
    state_hash: str
    lod: LOD
    active_fraction: float
    changed_fraction: float
    residual_norm: float
    fixed_point_delta: float
    lifecycle: str


_RENDER_ONLY = frozenset({CommandKind.SET_VIEW, CommandKind.SET_LOD})
_REQUIRED_CAPABILITY = {
    CommandKind.EXECUTE_STEP: "os.execute",
    CommandKind.RUN_BATCH: "os.execute",
    CommandKind.META_OPTIMIZE: "os.meta_optimize",
    CommandKind.SHUTDOWN: "os.shutdown",
}
_REFRESH_MS = {
    PanelVisibility.FOREGROUND: 16,
    PanelVisibility.VISIBLE: 100,
    PanelVisibility.INACTIVE: 1000,
    PanelVisibility.OFFSCREEN: None,
}


class GUIControlPlane:
    """Bounded, deterministic command bus and pure panel projection logic."""

    def __init__(self, *, max_queue: int = 256, max_batch_steps: int = 1024) -> None:
        if max_queue <= 0:
            raise GUIControlError("max_queue must be positive")
        if max_batch_steps <= 0:
            raise GUIControlError("max_batch_steps must be positive")
        self.max_queue = max_queue
        self.max_batch_steps = max_batch_steps
        self._queue: Deque[GUICommand] = deque()
        self._seen_ids: set[str] = set()

    @property
    def queued(self) -> int:
        return len(self._queue)

    def submit(self, command: GUICommand) -> None:
        if command.command_id in self._seen_ids:
            raise DuplicateCommandError(
                f"command_id {command.command_id!r} has already been submitted"
            )
        if len(self._queue) >= self.max_queue:
            raise QueueFullError("GUI command queue is full")
        self._queue.append(command)
        self._seen_ids.add(command.command_id)

    def drain(self, *, limit: int | None = None) -> tuple[GUICommand, ...]:
        if limit is not None and limit < 0:
            raise GUIControlError("limit must be non-negative")
        count = len(self._queue) if limit is None else min(limit, len(self._queue))
        return tuple(self._queue.popleft() for _ in range(count))

    def propose(self, snapshot: EngineSnapshot, command: GUICommand) -> CommandProposal:
        if command.expected_epoch != snapshot.epoch:
            raise StaleSnapshotError(
                f"command epoch {command.expected_epoch} != snapshot epoch {snapshot.epoch}"
            )

        payload = command.payload_dict()
        if command.kind is CommandKind.RUN_BATCH:
            steps = payload.get("steps", 1)
            if isinstance(steps, bool) or not isinstance(steps, int):
                raise GUIControlError("RUN_BATCH steps must be an integer")
            if not 1 <= steps <= self.max_batch_steps:
                raise GUIControlError(
                    f"RUN_BATCH steps must be within [1, {self.max_batch_steps}]"
                )

        render_only = command.kind in _RENDER_ONLY
        return CommandProposal(
            command_id=command.command_id,
            kind=command.kind,
            expected_epoch=command.expected_epoch,
            render_only=render_only,
            requires_pi_lambda=not render_only,
            required_capability=_REQUIRED_CAPABILITY.get(command.kind),
            payload=command.payload,
        )

    @staticmethod
    def should_render(panel: PanelContext, *, now_ms: int) -> bool:
        if now_ms < 0:
            raise GUIControlError("now_ms must be non-negative")
        refresh_ms = _REFRESH_MS[panel.visibility]
        if refresh_ms is None:
            return False
        if panel.last_render_ms is None:
            return True
        if now_ms < panel.last_render_ms:
            raise GUIControlError("now_ms cannot precede last_render_ms")
        return now_ms - panel.last_render_ms >= refresh_ms

    @staticmethod
    def select_lod(
        *,
        camera_distance: float,
        active_fraction: float,
    ) -> LOD:
        if camera_distance < 0:
            raise GUIControlError("camera_distance must be non-negative")
        if not 0.0 <= active_fraction <= 1.0:
            raise GUIControlError("active_fraction must be within [0, 1]")

        # Very close, sparse views can afford exact bit detail.
        if camera_distance <= 1.5 and active_fraction <= 0.25:
            return LOD.BIT
        if camera_distance <= 4.0:
            return LOD.VOXEL
        if camera_distance <= 12.0:
            return LOD.TILE
        return LOD.REGION

    @classmethod
    def project(
        cls,
        snapshot: EngineSnapshot,
        panel: PanelContext,
    ) -> PanelProjection:
        lod = cls.select_lod(
            camera_distance=panel.camera_distance,
            active_fraction=snapshot.active_fraction,
        )
        return PanelProjection(
            panel_id=panel.panel_id,
            epoch=snapshot.epoch,
            state_hash=snapshot.state_hash,
            lod=lod,
            active_fraction=snapshot.active_fraction,
            changed_fraction=snapshot.changed_fraction,
            residual_norm=snapshot.residual_norm,
            fixed_point_delta=snapshot.fixed_point_delta,
            lifecycle=snapshot.lifecycle,
        )


def snapshots_are_monotonic(snapshots: Iterable[EngineSnapshot]) -> bool:
    """Return True when epochs never decrease across a GUI telemetry stream."""

    last_epoch = -1
    for snapshot in snapshots:
        if snapshot.epoch < last_epoch:
            return False
        last_epoch = snapshot.epoch
    return True
