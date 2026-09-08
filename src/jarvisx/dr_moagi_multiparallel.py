"""Sparse 1000x1000 multiparallel inward/outward 3D Dr Moagi runtime.

The runtime models one million *logical* auto-encoding/decoding loops arranged on
an X/Y lattice.  Recursive encode/decode depth is the third spatial axis.  Only
active loop/depth cells are materialized, so the reference implementation never
allocates a dense 1000x1000xK tensor.

One transaction performs:

    surface -> kinetic permeation -> inward fold -> global core -> outward fold
            -> reconstruction/error -> memory update -> commit/rollback

The implementation is deterministic and CPU-oriented.  The state layout is
chosen so the same equations can be tensorized on GPU/TPU backends later.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, Mapping, Sequence

LoopCoordinate = tuple[int, int]
VolumeCoordinate = tuple[int, int, int]
Vector = tuple[float, ...]
SparseLoopField = dict[LoopCoordinate, Vector]
SparseVolume3D = dict[VolumeCoordinate, Vector]
Validator = Callable[[Mapping[LoopCoordinate, Vector], "MultiparallelStepMetrics"], bool]


@dataclass(frozen=True)
class MultiparallelConfig:
    """Numerical, topology and resource contract for the sparse logical lattice."""

    side: int = 1000
    depth: int = 8
    max_active_loops: int = 100_000
    max_channels: int = 64
    dt: float = 0.05
    damping: float = 0.90
    diffusion: float = 0.08
    memory_gain: float = 0.10
    memory_decay: float = 0.90
    encode_scale: float = 0.5
    quantization: float = 1.0e-6
    global_coupling: float = 0.05
    prune_epsilon: float = 1.0e-12
    expand_halo: bool = True
    max_reconstruction_mse: float = 0.25

    def __post_init__(self) -> None:
        if isinstance(self.side, bool) or not isinstance(self.side, int) or self.side <= 0:
            raise ValueError("side must be a positive integer")
        if isinstance(self.depth, bool) or not isinstance(self.depth, int) or self.depth <= 0:
            raise ValueError("depth must be a positive integer")
        if (
            isinstance(self.max_active_loops, bool)
            or not isinstance(self.max_active_loops, int)
            or self.max_active_loops <= 0
        ):
            raise ValueError("max_active_loops must be a positive integer")
        if (
            isinstance(self.max_channels, bool)
            or not isinstance(self.max_channels, int)
            or self.max_channels <= 0
        ):
            raise ValueError("max_channels must be a positive integer")

        for name in (
            "dt",
            "damping",
            "diffusion",
            "memory_gain",
            "memory_decay",
            "encode_scale",
            "quantization",
            "global_coupling",
            "prune_epsilon",
            "max_reconstruction_mse",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")

        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if not 0.0 <= self.damping <= 1.0:
            raise ValueError("damping must be in [0, 1]")
        if self.diffusion < 0.0 or self.memory_gain < 0.0:
            raise ValueError("diffusion and memory_gain must be non-negative")
        if not 0.0 <= self.memory_decay <= 1.0:
            raise ValueError("memory_decay must be in [0, 1]")
        if not 0.0 < self.encode_scale <= 1.0:
            raise ValueError("encode_scale must be in (0, 1]")
        if self.quantization <= 0.0:
            raise ValueError("quantization must be positive")
        if not 0.0 <= self.global_coupling <= 1.0:
            raise ValueError("global_coupling must be in [0, 1]")
        if self.prune_epsilon < 0.0:
            raise ValueError("prune_epsilon must be non-negative")
        if self.max_reconstruction_mse < 0.0:
            raise ValueError("max_reconstruction_mse must be non-negative")

    @property
    def logical_loop_count(self) -> int:
        return self.side * self.side

    @property
    def logical_volume_cell_count(self) -> int:
        return self.logical_loop_count * (self.depth + 1)


@dataclass(frozen=True)
class MultiparallelStepMetrics:
    cycle: int
    support_loops: int
    active_loops_before: int
    active_loops_after: int
    allocated_volume_cells: int
    channel_count: int
    reconstruction_mse: float
    max_abs_error: float
    kinetic_energy: float
    global_core: Vector
    committed: bool
    rejection_reason: str | None = None


class DrMoagiMultiparallel3D:
    """Sparse reference runtime for 1000x1000 logical AE/AD loops.

    X/Y identify a logical loop.  Encode/decode depth is Z.  Each loop carries a
    fixed-width multimodal vector; channel semantics are intentionally external
    (for example image/audio/text/video/geometry/code projections may all feed
    the same vector contract).
    """

    _NEIGHBOURS: tuple[LoopCoordinate, ...] = (
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
    )

    def __init__(self, config: MultiparallelConfig | None = None) -> None:
        self.config = config or MultiparallelConfig()
        self._surface: SparseLoopField = {}
        self._velocity: SparseLoopField = {}
        self._memory: SparseLoopField = {}
        self._error: SparseLoopField = {}
        self._volume: SparseVolume3D = {}
        self._global_core: Vector = ()
        self._channels = 0
        self._cycle = 0

    @property
    def cycle(self) -> int:
        return self._cycle

    @property
    def logical_loop_count(self) -> int:
        return self.config.logical_loop_count

    @property
    def logical_volume_cell_count(self) -> int:
        return self.config.logical_volume_cell_count

    @property
    def active_loop_count(self) -> int:
        return len(self._surface)

    @property
    def allocated_volume_cell_count(self) -> int:
        return len(self._volume)

    @property
    def channel_count(self) -> int:
        return self._channels

    @property
    def global_core(self) -> Vector:
        return self._global_core

    def snapshot_surface(self) -> SparseLoopField:
        return dict(self._surface)

    def snapshot_volume(self) -> SparseVolume3D:
        return dict(self._volume)

    def snapshot_error(self) -> SparseLoopField:
        return dict(self._error)

    def load(self, field: Mapping[LoopCoordinate, Sequence[float]]) -> SparseLoopField:
        parsed: SparseLoopField = {}
        channels: int | None = None
        for raw_coordinate, raw_vector in field.items():
            coordinate = self._validate_coordinate(raw_coordinate)
            vector = self._validate_vector(raw_vector)
            if channels is None:
                channels = len(vector)
            elif len(vector) != channels:
                raise ValueError("all loop vectors must have the same channel count")
            if self._active(vector):
                parsed[coordinate] = vector
            if len(parsed) > self.config.max_active_loops:
                raise RuntimeError("active-loop budget exceeded during load")

        self._surface = parsed
        self._channels = channels or 0
        zero = self._zero()
        self._velocity = {coordinate: zero for coordinate in parsed}
        self._memory = {coordinate: zero for coordinate in parsed}
        self._error = {coordinate: zero for coordinate in parsed}
        self._volume, self._global_core = self._fold_volume(parsed)
        self._cycle = 0
        return self.snapshot_surface()

    def run(
        self,
        cycles: int,
        *,
        validator: Validator | None = None,
    ) -> tuple[MultiparallelStepMetrics, ...]:
        if isinstance(cycles, bool) or not isinstance(cycles, int) or cycles <= 0:
            raise ValueError("cycles must be a positive integer")
        return tuple(self.step(validator=validator) for _ in range(cycles))

    def step(self, validator: Validator | None = None) -> MultiparallelStepMetrics:
        self._require_loaded()
        snapshot = dict(self._surface)
        support = self._support(snapshot)
        cycle = self._cycle + 1

        candidate: SparseLoopField = {}
        candidate_velocity: SparseLoopField = {}
        zero = self._zero()
        kinetic_energy = 0.0

        for coordinate in support:
            current = snapshot.get(coordinate, zero)
            laplacian = self._laplacian(snapshot, coordinate)
            memory = self._memory.get(coordinate, zero)
            previous_velocity = self._velocity.get(coordinate, zero)
            force = self._add(
                self._scale(laplacian, self.config.diffusion),
                self._scale(memory, self.config.memory_gain),
            )
            velocity = self._add(
                self._scale(previous_velocity, self.config.damping),
                force,
            )
            value = self._add(current, self._scale(velocity, self.config.dt))
            if self._active(value):
                candidate[coordinate] = value
                candidate_velocity[coordinate] = velocity
                kinetic_energy += 0.5 * self._dot(velocity, velocity)

        if len(candidate) > self.config.max_active_loops:
            self._cycle = cycle
            return self._metrics(
                cycle=cycle,
                support=support,
                before=snapshot,
                after=candidate,
                volume={},
                global_core=(),
                reconstruction={},
                kinetic_energy=kinetic_energy,
                committed=False,
                rejection_reason="active-loop budget exceeded",
            )

        volume, global_core = self._fold_volume(candidate)
        reconstruction = self._unfold_volume(volume, global_core, tuple(candidate))
        provisional = self._metrics(
            cycle=cycle,
            support=support,
            before=snapshot,
            after=candidate,
            volume=volume,
            global_core=global_core,
            reconstruction=reconstruction,
            kinetic_energy=kinetic_energy,
            committed=False,
        )

        if provisional.reconstruction_mse > self.config.max_reconstruction_mse:
            self._cycle = cycle
            return replace(
                provisional, rejection_reason="reconstruction error budget exceeded"
            )
        if validator is not None and not bool(validator(candidate, provisional)):
            self._cycle = cycle
            return replace(provisional, rejection_reason="validator rejected candidate")

        error = {
            coordinate: self._sub(candidate[coordinate], reconstruction[coordinate])
            for coordinate in candidate
        }
        new_memory: SparseLoopField = {}
        for coordinate, value in candidate.items():
            old = self._memory.get(coordinate, zero)
            err = error[coordinate]
            memory = self._add(
                self._scale(old, self.config.memory_decay),
                self._scale(err, 1.0 - self.config.memory_decay),
            )
            if self._active(memory):
                new_memory[coordinate] = memory

        self._surface = candidate
        self._velocity = candidate_velocity
        self._memory = new_memory
        self._error = error
        self._volume = volume
        self._global_core = global_core
        self._cycle = cycle

        return self._metrics(
            cycle=cycle,
            support=support,
            before=snapshot,
            after=candidate,
            volume=volume,
            global_core=global_core,
            reconstruction=reconstruction,
            kinetic_energy=kinetic_energy,
            committed=True,
        )

    def reconstruct(self) -> SparseLoopField:
        self._require_loaded()
        if not self._surface:
            return {}
        if not self._volume:
            self._volume, self._global_core = self._fold_volume(self._surface)
        return self._unfold_volume(self._volume, self._global_core, tuple(self._surface))

    def status(self) -> dict[str, object]:
        return {
            "cycle": self._cycle,
            "side": self.config.side,
            "depth": self.config.depth,
            "logical_loops": self.logical_loop_count,
            "logical_volume_cells": self.logical_volume_cell_count,
            "active_loops": self.active_loop_count,
            "allocated_volume_cells": self.allocated_volume_cell_count,
            "channels": self.channel_count,
            "global_core": self.global_core,
            "sparsity": 1.0 - self.active_loop_count / self.logical_loop_count,
        }

    def _fold_volume(self, surface: Mapping[LoopCoordinate, Vector]) -> tuple[SparseVolume3D, Vector]:
        volume: SparseVolume3D = {}
        deepest: list[Vector] = []
        for coordinate in sorted(surface, key=self._linear_address):
            value = tuple(surface[coordinate])
            volume[(coordinate[0], coordinate[1], 0)] = value
            z = value
            for depth in range(1, self.config.depth + 1):
                z = self._encode(z)
                volume[(coordinate[0], coordinate[1], depth)] = z
            deepest.append(z)
        core = self._mean(deepest) if deepest else self._zero()
        return volume, core

    def _unfold_volume(
        self,
        volume: Mapping[VolumeCoordinate, Vector],
        global_core: Vector,
        support: Sequence[LoopCoordinate],
    ) -> SparseLoopField:
        output: SparseLoopField = {}
        coupling = self.config.global_coupling
        for i, j in support:
            z = volume[(i, j, self.config.depth)]
            if global_core:
                z = self._add(self._scale(z, 1.0 - coupling), self._scale(global_core, coupling))
            for _ in range(self.config.depth):
                z = self._decode(z)
            output[(i, j)] = z
        return output

    def _encode(self, vector: Vector) -> Vector:
        scaled = self._scale(vector, self.config.encode_scale)
        q = self.config.quantization
        return tuple(round(value / q) * q for value in scaled)

    def _decode(self, vector: Vector) -> Vector:
        return self._scale(vector, 1.0 / self.config.encode_scale)

    def _support(self, field: Mapping[LoopCoordinate, Vector]) -> tuple[LoopCoordinate, ...]:
        support = set(field)
        if self.config.expand_halo:
            for coordinate in tuple(support):
                support.update(self._iter_neighbours(coordinate))
        if len(support) > self.config.max_active_loops:
            raise RuntimeError("support-closure budget exceeded")
        return tuple(sorted(support, key=self._linear_address))

    def _laplacian(self, field: Mapping[LoopCoordinate, Vector], coordinate: LoopCoordinate) -> Vector:
        center = field.get(coordinate, self._zero())
        neighbours = tuple(self._iter_neighbours(coordinate))
        if not neighbours:
            return self._zero()
        total = self._zero()
        for neighbour in neighbours:
            total = self._add(total, field.get(neighbour, self._zero()))
        return self._sub(total, self._scale(center, float(len(neighbours))))

    def _iter_neighbours(self, coordinate: LoopCoordinate):
        i, j = coordinate
        for di, dj in self._NEIGHBOURS:
            neighbour = (i + di, j + dj)
            if 0 <= neighbour[0] < self.config.side and 0 <= neighbour[1] < self.config.side:
                yield neighbour

    def _metrics(
        self,
        *,
        cycle: int,
        support: Sequence[LoopCoordinate],
        before: Mapping[LoopCoordinate, Vector],
        after: Mapping[LoopCoordinate, Vector],
        volume: Mapping[VolumeCoordinate, Vector],
        global_core: Vector,
        reconstruction: Mapping[LoopCoordinate, Vector],
        kinetic_energy: float,
        committed: bool,
        rejection_reason: str | None = None,
    ) -> MultiparallelStepMetrics:
        squared_error = 0.0
        max_abs_error = 0.0
        samples = 0
        for coordinate, value in after.items():
            reconstructed = reconstruction.get(coordinate, self._zero())
            for actual, predicted in zip(value, reconstructed):
                error = actual - predicted
                squared_error += error * error
                max_abs_error = max(max_abs_error, abs(error))
                samples += 1
        mse = squared_error / samples if samples else 0.0
        return MultiparallelStepMetrics(
            cycle=cycle,
            support_loops=len(support),
            active_loops_before=len(before),
            active_loops_after=len(after),
            allocated_volume_cells=len(volume),
            channel_count=self._channels,
            reconstruction_mse=mse,
            max_abs_error=max_abs_error,
            kinetic_energy=kinetic_energy,
            global_core=global_core,
            committed=committed,
            rejection_reason=rejection_reason,
        )

    def _validate_coordinate(self, coordinate: object) -> LoopCoordinate:
        if not isinstance(coordinate, (tuple, list)) or len(coordinate) != 2:
            raise TypeError("loop coordinate must contain exactly two integer axes")
        i, j = coordinate
        if isinstance(i, bool) or not isinstance(i, int):
            raise TypeError("loop coordinate axes must be integers")
        if isinstance(j, bool) or not isinstance(j, int):
            raise TypeError("loop coordinate axes must be integers")
        if not 0 <= i < self.config.side or not 0 <= j < self.config.side:
            raise ValueError("loop coordinate outside logical 1000x1000 lattice")
        return i, j

    def _validate_vector(self, vector: object) -> Vector:
        if isinstance(vector, (str, bytes)) or not isinstance(vector, Sequence):
            raise TypeError("loop value must be a finite numeric sequence")
        if not 1 <= len(vector) <= self.config.max_channels:
            raise ValueError("loop vector channel count outside configured bounds")
        values: list[float] = []
        for raw in vector:
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise TypeError("loop channels must be numeric")
            value = float(raw)
            if not math.isfinite(value):
                raise ValueError("loop channels must be finite")
            values.append(value)
        return tuple(values)

    def _require_loaded(self) -> None:
        if self._channels == 0:
            raise RuntimeError("load a non-empty field before executing the runtime")

    def _zero(self) -> Vector:
        return (0.0,) * self._channels

    def _active(self, vector: Vector) -> bool:
        return any(abs(value) > self.config.prune_epsilon for value in vector)

    def _linear_address(self, coordinate: LoopCoordinate) -> int:
        return coordinate[0] * self.config.side + coordinate[1]

    @staticmethod
    def _add(left: Vector, right: Vector) -> Vector:
        return tuple(a + b for a, b in zip(left, right))

    @staticmethod
    def _sub(left: Vector, right: Vector) -> Vector:
        return tuple(a - b for a, b in zip(left, right))

    @staticmethod
    def _scale(vector: Vector, scalar: float) -> Vector:
        return tuple(scalar * value for value in vector)

    @staticmethod
    def _dot(left: Vector, right: Vector) -> float:
        return sum(a * b for a, b in zip(left, right))

    def _mean(self, vectors: Sequence[Vector]) -> Vector:
        if not vectors:
            return self._zero()
        total = self._zero()
        for vector in vectors:
            total = self._add(total, vector)
        return self._scale(total, 1.0 / len(vectors))
