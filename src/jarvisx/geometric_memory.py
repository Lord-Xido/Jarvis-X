"""Deterministic 3D visual-memory reference runtime.

Pipeline:
    Volume3D -> geometric encoder -> latent lattice -> associative memory
             -> spatial residual refinement -> decoder -> bounded optimiser

This module is the executable semantic oracle for later tensor, GPU, and learned
implementations. Every candidate runs from an identical memory snapshot, and
only a strictly better measured objective may replace the baseline mechanics.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Callable, List, Optional, Sequence, Tuple

Shape3D = Tuple[int, int, int]
Vector = Tuple[float, ...]


def _prod(shape: Shape3D) -> int:
    return shape[0] * shape[1] * shape[2]


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _norm(xs: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in xs))


def _softmax(xs: Sequence[float]) -> List[float]:
    peak = max(xs)
    exps = [math.exp(x - peak) for x in xs]
    total = sum(exps)
    return [x / total for x in exps]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    scale = _norm(a) * _norm(b)
    if scale <= 1e-12:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / scale


@dataclass(frozen=True)
class Volume3D:
    shape: Shape3D
    values: Tuple[float, ...]

    def __post_init__(self) -> None:
        if any(axis <= 0 for axis in self.shape):
            raise ValueError("volume axes must be positive")
        if len(self.values) != _prod(self.shape):
            raise ValueError("volume data length does not match shape")
        if not all(math.isfinite(value) for value in self.values):
            raise ValueError("volume values must be finite")

    @classmethod
    def from_function(
        cls, shape: Shape3D, fn: Callable[[int, int, int], float]
    ) -> "Volume3D":
        d, h, w = shape
        values = tuple(
            float(fn(z, y, x))
            for z in range(d)
            for y in range(h)
            for x in range(w)
        )
        return cls(shape, values)

    def at(self, z: int, y: int, x: int) -> float:
        d, h, w = self.shape
        if not (0 <= z < d and 0 <= y < h and 0 <= x < w):
            raise IndexError("voxel coordinate outside volume")
        return self.values[x + w * (y + h * z)]

    def mse(self, other: "Volume3D") -> float:
        if self.shape != other.shape:
            raise ValueError("MSE requires equal shapes")
        return _mean([(a - b) ** 2 for a, b in zip(self.values, other.values)])

    def subtract(self, other: "Volume3D") -> "Volume3D":
        if self.shape != other.shape:
            raise ValueError("subtraction requires equal shapes")
        return Volume3D(
            self.shape,
            tuple(a - b for a, b in zip(self.values, other.values)),
        )


@dataclass(frozen=True)
class LatentField:
    shape: Shape3D
    channels: int
    values: Tuple[float, ...]

    def __post_init__(self) -> None:
        if self.channels <= 0:
            raise ValueError("latent channels must be positive")
        if len(self.values) != _prod(self.shape) * self.channels:
            raise ValueError("latent data length does not match geometry")

    def vectors(self) -> List[Vector]:
        return [
            self.values[i : i + self.channels]
            for i in range(0, len(self.values), self.channels)
        ]

    def vector(self, z: int, y: int, x: int) -> Vector:
        d, h, w = self.shape
        if not (0 <= z < d and 0 <= y < h and 0 <= x < w):
            raise IndexError("latent coordinate outside field")
        start = self.channels * (x + w * (y + h * z))
        return self.values[start : start + self.channels]

    @classmethod
    def build(
        cls, shape: Shape3D, vectors: Sequence[Sequence[float]]
    ) -> "LatentField":
        if not vectors:
            raise ValueError("latent field cannot be empty")
        channels = len(vectors[0])
        if len(vectors) != _prod(shape):
            raise ValueError("latent vector count does not match shape")
        if any(len(vector) != channels for vector in vectors):
            raise ValueError("latent vector widths do not match")
        return cls(
            shape,
            channels,
            tuple(value for vector in vectors for value in vector),
        )


@dataclass(frozen=True)
class GeometricConfig:
    latent_shape: Shape3D = (4, 4, 4)
    channels: int = 6
    memory_slots: int = 8
    refinement_steps: int = 4
    learning_rate: float = 0.18
    residual_gain: float = 0.35
    memory_gain: float = 0.12
    max_abs_latent: float = 1.5
    cost_weight: float = 1e-7
    max_candidate_steps: int = 8

    def validate(self) -> None:
        if any(axis <= 0 for axis in self.latent_shape):
            raise ValueError("latent axes must be positive")
        if self.channels < 2 or self.memory_slots <= 0:
            raise ValueError("invalid channel or memory dimensions")
        if not 1 <= self.refinement_steps <= self.max_candidate_steps:
            raise ValueError("refinement_steps outside declared bounds")
        if not 0.0 < self.learning_rate <= 1.0:
            raise ValueError("learning_rate outside (0, 1]")
        if not 0.0 <= self.residual_gain <= 1.0:
            raise ValueError("residual_gain outside [0, 1]")
        if not 0.0 <= self.memory_gain <= 1.0:
            raise ValueError("memory_gain outside [0, 1]")
        if self.max_abs_latent <= 0.0 or self.cost_weight < 0.0:
            raise ValueError("invalid projection or cost bound")


@dataclass(frozen=True)
class MemorySlot:
    key: Vector
    value: Vector
    usage: float = 0.0


class SpatialMemory:
    """Finite content-addressed memory with anti-monopoly usage pressure."""

    def __init__(self, count: int, channels: int) -> None:
        self.channels = channels
        self._slots = [
            MemorySlot(
                tuple(
                    0.05 * math.sin((i + 1) * (j + 1))
                    for j in range(channels)
                ),
                (0.0,) * channels,
            )
            for i in range(count)
        ]

    def clone(self) -> "SpatialMemory":
        clone = SpatialMemory(len(self._slots), self.channels)
        clone._slots = list(self._slots)
        return clone

    def snapshot(self) -> Tuple[MemorySlot, ...]:
        return tuple(self._slots)

    def _attention(self, query: Sequence[float]) -> List[float]:
        if len(query) != self.channels:
            raise ValueError("memory query width mismatch")
        scores = [
            4.0 * _cosine(query, slot.key) - 0.05 * slot.usage
            for slot in self._slots
        ]
        return _softmax(scores)

    def read(self, query: Sequence[float]) -> Vector:
        weights = self._attention(query)
        return tuple(
            sum(
                weight * slot.value[channel]
                for weight, slot in zip(weights, self._slots)
            )
            for channel in range(self.channels)
        )

    def write(
        self,
        key: Sequence[float],
        value: Sequence[float],
        rate: float,
    ) -> int:
        if len(value) != self.channels:
            raise ValueError("memory value width mismatch")
        weights = self._attention(key)
        index = max(
            range(len(weights)),
            key=lambda i: (weights[i], -self._slots[i].usage, -i),
        )
        old = self._slots[index]
        self._slots[index] = MemorySlot(
            tuple(
                (1.0 - rate) * previous + rate * current
                for previous, current in zip(old.key, key)
            ),
            tuple(
                (1.0 - rate) * previous + rate * current
                for previous, current in zip(old.value, value)
            ),
            old.usage + 1.0,
        )
        return index


@dataclass(frozen=True)
class RefinementTrace:
    """Auditable numerical telemetry, not a textual reasoning trace."""

    step: int
    reconstruction_loss: float
    latent_norm: float
    residual_norm: float
    recalled_norm: float
    memory_slot: int


@dataclass(frozen=True)
class CandidateMeasurement:
    config: GeometricConfig
    reconstruction_loss: float
    estimated_operations: int
    objective: float


@dataclass(frozen=True)
class PermeationResult:
    reconstruction: Volume3D
    latent: LatentField
    trace: Tuple[RefinementTrace, ...]
    selected: CandidateMeasurement
    baseline: CandidateMeasurement
    candidates: Tuple[CandidateMeasurement, ...]
    mechanics_changed: bool

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)

    def summary(self) -> dict:
        return {
            "shape": self.reconstruction.shape,
            "latent_shape": self.latent.shape,
            "channels": self.latent.channels,
            "refinement_steps": self.selected.config.refinement_steps,
            "loss": self.selected.reconstruction_loss,
            "operations": self.selected.estimated_operations,
            "objective": self.selected.objective,
            "mechanics_changed": self.mechanics_changed,
            "candidate_count": self.candidate_count,
            "candidate_objectives": [
                measurement.objective for measurement in self.candidates
            ],
            "trace": [item.__dict__ for item in self.trace],
        }


class GeometricCodec:
    def __init__(self, config: GeometricConfig) -> None:
        self.config = config

    @staticmethod
    def _bounds(source: int, latent: int, index: int) -> Tuple[int, int]:
        start = index * source // latent
        return start, max(start + 1, (index + 1) * source // latent)

    def _regions(self, volume: Volume3D):
        sd, sh, sw = volume.shape
        ld, lh, lw = self.config.latent_shape
        for lz in range(ld):
            z0, z1 = self._bounds(sd, ld, lz)
            for ly in range(lh):
                y0, y1 = self._bounds(sh, lh, ly)
                for lx in range(lw):
                    x0, x1 = self._bounds(sw, lw, lx)
                    samples = [
                        volume.at(z, y, x)
                        for z in range(z0, min(z1, sd))
                        for y in range(y0, min(y1, sh))
                        for x in range(x0, min(x1, sw))
                    ]
                    cz = min(sd - 1, (z0 + z1 - 1) // 2)
                    cy = min(sh - 1, (y0 + y1 - 1) // 2)
                    cx = min(sw - 1, (x0 + x1 - 1) // 2)
                    gradients = (
                        volume.at(min(sd - 1, cz + 1), cy, cx)
                        - volume.at(max(0, cz - 1), cy, cx),
                        volume.at(cz, min(sh - 1, cy + 1), cx)
                        - volume.at(cz, max(0, cy - 1), cx),
                        volume.at(cz, cy, min(sw - 1, cx + 1))
                        - volume.at(cz, cy, max(0, cx - 1)),
                    )
                    yield lz, ly, lx, samples, gradients

    def _expand_basis(self, basis: Sequence[float]) -> Vector:
        vector = [
            basis[channel]
            if channel < len(basis)
            else _mean(
                [
                    value * math.sin((channel + 1) * (index + 1))
                    for index, value in enumerate(basis)
                ]
            )
            for channel in range(self.config.channels)
        ]
        return tuple(
            _clamp(
                value,
                -self.config.max_abs_latent,
                self.config.max_abs_latent,
            )
            for value in vector
        )

    def encode(self, volume: Volume3D) -> LatentField:
        ld, lh, lw = self.config.latent_shape
        vectors: List[Vector] = []
        for lz, ly, lx, samples, gradients in self._regions(volume):
            mean = _mean(samples)
            variance = _mean([(value - mean) ** 2 for value in samples])
            radius = math.sqrt(
                ((2 * lz + 1) / ld - 1) ** 2
                + ((2 * ly + 1) / lh - 1) ** 2
                + ((2 * lx + 1) / lw - 1) ** 2
            )
            basis = [
                2.0 * mean - 1.0,
                math.sqrt(max(0.0, variance)),
                gradients[2],
                gradients[1],
                gradients[0],
                radius,
            ]
            vectors.append(self._expand_basis(basis))
        return LatentField.build(self.config.latent_shape, vectors)

    def encode_residual(self, residual: Volume3D) -> LatentField:
        """Project spatial reconstruction error into matching latent cells."""

        vectors: List[Vector] = []
        for _, _, _, samples, gradients in self._regions(residual):
            mean = _mean(samples)
            rms = math.sqrt(_mean([value * value for value in samples]))
            basis = [
                mean,
                rms,
                gradients[2],
                gradients[1],
                gradients[0],
                _mean([abs(value) for value in samples]),
            ]
            vectors.append(self._expand_basis(basis))
        return LatentField.build(self.config.latent_shape, vectors)

    def decode(self, latent: LatentField, shape: Shape3D) -> Volume3D:
        d, h, w = shape
        ld, lh, lw = latent.shape

        def sample(z: int, y: int, x: int) -> float:
            vector = latent.vector(
                min(ld - 1, z * ld // d),
                min(lh - 1, y * lh // h),
                min(lw - 1, x * lw // w),
            )
            detail = 0.025 * sum(vector[2:5]) if len(vector) > 2 else 0.0
            return _clamp(
                0.5 * (vector[0] + 1.0) + detail,
                0.0,
                1.0,
            )

        return Volume3D.from_function(shape, sample)


class VisualMemoryANN:
    """Transactional 3D visual memory with bounded inward optimisation."""

    def __init__(self, config: Optional[GeometricConfig] = None) -> None:
        self.config = config or GeometricConfig()
        self.config.validate()
        self.memory = SpatialMemory(
            self.config.memory_slots,
            self.config.channels,
        )
        self.journal: List[Tuple[CandidateMeasurement, ...]] = []

    @staticmethod
    def _global(latent: LatentField) -> Vector:
        vectors = latent.vectors()
        return tuple(
            _mean([vector[channel] for vector in vectors])
            for channel in range(latent.channels)
        )

    @staticmethod
    def _operations(volume: Volume3D, config: GeometricConfig) -> int:
        voxels = _prod(volume.shape)
        cells = _prod(config.latent_shape)
        encode_decode = voxels * (2 * config.channels + 3)
        refine = (
            config.refinement_steps
            * cells
            * config.channels
            * (config.memory_slots + 10)
        )
        return encode_decode + refine

    def _run(
        self,
        observed: Volume3D,
        target: Volume3D,
        config: GeometricConfig,
        memory: SpatialMemory,
    ):
        codec = GeometricCodec(config)
        latent = codec.encode(observed)
        trace: List[RefinementTrace] = []

        for step in range(config.refinement_steps):
            query = self._global(latent)
            recalled = memory.read(query)
            reconstruction = codec.decode(latent, target.shape)
            residual_volume = target.subtract(reconstruction)
            residual_latent = codec.encode_residual(residual_volume)
            residual_summary = self._global(residual_latent)
            slot = memory.write(
                query,
                residual_summary,
                config.learning_rate,
            )

            vectors = []
            for vector, local_residual in zip(
                latent.vectors(),
                residual_latent.vectors(),
            ):
                updated = tuple(
                    _clamp(
                        value
                        + config.learning_rate
                        * (
                            config.residual_gain * local_residual[channel]
                            + config.memory_gain
                            * (recalled[channel] - value)
                        ),
                        -config.max_abs_latent,
                        config.max_abs_latent,
                    )
                    for channel, value in enumerate(vector)
                )
                vectors.append(updated)

            latent = LatentField.build(latent.shape, vectors)
            post = codec.decode(latent, target.shape)
            trace.append(
                RefinementTrace(
                    step=step,
                    reconstruction_loss=post.mse(target),
                    latent_norm=_norm(latent.values)
                    / math.sqrt(len(latent.values)),
                    residual_norm=_norm(residual_volume.values)
                    / math.sqrt(len(residual_volume.values)),
                    recalled_norm=_norm(recalled),
                    memory_slot=slot,
                )
            )

        reconstruction = codec.decode(latent, target.shape)
        loss = reconstruction.mse(target)
        operations = self._operations(observed, config)
        measurement = CandidateMeasurement(
            config,
            loss,
            operations,
            loss + config.cost_weight * operations,
        )
        return reconstruction, latent, tuple(trace), measurement, memory

    def _candidates(self) -> Tuple[GeometricConfig, ...]:
        current = self.config
        proposals = [
            current,
            replace(
                current,
                learning_rate=max(0.01, current.learning_rate * 0.75),
            ),
            replace(
                current,
                learning_rate=min(1.0, current.learning_rate * 1.25),
            ),
            replace(
                current,
                residual_gain=min(1.0, current.residual_gain + 0.1),
            ),
            replace(
                current,
                memory_gain=max(0.0, current.memory_gain * 0.75),
            ),
            replace(
                current,
                memory_gain=min(1.0, current.memory_gain * 1.25),
            ),
        ]
        if current.refinement_steps < current.max_candidate_steps:
            proposals.append(
                replace(
                    current,
                    refinement_steps=current.refinement_steps + 1,
                )
            )

        unique: List[GeometricConfig] = []
        for proposal in proposals:
            proposal.validate()
            if proposal not in unique:
                unique.append(proposal)
        return tuple(unique)

    def permeate(
        self,
        observed: Volume3D,
        target: Optional[Volume3D] = None,
        auto_optimize: bool = True,
    ) -> PermeationResult:
        target = target or observed
        if observed.shape != target.shape:
            raise ValueError("observed and target volumes must have equal shapes")

        candidates = self._candidates() if auto_optimize else (self.config,)
        snapshot = self.memory.clone()
        runs = [
            self._run(
                observed,
                target,
                candidate,
                snapshot.clone(),
            )
            for candidate in candidates
        ]
        baseline = runs[0]
        index = min(
            range(len(runs)),
            key=lambda candidate_index: (
                runs[candidate_index][3].objective,
                candidate_index,
            ),
        )
        selected = runs[index]
        changed = (
            index != 0
            and selected[3].objective < baseline[3].objective
        )

        self.config = selected[3].config
        self.memory = selected[4]
        measurements = tuple(run[3] for run in runs)
        self.journal.append(measurements)

        return PermeationResult(
            reconstruction=selected[0],
            latent=selected[1],
            trace=selected[2],
            selected=selected[3],
            baseline=baseline[3],
            candidates=measurements,
            mechanics_changed=changed,
        )


def make_demo_volume(size: int = 12) -> Volume3D:
    if size < 4:
        raise ValueError("demo size must be at least 4")
    centre = (size - 1) / 2.0
    scale = max(1.0, (size - 1) / 2.0)

    def field(z: int, y: int, x: int) -> float:
        nx = (x - centre) / scale
        ny = (y - centre) / scale
        nz = (z - centre) / scale
        radius = math.sqrt(nx * nx + ny * ny + nz * nz)
        shell = math.exp(-10.0 * (radius - 0.58) ** 2)
        wave = 0.12 * (
            math.sin(5 * nx)
            * math.cos(4 * ny)
            * math.sin(3 * nz)
            + 1.0
        )
        return _clamp(0.82 * shell + wave, 0.0, 1.0)

    return Volume3D.from_function((size, size, size), field)
