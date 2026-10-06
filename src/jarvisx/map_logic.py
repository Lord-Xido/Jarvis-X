"""Map Logic: deterministic toroidal mapping and inward refinement.

This module turns named numeric vectors into a bounded 3D toroidal execution
field.  It is an operational software model, not a claim about physical
spacetime.  The closed loop is:

    input -> canonical encode -> T^3 map -> inward update
          -> residual memory -> constitutional gate -> receipt

The update reuses the stability contract from :mod:`jarvisx.permeation` while
using periodic neighbours so the execution surface has no privileged edge.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping, Sequence

from jarvisx.permeation import (
    PermeationConstitution,
    constitutional_gate,
    distance_to_core,
    saturation_fraction,
)

Coordinate3D = tuple[int, int, int]
Vector = tuple[float, ...]
NamedVectors = Mapping[str, Sequence[float]]


class MapLogicError(ValueError):
    """Raised when Map Logic inputs violate the execution contract."""


@dataclass(frozen=True, slots=True)
class MapLogicConfig:
    """Bounded geometry, feedback, and convergence settings."""

    side: int = 8
    fold_steps: int = 8
    memory_decay: float = 0.88
    residual_gain: float = 0.35
    saturation_tolerance: float = 1.0e-3
    constitution: PermeationConstitution = PermeationConstitution()

    def __post_init__(self) -> None:
        if self.side < 2:
            raise MapLogicError("side must be at least 2")
        if self.fold_steps < 0:
            raise MapLogicError("fold_steps must be non-negative")
        if not (0.0 <= self.memory_decay < 1.0):
            raise MapLogicError("memory_decay must be in [0, 1)")
        if not (0.0 <= self.residual_gain <= 1.0):
            raise MapLogicError("residual_gain must be in [0, 1]")
        if (
            not math.isfinite(self.saturation_tolerance)
            or self.saturation_tolerance < 0.0
        ):
            raise MapLogicError("saturation_tolerance must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class MapLogicReceipt:
    """Deterministic evidence emitted by one closed-loop execution."""

    mapped_coordinates: tuple[tuple[str, Coordinate3D], ...]
    core_signature: Vector
    objective_history: tuple[float, ...]
    accepted_steps: int
    rejected_steps: int
    fixed_point_delta: float
    saturation_fraction: float
    memory_rms: float
    theoretical_spectral_ceiling: float
    semantic_hbar: float
    verified: bool
    field_digest: str

    def to_dict(self) -> dict[str, object]:
        return {
            "mapped_coordinates": [
                {"name": name, "coordinate": list(coordinate)}
                for name, coordinate in self.mapped_coordinates
            ],
            "core_signature": list(self.core_signature),
            "objective_history": list(self.objective_history),
            "accepted_steps": self.accepted_steps,
            "rejected_steps": self.rejected_steps,
            "fixed_point_delta": self.fixed_point_delta,
            "saturation_fraction": self.saturation_fraction,
            "memory_rms": self.memory_rms,
            "theoretical_spectral_ceiling": self.theoretical_spectral_ceiling,
            "semantic_hbar": self.semantic_hbar,
            "verified": self.verified,
            "field_digest": self.field_digest,
        }


def _finite_vector(value: Sequence[float], *, name: str) -> Vector:
    if len(value) == 0:
        raise MapLogicError(f"{name} must contain at least one value")
    encoded: list[float] = []
    for index, component in enumerate(value):
        if isinstance(component, bool) or not isinstance(component, (int, float)):
            raise MapLogicError(f"{name}[{index}] must be numeric")
        scalar = float(component)
        if not math.isfinite(scalar):
            raise MapLogicError(f"{name}[{index}] must be finite")
        encoded.append(math.tanh(scalar))
    return tuple(encoded)


def canonical_encode(samples: NamedVectors) -> dict[str, Vector]:
    """Validate and squash input vectors into a bounded canonical representation."""

    if not samples:
        raise MapLogicError("samples must not be empty")
    encoded = {
        str(name): _finite_vector(vector, name=f"samples[{name!r}]")
        for name, vector in samples.items()
    }
    widths = {len(vector) for vector in encoded.values()}
    if len(widths) != 1:
        raise MapLogicError("all vectors must have the same width")
    if len(encoded) != len(samples):
        raise MapLogicError("sample names must be unique after string normalization")
    return dict(sorted(encoded.items()))


def map_coordinate(name: str, side: int) -> Coordinate3D:
    """Map one stable name to a coordinate on Z_side x Z_side x Z_side."""

    if side < 2:
        raise MapLogicError("side must be at least 2")
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return (
        int.from_bytes(digest[0:8], "big") % side,
        int.from_bytes(digest[8:16], "big") % side,
        int.from_bytes(digest[16:24], "big") % side,
    )


def _zero_vector(width: int) -> Vector:
    return tuple(0.0 for _ in range(width))


def _mean_vector(vectors: Sequence[Vector]) -> Vector:
    if not vectors:
        raise MapLogicError("cannot compute the mean of an empty vector set")
    width = len(vectors[0])
    return tuple(
        sum(vector[channel] for vector in vectors) / len(vectors)
        for channel in range(width)
    )


def build_toroidal_field(
    encoded: Mapping[str, Vector],
    *,
    side: int,
) -> tuple[dict[Coordinate3D, Vector], tuple[tuple[str, Coordinate3D], ...], Vector]:
    """Materialize a bounded dense T^3 field and place encoded samples into it."""

    if not encoded:
        raise MapLogicError("encoded samples must not be empty")
    width = len(next(iter(encoded.values())))
    zero = _zero_vector(width)
    field: dict[Coordinate3D, Vector] = {
        (x, y, z): zero
        for x in range(side)
        for y in range(side)
        for z in range(side)
    }

    mapped: list[tuple[str, Coordinate3D]] = []
    buckets: dict[Coordinate3D, list[Vector]] = {}
    for name, vector in encoded.items():
        coordinate = map_coordinate(name, side)
        mapped.append((name, coordinate))
        buckets.setdefault(coordinate, []).append(vector)

    for coordinate, vectors in buckets.items():
        field[coordinate] = _mean_vector(vectors)

    core = _mean_vector(list(encoded.values()))
    return field, tuple(mapped), core


def toroidal_step(
    field: Mapping[Coordinate3D, Sequence[float]],
    core_signature: Sequence[float],
    *,
    side: int,
    constitution: PermeationConstitution,
) -> dict[Coordinate3D, Vector]:
    """Apply one periodic six-neighbour diffusion/relaxation update."""

    width = len(core_signature)
    if width == 0:
        raise MapLogicError("core_signature must not be empty")
    if side < 2:
        raise MapLogicError("side must be at least 2")

    rho = constitution.target_spectral_radius
    alpha_dt = constitution.alpha * constitution.dt
    source_weight = 1.0 - rho
    output: dict[Coordinate3D, Vector] = {}

    for coordinate, raw_center in field.items():
        if len(raw_center) != width:
            raise MapLogicError("field vectors must match core_signature width")
        x, y, z = coordinate
        center = tuple(float(component) for component in raw_center)
        neighbours = (
            ((x + 1) % side, y, z),
            ((x - 1) % side, y, z),
            (x, (y + 1) % side, z),
            (x, (y - 1) % side, z),
            (x, y, (z + 1) % side),
            (x, y, (z - 1) % side),
        )
        laplacian = [0.0] * width
        for neighbour_coordinate in neighbours:
            neighbour = field[neighbour_coordinate]
            for channel in range(width):
                laplacian[channel] += float(neighbour[channel]) - center[channel]

        output[coordinate] = tuple(
            rho * center[channel]
            + alpha_dt * laplacian[channel]
            + source_weight * float(core_signature[channel])
            for channel in range(width)
        )
    return output


def _feedback_correct(
    candidate: Mapping[Coordinate3D, Vector],
    memory: Mapping[Coordinate3D, Vector],
    core_signature: Vector,
    config: MapLogicConfig,
) -> tuple[dict[Coordinate3D, Vector], dict[Coordinate3D, Vector]]:
    width = len(core_signature)
    next_memory: dict[Coordinate3D, Vector] = {}
    corrected: dict[Coordinate3D, Vector] = {}

    for coordinate, value in candidate.items():
        previous = memory[coordinate]
        residual = tuple(
            core_signature[channel] - value[channel] for channel in range(width)
        )
        omega = tuple(
            config.memory_decay * previous[channel]
            + (1.0 - config.memory_decay) * residual[channel]
            for channel in range(width)
        )
        next_memory[coordinate] = omega
        corrected[coordinate] = tuple(
            value[channel] + config.residual_gain * omega[channel]
            for channel in range(width)
        )
    return corrected, next_memory


def _field_delta(
    left: Mapping[Coordinate3D, Vector],
    right: Mapping[Coordinate3D, Vector],
) -> float:
    if left.keys() != right.keys():
        raise MapLogicError("field topology changed during fixed-point iteration")
    total = 0.0
    count = 0
    for coordinate in left:
        a = left[coordinate]
        b = right[coordinate]
        if len(a) != len(b):
            raise MapLogicError("field width changed during fixed-point iteration")
        for av, bv in zip(a, b):
            total += (av - bv) ** 2
            count += 1
    return math.sqrt(total / count) if count else 0.0


def _field_rms(field: Mapping[Coordinate3D, Vector]) -> float:
    total = 0.0
    count = 0
    for vector in field.values():
        for value in vector:
            total += value * value
            count += 1
    return math.sqrt(total / count) if count else 0.0


def _field_digest(field: Mapping[Coordinate3D, Vector]) -> str:
    payload = [
        [coordinate[0], coordinate[1], coordinate[2], *field[coordinate]]
        for coordinate in sorted(field)
    ]
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class MapLogicEngine:
    """Execute the bounded Map Logic state transition to a verified receipt."""

    def __init__(self, config: MapLogicConfig | None = None) -> None:
        self.config = config or MapLogicConfig()

    def execute(self, samples: NamedVectors) -> MapLogicReceipt:
        encoded = canonical_encode(samples)
        field, mapped, core = build_toroidal_field(encoded, side=self.config.side)
        memory = {
            coordinate: _zero_vector(len(core))
            for coordinate in field
        }

        objective = distance_to_core(field, core)
        history = [objective]
        accepted = 0
        rejected = 0
        fixed_point_delta = 0.0

        for _ in range(self.config.fold_steps):
            raw_candidate = toroidal_step(
                field,
                core,
                side=self.config.side,
                constitution=self.config.constitution,
            )
            candidate, next_memory = _feedback_correct(
                raw_candidate,
                memory,
                core,
                self.config,
            )
            candidate_objective = distance_to_core(candidate, core)
            if not constitutional_gate(
                objective,
                candidate_objective,
                measured_spectral_radius=self.config.constitution.theoretical_spectral_ceiling,
                constitution=self.config.constitution,
            ):
                rejected += 1
                break

            fixed_point_delta = _field_delta(field, candidate)
            field = candidate
            memory = next_memory
            objective = candidate_objective
            history.append(objective)
            accepted += 1

        monotone = all(
            later <= earlier + self.config.constitution.non_regression_tolerance
            for earlier, later in zip(history, history[1:])
        )
        verified = (
            monotone
            and self.config.constitution.theoretical_spectral_ceiling < 1.0
            and self.config.constitution.semantic_hbar > 0.0
        )

        return MapLogicReceipt(
            mapped_coordinates=mapped,
            core_signature=core,
            objective_history=tuple(history),
            accepted_steps=accepted,
            rejected_steps=rejected,
            fixed_point_delta=fixed_point_delta,
            saturation_fraction=saturation_fraction(
                field,
                core,
                tolerance=self.config.saturation_tolerance,
            ),
            memory_rms=_field_rms(memory),
            theoretical_spectral_ceiling=self.config.constitution.theoretical_spectral_ceiling,
            semantic_hbar=self.config.constitution.semantic_hbar,
            verified=verified,
            field_digest=_field_digest(field),
        )


def demo_receipt() -> MapLogicReceipt:
    """Run a deterministic three-sample demonstration."""

    engine = MapLogicEngine()
    return engine.execute(
        {
            "encode": (1.0, 0.0, 0.25),
            "transform": (0.25, 1.0, -0.25),
            "verify": (-0.5, 0.25, 1.0),
        }
    )


def main() -> int:
    """CLI entry point for a deterministic operational smoke run."""

    print(json.dumps(demo_receipt().to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "Coordinate3D",
    "MapLogicConfig",
    "MapLogicEngine",
    "MapLogicError",
    "MapLogicReceipt",
    "NamedVectors",
    "Vector",
    "build_toroidal_field",
    "canonical_encode",
    "demo_receipt",
    "map_coordinate",
    "toroidal_step",
]
