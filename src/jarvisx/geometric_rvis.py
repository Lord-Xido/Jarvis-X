"""Transactional 3D geometric auto-encoding/decoding feedback runtime.

The runtime converts public shell values into a signed-3-bit voxel lattice,
builds a multiresolution geometric pyramid, evaluates independent candidate
lanes in parallel, validates the candidates through Lambda constraints, and
commits or rolls back atomically. Committed decoded output may be fed into the
next cycle to form an inward recursive loop.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

Number = Union[int, float]
Coordinate = Tuple[int, int, int]
Q3_MIN = -4
Q3_MAX = 3


def _clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, int(value)))


def _round_half_away_from_zero(value: float) -> int:
    if value >= 0:
        return int(math.floor(value + 0.5))
    return int(math.ceil(value - 0.5))


def _div_round_nearest(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    if numerator >= 0:
        return (numerator + denominator // 2) // denominator
    return -((-numerator + denominator // 2) // denominator)


def quantize_q3(value: Number, scale: float = 1.0) -> int:
    """Quantize a finite scalar into the signed three-bit domain."""
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError("geometric input values must be finite")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be positive and finite")
    return _clamp(_round_half_away_from_zero(scalar * scale), Q3_MIN, Q3_MAX)


def coordinate_to_index(coord: Coordinate, shape: Coordinate) -> int:
    """Map (x, y, z) to a row-major linear address."""
    x, y, z = coord
    width, height, depth = shape
    if not (0 <= x < width and 0 <= y < height and 0 <= z < depth):
        raise ValueError("coordinate outside lattice bounds")
    return x + width * (y + height * z)


def index_to_coordinate(index: int, shape: Coordinate) -> Coordinate:
    """Inverse of :func:`coordinate_to_index`."""
    width, height, depth = shape
    volume = width * height * depth
    if index < 0 or index >= volume:
        raise ValueError("index outside lattice bounds")
    x = index % width
    yz = index // width
    y = yz % height
    z = yz // height
    return x, y, z


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


@dataclass(frozen=True)
class GeometricConfig:
    shape: Coordinate = (4, 4, 4)
    input_scale: float = 1.0
    max_input_values: Optional[int] = None
    parallel_workers: int = 4
    feedback_cycles: int = 4
    retention_numerator: int = 7
    retention_denominator: int = 8
    learning_numerator: int = 1
    learning_denominator: int = 2
    omega_limit: int = 127
    max_reconstruction_l1: Optional[int] = None
    max_active_voxels: Optional[int] = None
    journal_limit: int = 128

    def validate(self) -> None:
        if any(not _is_power_of_two(axis) for axis in self.shape):
            raise ValueError("shape axes must be positive powers of two")
        if not math.isfinite(self.input_scale) or self.input_scale <= 0:
            raise ValueError("input_scale must be positive and finite")
        if self.parallel_workers < 1:
            raise ValueError("parallel_workers must be positive")
        if self.feedback_cycles < 1:
            raise ValueError("feedback_cycles must be positive")
        if self.retention_denominator <= 0 or self.learning_denominator <= 0:
            raise ValueError("update denominators must be positive")
        if not 0 <= self.retention_numerator <= self.retention_denominator:
            raise ValueError("retention ratio must be in [0, 1]")
        if self.learning_numerator < 0:
            raise ValueError("learning numerator must be non-negative")
        if self.omega_limit < 1:
            raise ValueError("omega_limit must be positive")
        if self.journal_limit < 1:
            raise ValueError("journal_limit must be positive")
        volume = self.shape[0] * self.shape[1] * self.shape[2]
        if self.max_input_values is not None and not 1 <= self.max_input_values <= volume:
            raise ValueError("max_input_values must fit inside the lattice")

    @property
    def volume(self) -> int:
        return self.shape[0] * self.shape[1] * self.shape[2]

    @property
    def input_limit(self) -> int:
        return self.max_input_values or self.volume


@dataclass(frozen=True)
class GeometricLevel:
    shape: Coordinate
    values: Tuple[int, ...]


@dataclass(frozen=True)
class LaneResult:
    name: str
    evolved: Tuple[int, ...]
    hierarchy: Tuple[GeometricLevel, ...]
    decoded: Tuple[int, ...]
    reconstruction_l1: int
    active_voxels: int
    valid: bool
    reason: str


@dataclass(frozen=True)
class GeometricState:
    cycle: int = 0
    input_length: int = 0
    encoded: Tuple[int, ...] = field(default_factory=tuple)
    evolved: Tuple[int, ...] = field(default_factory=tuple)
    hierarchy: Tuple[GeometricLevel, ...] = field(default_factory=tuple)
    decoded: Tuple[int, ...] = field(default_factory=tuple)
    omega: Tuple[int, ...] = field(default_factory=tuple)
    selected_lane: str = "GENESIS"
    state_hash: str = "GENESIS"


@dataclass(frozen=True)
class GeometricCycleResult:
    cycle: int
    committed: bool
    reason: str
    selected_lane: Optional[str]
    encoded: Tuple[int, ...]
    evolved: Tuple[int, ...]
    hierarchy: Tuple[GeometricLevel, ...]
    decoded: Tuple[int, ...]
    output: Tuple[int, ...]
    omega_before: Tuple[int, ...]
    omega_after: Tuple[int, ...]
    lanes: Tuple[LaneResult, ...]
    metrics: Dict[str, float]
    events: Tuple[Dict[str, object], ...]
    previous_hash: str
    candidate_hash: str
    state_hash: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class GeometricFeedbackRuntime:
    """3D geometric transactional runtime with deterministic parallel lanes."""

    LANE_ORDER = ("identity", "diffusion", "memory", "hybrid")

    def __init__(self, config: Optional[GeometricConfig] = None) -> None:
        self.config = config or GeometricConfig()
        self.config.validate()
        self.state = GeometricState()
        self.journal: List[GeometricCycleResult] = []

    def encode(self, values: Union[Number, Sequence[Number]]) -> Tuple[Tuple[int, ...], int]:
        if isinstance(values, (int, float)):
            source: Iterable[Number] = (values,)
        else:
            source = values
        materialized = tuple(source)
        if not materialized:
            raise ValueError("at least one geometric input value is required")
        if len(materialized) > self.config.input_limit:
            raise ValueError("input exceeds configured geometric lattice capacity")
        encoded = [quantize_q3(value, self.config.input_scale) for value in materialized]
        encoded.extend([0] * (self.config.volume - len(encoded)))
        return tuple(encoded), len(materialized)

    def _pool_level(self, level: GeometricLevel) -> GeometricLevel:
        width, height, depth = level.shape
        parent_shape = (width // 2, height // 2, depth // 2)
        parent: List[int] = []
        for pz in range(parent_shape[2]):
            for py in range(parent_shape[1]):
                for px in range(parent_shape[0]):
                    group: List[int] = []
                    for dz in (0, 1):
                        for dy in (0, 1):
                            for dx in (0, 1):
                                child = (2 * px + dx, 2 * py + dy, 2 * pz + dz)
                                group.append(level.values[coordinate_to_index(child, level.shape)])
                    parent.append(_clamp(_div_round_nearest(sum(group), 8), Q3_MIN, Q3_MAX))
        return GeometricLevel(parent_shape, tuple(parent))

    def condense(self, values: Tuple[int, ...]) -> Tuple[GeometricLevel, ...]:
        levels: List[GeometricLevel] = [GeometricLevel(self.config.shape, values)]
        while levels[-1].shape != (1, 1, 1):
            levels.append(self._pool_level(levels[-1]))
        return tuple(levels)

    def decode(self, hierarchy: Tuple[GeometricLevel, ...]) -> Tuple[int, ...]:
        reconstructed = hierarchy[-1]
        for target in reversed(hierarchy[:-1]):
            values: List[int] = []
            for z in range(target.shape[2]):
                for y in range(target.shape[1]):
                    for x in range(target.shape[0]):
                        parent = (x // 2, y // 2, z // 2)
                        values.append(
                            reconstructed.values[coordinate_to_index(parent, reconstructed.shape)]
                        )
            reconstructed = GeometricLevel(target.shape, tuple(values))
        return reconstructed.values

    def _neighbor_mean(self, values: Tuple[int, ...], coord: Coordinate) -> int:
        x, y, z = coord
        width, height, depth = self.config.shape
        neighbors: List[int] = []
        for dx, dy, dz in (
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ):
            nx, ny, nz = x + dx, y + dy, z + dz
            if 0 <= nx < width and 0 <= ny < height and 0 <= nz < depth:
                neighbors.append(values[coordinate_to_index((nx, ny, nz), self.config.shape)])
        if not neighbors:
            return values[coordinate_to_index(coord, self.config.shape)]
        return _div_round_nearest(sum(neighbors), len(neighbors))

    def _evolve_lane(self, name: str, encoded: Tuple[int, ...]) -> Tuple[int, ...]:
        evolved: List[int] = []
        for index, value in enumerate(encoded):
            coord = index_to_coordinate(index, self.config.shape)
            neighbor = self._neighbor_mean(encoded, coord)
            memory = self.state.omega[index] if index < len(self.state.omega) else 0
            correction = _div_round_nearest(memory, 4)
            if name == "identity":
                candidate = value
            elif name == "diffusion":
                candidate = _div_round_nearest(3 * value + neighbor, 4)
            elif name == "memory":
                candidate = value + correction
            elif name == "hybrid":
                candidate = _div_round_nearest(3 * value + neighbor + correction, 5)
            else:
                raise ValueError("unknown geometric lane")
            evolved.append(_clamp(candidate, Q3_MIN, Q3_MAX))
        return tuple(evolved)

    def _validate_lane(
        self,
        decoded: Tuple[int, ...],
        encoded: Tuple[int, ...],
    ) -> Tuple[bool, str, int, int]:
        reconstruction_l1 = sum(abs(a - b) for a, b in zip(encoded, decoded))
        active_voxels = sum(1 for value in decoded if value != 0)
        if (
            self.config.max_reconstruction_l1 is not None
            and reconstruction_l1 > self.config.max_reconstruction_l1
        ):
            return False, "reconstruction budget exceeded", reconstruction_l1, active_voxels
        if self.config.max_active_voxels is not None and active_voxels > self.config.max_active_voxels:
            return False, "active-voxel budget exceeded", reconstruction_l1, active_voxels
        return True, "admissible", reconstruction_l1, active_voxels

    def _run_lane(self, name: str, encoded: Tuple[int, ...]) -> LaneResult:
        evolved = self._evolve_lane(name, encoded)
        hierarchy = self.condense(evolved)
        decoded = self.decode(hierarchy)
        valid, reason, error, active = self._validate_lane(decoded, encoded)
        return LaneResult(name, evolved, hierarchy, decoded, error, active, valid, reason)

    def run_parallel_lanes(self, encoded: Tuple[int, ...]) -> Tuple[LaneResult, ...]:
        workers = min(self.config.parallel_workers, len(self.LANE_ORDER))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                name: executor.submit(self._run_lane, name, encoded) for name in self.LANE_ORDER
            }
            results = {name: future.result() for name, future in futures.items()}
        return tuple(results[name] for name in self.LANE_ORDER)

    def _update_omega(
        self,
        encoded: Tuple[int, ...],
        decoded: Tuple[int, ...],
    ) -> Tuple[int, ...]:
        updated: List[int] = []
        for index, (actual, reconstructed) in enumerate(zip(encoded, decoded)):
            old = self.state.omega[index] if index < len(self.state.omega) else 0
            residual = actual - reconstructed
            retained = _div_round_nearest(
                old * self.config.retention_numerator,
                self.config.retention_denominator,
            )
            learned = _div_round_nearest(
                residual * self.config.learning_numerator,
                self.config.learning_denominator,
            )
            updated.append(
                _clamp(retained + learned, -self.config.omega_limit, self.config.omega_limit)
            )
        return tuple(updated)

    def _events(
        self,
        cycle: int,
        input_length: int,
        lanes: Tuple[LaneResult, ...],
        selected: Optional[LaneResult],
        committed: bool,
        reason: str,
    ) -> Tuple[Dict[str, object], ...]:
        events: List[Dict[str, object]] = [
            {
                "seq": 0,
                "cycle": cycle,
                "phase": "GEOM_ENCODE",
                "summary": "Mapped public shell values into a bounded signed-3-bit 3D lattice.",
                "shape": self.config.shape,
                "input_length": input_length,
                "committed": False,
            }
        ]
        for lane in lanes:
            events.append(
                {
                    "seq": len(events),
                    "cycle": cycle,
                    "phase": "MULTIPARALLEL_LANE",
                    "lane": lane.name,
                    "summary": "Encoded, evolved, condensed, and decoded one independent geometric candidate.",
                    "reconstruction_l1": lane.reconstruction_l1,
                    "valid": lane.valid,
                    "reason": lane.reason,
                    "committed": False,
                }
            )
        events.append(
            {
                "seq": len(events),
                "cycle": cycle,
                "phase": "LAMBDA_PROJECT",
                "lane": selected.name if selected else None,
                "summary": reason,
                "passed": committed,
                "committed": False,
            }
        )
        events.append(
            {
                "seq": len(events),
                "cycle": cycle,
                "phase": "COMMIT" if committed else "ROLLBACK",
                "lane": selected.name if selected else None,
                "summary": "Committed the best admissible geometric branch."
                if committed
                else "Restored the previously committed geometric state.",
                "committed": committed,
            }
        )
        return tuple(events)

    def _hash_candidate(self, payload: Dict[str, object]) -> str:
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def step(self, values: Union[Number, Sequence[Number]]) -> GeometricCycleResult:
        previous_hash = self.state.state_hash
        omega_before = self.state.omega
        encoded, input_length = self.encode(values)
        lanes = self.run_parallel_lanes(encoded)
        admissible = tuple(lane for lane in lanes if lane.valid)
        selected = (
            min(
                admissible,
                key=lambda lane: (
                    lane.reconstruction_l1,
                    self.LANE_ORDER.index(lane.name),
                ),
            )
            if admissible
            else None
        )
        committed = selected is not None
        reason = (
            "selected minimum-error admissible lane"
            if committed
            else "no lane passed Lambda projection"
        )
        cycle = self.state.cycle + 1

        if selected is None:
            evolved: Tuple[int, ...] = tuple()
            hierarchy: Tuple[GeometricLevel, ...] = tuple()
            decoded: Tuple[int, ...] = tuple()
            omega_after = self.state.omega
        else:
            evolved = selected.evolved
            hierarchy = selected.hierarchy
            decoded = selected.decoded
            omega_after = self._update_omega(encoded, decoded)

        candidate_payload: Dict[str, object] = {
            "config": asdict(self.config),
            "cycle": cycle,
            "input_length": input_length,
            "encoded": encoded,
            "selected_lane": selected.name if selected else None,
            "evolved": evolved,
            "hierarchy": [asdict(level) for level in hierarchy],
            "decoded": decoded,
            "omega": omega_after,
            "previous_hash": previous_hash,
        }
        candidate_hash = self._hash_candidate(candidate_payload)
        if committed and selected is not None:
            self.state = GeometricState(
                cycle=cycle,
                input_length=input_length,
                encoded=encoded,
                evolved=evolved,
                hierarchy=hierarchy,
                decoded=decoded,
                omega=omega_after,
                selected_lane=selected.name,
                state_hash=candidate_hash,
            )

        events = self._events(cycle, input_length, lanes, selected, committed, reason)
        best_error = float(selected.reconstruction_l1) if selected else math.inf
        metrics = {
            "input_values": float(input_length),
            "lattice_voxels": float(self.config.volume),
            "hierarchy_levels": float(len(hierarchy)),
            "parallel_lanes": float(len(lanes)),
            "admissible_lanes": float(len(admissible)),
            "best_reconstruction_l1": best_error,
            "active_voxels": float(selected.active_voxels) if selected else 0.0,
            "memory_l1": float(sum(abs(value) for value in omega_after)),
        }
        result = GeometricCycleResult(
            cycle=cycle,
            committed=committed,
            reason=reason,
            selected_lane=selected.name if selected else None,
            encoded=encoded,
            evolved=evolved,
            hierarchy=hierarchy,
            decoded=decoded,
            output=decoded[:input_length],
            omega_before=omega_before,
            omega_after=omega_after,
            lanes=lanes,
            metrics=metrics,
            events=events,
            previous_hash=previous_hash,
            candidate_hash=candidate_hash,
            state_hash=self.state.state_hash,
        )
        self.journal.append(result)
        if len(self.journal) > self.config.journal_limit:
            del self.journal[: len(self.journal) - self.config.journal_limit]
        return result

    def run_feedback(
        self,
        values: Union[Number, Sequence[Number]],
        cycles: Optional[int] = None,
    ) -> List[GeometricCycleResult]:
        count = cycles or self.config.feedback_cycles
        if count < 1:
            raise ValueError("feedback cycles must be positive")
        current: Union[Number, Sequence[Number]] = values
        results: List[GeometricCycleResult] = []
        for _ in range(count):
            result = self.step(current)
            results.append(result)
            if not result.committed:
                break
            current = result.output
        return results

    def snapshot(self) -> Dict[str, object]:
        return asdict(self.state)
