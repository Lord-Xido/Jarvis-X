"""Deterministic operational reference for the Dr Moagi inward 3D system.

The runtime implements a bounded sparse voxel field with 26-neighbour message passing,
front-projection decode, prior-preserving re-encoding, relaxed fixed-point iteration,
bounded analytic parameter updates, exact trace-macro promotion, and deterministic
verification. Operator promotion compresses execution descriptions; acceleration is measured
separately by an execution backend.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from typing import Dict, Iterable, Mapping, Sequence, Tuple

Coordinate = Tuple[int, int, int]
Pixel = Tuple[int, int]
Vector = Tuple[float, ...]

PRIMITIVE_TRACE: Tuple[str, ...] = (
    "LOAD_SELF",
    "AGGREGATE_26",
    "DECODE_FRONT",
    "LOAD_INPUT",
    "AFFINE",
    "TANH",
    "RELAX",
)


@dataclass(frozen=True)
class DMSOConfig:
    side: int = 16
    channels: int = 1
    alpha: float = 0.25
    learning_rate: float = 0.025
    tolerance: float = 1.0e-6
    max_settle_steps: int = 128
    max_active_cells: int = 4096
    promotion_repeats: int = 4
    max_operator_depth: int = 4
    auto_approve_depth: int = 1
    parameter_limit: float = 4.0

    def __post_init__(self) -> None:
        integer_fields = (
            "side",
            "channels",
            "max_settle_steps",
            "max_active_cells",
            "promotion_repeats",
            "max_operator_depth",
            "auto_approve_depth",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.auto_approve_depth > self.max_operator_depth:
            raise ValueError("auto_approve_depth cannot exceed max_operator_depth")
        for name in ("alpha", "learning_rate", "tolerance", "parameter_limit"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if self.learning_rate < 0.0:
            raise ValueError("learning_rate must be non-negative")
        if self.tolerance < 0.0:
            raise ValueError("tolerance must be non-negative")
        if self.parameter_limit <= 0.0:
            raise ValueError("parameter_limit must be positive")


@dataclass(frozen=True)
class DMSOParameters:
    self_gain: float = 0.75
    neighbour_gain: float = 0.20
    projection_gain: float = 0.05
    input_gain: float = 0.50
    bias: float = 0.0


@dataclass(frozen=True)
class OperatorDefinition:
    operator_id: str
    expansion: Tuple[str, ...]
    depth: int
    observed_repeats: int
    utility: float
    verified: bool
    human_approved: bool


@dataclass(frozen=True)
class DMSOMetrics:
    cycle: int
    active_cells: int
    fixed_point_residual: float
    task_loss: float
    stable: bool
    parameter_version: int
    operator_count: int
    trace_bytes_raw: int
    trace_bytes_encoded: int
    description_compression_ratio: float
    state_digest: str


class DMSORuntime:
    """Sparse deterministic inward 3D reference runtime."""

    NEIGHBOUR_OFFSETS: Tuple[Coordinate, ...] = tuple(
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if (dx, dy, dz) != (0, 0, 0)
    )

    def __init__(
        self,
        config: DMSOConfig | None = None,
        parameters: DMSOParameters | None = None,
    ) -> None:
        self.config = config or DMSOConfig()
        self.parameters = parameters or DMSOParameters()
        self._validate_parameters(self.parameters)
        self._state: Dict[Coordinate, Vector] = {}
        self._cycle = 0
        self._parameter_version = 0
        self._trace_counts: Dict[Tuple[str, ...], int] = {}
        self._operators: Dict[str, OperatorDefinition] = {}
        self._raw_trace_occurrences = 0
        self._encoded_trace_occurrences = 0

    @property
    def cycle(self) -> int:
        return self._cycle

    @property
    def state(self) -> Dict[Coordinate, Vector]:
        return dict(self._state)

    @property
    def operators(self) -> Tuple[OperatorDefinition, ...]:
        return tuple(self._operators[key] for key in sorted(self._operators))

    @classmethod
    def neighbour_offsets(cls) -> Tuple[Coordinate, ...]:
        return cls.NEIGHBOUR_OFFSETS

    def seed(self, values: Mapping[Coordinate, Sequence[float] | float]) -> None:
        normalized = self._normalize_field(values, "state")
        if len(normalized) > self.config.max_active_cells:
            raise RuntimeError("active-cell budget exceeded")
        self._state = normalized

    def decode(self, state: Mapping[Coordinate, Vector] | None = None) -> Dict[Pixel, Vector]:
        """Front-project by selecting the smallest occupied z for each (x, y)."""

        source = self._state if state is None else state
        front: Dict[Pixel, Tuple[int, Vector]] = {}
        for (x, y, z), value in source.items():
            current = front.get((x, y))
            if current is None or z < current[0]:
                front[(x, y)] = (z, value)
        return {pixel: item[1] for pixel, item in front.items()}

    def encode(
        self,
        decoded: Mapping[Pixel, Vector],
        external: Mapping[Coordinate, Vector],
        prior: Mapping[Coordinate, Vector],
        support: Iterable[Coordinate],
    ) -> Dict[Coordinate, Vector]:
        """Evaluate E(D(S), U, S) on bounded support without authoritative mutation."""

        visible_depth = self._visible_depth(prior)
        candidate: Dict[Coordinate, Vector] = {}
        for coordinate in support:
            x, y, z = coordinate
            current = prior.get(coordinate, self._zero())
            neighbours = self._neighbour_mean(prior, coordinate)
            projected = (
                decoded.get((x, y), self._zero())
                if visible_depth.get((x, y)) == z
                else self._zero()
            )
            stimulus = external.get(coordinate, self._zero())
            candidate[coordinate] = tuple(
                math.tanh(
                    self.parameters.self_gain * current[channel]
                    + self.parameters.neighbour_gain * neighbours[channel]
                    + self.parameters.projection_gain * projected[channel]
                    + self.parameters.input_gain * stimulus[channel]
                    + self.parameters.bias
                )
                for channel in range(self.config.channels)
            )
        return candidate

    def step(
        self,
        external: Mapping[Coordinate, Sequence[float] | float] | None = None,
        targets: Mapping[Coordinate, Sequence[float] | float] | None = None,
        *,
        learn: bool = False,
    ) -> DMSOMetrics:
        """Run one transactional relaxed update and optional bounded mechanics-learning step."""

        normalized_external = self._normalize_field(external or {}, "external")
        normalized_targets = self._normalize_field(targets or {}, "target")
        support = set(self._state).union(normalized_external).union(normalized_targets)
        if len(support) > self.config.max_active_cells:
            raise RuntimeError("active-cell budget exceeded")

        snapshot = dict(self._state)
        decoded = self.decode(snapshot)
        mapped = self.encode(decoded, normalized_external, snapshot, sorted(support))
        staged: Dict[Coordinate, Vector] = {}
        squared_delta = 0.0
        scalar_count = max(1, len(support) * self.config.channels)

        for coordinate in sorted(support):
            current = snapshot.get(coordinate, self._zero())
            mapped_value = mapped[coordinate]
            relaxed = tuple(
                current[channel]
                + self.config.alpha * (mapped_value[channel] - current[channel])
                for channel in range(self.config.channels)
            )
            self._require_finite_vector(relaxed, "candidate")
            staged[coordinate] = relaxed
            squared_delta += sum(
                (relaxed[channel] - current[channel]) ** 2
                for channel in range(self.config.channels)
            )

        residual = math.sqrt(squared_delta / scalar_count)
        task_loss = self._task_loss(staged, normalized_targets)
        staged_parameters = self.parameters
        if learn and normalized_targets:
            staged_parameters = self._learn_parameters(
                snapshot,
                decoded,
                normalized_external,
                normalized_targets,
                support,
            )

        self._validate_parameters(staged_parameters)
        if not math.isfinite(residual) or not math.isfinite(task_loss):
            raise RuntimeError("non-finite runtime metric; transaction rolled back")

        self._state = staged
        if staged_parameters != self.parameters:
            self.parameters = staged_parameters
            self._parameter_version += 1
        self._cycle += 1
        self._observe_trace(PRIMITIVE_TRACE, max(1, len(support)))
        return self.metrics(task_loss=task_loss, residual=residual)

    def settle(
        self,
        external: Mapping[Coordinate, Sequence[float] | float] | None = None,
    ) -> DMSOMetrics:
        metrics = self.metrics()
        for _ in range(self.config.max_settle_steps):
            metrics = self.step(external=external)
            if metrics.stable:
                return metrics
        return metrics

    def promote_operator(
        self,
        expansion: Sequence[str],
        *,
        human_approved: bool = False,
        observed_repeats: int = 1,
    ) -> OperatorDefinition:
        normalized = tuple(expansion)
        if len(normalized) < 2:
            raise ValueError("composite operator requires at least two children")
        child_depths = []
        for child in normalized:
            if child in PRIMITIVE_TRACE:
                child_depths.append(0)
            elif child in self._operators:
                child_depths.append(self._operators[child].depth)
            else:
                raise ValueError(f"unknown operator child: {child}")
        depth = 1 + max(child_depths)
        if depth > self.config.max_operator_depth:
            raise RuntimeError("maximum operator abstraction depth exceeded")
        if depth > self.config.auto_approve_depth and not human_approved:
            raise PermissionError("human approval required for this abstraction depth")

        digest = hashlib.sha256("\0".join(normalized).encode("utf-8")).hexdigest()[:12]
        operator_id = f"DMSO_{depth}_{digest}"
        repeats = max(1, int(observed_repeats))
        utility = float((len(normalized) - 1) * repeats - len(normalized))
        definition = OperatorDefinition(
            operator_id=operator_id,
            expansion=normalized,
            depth=depth,
            observed_repeats=repeats,
            utility=utility,
            verified=True,
            human_approved=human_approved or depth <= self.config.auto_approve_depth,
        )
        current = self._operators.get(operator_id)
        if current is None or definition.observed_repeats > current.observed_repeats:
            self._operators[operator_id] = definition
        return self._operators[operator_id]

    def verify(self) -> bool:
        self._validate_parameters(self.parameters)
        if len(self._state) > self.config.max_active_cells:
            return False
        for coordinate, vector in self._state.items():
            self._validate_coordinate(coordinate)
            self._require_finite_vector(vector, "state")
        known = set(PRIMITIVE_TRACE).union(self._operators)
        for definition in self._operators.values():
            if not definition.verified or definition.depth > self.config.max_operator_depth:
                return False
            if any(child not in known for child in definition.expansion):
                return False
            if definition.depth > self.config.auto_approve_depth and not definition.human_approved:
                return False
        return True

    def metrics(
        self,
        *,
        task_loss: float = 0.0,
        residual: float | None = None,
    ) -> DMSOMetrics:
        current_residual = self._fixed_point_residual() if residual is None else residual
        raw = self._raw_trace_occurrences * len(PRIMITIVE_TRACE) * 4
        encoded = self._encoded_trace_occurrences * 4
        unresolved = self._raw_trace_occurrences - self._encoded_trace_occurrences
        encoded += unresolved * len(PRIMITIVE_TRACE) * 4
        ratio = raw / encoded if encoded else 1.0
        return DMSOMetrics(
            cycle=self._cycle,
            active_cells=len(self._state),
            fixed_point_residual=current_residual,
            task_loss=task_loss,
            stable=current_residual <= self.config.tolerance,
            parameter_version=self._parameter_version,
            operator_count=len(self._operators),
            trace_bytes_raw=raw,
            trace_bytes_encoded=encoded,
            description_compression_ratio=ratio,
            state_digest=self.state_digest(),
        )

    def state_digest(self) -> str:
        payload = {
            "config": asdict(self.config),
            "parameters": asdict(self.parameters),
            "cycle": self._cycle,
            "state": [
                [list(coordinate), list(self._state[coordinate])]
                for coordinate in sorted(self._state)
            ],
            "operators": [asdict(item) for item in self.operators],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(b"jarvisx-dmso-v2\0" + encoded).hexdigest()

    def _fixed_point_residual(self) -> float:
        if not self._state:
            return 0.0
        decoded = self.decode(self._state)
        mapped = self.encode(decoded, {}, self._state, sorted(self._state))
        squared = 0.0
        count = len(self._state) * self.config.channels
        for coordinate, current in self._state.items():
            squared += sum(
                (mapped[coordinate][channel] - current[channel]) ** 2
                for channel in range(self.config.channels)
            )
        return math.sqrt(squared / max(1, count))

    def _learn_parameters(
        self,
        snapshot: Mapping[Coordinate, Vector],
        decoded: Mapping[Pixel, Vector],
        external: Mapping[Coordinate, Vector],
        targets: Mapping[Coordinate, Vector],
        support: Iterable[Coordinate],
    ) -> DMSOParameters:
        visible_depth = self._visible_depth(snapshot)
        gradients: Dict[str, float] = {
            "self_gain": 0.0,
            "neighbour_gain": 0.0,
            "projection_gain": 0.0,
            "input_gain": 0.0,
            "bias": 0.0,
        }
        scalar_count = max(1, len(targets) * self.config.channels)
        for coordinate in sorted(support):
            target = targets.get(coordinate)
            if target is None:
                continue
            x, y, z = coordinate
            current = snapshot.get(coordinate, self._zero())
            neighbours = self._neighbour_mean(snapshot, coordinate)
            projected = (
                decoded.get((x, y), self._zero())
                if visible_depth.get((x, y)) == z
                else self._zero()
            )
            stimulus = external.get(coordinate, self._zero())
            for channel in range(self.config.channels):
                features = {
                    "self_gain": current[channel],
                    "neighbour_gain": neighbours[channel],
                    "projection_gain": projected[channel],
                    "input_gain": stimulus[channel],
                    "bias": 1.0,
                }
                preactivation = sum(
                    getattr(self.parameters, name) * feature
                    for name, feature in features.items()
                )
                mapped_value = math.tanh(preactivation)
                relaxed = current[channel] + self.config.alpha * (
                    mapped_value - current[channel]
                )
                dloss = 2.0 * (relaxed - target[channel]) / scalar_count
                common = dloss * self.config.alpha * (1.0 - mapped_value * mapped_value)
                for name, feature in features.items():
                    gradients[name] += common * feature

        updates: Dict[str, float] = {}
        limit = self.config.parameter_limit
        for name, gradient in gradients.items():
            value = getattr(self.parameters, name) - self.config.learning_rate * gradient
            updates[name] = max(-limit, min(limit, value))
        return replace(self.parameters, **updates)

    def _observe_trace(self, trace: Tuple[str, ...], occurrences: int) -> None:
        previous = self._trace_counts.get(trace, 0)
        total = previous + occurrences
        self._trace_counts[trace] = total
        self._raw_trace_occurrences += occurrences
        existing = self._operator_for_expansion(trace)
        if existing is None and total >= self.config.promotion_repeats:
            existing = self.promote_operator(trace, observed_repeats=total)
        if existing is not None:
            self._encoded_trace_occurrences += occurrences

    def _operator_for_expansion(self, trace: Tuple[str, ...]) -> OperatorDefinition | None:
        return next(
            (definition for definition in self._operators.values() if definition.expansion == trace),
            None,
        )

    def _task_loss(
        self,
        state: Mapping[Coordinate, Vector],
        targets: Mapping[Coordinate, Vector],
    ) -> float:
        if not targets:
            return 0.0
        squared = 0.0
        count = 0
        for coordinate, target in targets.items():
            value = state.get(coordinate, self._zero())
            for channel in range(self.config.channels):
                squared += (value[channel] - target[channel]) ** 2
                count += 1
        return squared / max(1, count)

    @staticmethod
    def _visible_depth(state: Mapping[Coordinate, Vector]) -> Dict[Pixel, int]:
        visible: Dict[Pixel, int] = {}
        for x, y, z in state:
            pixel = (x, y)
            current = visible.get(pixel)
            if current is None or z < current:
                visible[pixel] = z
        return visible

    def _neighbour_mean(
        self,
        state: Mapping[Coordinate, Vector],
        coordinate: Coordinate,
    ) -> Vector:
        x, y, z = coordinate
        totals = [0.0] * self.config.channels
        count = 0
        for dx, dy, dz in self.NEIGHBOUR_OFFSETS:
            neighbour = (x + dx, y + dy, z + dz)
            if self._coordinate_in_bounds(neighbour):
                vector = state.get(neighbour, self._zero())
                for channel in range(self.config.channels):
                    totals[channel] += vector[channel]
                count += 1
        if count == 0:
            return self._zero()
        return tuple(value / count for value in totals)

    def _normalize_field(
        self,
        values: Mapping[Coordinate, Sequence[float] | float],
        name: str,
    ) -> Dict[Coordinate, Vector]:
        if not isinstance(values, Mapping):
            raise TypeError(f"{name} must be a mapping")
        normalized: Dict[Coordinate, Vector] = {}
        for coordinate, value in values.items():
            checked = self._validate_coordinate(coordinate)
            vector: Vector
            if (
                self.config.channels == 1
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            ):
                vector = (float(value),)
            else:
                if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
                    raise TypeError(f"{name} values must match channel count")
                vector = tuple(float(item) for item in value)
            if len(vector) != self.config.channels:
                raise ValueError(
                    f"{name} values must contain exactly {self.config.channels} channels"
                )
            self._require_finite_vector(vector, name)
            normalized[checked] = vector
        return normalized

    def _validate_coordinate(self, coordinate: Coordinate) -> Coordinate:
        if not isinstance(coordinate, tuple) or len(coordinate) != 3:
            raise TypeError("coordinate must be a three-integer tuple")
        if not all(isinstance(item, int) and not isinstance(item, bool) for item in coordinate):
            raise TypeError("coordinate components must be integers")
        if not self._coordinate_in_bounds(coordinate):
            raise ValueError("coordinate is outside the configured lattice")
        return coordinate

    def _coordinate_in_bounds(self, coordinate: Coordinate) -> bool:
        return all(0 <= item < self.config.side for item in coordinate)

    def _validate_parameters(self, parameters: DMSOParameters) -> None:
        limit = self.config.parameter_limit
        for name, value in asdict(parameters).items():
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                raise ValueError(f"parameter {name} must be finite")
            if abs(numeric_value) > limit:
                raise ValueError(f"parameter {name} exceeds configured bound")

    @staticmethod
    def _require_finite_vector(vector: Sequence[float], name: str) -> None:
        if not all(math.isfinite(float(value)) for value in vector):
            raise ValueError(f"{name} contains a non-finite value")

    def _zero(self) -> Vector:
        return (0.0,) * self.config.channels
