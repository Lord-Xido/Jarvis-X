"""Bounded multimodal 3D auto-encoding/decoding reference runtime.

This module operationalizes the canonical multimodal Dr Moagi operator

    G_{Omega Xi,IO}^3D

as a deterministic, sparse, transactional research-layer runtime. It does not
grant adaptive code authority over the canonical VM core. Physical channels
remain adapters: sensors encode observations into a common 3D field and
actuators decode committed field state back into medium-specific commands.

The reference implementation is intentionally small and auditable:
- sparse 3D coordinates prevent accidental dense materialization;
- all numeric values are finite and projected to bounded ranges;
- candidate transitions are atomic and optionally validator-gated;
- persistent Omega memory is updated only on committed transitions;
- target/output feedback is explicit rather than inferred from capability
  metadata;
- all adaptive terms are bounded by Pi_Lambda before commit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Protocol, Sequence

Coordinate = tuple[int, int, int]
Vector = tuple[float, ...]
SparseVectorField = dict[Coordinate, Vector]
Validator = Callable[
    [Mapping[Coordinate, Vector], "MultimodalStepMetrics"],
    bool,
]


class MediumAdapter(Protocol):
    """Bidirectional medium boundary.

    ``encode_input`` maps a physical/digital observation into the common 3D
    field. ``decode_output`` maps a committed field to a medium-specific
    command or representation. ``observe_output`` is optional physical or
    digital loopback used to verify what was actually produced.
    """

    def encode_input(self, observation: Any) -> Mapping[Coordinate, Sequence[float]]:
        ...

    def decode_output(self, field: Mapping[Coordinate, Vector]) -> Any:
        ...

    def observe_output(self, output: Any) -> Mapping[Coordinate, Sequence[float]]:
        ...


class IdentityMediumAdapter:
    """Deterministic field adapter useful for tests and digital loopback."""

    def encode_input(
        self, observation: Mapping[Coordinate, Sequence[float]]
    ) -> Mapping[Coordinate, Sequence[float]]:
        return dict(observation)

    def decode_output(self, field: Mapping[Coordinate, Vector]) -> SparseVectorField:
        return dict(field)

    def observe_output(
        self, output: Mapping[Coordinate, Sequence[float]]
    ) -> Mapping[Coordinate, Sequence[float]]:
        return dict(output)


class Predictor(Protocol):
    """Predicts a bounded 3D field contribution from current state and input."""

    def predict(
        self,
        state: Mapping[Coordinate, Vector],
        fused_input: Mapping[Coordinate, Vector],
    ) -> Mapping[Coordinate, Sequence[float]]:
        ...


class ZeroPredictor:
    """Reference predictor that contributes no speculative motion."""

    def predict(
        self,
        state: Mapping[Coordinate, Vector],
        fused_input: Mapping[Coordinate, Vector],
    ) -> SparseVectorField:
        return {}


@dataclass(frozen=True)
class MediumChannel:
    """One input/output medium attached to the common 3D field."""

    name: str
    adapter: MediumAdapter
    input_weight: float = 1.0
    output_weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError("channel name must be a non-empty string")
        for attr in ("input_weight", "output_weight"):
            value = getattr(self, attr)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{attr} must be numeric")
            if not math.isfinite(float(value)) or value < 0.0:
                raise ValueError(f"{attr} must be finite and non-negative")


@dataclass(frozen=True)
class DrMoagiMultimodalConfig:
    """Pi_Lambda contract for the multimodal I/O runtime."""

    side: int = 64
    vector_width: int = 4
    dt: float = 0.05
    input_gain: float = 1.0
    prediction_gain: float = 0.25
    error_gain: float = 0.5
    memory_gain: float = 0.25
    memory_decay: float = 0.95
    memory_error_gain: float = 0.1
    value_min: float = -1.0
    value_max: float = 1.0
    max_active_cells: int = 100_000
    max_channels: int = 32
    distortion_weight: float = 1.0
    latency_weight: float = 0.0
    bandwidth_weight: float = 0.0
    energy_weight: float = 0.0
    compute_weight: float = 0.0

    def __post_init__(self) -> None:
        for name in ("side", "vector_width", "max_active_cells", "max_channels"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        for name in (
            "dt",
            "input_gain",
            "prediction_gain",
            "error_gain",
            "memory_gain",
            "memory_decay",
            "memory_error_gain",
            "value_min",
            "value_max",
            "distortion_weight",
            "latency_weight",
            "bandwidth_weight",
            "energy_weight",
            "compute_weight",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")

        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.value_min >= self.value_max:
            raise ValueError("value_min must be smaller than value_max")
        if not 0.0 <= self.memory_decay <= 1.0:
            raise ValueError("memory_decay must be in [0, 1]")
        for name in (
            "input_gain",
            "prediction_gain",
            "error_gain",
            "memory_gain",
            "memory_error_gain",
            "distortion_weight",
            "latency_weight",
            "bandwidth_weight",
            "energy_weight",
            "compute_weight",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class MultimodalStepMetrics:
    """Telemetry emitted for every attempted macro-instruction."""

    cycle: int
    channels_seen: int
    active_cells_before: int
    fused_input_cells: int
    prediction_cells: int
    error_cells: int
    active_cells_after: int
    distortion_mse: float
    memory_l2: float
    approximate_scalar_ops: int
    loss: float
    committed: bool
    rejection_reason: str | None = None


@dataclass(frozen=True)
class MultimodalStepResult:
    """Atomic result of one G_{Omega Xi,IO}^3D invocation."""

    metrics: MultimodalStepMetrics
    outputs: Mapping[str, Any]


class DrMoagiMultimodalRuntime:
    """Transactional reference implementation of the multimodal 3D I/O law.

    The runtime keeps one shared 3D state Xi and one persistent memory field
    Omega. Inputs from all registered media are encoded and fused. A predictor
    adds the inward-looking term P^circlearrowleft. Optional targets and
    loopback observations produce an explicit residual E. The candidate state
    is then projected through Pi_Lambda and atomically committed.
    """

    def __init__(
        self,
        channels: Sequence[MediumChannel],
        *,
        predictor: Predictor | None = None,
        config: DrMoagiMultimodalConfig | None = None,
    ) -> None:
        self.config = config or DrMoagiMultimodalConfig()
        if len(channels) > self.config.max_channels:
            raise ValueError("channel budget exceeded")
        names = [channel.name for channel in channels]
        if len(set(names)) != len(names):
            raise ValueError("channel names must be unique")
        self.channels = {channel.name: channel for channel in channels}
        self.predictor = predictor or ZeroPredictor()
        self._state: SparseVectorField = {}
        self._memory: SparseVectorField = {}
        self._cycle = 0

    @property
    def cycle(self) -> int:
        return self._cycle

    @property
    def virtual_cell_count(self) -> int:
        return self.config.side**3

    def snapshot(self) -> SparseVectorField:
        return dict(self._state)

    def memory_snapshot(self) -> SparseVectorField:
        return dict(self._memory)

    def load(
        self,
        state: Mapping[Coordinate, Sequence[float]],
        *,
        memory: Mapping[Coordinate, Sequence[float]] | None = None,
    ) -> None:
        projected_state = self._project_field(state, "state")
        projected_memory = self._project_field(memory or {}, "memory")
        self._enforce_cell_budget(projected_state, "state")
        self._enforce_cell_budget(projected_memory, "memory")
        self._state = projected_state
        self._memory = projected_memory
        self._cycle = 0

    def step(
        self,
        inputs: Mapping[str, Any],
        *,
        targets: Mapping[str, Any] | None = None,
        resource_costs: Mapping[str, float] | None = None,
        validator: Validator | None = None,
    ) -> MultimodalStepResult:
        """Attempt one complete multimodal sense->encode->fuse->decode cycle."""

        unknown_inputs = set(inputs) - set(self.channels)
        if unknown_inputs:
            raise KeyError(f"unknown input channels: {sorted(unknown_inputs)!r}")
        if targets is not None:
            unknown_targets = set(targets) - set(self.channels)
            if unknown_targets:
                raise KeyError(f"unknown target channels: {sorted(unknown_targets)!r}")

        cycle = self._cycle + 1
        snapshot = dict(self._state)
        memory_snapshot = dict(self._memory)

        encoded_fields: list[tuple[float, SparseVectorField]] = []
        for name in sorted(inputs):
            channel = self.channels[name]
            encoded = self._project_field(
                channel.adapter.encode_input(inputs[name]),
                f"{name} encoded input",
            )
            self._enforce_cell_budget(encoded, f"{name} encoded input")
            if channel.input_weight > 0.0:
                encoded_fields.append((channel.input_weight, encoded))

        fused = self._weighted_fuse(encoded_fields)
        self._enforce_cell_budget(fused, "fused input")

        prediction = self._project_field(
            self.predictor.predict(snapshot, fused),
            "prediction",
        )
        self._enforce_cell_budget(prediction, "prediction")

        current_outputs = self._decode_outputs(snapshot)
        error = self._feedback_error(current_outputs, targets or {})
        self._enforce_cell_budget(error, "feedback error")

        memory_candidate = self._memory_update(memory_snapshot, error)
        self._enforce_cell_budget(memory_candidate, "memory candidate")

        candidate = self._transition(
            snapshot=snapshot,
            fused=fused,
            prediction=prediction,
            error=error,
            memory=memory_candidate,
        )
        self._enforce_cell_budget(candidate, "candidate")

        approximate_ops = (
            self.config.vector_width
            * (
                len(snapshot)
                + len(fused)
                + len(prediction)
                + len(error)
                + len(memory_candidate)
                + len(candidate)
            )
        )
        distortion = self._mse(error)
        loss = self._loss(distortion, approximate_ops, resource_costs or {})

        provisional = MultimodalStepMetrics(
            cycle=cycle,
            channels_seen=len(inputs),
            active_cells_before=len(snapshot),
            fused_input_cells=len(fused),
            prediction_cells=len(prediction),
            error_cells=len(error),
            active_cells_after=len(candidate),
            distortion_mse=distortion,
            memory_l2=self._l2(memory_candidate),
            approximate_scalar_ops=approximate_ops,
            loss=loss,
            committed=False,
        )

        if validator is not None and not bool(validator(candidate, provisional)):
            self._cycle = cycle
            return MultimodalStepResult(
                metrics=replace(
                    provisional,
                    active_cells_after=len(snapshot),
                    committed=False,
                    rejection_reason="validator rejected candidate",
                ),
                outputs=current_outputs,
            )

        self._state = candidate
        self._memory = memory_candidate
        self._cycle = cycle
        committed_outputs = self._decode_outputs(candidate)
        return MultimodalStepResult(
            metrics=replace(provisional, committed=True),
            outputs=committed_outputs,
        )

    def _decode_outputs(self, field: Mapping[Coordinate, Vector]) -> dict[str, Any]:
        outputs: dict[str, Any] = {}
        for name in sorted(self.channels):
            channel = self.channels[name]
            if channel.output_weight == 0.0:
                continue
            weighted = self._scale_field(field, channel.output_weight)
            outputs[name] = channel.adapter.decode_output(weighted)
        return outputs

    def _feedback_error(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> SparseVectorField:
        fields: list[tuple[float, SparseVectorField]] = []
        for name in sorted(targets):
            channel = self.channels[name]
            observed = self._project_field(
                channel.adapter.observe_output(outputs.get(name)),
                f"{name} observed output",
            )
            target = self._project_field(
                channel.adapter.encode_input(targets[name]),
                f"{name} target",
            )
            residual = self._field_subtract(target, observed)
            fields.append((max(channel.output_weight, 1.0), residual))
        return self._weighted_fuse(fields)

    def _transition(
        self,
        *,
        snapshot: Mapping[Coordinate, Vector],
        fused: Mapping[Coordinate, Vector],
        prediction: Mapping[Coordinate, Vector],
        error: Mapping[Coordinate, Vector],
        memory: Mapping[Coordinate, Vector],
    ) -> SparseVectorField:
        support = set(snapshot) | set(fused) | set(prediction) | set(error) | set(memory)
        candidate: SparseVectorField = {}
        for coordinate in sorted(support, key=self._linear_address):
            state_v = self._value(snapshot, coordinate)
            fused_v = self._value(fused, coordinate)
            prediction_v = self._value(prediction, coordinate)
            error_v = self._value(error, coordinate)
            memory_v = self._value(memory, coordinate)
            innovation = tuple(fused_v[i] - state_v[i] for i in range(self.config.vector_width))
            next_v = tuple(
                state_v[i]
                + self.config.dt
                * (
                    self.config.input_gain * innovation[i]
                    + self.config.prediction_gain * prediction_v[i]
                    + self.config.error_gain * error_v[i]
                    + self.config.memory_gain * memory_v[i]
                )
                for i in range(self.config.vector_width)
            )
            projected = self._project_vector(next_v, "candidate vector")
            if any(value != 0.0 for value in projected):
                candidate[coordinate] = projected
        return candidate

    def _memory_update(
        self,
        memory: Mapping[Coordinate, Vector],
        error: Mapping[Coordinate, Vector],
    ) -> SparseVectorField:
        support = set(memory) | set(error)
        updated: SparseVectorField = {}
        for coordinate in sorted(support, key=self._linear_address):
            old_v = self._value(memory, coordinate)
            error_v = self._value(error, coordinate)
            next_v = tuple(
                self.config.memory_decay * old_v[i]
                + self.config.memory_error_gain * error_v[i]
                for i in range(self.config.vector_width)
            )
            projected = self._project_vector(next_v, "memory vector")
            if any(value != 0.0 for value in projected):
                updated[coordinate] = projected
        return updated

    def _weighted_fuse(
        self,
        fields: Sequence[tuple[float, Mapping[Coordinate, Vector]]],
    ) -> SparseVectorField:
        numerator: dict[Coordinate, list[float]] = {}
        denominator: dict[Coordinate, float] = {}
        for weight, field in fields:
            if weight <= 0.0:
                continue
            for coordinate, vector in field.items():
                bucket = numerator.setdefault(
                    coordinate, [0.0] * self.config.vector_width
                )
                for i, value in enumerate(vector):
                    bucket[i] += weight * value
                denominator[coordinate] = denominator.get(coordinate, 0.0) + weight

        fused: SparseVectorField = {}
        for coordinate in sorted(numerator, key=self._linear_address):
            scale = denominator[coordinate]
            vector = tuple(value / scale for value in numerator[coordinate])
            projected = self._project_vector(vector, "fused vector")
            if any(value != 0.0 for value in projected):
                fused[coordinate] = projected
        return fused

    def _field_subtract(
        self,
        a: Mapping[Coordinate, Vector],
        b: Mapping[Coordinate, Vector],
    ) -> SparseVectorField:
        support = set(a) | set(b)
        result: SparseVectorField = {}
        for coordinate in sorted(support, key=self._linear_address):
            av = self._value(a, coordinate)
            bv = self._value(b, coordinate)
            vector = tuple(av[i] - bv[i] for i in range(self.config.vector_width))
            projected = self._project_vector(vector, "residual vector")
            if any(value != 0.0 for value in projected):
                result[coordinate] = projected
        return result

    def _scale_field(
        self,
        field: Mapping[Coordinate, Vector],
        scale: float,
    ) -> SparseVectorField:
        return {
            coordinate: self._project_vector(
                tuple(scale * value for value in vector),
                "scaled output vector",
            )
            for coordinate, vector in field.items()
        }

    def _project_field(
        self,
        field: Mapping[Coordinate, Sequence[float]],
        name: str,
    ) -> SparseVectorField:
        if not isinstance(field, Mapping):
            raise TypeError(f"{name} must be a mapping")
        projected: SparseVectorField = {}
        for coordinate, vector in field.items():
            coordinate = self._validate_coordinate(coordinate)
            projected_vector = self._project_vector(vector, name)
            # Preserve explicitly supplied zero vectors at I/O boundaries:
            # absence and a measured zero are not equivalent during weighted
            # multimodal fusion. Candidate state and memory still prune zeros.
            projected[coordinate] = projected_vector
        return projected

    def _project_vector(self, vector: Sequence[float], name: str) -> Vector:
        if isinstance(vector, (str, bytes)) or not isinstance(vector, Sequence):
            raise TypeError(f"{name} vectors must be sequences")
        if len(vector) != self.config.vector_width:
            raise ValueError(
                f"{name} vectors must have width {self.config.vector_width}"
            )
        projected: list[float] = []
        for raw in vector:
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise TypeError(f"{name} vector entries must be numeric")
            value = float(raw)
            if not math.isfinite(value):
                raise ValueError(f"{name} vector entries must be finite")
            projected.append(
                min(self.config.value_max, max(self.config.value_min, value))
            )
        return tuple(projected)

    def _value(
        self,
        field: Mapping[Coordinate, Vector],
        coordinate: Coordinate,
    ) -> Vector:
        return field.get(coordinate, (0.0,) * self.config.vector_width)

    def _validate_coordinate(self, coordinate: Coordinate) -> Coordinate:
        if (
            not isinstance(coordinate, tuple)
            or len(coordinate) != 3
            or any(isinstance(v, bool) or not isinstance(v, int) for v in coordinate)
        ):
            raise TypeError("coordinates must be integer (x, y, z) tuples")
        x, y, z = coordinate
        side = self.config.side
        if not (0 <= x < side and 0 <= y < side and 0 <= z < side):
            raise ValueError("coordinate is outside the logical lattice")
        return coordinate

    def _linear_address(self, coordinate: Coordinate) -> int:
        x, y, z = coordinate
        side = self.config.side
        return x + side * (y + side * z)

    def _enforce_cell_budget(
        self,
        field: Mapping[Coordinate, Vector],
        name: str,
    ) -> None:
        if len(field) > self.config.max_active_cells:
            raise RuntimeError(f"{name} active-cell budget exceeded")

    def _mse(self, field: Mapping[Coordinate, Vector]) -> float:
        if not field:
            return 0.0
        squared = sum(
            value * value
            for vector in field.values()
            for value in vector
        )
        return squared / (len(field) * self.config.vector_width)

    @staticmethod
    def _l2(field: Mapping[Coordinate, Vector]) -> float:
        return math.sqrt(
            sum(value * value for vector in field.values() for value in vector)
        )

    def _loss(
        self,
        distortion: float,
        approximate_ops: int,
        resource_costs: Mapping[str, float],
    ) -> float:
        costs: dict[str, float] = {}
        for key in ("latency", "bandwidth", "energy"):
            raw = resource_costs.get(key, 0.0)
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise TypeError(f"{key} cost must be numeric")
            value = float(raw)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{key} cost must be finite and non-negative")
            costs[key] = value

        return (
            self.config.distortion_weight * distortion
            + self.config.latency_weight * costs["latency"]
            + self.config.bandwidth_weight * costs["bandwidth"]
            + self.config.energy_weight * costs["energy"]
            + self.config.compute_weight * float(approximate_ops)
        )
