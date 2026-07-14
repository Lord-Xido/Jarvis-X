"""CHRYSALIS-Theta: deterministic sparse multimodal arithmetic on a 3D grid.

The reference runtime is dependency-free and Python 3.8 compatible. It implements
true top-k expert dispatch, coordinate-correct [D, H, W] geometry, multimodal
cross-attention, and a sparse ConvLSTM-like recurrent voxel state.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

Number = Union[int, float]
ModalityInput = Union[str, Sequence[Number]]
Vector = Tuple[float, ...]
Coordinate = Tuple[int, int, int]

_MASK64 = (1 << 64) - 1


def _mix64(value: int) -> int:
    value &= _MASK64
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & _MASK64
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & _MASK64
    value ^= value >> 31
    return value & _MASK64


def _stable_int(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def _unit(seed: int, *parts: int) -> float:
    value = seed & _MASK64
    for part in parts:
        value = _mix64(value ^ _mix64(int(part) + 0x9E3779B97F4A7C15))
    return (value / float(_MASK64)) * 2.0 - 1.0


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _softmax(values: Sequence[float]) -> Tuple[float, ...]:
    if not values:
        return ()
    maximum = max(values)
    exps = [math.exp(value - maximum) for value in values]
    total = sum(exps)
    if total <= 0.0 or not math.isfinite(total):
        raise ValueError("softmax normalization failed")
    return tuple(value / total for value in exps)


def _l2_normalize(values: Sequence[float], epsilon: float = 1e-12) -> Vector:
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= epsilon:
        return tuple(0.0 for _ in values)
    return tuple(value / norm for value in values)


def _ensure_finite(values: Iterable[float], name: str) -> None:
    if not all(math.isfinite(value) for value in values):
        raise ValueError("%s contains non-finite values" % name)


@dataclass(frozen=True)
class GridShape:
    width: int = 4
    height: int = 4
    depth: int = 4

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0 or self.depth <= 0:
            raise ValueError("grid dimensions must be positive")

    @property
    def positions(self) -> int:
        return self.width * self.height * self.depth

    def index(self, coordinate: Coordinate) -> int:
        x, y, z = coordinate
        if not (0 <= x < self.width and 0 <= y < self.height and 0 <= z < self.depth):
            raise IndexError("coordinate is outside the grid")
        return (z * self.height + y) * self.width + x

    def coordinate(self, index: int) -> Coordinate:
        if not 0 <= index < self.positions:
            raise IndexError("grid index is outside the grid")
        x = index % self.width
        rest = index // self.width
        y = rest % self.height
        z = rest // self.height
        return (x, y, z)

    def neighbors6(self, index: int) -> Tuple[int, ...]:
        x, y, z = self.coordinate(index)
        result = []
        for dx, dy, dz in (
            (-1, 0, 0),
            (1, 0, 0),
            (0, -1, 0),
            (0, 1, 0),
            (0, 0, -1),
            (0, 0, 1),
        ):
            nx, ny, nz = x + dx, y + dy, z + dz
            if 0 <= nx < self.width and 0 <= ny < self.height and 0 <= nz < self.depth:
                result.append(self.index((nx, ny, nz)))
        return tuple(result)


@dataclass(frozen=True)
class ChrysalisConfig:
    grid: GridShape = field(default_factory=GridShape)
    d_model: int = 512
    top_k: int = 2
    expert_rank: int = 8
    num_heads: int = 8
    seed: int = 0xC485A115
    state_decay: float = 0.995
    output_residual: float = 0.5

    def __post_init__(self) -> None:
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if not 1 <= self.top_k <= self.grid.positions:
            raise ValueError("top_k must be inside [1, positions]")
        if self.expert_rank <= 0:
            raise ValueError("expert_rank must be positive")
        if self.num_heads <= 0 or self.d_model % self.num_heads != 0:
            raise ValueError("num_heads must divide d_model")
        if not 0.0 <= self.state_decay <= 1.0:
            raise ValueError("state_decay must be inside [0, 1]")
        if not 0.0 <= self.output_residual <= 1.0:
            raise ValueError("output_residual must be inside [0, 1]")


@dataclass(frozen=True)
class ExpertActivation:
    index: int
    coordinate: Coordinate
    logit: float
    weight: float


@dataclass(frozen=True)
class ArithmeticReport:
    positions: int
    active_experts: int
    activation_ratio: float
    grid_projection_ops: int
    router_ops: int
    expert_ops: int
    attention_ops: int
    recurrent_ops: int
    total_scalar_ops: int
    recurrent_state_values: int


@dataclass(frozen=True)
class ChrysalisResult:
    output: Vector
    activations: Tuple[ExpertActivation, ...]
    modality_attention: Tuple[Tuple[float, ...], ...]
    state_hash: str
    step: int
    arithmetic: ArithmeticReport


class ChrysalisTheta3D:
    """Dependency-free operational reference for CHRYSALIS-Theta.

    Input modalities can be d_model embeddings, arbitrary numeric vectors, or text.
    Arbitrary inputs are deterministically feature-hashed into d_model. The core then:

    1. projects modalities into a [D, H, W, d_model] latent field;
    2. computes a locally smoothed scalar routing field;
    3. executes only top-k independent coordinate-seeded low-rank experts;
    4. applies multi-head cross-modal attention;
    5. updates selected voxels with a sparse ConvLSTM-like recurrence.
    """

    def __init__(self, config: Optional[ChrysalisConfig] = None) -> None:
        self.config = config or ChrysalisConfig()
        p = self.config.grid.positions
        d = self.config.d_model
        self._hidden: List[List[float]] = [[0.0] * d for _ in range(p)]
        self._cell: List[List[float]] = [[0.0] * d for _ in range(p)]
        self._step = 0

    @property
    def step_index(self) -> int:
        return self._step

    def reset(self) -> None:
        for vector in self._hidden:
            vector[:] = [0.0] * self.config.d_model
        for vector in self._cell:
            vector[:] = [0.0] * self.config.d_model
        self._step = 0

    def encode_modality(self, value: ModalityInput, name: str) -> Vector:
        d = self.config.d_model
        salt = _stable_int(name)
        if isinstance(value, str):
            raw = value.encode("utf-8")
            accumulator = [0.0] * d
            if not raw:
                return tuple(accumulator)
            for index, byte in enumerate(raw):
                key = _mix64(salt ^ _mix64(index + 1) ^ byte)
                bucket = key % d
                sign = 1.0 if (key >> 8) & 1 else -1.0
                accumulator[bucket] += sign * (byte / 255.0)
            return _l2_normalize(accumulator)

        numeric = [float(item) for item in value]
        _ensure_finite(numeric, name)
        if len(numeric) == d:
            return _l2_normalize(numeric)
        accumulator = [0.0] * d
        if not numeric:
            return tuple(accumulator)
        scale = 1.0 / math.sqrt(float(len(numeric)))
        for index, item in enumerate(numeric):
            key1 = _mix64(salt ^ _mix64(index + 1))
            key2 = _mix64(key1 ^ 0xA5A5A5A5A5A5A5A5)
            bucket1 = key1 % d
            bucket2 = key2 % d
            accumulator[bucket1] += item * scale * (1.0 if (key1 >> 9) & 1 else -1.0)
            accumulator[bucket2] += (
                item * scale * 0.5 * (1.0 if (key2 >> 9) & 1 else -1.0)
            )
        return _l2_normalize(accumulator)

    def _prepare_modalities(
        self, modalities: Mapping[str, ModalityInput]
    ) -> Tuple[Tuple[str, Vector], ...]:
        if not modalities:
            raise ValueError("at least one modality is required")
        prepared = tuple(
            (name, self.encode_modality(value, name))
            for name, value in sorted(modalities.items())
        )
        return prepared

    def _grid_projection(
        self, modalities: Tuple[Tuple[str, Vector], ...]
    ) -> List[List[float]]:
        config = self.config
        d = config.d_model
        count = float(len(modalities))
        result: List[List[float]] = []
        for position in range(config.grid.positions):
            x, y, z = config.grid.coordinate(position)
            coordinate_bias = (
                (x + 0.5) / config.grid.width
                + (y + 0.5) / config.grid.height
                + (z + 0.5) / config.grid.depth
            ) / 3.0
            vector = [0.0] * d
            for name, embedding in modalities:
                salt = _stable_int(name)
                offset = _mix64(salt ^ position) % d
                stride = 1 + (_mix64(salt ^ (position + 1)) % max(1, d - 1))
                for channel in range(d):
                    source = (offset + channel * stride) % d
                    sign = (
                        1.0
                        if _unit(config.seed ^ salt, position, channel) >= 0.0
                        else -1.0
                    )
                    vector[channel] += sign * embedding[source] / count
            for channel in range(d):
                positional = 0.125 * math.sin(
                    (channel + 1) * (1.0 + coordinate_bias) + position * 0.173
                )
                vector[channel] = math.tanh(vector[channel] + positional)
            result.append(vector)
        return result

    def _router(self, grid: Sequence[Sequence[float]]) -> Tuple[ExpertActivation, ...]:
        config = self.config
        d = config.d_model
        raw = []
        for position, vector in enumerate(grid):
            total = 0.0
            for channel, value in enumerate(vector):
                total += value * _unit(config.seed ^ 0xA11CE, position, channel)
            raw.append(total / math.sqrt(float(d)))

        smoothed = []
        for position, own in enumerate(raw):
            neighbors = config.grid.neighbors6(position)
            neighborhood = sum(raw[item] for item in neighbors) / float(
                len(neighbors) or 1
            )
            smoothed.append(0.75 * own + 0.25 * neighborhood)

        selected = sorted(
            range(config.grid.positions),
            key=lambda index: (-smoothed[index], index),
        )[: config.top_k]
        weights = _softmax([smoothed[index] for index in selected])
        return tuple(
            ExpertActivation(
                index=index,
                coordinate=config.grid.coordinate(index),
                logit=smoothed[index],
                weight=weight,
            )
            for index, weight in zip(selected, weights)
        )

    def _expert(self, position: int, vector: Sequence[float]) -> Vector:
        config = self.config
        d = config.d_model
        rank = config.expert_rank
        hidden = [0.0] * rank
        inv_d = 1.0 / math.sqrt(float(d))
        for component in range(rank):
            total = 0.0
            for channel, value in enumerate(vector):
                total += value * _unit(
                    config.seed ^ 0xE1A1, position, component, channel
                )
            hidden[component] = max(0.0, total * inv_d)

        inv_rank = 1.0 / math.sqrt(float(rank))
        output = [0.0] * d
        for channel in range(d):
            correction = 0.0
            for component, value in enumerate(hidden):
                correction += value * _unit(
                    config.seed ^ 0xE2A2, position, channel, component
                )
            output[channel] = math.tanh(vector[channel] + correction * inv_rank)
        return tuple(output)

    def _dispatch(
        self,
        grid: Sequence[Sequence[float]],
        activations: Sequence[ExpertActivation],
    ) -> Vector:
        output = [0.0] * self.config.d_model
        for activation in activations:
            expert_output = self._expert(activation.index, grid[activation.index])
            for channel, value in enumerate(expert_output):
                output[channel] += activation.weight * value
        return tuple(output)

    def _attention(
        self,
        query: Sequence[float],
        modalities: Tuple[Tuple[str, Vector], ...],
    ) -> Tuple[Vector, Tuple[Tuple[float, ...], ...]]:
        config = self.config
        head_size = config.d_model // config.num_heads
        output = [0.0] * config.d_model
        head_weights = []
        for head in range(config.num_heads):
            start = head * head_size
            end = start + head_size
            scores = []
            for _, embedding in modalities:
                score = sum(
                    query[index] * embedding[index] for index in range(start, end)
                )
                scores.append(score / math.sqrt(float(head_size)))
            weights = _softmax(scores)
            head_weights.append(weights)
            for index in range(start, end):
                output[index] = sum(
                    weight * embedding[index]
                    for weight, (_, embedding) in zip(weights, modalities)
                )
        return tuple(output), tuple(head_weights)

    def _neighbor_component_mean(self, position: int, channel: int) -> float:
        neighbors = self.config.grid.neighbors6(position)
        if not neighbors:
            return 0.0
        return sum(self._hidden[index][channel] for index in neighbors) / float(
            len(neighbors)
        )

    def _update_recurrence(
        self,
        grid: Sequence[Sequence[float]],
        activations: Sequence[ExpertActivation],
    ) -> None:
        config = self.config
        active = {activation.index: activation.weight for activation in activations}
        for position in range(config.grid.positions):
            if position not in active:
                hidden = self._hidden[position]
                cell = self._cell[position]
                for channel in range(config.d_model):
                    hidden[channel] *= config.state_decay
                    cell[channel] *= config.state_decay
                continue

            weight = active[position]
            previous_hidden = self._hidden[position][:]
            previous_cell = self._cell[position][:]
            for channel in range(config.d_model):
                x_value = grid[position][channel]
                neighbor = self._neighbor_component_mean(position, channel)
                h_value = previous_hidden[channel]
                gate_bias = _unit(config.seed ^ 0xC311, position, channel)
                input_gate = _sigmoid(
                    1.15 * x_value + 0.45 * h_value + 0.25 * neighbor + gate_bias
                )
                forget_gate = _sigmoid(
                    -0.35 * x_value + 0.85 * h_value + 0.20 * neighbor + 0.5
                )
                output_gate = _sigmoid(
                    0.65 * x_value + 0.55 * h_value + 0.15 * neighbor
                )
                candidate = math.tanh(x_value + 0.50 * neighbor + 0.20 * gate_bias)
                cell_value = (
                    forget_gate * previous_cell[channel]
                    + input_gate * candidate * weight
                )
                hidden_value = output_gate * math.tanh(cell_value)
                self._cell[position][channel] = cell_value
                self._hidden[position][channel] = hidden_value

    def _state_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(struct.pack(">Q", self._step))
        for matrix in (self._hidden, self._cell):
            for vector in matrix:
                for value in vector:
                    digest.update(struct.pack(">d", value))
        return digest.hexdigest()

    def arithmetic_report(self, modality_count: int) -> ArithmeticReport:
        config = self.config
        p = config.grid.positions
        d = config.d_model
        k = config.top_k
        rank = config.expert_rank
        grid_ops = p * d * max(1, modality_count) * 3
        router_ops = p * d * 2 + p * 12
        expert_ops = k * (2 * d * rank * 2 + d * 4)
        attention_ops = (
            config.num_heads * max(1, modality_count) * (2 * (d // config.num_heads))
        )
        recurrent_ops = k * d * 28 + (p - k) * d * 2
        total = grid_ops + router_ops + expert_ops + attention_ops + recurrent_ops
        return ArithmeticReport(
            positions=p,
            active_experts=k,
            activation_ratio=k / float(p),
            grid_projection_ops=grid_ops,
            router_ops=router_ops,
            expert_ops=expert_ops,
            attention_ops=attention_ops,
            recurrent_ops=recurrent_ops,
            total_scalar_ops=total,
            recurrent_state_values=2 * p * d,
        )

    def step(self, modalities: Mapping[str, ModalityInput]) -> ChrysalisResult:
        prepared = self._prepare_modalities(modalities)
        grid = self._grid_projection(prepared)
        activations = self._router(grid)
        moe_output = self._dispatch(grid, activations)
        attended, attention_weights = self._attention(moe_output, prepared)
        residual = self.config.output_residual
        output = _l2_normalize(
            [
                residual * moe_value + (1.0 - residual) * attended_value
                for moe_value, attended_value in zip(moe_output, attended)
            ]
        )
        self._update_recurrence(grid, activations)
        self._step += 1
        return ChrysalisResult(
            output=output,
            activations=tuple(activations),
            modality_attention=attention_weights,
            state_hash=self._state_hash(),
            step=self._step,
            arithmetic=self.arithmetic_report(len(prepared)),
        )

    def snapshot(self) -> Dict[str, object]:
        return {
            "step": self._step,
            "grid": {
                "width": self.config.grid.width,
                "height": self.config.grid.height,
                "depth": self.config.grid.depth,
            },
            "d_model": self.config.d_model,
            "top_k": self.config.top_k,
            "state_hash": self._state_hash(),
        }

    def snapshot_json(self) -> str:
        return json.dumps(self.snapshot(), sort_keys=True, separators=(",", ":"))
