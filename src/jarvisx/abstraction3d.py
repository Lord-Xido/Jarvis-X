"""Sparse 3D mathematical abstraction ANN core.

The core maps arbitrary vectors into a normalized feature manifold, routes each
observation to at most eight trilinear lattice nodes, performs sparse attention,
updates local prototypes and residual memory, and decodes an output vector.
All execution costs are proportional to feature width and routed nodes, not to
the full side**3 lattice.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

FEATURE_WIDTH = 16
DEFAULT_SIDE = 64
DEFAULT_ROUTE_LIMIT = 8
EPSILON = 1e-12

Number = Union[int, float]
Vector = Tuple[float, ...]
Coordinate3D = Tuple[int, int, int]


def _zeros(width: int = FEATURE_WIDTH) -> Vector:
    return (0.0,) * width


def _finite_vector(values: Sequence[Number], name: str) -> Vector:
    result = tuple(float(value) for value in values)
    if not result:
        raise ValueError(f"{name} must not be empty")
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain finite values")
    return result


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def _normalize(values: Sequence[float]) -> Vector:
    magnitude = _norm(values)
    if magnitude <= EPSILON:
        return tuple(0.0 for _ in values)
    return tuple(value / magnitude for value in values)


def _softmax(scores: Sequence[float]) -> Vector:
    if not scores:
        return ()
    maximum = max(scores)
    exponents = [math.exp(score - maximum) for score in scores]
    denominator = sum(exponents)
    return tuple(value / denominator for value in exponents)


def _basis(feature_index: int, source_index: int) -> float:
    phase = (feature_index + 1) * (source_index + 1) * 0.16180339887498948
    return math.sin(phase) + 0.5 * math.cos(phase * 0.5)


def _axis_basis(axis: int, feature_index: int) -> float:
    phase = (axis + 1) * (feature_index + 1) * 0.2718281828459045
    return math.cos(phase) + 0.25 * math.sin(phase * 1.5)


def _readout_basis(feature_index: int) -> float:
    phase = (feature_index + 1) * 0.6180339887498948
    return math.sin(phase) + 0.5 * math.cos(phase * 0.5)


def project_features(values: Sequence[Number], width: int = FEATURE_WIDTH) -> Vector:
    """Project any finite input vector into a normalized fixed-width feature vector."""
    source = _finite_vector(values, "input")
    scale = math.sqrt(float(len(source)))
    projected = []
    for feature_index in range(width):
        total = sum(
            value * _basis(feature_index, source_index)
            for source_index, value in enumerate(source)
        )
        projected.append(total / scale)
    return _normalize(projected)


def continuous_coordinate(features: Sequence[float], side: int = DEFAULT_SIDE) -> Vector:
    """Map normalized features onto a continuous coordinate in [0, side-1]^3."""
    if side < 2:
        raise ValueError("side must be at least 2")
    if len(features) != FEATURE_WIDTH:
        raise ValueError(f"features must contain {FEATURE_WIDTH} values")
    upper = float(side - 1)
    coordinates = []
    for axis in range(3):
        score = _dot(
            features,
            tuple(_axis_basis(axis, index) for index in range(FEATURE_WIDTH)),
        ) / math.sqrt(float(FEATURE_WIDTH))
        coordinates.append(0.5 * (math.tanh(score) + 1.0) * upper)
    return tuple(coordinates)


def trilinear_route(
    point: Sequence[float], side: int = DEFAULT_SIDE
) -> Tuple[Tuple[Coordinate3D, float], ...]:
    """Return at most eight lattice corners and normalized interpolation weights."""
    if len(point) != 3:
        raise ValueError("point must contain exactly three values")
    axes = []
    for value in point:
        value = max(0.0, min(float(side - 1), float(value)))
        lower = int(math.floor(value))
        upper = min(side - 1, lower + 1)
        fraction = value - lower
        axes.append(((lower, 1.0 - fraction), (upper, fraction)))
    merged: Dict[Coordinate3D, float] = {}
    for x, wx in axes[0]:
        for y, wy in axes[1]:
            for z, wz in axes[2]:
                weight = wx * wy * wz
                coordinate = (x, y, z)
                merged[coordinate] = merged.get(coordinate, 0.0) + weight
    positive = [
        (coordinate, weight)
        for coordinate, weight in merged.items()
        if weight > EPSILON
    ]
    total = sum(weight for _, weight in positive)
    return tuple(
        sorted(
            ((coordinate, weight / total) for coordinate, weight in positive),
            key=lambda item: item[0],
        )
    )


@dataclass
class AbstractionNode3D:
    prototype: Vector = field(default_factory=_zeros)
    memory: Vector = field(default_factory=_zeros)
    activation: float = 0.0
    confidence: float = 0.0
    visits: int = 0

    def clone(self) -> "AbstractionNode3D":
        return AbstractionNode3D(
            prototype=tuple(self.prototype),
            memory=tuple(self.memory),
            activation=self.activation,
            confidence=self.confidence,
            visits=self.visits,
        )


class SparseAbstractionLattice3D:
    def __init__(self, side: int = DEFAULT_SIDE, max_active_nodes: int = 100000) -> None:
        if side < 2:
            raise ValueError("side must be at least 2")
        if max_active_nodes <= 0:
            raise ValueError("max_active_nodes must be positive")
        self.side = int(side)
        self.max_active_nodes = int(max_active_nodes)
        self._nodes: Dict[Coordinate3D, AbstractionNode3D] = {}

    @property
    def theoretical_nodes(self) -> int:
        return self.side ** 3

    @property
    def active_nodes(self) -> int:
        return len(self._nodes)

    def peek(self, coordinate: Coordinate3D) -> Optional[AbstractionNode3D]:
        return self._nodes.get(coordinate)

    def touch(self, coordinate: Coordinate3D) -> AbstractionNode3D:
        if len(coordinate) != 3 or any(
            value < 0 or value >= self.side for value in coordinate
        ):
            raise ValueError("coordinate outside 3D lattice")
        node = self._nodes.get(coordinate)
        if node is None:
            if self.active_nodes >= self.max_active_nodes:
                raise MemoryError("3D abstraction active-node quota exceeded")
            node = AbstractionNode3D()
            self._nodes[coordinate] = node
        return node

    def remove(self, coordinate: Coordinate3D) -> None:
        self._nodes.pop(coordinate, None)


class Opcode3D(str, Enum):
    LOAD = "LOAD3D"
    ABSTRACT = "ABSTRACT3D"
    ROUTE = "ROUTE3D"
    ATTEND = "ATTEND3D"
    PREDICT = "PREDICT3D"
    COMPARE = "COMPARE3D"
    LEARN = "LEARN3D"
    PROJECT = "PROJECT3D"
    DECODE = "DECODE3D"
    HALT = "HALT3D"


@dataclass(frozen=True)
class Instruction3D:
    opcode: Opcode3D


@dataclass
class AbstractionSnapshot3D:
    dimensions: int
    feature_width: int
    side: int
    theoretical_nodes: int
    active_nodes: int
    route: Tuple[Tuple[Coordinate3D, float], ...]
    attention: Tuple[float, ...]
    prediction: float
    residual: float
    loss: float
    memory_norm: float
    output: Tuple[float, ...]
    cycles: int
    halted: bool


class AbstractionANNCore3D:
    """Deterministic sparse 3D abstraction and associative learning engine."""

    def __init__(
        self,
        side: int = DEFAULT_SIDE,
        feature_width: int = FEATURE_WIDTH,
        learning_rate: float = 0.12,
        memory_retention: float = 0.97,
        attention_temperature: float = 3.0,
        max_active_nodes: int = 100000,
        max_input_length: int = 4096,
        max_abs_input: float = 1000000.0,
    ) -> None:
        if feature_width != FEATURE_WIDTH:
            raise ValueError(f"feature_width must be {FEATURE_WIDTH}")
        if not 0.0 < learning_rate <= 1.0:
            raise ValueError("learning_rate must be inside (0, 1]")
        if not 0.0 <= memory_retention <= 1.0:
            raise ValueError("memory_retention must be inside [0, 1]")
        if attention_temperature <= 0.0:
            raise ValueError("attention_temperature must be positive")
        self.lattice = SparseAbstractionLattice3D(side, max_active_nodes)
        self.feature_width = feature_width
        self.learning_rate = float(learning_rate)
        self.memory_retention = float(memory_retention)
        self.attention_temperature = float(attention_temperature)
        self.max_input_length = int(max_input_length)
        self.max_abs_input = float(max_abs_input)
        self.registers: Dict[str, object] = {}
        self.cycles = 0
        self.halted = False
        self.reset_run_state()

    @staticmethod
    def default_program() -> Tuple[Instruction3D, ...]:
        return tuple(Instruction3D(opcode) for opcode in Opcode3D)

    def reset_run_state(self) -> None:
        self.registers = {
            "INPUT": (),
            "TARGET": 0.0,
            "FEATURES": None,
            "POINT": None,
            "ROUTE": (),
            "ATTENTION": (),
            "CONTEXT": _zeros(),
            "PREDICTION": 0.0,
            "RESIDUAL": 0.0,
            "LOSS": 0.0,
            "OUTPUT": (),
            "STAGE": 0,
        }
        self.cycles = 0
        self.halted = False

    def _require_stage(self, expected: int, operation: str) -> None:
        stage = int(self.registers["STAGE"])
        if stage != expected:
            raise RuntimeError(
                f"{operation} requires stage {expected}, current stage is {stage}"
            )

    def _validate_input(self, values: Sequence[Number]) -> Vector:
        source = _finite_vector(values, "input")
        if len(source) > self.max_input_length:
            raise ValueError("input exceeds configured length limit")
        if any(abs(value) > self.max_abs_input for value in source):
            raise ValueError("input value exceeds configured magnitude limit")
        return source

    def abstract(self, values: Sequence[Number]) -> Vector:
        return project_features(values, self.feature_width)

    def route(self, features: Sequence[float]) -> Tuple[Tuple[Coordinate3D, float], ...]:
        point = continuous_coordinate(features, self.lattice.side)
        return trilinear_route(point, self.lattice.side)

    def attend(
        self,
        features: Sequence[float],
        route: Sequence[Tuple[Coordinate3D, float]],
    ) -> Tuple[Vector, Vector]:
        if not route:
            raise RuntimeError("ROUTE3D must execute before ATTEND3D")
        scores: List[float] = []
        states: List[Vector] = []
        for coordinate, prior in route:
            node = self.lattice.touch(coordinate)
            state = tuple(
                node.prototype[index] + node.memory[index]
                for index in range(self.feature_width)
            )
            similarity = (
                _dot(features, _normalize(state)) if _norm(state) > EPSILON else 0.0
            )
            score = (
                self.attention_temperature * similarity
                + math.log(max(prior, EPSILON))
                + 0.25 * node.confidence
            )
            scores.append(score)
            states.append(state)
        attention = _softmax(scores)
        context = tuple(
            sum(
                attention[node_index] * states[node_index][feature_index]
                for node_index in range(len(states))
            )
            for feature_index in range(self.feature_width)
        )
        if _norm(context) <= EPSILON:
            context = tuple(features)
        return attention, _normalize(context)

    def predict(self, context: Sequence[float]) -> float:
        weights = tuple(_readout_basis(index) for index in range(self.feature_width))
        return math.tanh(
            _dot(context, weights) / math.sqrt(float(self.feature_width))
        )

    def learn(
        self,
        features: Sequence[float],
        route: Sequence[Tuple[Coordinate3D, float]],
        attention: Sequence[float],
        residual: float,
    ) -> None:
        if len(route) != len(attention):
            raise RuntimeError("route and attention lengths differ")
        error_direction = _normalize(
            tuple(_readout_basis(index) for index in range(self.feature_width))
        )
        for node_index, (coordinate, _) in enumerate(route):
            node = self.lattice.touch(coordinate)
            alpha = float(attention[node_index])
            rate = self.learning_rate * alpha
            candidate = tuple(
                (1.0 - rate) * node.prototype[index] + rate * features[index]
                for index in range(self.feature_width)
            )
            node.prototype = _normalize(candidate)
            node.memory = tuple(
                self.memory_retention * node.memory[index]
                + rate * residual * error_direction[index]
                for index in range(self.feature_width)
            )
            node.activation = min(1.0, max(0.0, 0.9 * node.activation + alpha))
            node.confidence = min(
                1.0,
                max(0.0, node.confidence + rate * (1.0 - abs(residual))),
            )
            node.visits += 1

    def project(self, route: Sequence[Tuple[Coordinate3D, float]]) -> None:
        for coordinate, _ in route:
            node = self.lattice.touch(coordinate)
            values = (
                *node.prototype,
                *node.memory,
                node.activation,
                node.confidence,
            )
            if not all(math.isfinite(value) for value in values):
                raise FloatingPointError("non-finite 3D abstraction state")
            node.prototype = _normalize(node.prototype)
            memory_norm = _norm(node.memory)
            if memory_norm > 4.0:
                node.memory = tuple(value * 4.0 / memory_norm for value in node.memory)
            node.activation = min(1.0, max(0.0, node.activation))
            node.confidence = min(1.0, max(0.0, node.confidence))

    def decode(self, context: Sequence[float], output_size: int) -> Tuple[float, ...]:
        if output_size <= 0:
            return ()
        output = []
        for source_index in range(output_size):
            reconstructed = sum(
                context[feature_index] * _basis(feature_index, source_index)
                for feature_index in range(self.feature_width)
            ) / math.sqrt(float(self.feature_width))
            output.append(math.tanh(reconstructed))
        return tuple(output)

    def _capture_route(self) -> Dict[Coordinate3D, Optional[AbstractionNode3D]]:
        route = self.registers.get("ROUTE", ())
        return {
            coordinate: (
                self.lattice.peek(coordinate).clone()
                if self.lattice.peek(coordinate) is not None
                else None
            )
            for coordinate, _ in route
        }

    def _rollback(
        self, captured: Dict[Coordinate3D, Optional[AbstractionNode3D]]
    ) -> None:
        for coordinate, node in captured.items():
            if node is None:
                self.lattice.remove(coordinate)
            else:
                self.lattice._nodes[coordinate] = node

    def execute(
        self,
        instruction: Instruction3D,
        input_vector: Optional[Sequence[Number]] = None,
        target: Optional[Number] = None,
    ) -> bool:
        registers_before = dict(self.registers)
        cycles_before = self.cycles
        halted_before = self.halted
        captured = self._capture_route()
        try:
            opcode = instruction.opcode
            if opcode == Opcode3D.LOAD:
                self._require_stage(0, "LOAD3D")
                if input_vector is None:
                    raise ValueError("LOAD3D requires an input vector")
                self.registers["INPUT"] = self._validate_input(input_vector)
                if target is not None:
                    target_value = float(target)
                    if not math.isfinite(target_value):
                        raise ValueError("target must be finite")
                    self.registers["TARGET"] = target_value
                self.registers["STAGE"] = 1

            elif opcode == Opcode3D.ABSTRACT:
                self._require_stage(1, "ABSTRACT3D")
                source = self.registers["INPUT"]
                if not source:
                    raise RuntimeError("LOAD3D must execute before ABSTRACT3D")
                features = self.abstract(source)
                self.registers["FEATURES"] = features
                self.registers["POINT"] = continuous_coordinate(
                    features, self.lattice.side
                )
                self.registers["STAGE"] = 2

            elif opcode == Opcode3D.ROUTE:
                self._require_stage(2, "ROUTE3D")
                features = self.registers["FEATURES"]
                if features is None:
                    raise RuntimeError("ABSTRACT3D must execute before ROUTE3D")
                self.registers["ROUTE"] = self.route(features)
                self.registers["STAGE"] = 3

            elif opcode == Opcode3D.ATTEND:
                self._require_stage(3, "ATTEND3D")
                features = self.registers["FEATURES"]
                if features is None:
                    raise RuntimeError("ABSTRACT3D must execute before ATTEND3D")
                route = self.registers["ROUTE"]
                attention, context = self.attend(features, route)
                self.registers["ATTENTION"] = attention
                self.registers["CONTEXT"] = context
                self.registers["STAGE"] = 4

            elif opcode == Opcode3D.PREDICT:
                self._require_stage(4, "PREDICT3D")
                context = self.registers["CONTEXT"]
                self.registers["PREDICTION"] = self.predict(context)
                self.registers["STAGE"] = 5

            elif opcode == Opcode3D.COMPARE:
                self._require_stage(5, "COMPARE3D")
                residual = float(self.registers["TARGET"]) - float(
                    self.registers["PREDICTION"]
                )
                self.registers["RESIDUAL"] = residual
                self.registers["LOSS"] = 0.5 * residual * residual
                self.registers["STAGE"] = 6

            elif opcode == Opcode3D.LEARN:
                self._require_stage(6, "LEARN3D")
                self.learn(
                    self.registers["FEATURES"],
                    self.registers["ROUTE"],
                    self.registers["ATTENTION"],
                    float(self.registers["RESIDUAL"]),
                )
                self.registers["STAGE"] = 7

            elif opcode == Opcode3D.PROJECT:
                self._require_stage(7, "PROJECT3D")
                self.project(self.registers["ROUTE"])
                self.registers["STAGE"] = 8

            elif opcode == Opcode3D.DECODE:
                self._require_stage(8, "DECODE3D")
                source = self.registers["INPUT"]
                self.registers["OUTPUT"] = self.decode(
                    self.registers["CONTEXT"], len(source)
                )
                self.registers["STAGE"] = 9

            elif opcode == Opcode3D.HALT:
                self._require_stage(9, "HALT3D")
                self.halted = True
                self.cycles += 1
                return False
            else:
                raise ValueError(f"unsupported 3D opcode: {opcode}")

            self.cycles += 1
            return True
        except Exception:
            self.registers = registers_before
            self.cycles = cycles_before
            self.halted = halted_before
            self._rollback(captured)
            raise

    def run(
        self,
        input_vector: Sequence[Number],
        target: Number = 0.0,
        program: Optional[Iterable[Instruction3D]] = None,
    ) -> AbstractionSnapshot3D:
        self.reset_run_state()
        selected = tuple(program) if program is not None else self.default_program()
        if not selected or selected[-1].opcode != Opcode3D.HALT:
            raise ValueError("3D abstraction program must terminate with HALT3D")
        for instruction in selected:
            if not self.execute(instruction, input_vector=input_vector, target=target):
                break
        return self.snapshot()

    def memory_norm(self) -> float:
        route = self.registers.get("ROUTE", ())
        if not route:
            return 0.0
        return sum(
            weight * _norm(self.lattice.touch(coordinate).memory)
            for coordinate, weight in route
        )

    def snapshot(self) -> AbstractionSnapshot3D:
        return AbstractionSnapshot3D(
            dimensions=3,
            feature_width=self.feature_width,
            side=self.lattice.side,
            theoretical_nodes=self.lattice.theoretical_nodes,
            active_nodes=self.lattice.active_nodes,
            route=tuple(self.registers["ROUTE"]),
            attention=tuple(self.registers["ATTENTION"]),
            prediction=float(self.registers["PREDICTION"]),
            residual=float(self.registers["RESIDUAL"]),
            loss=float(self.registers["LOSS"]),
            memory_norm=self.memory_norm(),
            output=tuple(self.registers["OUTPUT"]),
            cycles=self.cycles,
            halted=self.halted,
        )

    def state_hash(self) -> str:
        nodes = []
        for coordinate in sorted(self.lattice._nodes):
            node = self.lattice._nodes[coordinate]
            nodes.append(
                {
                    "coordinate": coordinate,
                    "prototype": node.prototype,
                    "memory": node.memory,
                    "activation": node.activation,
                    "confidence": node.confidence,
                    "visits": node.visits,
                }
            )
        payload = json.dumps(
            nodes, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
