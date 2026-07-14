"""Sparse 30-dimensional virtual ANN bytecode processor.

The runtime treats 30D as a virtual computational coordinate system. It never
allocates a dense ``side ** 30`` tensor. Only coordinates touched by bytecode
are materialized, so 30 independent state axes remain executable on ordinary
hardware.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

DIMENSIONS = 30
Q3_MIN = -4
Q3_MAX = 3
DEFAULT_SIDE = 8

Vector30 = Tuple[float, ...]
Latent30 = Tuple[int, ...]
Coordinate30 = Tuple[int, ...]
Number = Union[int, float]


def _require_length(values: Sequence[Number], expected: int, name: str) -> None:
    if len(values) != expected:
        raise ValueError("{} must contain exactly {} values".format(name, expected))


def quantize_q3(value: Number) -> int:
    """Round and clamp a scalar into the signed 3-bit set {-4, ..., 3}."""

    return max(Q3_MIN, min(Q3_MAX, int(round(float(value)))))


def quantize_vector30(values: Sequence[Number]) -> Latent30:
    """Quantize a 30-component vector into signed 3-bit latent symbols."""

    _require_length(values, DIMENSIONS, "values")
    return tuple(quantize_q3(value) for value in values)


def latent_to_coordinate(latent: Sequence[int], side: int = DEFAULT_SIDE) -> Coordinate30:
    """Map signed 3-bit symbols onto an addressable 30D coordinate."""

    _require_length(latent, DIMENSIONS, "latent")
    if side < DEFAULT_SIDE:
        raise ValueError("side must be at least 8 to represent all signed 3-bit symbols")
    return tuple((int(value) - Q3_MIN) % side for value in latent)


def coordinate_to_latent(coordinate: Sequence[int], side: int = DEFAULT_SIDE) -> Latent30:
    """Recover the canonical signed 3-bit symbol represented by a coordinate."""

    _require_length(coordinate, DIMENSIONS, "coordinate")
    if side < DEFAULT_SIDE:
        raise ValueError("side must be at least 8")
    return tuple(quantize_q3((int(value) % side) + Q3_MIN) for value in coordinate)


def flatten_coordinate(coordinate: Sequence[int], side: int = DEFAULT_SIDE) -> int:
    """Return the arbitrary-precision row-major address of a 30D coordinate."""

    _require_length(coordinate, DIMENSIONS, "coordinate")
    address = 0
    stride = 1
    for component in coordinate:
        component = int(component)
        if component < 0 or component >= side:
            raise ValueError("coordinate component {} is outside [0, {})".format(component, side))
        address += component * stride
        stride *= side
    return address


def _zero_vector30() -> Vector30:
    return (0.0,) * DIMENSIONS


@dataclass
class Cell30D:
    """Materialized neural and virtual-field state at one 30D coordinate."""

    activation: float = 0.0
    electric: Vector30 = field(default_factory=_zero_vector30)
    magnetic: Vector30 = field(default_factory=_zero_vector30)
    memory: float = 0.0
    prediction: float = 0.0
    residual: float = 0.0
    visits: int = 0

    def clone(self) -> "Cell30D":
        return Cell30D(
            activation=self.activation,
            electric=tuple(self.electric),
            magnetic=tuple(self.magnetic),
            memory=self.memory,
            prediction=self.prediction,
            residual=self.residual,
            visits=self.visits,
        )


class SparseField30D:
    """Sparse virtual lattice whose theoretical address space is ``side ** 30``."""

    def __init__(self, side: int = DEFAULT_SIDE) -> None:
        if side < DEFAULT_SIDE:
            raise ValueError("side must be at least 8")
        self.side = int(side)
        self._cells: Dict[Coordinate30, Cell30D] = {}

    @property
    def theoretical_cells(self) -> int:
        return self.side ** DIMENSIONS

    @property
    def active_cells(self) -> int:
        return len(self._cells)

    def _validate_coordinate(self, coordinate: Sequence[int]) -> Coordinate30:
        _require_length(coordinate, DIMENSIONS, "coordinate")
        normalized = tuple(int(value) for value in coordinate)
        if any(value < 0 or value >= self.side for value in normalized):
            raise ValueError("coordinate components must be inside the lattice")
        return normalized

    def peek(self, coordinate: Sequence[int]) -> Optional[Cell30D]:
        """Read without materializing an absent coordinate."""

        return self._cells.get(self._validate_coordinate(coordinate))

    def touch(self, coordinate: Sequence[int]) -> Cell30D:
        """Return a mutable cell, materializing it only when first accessed."""

        key = self._validate_coordinate(coordinate)
        if key not in self._cells:
            self._cells[key] = Cell30D()
        return self._cells[key]

    def deposit(self, coordinate: Sequence[int], value: Number) -> Cell30D:
        cell = self.touch(coordinate)
        cell.activation += float(value)
        cell.visits += 1
        return cell

    def axial_neighbor(self, coordinate: Coordinate30, axis: int, delta: int) -> Optional[Cell30D]:
        if axis < 0 or axis >= DIMENSIONS:
            raise ValueError("axis must be in [0, 30)")
        neighbor = list(coordinate)
        neighbor[axis] += int(delta)
        if neighbor[axis] < 0 or neighbor[axis] >= self.side:
            return None
        return self._cells.get(tuple(neighbor))

    def neighbor_activation_mean(self, coordinate: Coordinate30) -> float:
        total = 0.0
        count = 0
        for axis in range(DIMENSIONS):
            for delta in (-1, 1):
                neighbor = self.axial_neighbor(coordinate, axis, delta)
                if neighbor is not None:
                    total += neighbor.activation
                    count += 1
        return total / count if count else 0.0

    def snapshot(self) -> Mapping[Coordinate30, Cell30D]:
        return {coordinate: cell.clone() for coordinate, cell in self._cells.items()}


class Opcode30D(str, Enum):
    LOAD = "LOAD"
    ENCODE30 = "ENCODE30"
    PLACE30 = "PLACE30"
    FIELD30 = "FIELD30"
    PREDICT30 = "PREDICT30"
    COMPARE = "COMPARE"
    UPDATE_MEMORY = "UPDATE_MEMORY"
    PROJECT = "PROJECT"
    DECODE30 = "DECODE30"
    HALT = "HALT"


@dataclass(frozen=True)
class Instruction30D:
    opcode: Opcode30D
    operand: Optional[object] = None


@dataclass
class ProcessorSnapshot30D:
    dimensions: int
    side: int
    theoretical_cells: int
    active_cells: int
    coordinate: Optional[Coordinate30]
    address: Optional[int]
    latent: Optional[Latent30]
    prediction: float
    residual: float
    memory: float
    output: Tuple[float, ...]
    cycles: int
    halted: bool


class VirtualANNProcessor30D:
    """Deterministic sparse 30D ANN processor controlled by bytecode.

    Each materialized coordinate owns a scalar activation, two 30-component
    virtual field vectors, a local prediction residual, and persistent memory.
    """

    def __init__(
        self,
        side: int = DEFAULT_SIDE,
        learning_rate: float = 0.25,
        memory_retention: float = 0.98,
        field_dt: float = 0.1,
        field_damping: float = 0.97,
    ) -> None:
        self.field = SparseField30D(side=side)
        self.learning_rate = float(learning_rate)
        self.memory_retention = float(memory_retention)
        self.field_dt = float(field_dt)
        self.field_damping = float(field_damping)
        self.registers: Dict[str, object] = {
            "INPUT": (),
            "RAW_LATENT": None,
            "LATENT": None,
            "COORD": None,
            "PREDICTION": 0.0,
            "TARGET": 0.0,
            "ERROR": 0.0,
            "OUTPUT": (),
        }
        self.cycles = 0
        self.halted = False

    @staticmethod
    def default_program() -> Tuple[Instruction30D, ...]:
        return (
            Instruction30D(Opcode30D.LOAD),
            Instruction30D(Opcode30D.ENCODE30),
            Instruction30D(Opcode30D.PLACE30),
            Instruction30D(Opcode30D.FIELD30),
            Instruction30D(Opcode30D.PREDICT30),
            Instruction30D(Opcode30D.COMPARE),
            Instruction30D(Opcode30D.UPDATE_MEMORY),
            Instruction30D(Opcode30D.PROJECT),
            Instruction30D(Opcode30D.DECODE30),
            Instruction30D(Opcode30D.HALT),
        )

    @staticmethod
    def _projection_weight(dimension: int, input_index: int) -> float:
        phase = (dimension + 1) * (input_index + 1) * 0.17320508075688773
        return math.cos(phase) + 0.5 * math.sin(phase * 0.5)

    def encode(self, values: Sequence[Number]) -> Tuple[Vector30, Latent30]:
        if not values:
            raise ValueError("input vector must not be empty")
        source = tuple(float(value) for value in values)
        scale = math.sqrt(float(len(source)))
        raw = []
        for dimension in range(DIMENSIONS):
            projected = sum(
                value * self._projection_weight(dimension, index)
                for index, value in enumerate(source)
            ) / scale
            raw.append(projected)
        raw_latent = tuple(raw)
        return raw_latent, quantize_vector30(raw_latent)

    def _current_coordinate(self) -> Coordinate30:
        coordinate = self.registers["COORD"]
        if coordinate is None:
            raise RuntimeError("no active coordinate; execute ENCODE30 and PLACE30 first")
        return tuple(coordinate)  # type: ignore[arg-type]

    def _current_cell(self) -> Cell30D:
        return self.field.touch(self._current_coordinate())

    def field_step(self, coordinate: Coordinate30) -> None:
        """Advance coupled virtual electric/magnetic state across all 30 axes."""

        cell = self.field.touch(coordinate)
        old_e = cell.electric
        old_b = cell.magnetic
        new_e: List[float] = []
        new_b: List[float] = []

        for axis in range(DIMENSIONS):
            plus = self.field.axial_neighbor(coordinate, axis, 1)
            minus = self.field.axial_neighbor(coordinate, axis, -1)
            plus_a = plus.activation if plus is not None else 0.0
            minus_a = minus.activation if minus is not None else 0.0
            gradient = 0.5 * (plus_a - minus_a)
            injected_current = cell.activation / float(DIMENSIONS)
            electric = (
                self.field_damping * old_e[axis]
                + self.field_dt * (gradient + old_b[(axis + 1) % DIMENSIONS] - injected_current)
            )
            magnetic = (
                self.field_damping * old_b[axis]
                + self.field_dt * (old_e[(axis + 1) % DIMENSIONS] - old_e[axis - 1])
            )
            new_e.append(electric)
            new_b.append(magnetic)

        cell.electric = tuple(new_e)
        cell.magnetic = tuple(new_b)

    def predict(self, coordinate: Coordinate30) -> float:
        cell = self.field.touch(coordinate)
        mean_e = sum(cell.electric) / DIMENSIONS
        mean_b = sum(cell.magnetic) / DIMENSIONS
        neighbor_mean = self.field.neighbor_activation_mean(coordinate)
        preactivation = (
            0.35 * cell.activation
            + 0.15 * mean_e
            + 0.10 * mean_b
            + 0.40 * cell.memory
            + 0.10 * neighbor_mean
        )
        cell.prediction = math.tanh(preactivation)
        return cell.prediction

    def update_memory(self, coordinate: Coordinate30, residual: float) -> float:
        cell = self.field.touch(coordinate)
        cell.residual = float(residual)
        cell.memory = self.memory_retention * cell.memory + self.learning_rate * cell.residual
        return cell.memory

    def project(self, coordinate: Coordinate30) -> None:
        """Apply finite-state and bounded-energy invariants (the Lambda projector)."""

        cell = self.field.touch(coordinate)
        finite_values = (
            cell.activation,
            cell.memory,
            cell.prediction,
            cell.residual,
            *cell.electric,
            *cell.magnetic,
        )
        if not all(math.isfinite(value) for value in finite_values):
            raise FloatingPointError("non-finite 30D processor state")

        cell.activation = max(-4.0, min(3.0, cell.activation))
        cell.memory = max(-4.0, min(4.0, cell.memory))
        cell.electric = self._limit_norm(cell.electric, 8.0)
        cell.magnetic = self._limit_norm(cell.magnetic, 8.0)

    @staticmethod
    def _limit_norm(vector: Vector30, maximum: float) -> Vector30:
        norm = math.sqrt(sum(value * value for value in vector))
        if norm <= maximum or norm == 0.0:
            return vector
        scale = maximum / norm
        return tuple(value * scale for value in vector)

    def decode(self, latent: Latent30, output_size: int) -> Tuple[float, ...]:
        if output_size <= 0:
            return ()
        cell = self._current_cell()
        output = []
        for output_index in range(output_size):
            reconstructed = sum(
                latent[dimension] * self._projection_weight(dimension, output_index)
                for dimension in range(DIMENSIONS)
            ) / float(DIMENSIONS)
            output.append(math.tanh(reconstructed + 0.1 * cell.memory))
        return tuple(output)

    def execute(
        self,
        instruction: Instruction30D,
        input_vector: Optional[Sequence[Number]] = None,
        target: Optional[Number] = None,
    ) -> bool:
        opcode = instruction.opcode

        if opcode == Opcode30D.LOAD:
            source = instruction.operand if instruction.operand is not None else input_vector
            if source is None:
                raise ValueError("LOAD requires an input vector")
            self.registers["INPUT"] = tuple(float(value) for value in source)  # type: ignore[arg-type]
            if target is not None:
                self.registers["TARGET"] = float(target)

        elif opcode == Opcode30D.ENCODE30:
            raw, latent = self.encode(self.registers["INPUT"])  # type: ignore[arg-type]
            self.registers["RAW_LATENT"] = raw
            self.registers["LATENT"] = latent
            self.registers["COORD"] = latent_to_coordinate(latent, self.field.side)

        elif opcode == Opcode30D.PLACE30:
            latent = self.registers["LATENT"]
            if latent is None:
                raise RuntimeError("ENCODE30 must execute before PLACE30")
            amplitude = sum(abs(value) for value in latent) / float(DIMENSIONS)  # type: ignore[union-attr]
            self.field.deposit(self._current_coordinate(), amplitude or 1.0 / DIMENSIONS)

        elif opcode == Opcode30D.FIELD30:
            self.field_step(self._current_coordinate())

        elif opcode == Opcode30D.PREDICT30:
            self.registers["PREDICTION"] = self.predict(self._current_coordinate())

        elif opcode == Opcode30D.COMPARE:
            target_value = (
                float(instruction.operand)
                if instruction.operand is not None
                else float(self.registers["TARGET"])
            )
            self.registers["ERROR"] = target_value - float(self.registers["PREDICTION"])

        elif opcode == Opcode30D.UPDATE_MEMORY:
            self.update_memory(self._current_coordinate(), float(self.registers["ERROR"]))

        elif opcode == Opcode30D.PROJECT:
            self.project(self._current_coordinate())

        elif opcode == Opcode30D.DECODE30:
            latent = self.registers["LATENT"]
            if latent is None:
                raise RuntimeError("ENCODE30 must execute before DECODE30")
            output_size = len(self.registers["INPUT"])  # type: ignore[arg-type]
            self.registers["OUTPUT"] = self.decode(latent, output_size)  # type: ignore[arg-type]

        elif opcode == Opcode30D.HALT:
            self.halted = True
            self.cycles += 1
            return False

        else:
            raise ValueError("unsupported opcode: {}".format(opcode))

        self.cycles += 1
        return True

    def run(
        self,
        input_vector: Sequence[Number],
        target: Number = 0.0,
        program: Optional[Iterable[Instruction30D]] = None,
    ) -> ProcessorSnapshot30D:
        self.halted = False
        selected_program = tuple(program) if program is not None else self.default_program()
        for instruction in selected_program:
            if not self.execute(instruction, input_vector=input_vector, target=target):
                break
        return self.snapshot()

    def snapshot(self) -> ProcessorSnapshot30D:
        coordinate = self.registers["COORD"]
        cell = self.field.peek(coordinate) if coordinate is not None else None  # type: ignore[arg-type]
        return ProcessorSnapshot30D(
            dimensions=DIMENSIONS,
            side=self.field.side,
            theoretical_cells=self.field.theoretical_cells,
            active_cells=self.field.active_cells,
            coordinate=tuple(coordinate) if coordinate is not None else None,  # type: ignore[arg-type]
            address=flatten_coordinate(coordinate, self.field.side) if coordinate is not None else None,  # type: ignore[arg-type]
            latent=tuple(self.registers["LATENT"]) if self.registers["LATENT"] is not None else None,  # type: ignore[arg-type]
            prediction=float(self.registers["PREDICTION"]),
            residual=float(self.registers["ERROR"]),
            memory=cell.memory if cell is not None else 0.0,
            output=tuple(self.registers["OUTPUT"]),  # type: ignore[arg-type]
            cycles=self.cycles,
            halted=self.halted,
        )
