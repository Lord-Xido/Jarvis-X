"""Frontier research runtime for the Dr Moagi sparse 3D operating stack.

This module turns the "beyond SOTA" objective into measurable engineering
mechanisms rather than a declaration:

* Morton-ordered hierarchical sparse geometry for locality and multi-resolution
  occupancy accounting.
* An actual byte-producing entropy packet for quantized sparse fields.
* Anderson-accelerated implicit fixed-point solving over sparse 3D state.
* A rate-distortion-compute objective that chooses accelerated candidates only
  when they are no worse than a plain fixed-point baseline.
* A benchmark claim gate that refuses "SOTA" status without explicit external
  reference values and provenance.

The implementation is deliberately bounded and portable.  It does not claim to
outperform fVDB, learned point-cloud codecs, DEQ solvers, or coding agents until
a workload-matched benchmark demonstrates that result.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .dr_moagi_autoexec import (
    AutoExecPolicy,
    HashChainJournal,
    SparseBlockCodec3D,
    SparseParser3D,
)
from .dr_moagi_bitplane import fold_and_attenuate
from .dr_moagi_field_runtime import Coordinate, SparseField

FieldOperator = Callable[[Mapping[Coordinate, float]], SparseField]


def morton3_encode(x: int, y: int, z: int) -> int:
    """Interleave non-negative integer coordinate bits into a Morton code."""
    for name, value in (("x", x), ("y", y), ("z", z)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
    code = 0
    bit = 0
    a, b, c = x, y, z
    while a or b or c:
        code |= (a & 1) << (3 * bit)
        code |= (b & 1) << (3 * bit + 1)
        code |= (c & 1) << (3 * bit + 2)
        a >>= 1
        b >>= 1
        c >>= 1
        bit += 1
    return code


def morton3_decode(code: int) -> Coordinate:
    """Decode a Morton code back into a 3D integer coordinate."""
    if isinstance(code, bool) or not isinstance(code, int):
        raise TypeError("code must be an integer")
    if code < 0:
        raise ValueError("code must be non-negative")
    x = y = z = 0
    bit = 0
    value = code
    while value:
        x |= (value & 1) << bit
        y |= ((value >> 1) & 1) << bit
        z |= ((value >> 2) & 1) << bit
        value >>= 3
        bit += 1
    return x, y, z


@dataclass(frozen=True)
class HierarchicalSparseGrid3D:
    """Morton-ordered sparse field with hierarchical active-block accounting."""

    side: int
    leaves: tuple[tuple[int, float], ...]

    def __post_init__(self) -> None:
        if isinstance(self.side, bool) or not isinstance(self.side, int) or self.side <= 0:
            raise ValueError("side must be a positive integer")
        previous = -1
        for code, value in self.leaves:
            if isinstance(code, bool) or not isinstance(code, int) or code < 0:
                raise ValueError("Morton codes must be non-negative integers")
            if code <= previous:
                raise ValueError("Morton leaves must be unique and strictly ordered")
            coordinate = morton3_decode(code)
            if any(axis >= self.side for axis in coordinate):
                raise ValueError("Morton leaf lies outside logical lattice")
            if not math.isfinite(float(value)):
                raise ValueError("leaf values must be finite")
            previous = code

    @classmethod
    def from_field(
        cls,
        field: Mapping[Coordinate, float],
        *,
        side: int,
    ) -> "HierarchicalSparseGrid3D":
        leaves = []
        for coordinate, raw_value in field.items():
            x, y, z = coordinate
            if any(
                isinstance(axis, bool) or not isinstance(axis, int)
                for axis in (x, y, z)
            ):
                raise TypeError("coordinates must contain integers")
            if not (0 <= x < side and 0 <= y < side and 0 <= z < side):
                raise ValueError("coordinate outside logical lattice")
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError("field values must be finite")
            leaves.append((morton3_encode(x, y, z), value))
        leaves.sort(key=lambda item: item[0])
        return cls(side=side, leaves=tuple(leaves))

    @property
    def active_cells(self) -> int:
        return len(self.leaves)

    @property
    def levels(self) -> int:
        return max(1, (self.side - 1).bit_length())

    def to_field(self) -> SparseField:
        return {morton3_decode(code): float(value) for code, value in self.leaves}

    def active_blocks(self, level: int) -> int:
        """Return occupied blocks at ``2**level`` voxel edge length."""
        if isinstance(level, bool) or not isinstance(level, int):
            raise TypeError("level must be an integer")
        if not 0 <= level <= self.levels:
            raise ValueError("level outside hierarchy")
        if level == 0:
            return len(self.leaves)
        blocks = {
            (
                coordinate[0] >> level,
                coordinate[1] >> level,
                coordinate[2] >> level,
            )
            for coordinate in (morton3_decode(code) for code, _ in self.leaves)
        }
        return len(blocks)

    def occupancy_profile(self) -> tuple[int, ...]:
        return tuple(self.active_blocks(level) for level in range(self.levels + 1))

    def mean_morton_gap(self) -> float:
        if len(self.leaves) < 2:
            return 0.0
        gaps = [
            self.leaves[index][0] - self.leaves[index - 1][0]
            for index in range(1, len(self.leaves))
        ]
        return sum(gaps) / len(gaps)


def _encode_varint(value: int) -> bytes:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("varint value must be an integer")
    if value < 0:
        raise ValueError("varint value must be non-negative")
    output = bytearray()
    current = value
    while True:
        byte = current & 0x7F
        current >>= 7
        if current:
            output.append(byte | 0x80)
        else:
            output.append(byte)
            break
    return bytes(output)


def _decode_varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    cursor = offset
    while cursor < len(data):
        byte = data[cursor]
        cursor += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, cursor
        shift += 7
        if shift > 70:
            raise ValueError("varint is too large")
    raise ValueError("truncated varint")


def _zigzag_encode(value: int) -> int:
    return value * 2 if value >= 0 else (-value * 2) - 1


def _zigzag_decode(value: int) -> int:
    return value // 2 if value % 2 == 0 else -((value // 2) + 1)


@dataclass(frozen=True)
class EntropyPacket3D:
    """Actual compressed byte packet for a quantized sparse scalar field."""

    side: int
    quantization: float
    active_cells: int
    raw_float32_bytes: int
    payload: bytes
    checksum_sha256: str

    @property
    def encoded_bytes(self) -> int:
        return len(self.payload)

    @property
    def bits_per_active(self) -> float:
        return 0.0 if self.active_cells == 0 else (8.0 * self.encoded_bytes) / self.active_cells

    @property
    def compression_ratio(self) -> float:
        if self.encoded_bytes == 0:
            return math.inf
        return self.raw_float32_bytes / self.encoded_bytes

    def as_dict(self) -> dict[str, object]:
        return {
            "side": self.side,
            "quantization": self.quantization,
            "active_cells": self.active_cells,
            "raw_float32_bytes": self.raw_float32_bytes,
            "encoded_bytes": self.encoded_bytes,
            "bits_per_active": self.bits_per_active,
            "compression_ratio": self.compression_ratio,
            "checksum_sha256": self.checksum_sha256,
        }


class SparseEntropyCodec3D:
    """Morton-delta + zig-zag + DEFLATE reference entropy codec."""

    MAGIC = b"DMF1"

    def encode(
        self,
        field: Mapping[Coordinate, float],
        *,
        side: int,
        quantization: float,
    ) -> EntropyPacket3D:
        if isinstance(side, bool) or not isinstance(side, int) or side <= 0:
            raise ValueError("side must be a positive integer")
        if (
            isinstance(quantization, bool)
            or not isinstance(quantization, (int, float))
            or not math.isfinite(float(quantization))
            or quantization <= 0.0
        ):
            raise ValueError("quantization must be finite and positive")

        grid = HierarchicalSparseGrid3D.from_field(field, side=side)
        raw = bytearray(self.MAGIC)
        raw.extend(_encode_varint(side))
        raw.extend(struct.pack(">d", float(quantization)))
        raw.extend(_encode_varint(grid.active_cells))

        previous_code = 0
        first = True
        for code, value in grid.leaves:
            delta = code if first else code - previous_code
            first = False
            previous_code = code
            quantized = int(round(float(value) / float(quantization)))
            raw.extend(_encode_varint(delta))
            raw.extend(_encode_varint(_zigzag_encode(quantized)))

        payload = zlib.compress(bytes(raw), level=9)
        checksum = hashlib.sha256(payload).hexdigest()
        return EntropyPacket3D(
            side=side,
            quantization=float(quantization),
            active_cells=grid.active_cells,
            raw_float32_bytes=grid.active_cells * 16,
            payload=payload,
            checksum_sha256=checksum,
        )

    def decode(self, packet: EntropyPacket3D) -> SparseField:
        if hashlib.sha256(packet.payload).hexdigest() != packet.checksum_sha256:
            raise ValueError("entropy packet checksum mismatch")
        raw = zlib.decompress(packet.payload)
        if not raw.startswith(self.MAGIC):
            raise ValueError("invalid entropy packet magic")
        cursor = len(self.MAGIC)
        side, cursor = _decode_varint(raw, cursor)
        if side != packet.side:
            raise ValueError("entropy packet side mismatch")
        if cursor + 8 > len(raw):
            raise ValueError("truncated entropy packet")
        quantization = struct.unpack(">d", raw[cursor : cursor + 8])[0]
        cursor += 8
        if not math.isclose(quantization, packet.quantization, rel_tol=0.0, abs_tol=1.0e-15):
            raise ValueError("entropy packet quantization mismatch")
        count, cursor = _decode_varint(raw, cursor)

        field: SparseField = {}
        code = 0
        for index in range(count):
            delta, cursor = _decode_varint(raw, cursor)
            code = delta if index == 0 else code + delta
            encoded_value, cursor = _decode_varint(raw, cursor)
            quantized = _zigzag_decode(encoded_value)
            coordinate = morton3_decode(code)
            if any(axis >= side for axis in coordinate):
                raise ValueError("decoded coordinate outside logical lattice")
            field[coordinate] = quantized * quantization

        if cursor != len(raw):
            raise ValueError("entropy packet contains trailing bytes")
        if len(field) != packet.active_cells:
            raise ValueError("entropy packet active-cell count mismatch")
        return field


@dataclass(frozen=True)
class SolverConfig:
    tolerance: float = 1.0e-6
    max_iterations: int = 32
    depth: int = 4
    damping: float = 1.0
    regularization: float = 1.0e-8
    coefficient_clip: float = 8.0
    prune_epsilon: float = 0.0

    def __post_init__(self) -> None:
        if self.tolerance < 0.0 or not math.isfinite(self.tolerance):
            raise ValueError("tolerance must be finite and non-negative")
        if isinstance(self.max_iterations, bool) or not isinstance(self.max_iterations, int):
            raise TypeError("max_iterations must be an integer")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if isinstance(self.depth, bool) or not isinstance(self.depth, int):
            raise TypeError("depth must be an integer")
        if not 1 <= self.depth <= 16:
            raise ValueError("depth must be in [1, 16]")
        for name in ("damping", "regularization", "coefficient_clip", "prune_epsilon"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0.0 < self.damping <= 1.0:
            raise ValueError("damping must be in (0, 1]")
        if self.regularization <= 0.0:
            raise ValueError("regularization must be positive")
        if self.coefficient_clip <= 0.0:
            raise ValueError("coefficient_clip must be positive")
        if self.prune_epsilon < 0.0:
            raise ValueError("prune_epsilon must be non-negative")


@dataclass(frozen=True)
class EquilibriumResult:
    state: SparseField
    converged: bool
    iterations: int
    residual: float
    residual_history: tuple[float, ...]


def _rms_difference(
    left: Mapping[Coordinate, float],
    right: Mapping[Coordinate, float],
) -> float:
    support = set(left) | set(right)
    if not support:
        return 0.0
    mse = sum(
        (float(left.get(coordinate, 0.0)) - float(right.get(coordinate, 0.0))) ** 2
        for coordinate in support
    ) / len(support)
    return math.sqrt(mse)


def _difference(
    left: Mapping[Coordinate, float],
    right: Mapping[Coordinate, float],
) -> SparseField:
    result: SparseField = {}
    for coordinate in set(left) | set(right):
        value = float(left.get(coordinate, 0.0)) - float(right.get(coordinate, 0.0))
        if value != 0.0:
            result[coordinate] = value
    return result


def _linear_combination(
    fields: Sequence[Mapping[Coordinate, float]],
    coefficients: Sequence[float],
    *,
    prune_epsilon: float,
) -> SparseField:
    result: SparseField = {}
    support: set[Coordinate] = set()
    for field_value in fields:
        support.update(field_value)
    for coordinate in support:
        value = sum(
            float(coefficient) * float(field_value.get(coordinate, 0.0))
            for field_value, coefficient in zip(fields, coefficients)
        )
        if abs(value) > prune_epsilon:
            result[coordinate] = value
    return result


def _solve_linear_system(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    size = len(rhs)
    augmented = [list(matrix[row]) + [rhs[row]] for row in range(size)]
    for pivot in range(size):
        best = max(range(pivot, size), key=lambda row: abs(augmented[row][pivot]))
        if abs(augmented[best][pivot]) < 1.0e-14:
            raise ValueError("singular linear system")
        if best != pivot:
            augmented[pivot], augmented[best] = augmented[best], augmented[pivot]
        scale = augmented[pivot][pivot]
        augmented[pivot] = [value / scale for value in augmented[pivot]]
        for row in range(size):
            if row == pivot:
                continue
            factor = augmented[row][pivot]
            if factor == 0.0:
                continue
            augmented[row] = [
                augmented[row][column] - factor * augmented[pivot][column]
                for column in range(size + 1)
            ]
    return [augmented[row][-1] for row in range(size)]


class SparseAndersonSolver3D:
    """Small-memory Anderson acceleration over sparse field fixed points."""

    def __init__(self, config: SolverConfig | None = None) -> None:
        self.config = config or SolverConfig()

    def solve(self, operator: FieldOperator, initial: Mapping[Coordinate, float]) -> EquilibriumResult:
        current: SparseField = {coordinate: float(value) for coordinate, value in initial.items()}
        xs: list[SparseField] = []
        fs: list[SparseField] = []
        residuals: list[SparseField] = []
        history: list[float] = []

        for iteration in range(1, self.config.max_iterations + 1):
            mapped = operator(current)
            residual = _rms_difference(mapped, current)
            history.append(residual)
            if residual <= self.config.tolerance:
                return EquilibriumResult(
                    state=dict(mapped),
                    converged=True,
                    iterations=iteration,
                    residual=residual,
                    residual_history=tuple(history),
                )

            xs.append(dict(current))
            fs.append(dict(mapped))
            residuals.append(_difference(mapped, current))
            xs = xs[-self.config.depth :]
            fs = fs[-self.config.depth :]
            residuals = residuals[-self.config.depth :]

            if len(fs) < 2:
                current = self._damped(mapped, current)
                continue

            try:
                coefficients = self._coefficients(residuals)
                mixed = _linear_combination(
                    fs,
                    coefficients,
                    prune_epsilon=self.config.prune_epsilon,
                )
                current = self._damped(mixed, mapped)
            except ValueError:
                current = self._damped(mapped, current)

        final_mapped = operator(current)
        final_residual = _rms_difference(final_mapped, current)
        return EquilibriumResult(
            state=dict(current),
            converged=final_residual <= self.config.tolerance,
            iterations=self.config.max_iterations,
            residual=final_residual,
            residual_history=tuple(history),
        )

    def solve_plain(
        self,
        operator: FieldOperator,
        initial: Mapping[Coordinate, float],
    ) -> EquilibriumResult:
        current: SparseField = {coordinate: float(value) for coordinate, value in initial.items()}
        history: list[float] = []
        for iteration in range(1, self.config.max_iterations + 1):
            mapped = operator(current)
            residual = _rms_difference(mapped, current)
            history.append(residual)
            if residual <= self.config.tolerance:
                return EquilibriumResult(
                    state=dict(mapped),
                    converged=True,
                    iterations=iteration,
                    residual=residual,
                    residual_history=tuple(history),
                )
            current = self._damped(mapped, current)

        final_mapped = operator(current)
        final_residual = _rms_difference(final_mapped, current)
        return EquilibriumResult(
            state=dict(current),
            converged=final_residual <= self.config.tolerance,
            iterations=self.config.max_iterations,
            residual=final_residual,
            residual_history=tuple(history),
        )

    def _damped(
        self,
        target: Mapping[Coordinate, float],
        reference: Mapping[Coordinate, float],
    ) -> SparseField:
        beta = self.config.damping
        return _linear_combination(
            (target, reference),
            (beta, 1.0 - beta),
            prune_epsilon=self.config.prune_epsilon,
        )

    def _coefficients(self, residuals: Sequence[Mapping[Coordinate, float]]) -> list[float]:
        count = len(residuals)
        gram = [[0.0 for _ in range(count)] for _ in range(count)]
        supports = [set(residual) for residual in residuals]
        for row in range(count):
            for column in range(row, count):
                support = supports[row] | supports[column]
                value = sum(
                    float(residuals[row].get(coordinate, 0.0))
                    * float(residuals[column].get(coordinate, 0.0))
                    for coordinate in support
                )
                if row == column:
                    value += self.config.regularization
                gram[row][column] = value
                gram[column][row] = value

        size = count + 1
        matrix = [[0.0 for _ in range(size)] for _ in range(size)]
        rhs = [0.0 for _ in range(size)]
        rhs[-1] = 1.0
        for row in range(count):
            for column in range(count):
                matrix[row][column] = gram[row][column]
            matrix[row][-1] = 1.0
            matrix[-1][row] = 1.0

        solution = _solve_linear_system(matrix, rhs)
        raw = solution[:count]
        clipped = [
            max(-self.config.coefficient_clip, min(self.config.coefficient_clip, value))
            for value in raw
        ]
        total = sum(clipped)
        if abs(total) < 1.0e-12:
            raise ValueError("degenerate Anderson coefficients")
        return [value / total for value in clipped]


@dataclass(frozen=True)
class FrontierConfig:
    side: int = 64
    max_active_cells: int = 50_000
    policy: AutoExecPolicy = field(default_factory=AutoExecPolicy)
    contraction: float = 0.08
    attenuation: float = 0.10
    equilibrium_tolerance: float = 1.0e-6
    max_iterations: int = 32
    anderson_depth: int = 4
    anderson_damping: float = 1.0
    fixed_point_gain: float = 0.5
    rate_weight: float = 0.15
    distortion_weight: float = 0.70
    compute_weight: float = 0.15
    prune_epsilon: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.side, bool) or not isinstance(self.side, int) or self.side <= 0:
            raise ValueError("side must be a positive integer")
        if (
            isinstance(self.max_active_cells, bool)
            or not isinstance(self.max_active_cells, int)
            or self.max_active_cells <= 0
        ):
            raise ValueError("max_active_cells must be positive")
        if not 0.0 <= self.contraction < 1.0:
            raise ValueError("contraction must be in [0, 1)")
        if self.attenuation < 0.0:
            raise ValueError("attenuation must be non-negative")
        if not 0.0 < self.fixed_point_gain <= 1.0:
            raise ValueError("fixed_point_gain must be in (0, 1]")
        for name in (
            "equilibrium_tolerance",
            "anderson_damping",
            "rate_weight",
            "distortion_weight",
            "compute_weight",
            "prune_epsilon",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.equilibrium_tolerance < 0.0:
            raise ValueError("equilibrium_tolerance must be non-negative")
        if not 0.0 < self.anderson_damping <= 1.0:
            raise ValueError("anderson_damping must be in (0, 1]")
        if min(self.rate_weight, self.distortion_weight, self.compute_weight) < 0.0:
            raise ValueError("objective weights must be non-negative")
        if self.rate_weight + self.distortion_weight + self.compute_weight <= 0.0:
            raise ValueError("at least one objective weight must be positive")
        if self.prune_epsilon < 0.0:
            raise ValueError("prune_epsilon must be non-negative")


class FrontierOperator3D:
    """Pure sparse inward encode/decode operator suitable for implicit solving."""

    def __init__(self, config: FrontierConfig) -> None:
        self.config = config
        self.codec = SparseBlockCodec3D(config.policy)

    def __call__(self, field: Mapping[Coordinate, float]) -> SparseField:
        folded = fold_and_attenuate(
            field,
            side=self.config.side,
            contraction=self.config.contraction,
            attenuation=self.config.attenuation,
            prune_epsilon=self.config.prune_epsilon,
        )
        support = tuple(sorted(set(field) | set(folded)))
        latent = self.codec.encode(folded)
        decoded = self.codec.decode(latent, support)
        gain = self.config.fixed_point_gain
        result: SparseField = {}
        for coordinate in support:
            current = float(field.get(coordinate, 0.0))
            target = float(decoded.get(coordinate, 0.0))
            value = current + gain * (target - current)
            value = max(-1.0, min(1.0, value))
            if abs(value) > self.config.prune_epsilon:
                result[coordinate] = value
        return result


@dataclass(frozen=True)
class FrontierCycleReport:
    cycle: int
    committed: bool
    active_cells_before: int
    active_cells_after: int
    hierarchy_profile: tuple[int, ...]
    entropy_bytes: int
    bits_per_active: float
    distortion_mse: float
    plain_iterations: int
    accelerated_iterations: int
    iteration_speedup: float
    plain_residual: float
    accelerated_residual: float
    plain_objective: float
    accelerated_objective: float
    selected_objective: float
    selected_solver: str
    state_hash: str
    journal_hash: str
    rejection_reason: str | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class DrMoagiFrontierRuntime:
    """Transactional frontier runtime with empirical baseline selection."""

    def __init__(
        self,
        config: FrontierConfig | None = None,
        *,
        journal_path: str | Path | None = None,
    ) -> None:
        self.config = config or FrontierConfig()
        self.parser = SparseParser3D(
            side=self.config.side,
            max_active_cells=self.config.max_active_cells,
            value_min=-1.0,
            value_max=1.0,
            prune_epsilon=self.config.prune_epsilon,
        )
        solver_config = SolverConfig(
            tolerance=self.config.equilibrium_tolerance,
            max_iterations=self.config.max_iterations,
            depth=self.config.anderson_depth,
            damping=self.config.anderson_damping,
            prune_epsilon=self.config.prune_epsilon,
        )
        self.solver = SparseAndersonSolver3D(solver_config)
        self.operator = FrontierOperator3D(self.config)
        self.entropy_codec = SparseEntropyCodec3D()
        self.journal = HashChainJournal(journal_path)
        self._state: SparseField = {}
        self._loaded = False
        self._cycle = 0
        self.reports: list[FrontierCycleReport] = []

    def load(self, source: Mapping[Coordinate, float]) -> SparseField:
        self._state = self.parser.parse(source)
        self._cycle = 0
        self._loaded = True
        self.reports.clear()
        return self.snapshot()

    def snapshot(self) -> SparseField:
        self._require_loaded()
        return dict(self._state)

    def step(self) -> FrontierCycleReport:
        self._require_loaded()
        before = dict(self._state)
        plain = self.solver.solve_plain(self.operator, before)
        accelerated = self.solver.solve(self.operator, before)

        plain_packet = self.entropy_codec.encode(
            plain.state,
            side=self.config.side,
            quantization=self.config.policy.quantization,
        )
        accelerated_packet = self.entropy_codec.encode(
            accelerated.state,
            side=self.config.side,
            quantization=self.config.policy.quantization,
        )

        plain_distortion = self._mse(before, plain.state)
        accelerated_distortion = self._mse(before, accelerated.state)
        plain_objective = self._objective(
            distortion=plain_distortion,
            encoded_bytes=plain_packet.encoded_bytes,
            active_cells=max(1, len(plain.state)),
            iterations=plain.iterations,
        )
        accelerated_objective = self._objective(
            distortion=accelerated_distortion,
            encoded_bytes=accelerated_packet.encoded_bytes,
            active_cells=max(1, len(accelerated.state)),
            iterations=accelerated.iterations,
        )

        if accelerated_objective <= plain_objective:
            selected_name = "anderson"
            selected = accelerated
            selected_packet = accelerated_packet
            selected_distortion = accelerated_distortion
            selected_objective = accelerated_objective
        else:
            selected_name = "plain"
            selected = plain
            selected_packet = plain_packet
            selected_distortion = plain_distortion
            selected_objective = plain_objective

        rejection_reason = self._validate_candidate(selected.state)
        committed = rejection_reason is None
        if committed:
            self._state = dict(selected.state)
            self._cycle += 1

        hierarchy = HierarchicalSparseGrid3D.from_field(
            self._state,
            side=self.config.side,
        )
        speedup = plain.iterations / max(1, accelerated.iterations)
        provisional = FrontierCycleReport(
            cycle=self._cycle,
            committed=committed,
            active_cells_before=len(before),
            active_cells_after=len(self._state),
            hierarchy_profile=hierarchy.occupancy_profile(),
            entropy_bytes=selected_packet.encoded_bytes,
            bits_per_active=selected_packet.bits_per_active,
            distortion_mse=selected_distortion,
            plain_iterations=plain.iterations,
            accelerated_iterations=accelerated.iterations,
            iteration_speedup=speedup,
            plain_residual=plain.residual,
            accelerated_residual=accelerated.residual,
            plain_objective=plain_objective,
            accelerated_objective=accelerated_objective,
            selected_objective=selected_objective,
            selected_solver=selected_name,
            state_hash=self._state_hash(self._state),
            journal_hash="",
            rejection_reason=rejection_reason,
        )
        record = provisional.as_dict()
        record.pop("journal_hash", None)
        journal_hash = self.journal.append(record)
        report = FrontierCycleReport(
            **{**provisional.__dict__, "journal_hash": journal_hash}
        )
        self.reports.append(report)
        return report

    def run(self, cycles: int) -> tuple[FrontierCycleReport, ...]:
        if isinstance(cycles, bool) or not isinstance(cycles, int) or cycles <= 0:
            raise ValueError("cycles must be a positive integer")
        for _ in range(cycles):
            report = self.step()
            if not report.committed:
                break
            if (
                report.accelerated_residual <= self.config.equilibrium_tolerance
                and report.selected_solver == "anderson"
            ):
                break
        return tuple(self.reports)

    def status(self) -> dict[str, object]:
        self._require_loaded()
        grid = HierarchicalSparseGrid3D.from_field(self._state, side=self.config.side)
        packet = self.entropy_codec.encode(
            self._state,
            side=self.config.side,
            quantization=self.config.policy.quantization,
        )
        latest = self.reports[-1] if self.reports else None
        return {
            "system": "Dr Moagi Frontier Runtime",
            "claim_status": "frontier-candidate-until-external-benchmark-gate-passes",
            "cycle": self._cycle,
            "active_cells": len(self._state),
            "logical_cells": self.config.side**3,
            "hierarchy_profile": list(grid.occupancy_profile()),
            "entropy": packet.as_dict(),
            "journal_valid": self.journal.verify(),
            "state_hash": self._state_hash(self._state),
            "last_report": latest.as_dict() if latest else None,
        }

    def _objective(
        self,
        *,
        distortion: float,
        encoded_bytes: int,
        active_cells: int,
        iterations: int,
    ) -> float:
        rate_term = encoded_bytes / max(1.0, active_cells * 16.0)
        compute_term = iterations / max(1.0, float(self.config.max_iterations))
        total_weight = (
            self.config.distortion_weight
            + self.config.rate_weight
            + self.config.compute_weight
        )
        return (
            self.config.distortion_weight * distortion
            + self.config.rate_weight * rate_term
            + self.config.compute_weight * compute_term
        ) / total_weight

    def _validate_candidate(self, candidate: Mapping[Coordinate, float]) -> str | None:
        if len(candidate) > self.config.max_active_cells:
            return "active-cell budget exceeded"
        if not candidate and self._state:
            return "candidate removed all active state"
        for coordinate, raw_value in candidate.items():
            if any(axis < 0 or axis >= self.config.side for axis in coordinate):
                return "candidate coordinate outside logical lattice"
            if not math.isfinite(float(raw_value)):
                return "candidate contains non-finite value"
        return None

    @staticmethod
    def _mse(
        left: Mapping[Coordinate, float],
        right: Mapping[Coordinate, float],
    ) -> float:
        support = set(left) | set(right)
        if not support:
            return 0.0
        return sum(
            (float(left.get(coordinate, 0.0)) - float(right.get(coordinate, 0.0))) ** 2
            for coordinate in support
        ) / len(support)

    @staticmethod
    def _state_hash(field: Mapping[Coordinate, float]) -> str:
        canonical = [
            [x, y, z, float(field[(x, y, z)])]
            for x, y, z in sorted(field)
        ]
        payload = json.dumps(
            canonical,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _require_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("load a sparse 3D field before execution")


@dataclass(frozen=True)
class BenchmarkEvidence:
    """One workload-matched external comparison used by the claim gate."""

    metric: str
    candidate_value: float
    reference_value: float
    higher_is_better: bool
    minimum_relative_gain: float = 0.0
    source: str = ""

    def __post_init__(self) -> None:
        for name in ("candidate_value", "reference_value", "minimum_relative_gain"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.minimum_relative_gain < 0.0:
            raise ValueError("minimum_relative_gain must be non-negative")


@dataclass(frozen=True)
class ClaimGateResult:
    passed: bool
    checks: tuple[tuple[str, bool, float], ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "checks": [
                {"metric": metric, "passed": passed, "relative_gain": gain}
                for metric, passed, gain in self.checks
            ],
        }


class SOTAClaimGate:
    """Refuse a SOTA claim unless every supplied benchmark beats its reference."""

    def evaluate(self, evidence: Sequence[BenchmarkEvidence]) -> ClaimGateResult:
        if not evidence:
            return ClaimGateResult(passed=False, checks=())
        checks: list[tuple[str, bool, float]] = []
        for item in evidence:
            if not item.source.strip():
                checks.append((item.metric, False, 0.0))
                continue
            denominator = max(abs(item.reference_value), 1.0e-12)
            if item.higher_is_better:
                gain = (item.candidate_value - item.reference_value) / denominator
            else:
                gain = (item.reference_value - item.candidate_value) / denominator
            passed = gain >= item.minimum_relative_gain
            checks.append((item.metric, passed, gain))
        return ClaimGateResult(
            passed=all(passed for _, passed, _ in checks),
            checks=tuple(checks),
        )
