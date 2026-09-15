"""Q16.16x3 field-engine v2: numeric policy, packed storage, and field calculus.

This module strengthens the reference QVector architecture without changing the v1
wire format.  It adds explicit fixed-point rounding/status semantics, checked
signed-64 accumulation, a compact 12-byte-per-cell data plane, and deterministic
3D differential/convolution operators.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Iterator, Sequence

from .qvector3d import (
    Q_ONE,
    Q32_MAX,
    Q32_MIN,
    QVector3Q16,
    QVectorField3D,
    q16_from_float,
)

ACC64_MIN = -(1 << 63)
ACC64_MAX = (1 << 63) - 1
VECTOR_CELL_BYTES = 12


class QRoundMode(IntEnum):
    """Architectural rounding mode for fixed-point requantization."""

    TRUNCATE = 0
    NEAREST_AWAY = 1
    NEAREST_EVEN = 2


class QBoundaryMode(IntEnum):
    """Boundary policy for neighborhood field operations."""

    CLAMP = 0
    ZERO = 1
    WRAP = 2


@dataclass(frozen=True, slots=True)
class QNumericPolicy:
    """Numeric control register contract for the reference field engine."""

    rounding: QRoundMode = QRoundMode.NEAREST_EVEN
    saturate: bool = True
    accumulator_saturate: bool = True


@dataclass(slots=True)
class QArithmeticStatus:
    """Sticky arithmetic status flags, analogous to a fixed-point CSR."""

    saturated: bool = False
    accumulator_saturated: bool = False
    inexact: bool = False
    divide_by_zero: bool = False

    def clear(self) -> None:
        self.saturated = False
        self.accumulator_saturated = False
        self.inexact = False
        self.divide_by_zero = False


@dataclass(slots=True)
class QAccumulator64:
    """Checked signed-64 accumulator used by MAC/convolution operations."""

    value: int = 0

    def add(self, term: int, policy: QNumericPolicy, status: QArithmeticStatus) -> None:
        candidate = self.value + int(term)
        if ACC64_MIN <= candidate <= ACC64_MAX:
            self.value = candidate
            return
        status.accumulator_saturated = True
        if not policy.accumulator_saturate:
            raise OverflowError("signed 64-bit fixed-point accumulator overflow")
        self.value = ACC64_MAX if candidate > ACC64_MAX else ACC64_MIN

    def add_product(
        self,
        left_q16: int,
        right_q16: int,
        policy: QNumericPolicy,
        status: QArithmeticStatus,
    ) -> None:
        self.add(int(left_q16) * int(right_q16), policy, status)


def _round_div(
    numerator: int,
    denominator: int,
    mode: QRoundMode,
    status: QArithmeticStatus,
) -> int:
    if denominator == 0:
        status.divide_by_zero = True
        raise ZeroDivisionError("fixed-point division by zero")
    sign = -1 if (numerator < 0) ^ (denominator < 0) else 1
    n = abs(int(numerator))
    d = abs(int(denominator))
    q, r = divmod(n, d)
    if r:
        status.inexact = True
    if mode == QRoundMode.TRUNCATE:
        return sign * q
    twice = r * 2
    if mode == QRoundMode.NEAREST_AWAY:
        if twice >= d:
            q += 1
    elif mode == QRoundMode.NEAREST_EVEN:
        if twice > d or (twice == d and q & 1):
            q += 1
    else:  # pragma: no cover - IntEnum validation prevents this in normal use
        raise ValueError(f"unsupported Q16.16 rounding mode {mode!r}")
    return sign * q


def _q32(value: int, policy: QNumericPolicy, status: QArithmeticStatus) -> int:
    if Q32_MIN <= value <= Q32_MAX:
        return int(value)
    status.saturated = True
    if not policy.saturate:
        raise OverflowError("Q16.16 result does not fit signed 32 bits")
    return Q32_MAX if value > Q32_MAX else Q32_MIN


def requantize_product(
    accumulator_q32: int,
    policy: QNumericPolicy,
    status: QArithmeticStatus,
) -> int:
    """Convert a Q32.32 accumulator into one Q16.16 result exactly once."""

    raw = _round_div(accumulator_q32, Q_ONE, policy.rounding, status)
    return _q32(raw, policy, status)


def divide_q16_raw(
    numerator_q16: int,
    denominator_q16: int,
    policy: QNumericPolicy,
    status: QArithmeticStatus,
) -> int:
    """Q16.16 / Q16.16 -> Q16.16 using the architectural rounding policy."""

    raw = _round_div(int(numerator_q16) * Q_ONE, denominator_q16, policy.rounding, status)
    return _q32(raw, policy, status)


@dataclass(frozen=True, slots=True)
class QScalarKernel3D:
    """Odd-sized scalar Q16.16 kernel applied independently to x/y/z lanes."""

    weights_q16: tuple[int, ...]
    shape: tuple[int, int, int]

    def __post_init__(self) -> None:
        sx, sy, sz = self.shape
        if min(self.shape) < 1 or any(axis % 2 == 0 for axis in self.shape):
            raise ValueError("kernel dimensions must be positive odd integers")
        if len(self.weights_q16) != sx * sy * sz:
            raise ValueError("kernel weight count does not match kernel volume")
        for weight in self.weights_q16:
            if weight < Q32_MIN or weight > Q32_MAX:
                raise OverflowError("kernel weight does not fit signed Q16.16")

    @classmethod
    def from_floats(
        cls,
        weights: Sequence[float],
        shape: Sequence[int],
    ) -> "QScalarKernel3D":
        shape3 = (int(shape[0]), int(shape[1]), int(shape[2]))
        return cls(tuple(q16_from_float(float(value)) for value in weights), shape3)

    @classmethod
    def identity(cls) -> "QScalarKernel3D":
        return cls((Q_ONE,), (1, 1, 1))

    @property
    def center(self) -> tuple[int, int, int]:
        return tuple(axis // 2 for axis in self.shape)  # type: ignore[return-value]


class PackedQVectorField3D:
    """Compact mutable data plane with exactly 12 bytes per vector cell.

    The representation is a contiguous big-endian bytearray.  This avoids one
    Python object per cell while preserving the v1 binary contract and digest.
    """

    __slots__ = ("shape", "_buffer")

    def __init__(self, shape: Sequence[int], raw: bytes | bytearray | None = None) -> None:
        self.shape = (int(shape[0]), int(shape[1]), int(shape[2]))
        if min(self.shape) < 1:
            raise ValueError("packed field dimensions must be positive")
        expected = self.cells * VECTOR_CELL_BYTES
        if raw is None:
            self._buffer = bytearray(expected)
        else:
            if len(raw) != expected:
                raise ValueError(f"packed field contains {len(raw)} bytes; expected {expected}")
            self._buffer = bytearray(raw)

    @classmethod
    def from_field(cls, field: QVectorField3D) -> "PackedQVectorField3D":
        return cls(field.shape, field.to_bytes())

    @property
    def cells(self) -> int:
        return self.shape[0] * self.shape[1] * self.shape[2]

    @property
    def raw_bytes(self) -> int:
        return len(self._buffer)

    def _index(self, x: int, y: int, z: int) -> int:
        sx, sy, sz = self.shape
        if x < 0 or y < 0 or z < 0 or x >= sx or y >= sy or z >= sz:
            raise IndexError("packed vector coordinate is outside the field")
        return x + sx * (y + sy * z)

    def at(self, x: int, y: int, z: int) -> QVector3Q16:
        offset = self._index(x, y, z) * VECTOR_CELL_BYTES
        qx, qy, qz = struct.unpack_from(">iii", self._buffer, offset)
        return QVector3Q16(qx, qy, qz)

    def set(self, x: int, y: int, z: int, value: QVector3Q16) -> None:
        offset = self._index(x, y, z) * VECTOR_CELL_BYTES
        struct.pack_into(">iii", self._buffer, offset, value.x, value.y, value.z)

    def to_bytes(self) -> bytes:
        return bytes(self._buffer)

    def to_field(self) -> QVectorField3D:
        return QVectorField3D.from_bytes(self.to_bytes(), self.shape)

    @property
    def digest(self) -> str:
        prefix = b"Q16.16x3:" + b"x".join(str(axis).encode("ascii") for axis in self.shape) + b":"
        return hashlib.sha256(prefix + self.to_bytes()).hexdigest()

    def iter_tiles(
        self,
        tile_shape: Sequence[int],
    ) -> Iterator[tuple[tuple[int, int, int], tuple[int, int, int]]]:
        tx, ty, tz = (int(axis) for axis in tile_shape)
        if min(tx, ty, tz) < 1:
            raise ValueError("tile dimensions must be positive")
        sx, sy, sz = self.shape
        for z0 in range(0, sz, tz):
            for y0 in range(0, sy, ty):
                for x0 in range(0, sx, tx):
                    yield (
                        (x0, y0, z0),
                        (min(sx, x0 + tx), min(sy, y0 + ty), min(sz, z0 + tz)),
                    )


class QVectorFieldOps3D:
    """Deterministic fixed-point operators over QVectorField3D."""

    def __init__(
        self,
        *,
        policy: QNumericPolicy | None = None,
        boundary: QBoundaryMode = QBoundaryMode.CLAMP,
    ) -> None:
        self.policy = policy or QNumericPolicy()
        self.boundary = boundary
        self.status = QArithmeticStatus()

    def clear_status(self) -> None:
        self.status.clear()

    def _sample(self, field: QVectorField3D, x: int, y: int, z: int) -> QVector3Q16:
        sx, sy, sz = field.shape
        if 0 <= x < sx and 0 <= y < sy and 0 <= z < sz:
            return field.at(x, y, z)
        if self.boundary == QBoundaryMode.ZERO:
            return QVector3Q16.zero()
        if self.boundary == QBoundaryMode.WRAP:
            return field.at(x % sx, y % sy, z % sz)
        return field.at(
            min(sx - 1, max(0, x)),
            min(sy - 1, max(0, y)),
            min(sz - 1, max(0, z)),
        )

    def _derivative_component(
        self,
        minus: int,
        plus: int,
        spacing_q16: int,
    ) -> int:
        denominator = 2 * int(spacing_q16)
        raw = _round_div((int(plus) - int(minus)) * Q_ONE, denominator, self.policy.rounding, self.status)
        return _q32(raw, self.policy, self.status)

    def directional_derivative(
        self,
        field: QVectorField3D,
        axis: int,
        *,
        spacing_q16: int = Q_ONE,
    ) -> QVectorField3D:
        if axis not in (0, 1, 2):
            raise ValueError("derivative axis must be 0, 1, or 2")
        if spacing_q16 <= 0:
            raise ValueError("spacing must be positive Q16.16")
        sx, sy, sz = field.shape
        output: list[QVector3Q16] = []
        for z in range(sz):
            for y in range(sy):
                for x in range(sx):
                    offset = [0, 0, 0]
                    offset[axis] = 1
                    minus = self._sample(field, x - offset[0], y - offset[1], z - offset[2])
                    plus = self._sample(field, x + offset[0], y + offset[1], z + offset[2])
                    output.append(
                        QVector3Q16(
                            self._derivative_component(minus.x, plus.x, spacing_q16),
                            self._derivative_component(minus.y, plus.y, spacing_q16),
                            self._derivative_component(minus.z, plus.z, spacing_q16),
                        )
                    )
        return QVectorField3D(tuple(output), field.shape)

    def divergence(
        self,
        field: QVectorField3D,
        *,
        spacing_q16: int = Q_ONE,
    ) -> QVectorField3D:
        dx = self.directional_derivative(field, 0, spacing_q16=spacing_q16)
        dy = self.directional_derivative(field, 1, spacing_q16=spacing_q16)
        dz = self.directional_derivative(field, 2, spacing_q16=spacing_q16)
        values: list[QVector3Q16] = []
        for gx, gy, gz in zip(dx.vectors, dy.vectors, dz.vectors):
            div = _q32(gx.x + gy.y + gz.z, self.policy, self.status)
            values.append(QVector3Q16(div, div, div))
        return QVectorField3D(tuple(values), field.shape)

    def curl(
        self,
        field: QVectorField3D,
        *,
        spacing_q16: int = Q_ONE,
    ) -> QVectorField3D:
        dx = self.directional_derivative(field, 0, spacing_q16=spacing_q16)
        dy = self.directional_derivative(field, 1, spacing_q16=spacing_q16)
        dz = self.directional_derivative(field, 2, spacing_q16=spacing_q16)
        values: list[QVector3Q16] = []
        for gx, gy, gz in zip(dx.vectors, dy.vectors, dz.vectors):
            values.append(
                QVector3Q16(
                    _q32(gy.z - gz.y, self.policy, self.status),
                    _q32(gz.x - gx.z, self.policy, self.status),
                    _q32(gx.y - gy.x, self.policy, self.status),
                )
            )
        return QVectorField3D(tuple(values), field.shape)

    def laplacian(
        self,
        field: QVectorField3D,
        *,
        spacing_q16: int = Q_ONE,
    ) -> QVectorField3D:
        if spacing_q16 <= 0:
            raise ValueError("spacing must be positive Q16.16")
        sx, sy, sz = field.shape
        spacing_sq = int(spacing_q16) * int(spacing_q16)
        output: list[QVector3Q16] = []
        for z in range(sz):
            for y in range(sy):
                for x in range(sx):
                    center = field.at(x, y, z)
                    neighbors = (
                        self._sample(field, x - 1, y, z),
                        self._sample(field, x + 1, y, z),
                        self._sample(field, x, y - 1, z),
                        self._sample(field, x, y + 1, z),
                        self._sample(field, x, y, z - 1),
                        self._sample(field, x, y, z + 1),
                    )
                    lanes = []
                    for component in ("x", "y", "z"):
                        delta2 = sum(getattr(value, component) for value in neighbors) - 6 * getattr(center, component)
                        raw = _round_div(
                            delta2 * Q_ONE * Q_ONE,
                            spacing_sq,
                            self.policy.rounding,
                            self.status,
                        )
                        lanes.append(_q32(raw, self.policy, self.status))
                    output.append(QVector3Q16(lanes[0], lanes[1], lanes[2]))
        return QVectorField3D(tuple(output), field.shape)

    def convolve(
        self,
        field: QVectorField3D,
        kernel: QScalarKernel3D,
        *,
        bias: QVector3Q16 | None = None,
    ) -> QVectorField3D:
        """Apply one scalar kernel to all vector lanes with one final requantize.

        Every lane accumulates raw Q16.16 * Q16.16 products in a checked signed-64
        accumulator.  Requantization to Q16.16 occurs only after the full kernel
        MAC, avoiding repeated per-product saturation.
        """

        sx, sy, sz = field.shape
        kx, ky, kz = kernel.shape
        cx, cy, cz = kernel.center
        bias_value = bias or QVector3Q16.zero()
        output: list[QVector3Q16] = []
        for z in range(sz):
            for y in range(sy):
                for x in range(sx):
                    acc_x = QAccumulator64(int(bias_value.x) * Q_ONE)
                    acc_y = QAccumulator64(int(bias_value.y) * Q_ONE)
                    acc_z = QAccumulator64(int(bias_value.z) * Q_ONE)
                    weight_index = 0
                    for dz in range(kz):
                        for dy in range(ky):
                            for dx in range(kx):
                                sample = self._sample(
                                    field,
                                    x + dx - cx,
                                    y + dy - cy,
                                    z + dz - cz,
                                )
                                weight = kernel.weights_q16[weight_index]
                                weight_index += 1
                                acc_x.add_product(sample.x, weight, self.policy, self.status)
                                acc_y.add_product(sample.y, weight, self.policy, self.status)
                                acc_z.add_product(sample.z, weight, self.policy, self.status)
                    output.append(
                        QVector3Q16(
                            requantize_product(acc_x.value, self.policy, self.status),
                            requantize_product(acc_y.value, self.policy, self.status),
                            requantize_product(acc_z.value, self.policy, self.status),
                        )
                    )
        return QVectorField3D(tuple(output), field.shape)


__all__ = [
    "ACC64_MAX",
    "ACC64_MIN",
    "PackedQVectorField3D",
    "QAccumulator64",
    "QArithmeticStatus",
    "QBoundaryMode",
    "QNumericPolicy",
    "QRoundMode",
    "QScalarKernel3D",
    "QVectorFieldOps3D",
    "divide_q16_raw",
    "requantize_product",
]
