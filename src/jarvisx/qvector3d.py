"""Deterministic Q16.16 x Q16.16 x Q16.16 vector fields and autoencoder."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable, Sequence

Q_SHIFT = 16
Q_ONE = 1 << Q_SHIFT
Q32_MIN = -(1 << 31)
Q32_MAX = (1 << 31) - 1
VECTOR_COMPONENTS = 3
VECTOR_CELL_BYTES = 12


def _volume(shape: Sequence[int]) -> int:
    if len(shape) != 3:
        raise ValueError("shape must contain exactly three dimensions")
    x, y, z = (int(axis) for axis in shape)
    if min(x, y, z) < 1:
        raise ValueError("shape dimensions must be positive")
    return x * y * z


def _shape3(shape: Sequence[int]) -> tuple[int, int, int]:
    _volume(shape)
    return (int(shape[0]), int(shape[1]), int(shape[2]))


def _q32(value: int) -> int:
    return min(Q32_MAX, max(Q32_MIN, int(value)))


def _round_div_nearest(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    sign = -1 if numerator < 0 else 1
    magnitude = abs(numerator)
    quotient, remainder = divmod(magnitude, denominator)
    if remainder * 2 >= denominator:
        quotient += 1
    return sign * quotient


def _trunc_div(numerator: int, denominator: int) -> int:
    if denominator == 0:
        raise ZeroDivisionError("division by zero")
    sign = -1 if (numerator < 0) ^ (denominator < 0) else 1
    return sign * (abs(numerator) // abs(denominator))


def q16_from_float(value: float, *, saturate: bool = False) -> int:
    if not math.isfinite(value):
        raise ValueError("Q16.16 value must be finite")
    raw = int(round(value * Q_ONE))
    if saturate:
        return _q32(raw)
    if raw < Q32_MIN or raw > Q32_MAX:
        raise OverflowError("Q16.16 value does not fit signed 32 bits")
    return raw


def q16_to_float(raw: int) -> float:
    if raw < Q32_MIN or raw > Q32_MAX:
        raise OverflowError("raw Q16.16 value does not fit signed 32 bits")
    return int(raw) / Q_ONE


def q16_mul(left: int, right: int) -> int:
    return _q32(_trunc_div(int(left) * int(right), Q_ONE))


def q16_scale(value: int, scalar: int) -> int:
    return q16_mul(value, scalar)


@dataclass(frozen=True, slots=True)
class QVector3Q16:
    """One 96-bit logical vector: three signed 32-bit Q16.16 components."""

    x: int
    y: int
    z: int

    def __post_init__(self) -> None:
        for name, component in (("x", self.x), ("y", self.y), ("z", self.z)):
            if component < Q32_MIN or component > Q32_MAX:
                raise OverflowError(f"{name} component does not fit signed Q16.16")

    @classmethod
    def zero(cls) -> "QVector3Q16":
        return cls(0, 0, 0)

    @classmethod
    def from_floats(
        cls,
        x: float,
        y: float,
        z: float,
        *,
        saturate: bool = False,
    ) -> "QVector3Q16":
        return cls(
            q16_from_float(x, saturate=saturate),
            q16_from_float(y, saturate=saturate),
            q16_from_float(z, saturate=saturate),
        )

    def to_floats(self) -> tuple[float, float, float]:
        return (q16_to_float(self.x), q16_to_float(self.y), q16_to_float(self.z))

    def add(self, other: "QVector3Q16") -> "QVector3Q16":
        return QVector3Q16(
            _q32(self.x + other.x),
            _q32(self.y + other.y),
            _q32(self.z + other.z),
        )

    def sub(self, other: "QVector3Q16") -> "QVector3Q16":
        return QVector3Q16(
            _q32(self.x - other.x),
            _q32(self.y - other.y),
            _q32(self.z - other.z),
        )

    def hadamard(self, other: "QVector3Q16") -> "QVector3Q16":
        return QVector3Q16(
            q16_mul(self.x, other.x),
            q16_mul(self.y, other.y),
            q16_mul(self.z, other.z),
        )

    def scale(self, scalar_q16: int) -> "QVector3Q16":
        return QVector3Q16(
            q16_scale(self.x, scalar_q16),
            q16_scale(self.y, scalar_q16),
            q16_scale(self.z, scalar_q16),
        )

    def to_bytes(self) -> bytes:
        return b"".join(
            component.to_bytes(4, "big", signed=True)
            for component in (self.x, self.y, self.z)
        )

    @classmethod
    def from_bytes(cls, encoded: bytes) -> "QVector3Q16":
        if len(encoded) != VECTOR_CELL_BYTES:
            raise ValueError("one Q16.16x3 vector cell must contain exactly 12 bytes")
        return cls(
            int.from_bytes(encoded[0:4], "big", signed=True),
            int.from_bytes(encoded[4:8], "big", signed=True),
            int.from_bytes(encoded[8:12], "big", signed=True),
        )


@dataclass(frozen=True, slots=True)
class QVectorField3D:
    """Dense 3D field whose every voxel is one Q16.16x3 vector."""

    vectors: tuple[QVector3Q16, ...]
    shape: tuple[int, int, int]

    def __post_init__(self) -> None:
        expected = _volume(self.shape)
        if len(self.vectors) != expected:
            raise ValueError(f"field contains {len(self.vectors)} vectors; expected {expected}")

    @classmethod
    def from_vectors(
        cls,
        vectors: Iterable[Sequence[float] | QVector3Q16],
        shape: Sequence[int],
    ) -> "QVectorField3D":
        parsed: list[QVector3Q16] = []
        for vector in vectors:
            if isinstance(vector, QVector3Q16):
                parsed.append(vector)
                continue
            if len(vector) != VECTOR_COMPONENTS:
                raise ValueError("each vector must contain exactly three components")
            parsed.append(
                QVector3Q16.from_floats(
                    float(vector[0]),
                    float(vector[1]),
                    float(vector[2]),
                )
            )
        return cls(tuple(parsed), _shape3(shape))

    @classmethod
    def from_raw(
        cls,
        vectors_q16: Iterable[Sequence[int]],
        shape: Sequence[int],
    ) -> "QVectorField3D":
        parsed: list[QVector3Q16] = []
        for vector in vectors_q16:
            if len(vector) != VECTOR_COMPONENTS:
                raise ValueError("each raw vector must contain exactly three components")
            parsed.append(QVector3Q16(int(vector[0]), int(vector[1]), int(vector[2])))
        return cls(tuple(parsed), _shape3(shape))

    @property
    def cells(self) -> int:
        return len(self.vectors)

    @property
    def scalar_lanes(self) -> int:
        return self.cells * VECTOR_COMPONENTS

    @property
    def raw_bytes(self) -> int:
        return self.cells * VECTOR_CELL_BYTES

    @staticmethod
    def _index(x: int, y: int, z: int, shape: tuple[int, int, int]) -> int:
        sx, sy, sz = shape
        if x < 0 or y < 0 or z < 0 or x >= sx or y >= sy or z >= sz:
            raise IndexError("vector coordinate is outside the field")
        return x + sx * (y + sy * z)

    def at(self, x: int, y: int, z: int) -> QVector3Q16:
        return self.vectors[self._index(x, y, z, self.shape)]

    def replace(self, x: int, y: int, z: int, vector: QVector3Q16) -> "QVectorField3D":
        index = self._index(x, y, z, self.shape)
        values = list(self.vectors)
        values[index] = vector
        return QVectorField3D(tuple(values), self.shape)

    def to_bytes(self) -> bytes:
        return b"".join(vector.to_bytes() for vector in self.vectors)

    @classmethod
    def from_bytes(cls, encoded: bytes, shape: Sequence[int]) -> "QVectorField3D":
        shape3 = _shape3(shape)
        expected = _volume(shape3) * VECTOR_CELL_BYTES
        if len(encoded) != expected:
            raise ValueError(f"vector field contains {len(encoded)} bytes; expected {expected}")
        vectors = tuple(
            QVector3Q16.from_bytes(encoded[offset : offset + VECTOR_CELL_BYTES])
            for offset in range(0, len(encoded), VECTOR_CELL_BYTES)
        )
        return cls(vectors, shape3)

    @property
    def digest(self) -> str:
        prefix = (
            b"Q16.16x3:"
            + b"x".join(str(axis).encode("ascii") for axis in self.shape)
            + b":"
        )
        return hashlib.sha256(prefix + self.to_bytes()).hexdigest()

    def raw_payload(self) -> dict[str, object]:
        return {
            "shape": list(self.shape),
            "vectors_q16": [[vector.x, vector.y, vector.z] for vector in self.vectors],
        }


@dataclass(frozen=True, slots=True)
class QVectorRoundTrip3D:
    latent: QVectorField3D
    reconstruction: QVectorField3D
    axis_mse: tuple[float, float, float]
    component_mse: float
    vector_mse: float
    compression_ratio: float


class QVectorAutoencoder3D:
    """Block-mean 3D autoencoder operating entirely on raw Q16.16 triples."""

    @staticmethod
    def _latent_index(
        x: int,
        y: int,
        z: int,
        source_shape: tuple[int, int, int],
        latent_shape: tuple[int, int, int],
    ) -> int:
        sx, sy, sz = source_shape
        lx, ly, lz = latent_shape
        ix = min(lx - 1, (x * lx) // sx)
        iy = min(ly - 1, (y * ly) // sy)
        iz = min(lz - 1, (z * lz) // sz)
        return ix + lx * (iy + ly * iz)

    @staticmethod
    def _source_index(x: int, y: int, z: int, shape: tuple[int, int, int]) -> int:
        sx, sy, _ = shape
        return x + sx * (y + sy * z)

    def encode(self, field: QVectorField3D, latent_shape: Sequence[int]) -> QVectorField3D:
        latent_shape3 = _shape3(latent_shape)
        if any(latent > source for latent, source in zip(latent_shape3, field.shape)):
            raise ValueError("latent dimensions cannot exceed source dimensions")

        latent_cells = _volume(latent_shape3)
        sums_x = [0] * latent_cells
        sums_y = [0] * latent_cells
        sums_z = [0] * latent_cells
        counts = [0] * latent_cells
        sx, sy, sz = field.shape

        for z in range(sz):
            for y in range(sy):
                for x in range(sx):
                    source_index = self._source_index(x, y, z, field.shape)
                    latent_index = self._latent_index(x, y, z, field.shape, latent_shape3)
                    vector = field.vectors[source_index]
                    sums_x[latent_index] += vector.x
                    sums_y[latent_index] += vector.y
                    sums_z[latent_index] += vector.z
                    counts[latent_index] += 1

        vectors = tuple(
            QVector3Q16(
                _q32(_round_div_nearest(sums_x[index], counts[index])),
                _q32(_round_div_nearest(sums_y[index], counts[index])),
                _q32(_round_div_nearest(sums_z[index], counts[index])),
            )
            for index in range(latent_cells)
        )
        return QVectorField3D(vectors, latent_shape3)

    def decode(self, latent: QVectorField3D, output_shape: Sequence[int]) -> QVectorField3D:
        output_shape3 = _shape3(output_shape)
        if any(latent_axis > output for latent_axis, output in zip(latent.shape, output_shape3)):
            raise ValueError("latent dimensions cannot exceed output dimensions")

        sx, sy, sz = output_shape3
        vectors = [QVector3Q16.zero()] * _volume(output_shape3)
        for z in range(sz):
            for y in range(sy):
                for x in range(sx):
                    output_index = self._source_index(x, y, z, output_shape3)
                    latent_index = self._latent_index(x, y, z, output_shape3, latent.shape)
                    vectors[output_index] = latent.vectors[latent_index]
        return QVectorField3D(tuple(vectors), output_shape3)

    @staticmethod
    def error_metrics(
        source: QVectorField3D,
        reconstruction: QVectorField3D,
    ) -> tuple[tuple[float, float, float], float, float]:
        if source.shape != reconstruction.shape:
            raise ValueError("error metrics require vector fields with identical shapes")
        sse_x = 0
        sse_y = 0
        sse_z = 0
        for original, decoded in zip(source.vectors, reconstruction.vectors):
            dx = original.x - decoded.x
            dy = original.y - decoded.y
            dz = original.z - decoded.z
            sse_x += dx * dx
            sse_y += dy * dy
            sse_z += dz * dz
        scale_sq = float(Q_ONE * Q_ONE)
        n = source.cells
        axis_mse = (
            sse_x / (n * scale_sq),
            sse_y / (n * scale_sq),
            sse_z / (n * scale_sq),
        )
        component_mse = sum(axis_mse) / VECTOR_COMPONENTS
        vector_mse = sum(axis_mse)
        return axis_mse, component_mse, vector_mse

    def round_trip(
        self,
        field: QVectorField3D,
        latent_shape: Sequence[int],
    ) -> QVectorRoundTrip3D:
        latent = self.encode(field, latent_shape)
        reconstruction = self.decode(latent, field.shape)
        axis_mse, component_mse, vector_mse = self.error_metrics(field, reconstruction)
        return QVectorRoundTrip3D(
            latent=latent,
            reconstruction=reconstruction,
            axis_mse=axis_mse,
            component_mse=component_mse,
            vector_mse=vector_mse,
            compression_ratio=latent.cells / field.cells,
        )


__all__ = [
    "Q_ONE",
    "Q_SHIFT",
    "Q32_MAX",
    "Q32_MIN",
    "VECTOR_CELL_BYTES",
    "VECTOR_COMPONENTS",
    "QVector3Q16",
    "QVectorAutoencoder3D",
    "QVectorField3D",
    "QVectorRoundTrip3D",
    "q16_from_float",
    "q16_mul",
    "q16_scale",
    "q16_to_float",
]
