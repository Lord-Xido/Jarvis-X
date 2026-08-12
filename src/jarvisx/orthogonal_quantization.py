"""Deterministic orthogonal-transform quantization verification for Jarvis-X.

The reference implements the canonical precision law

    ||x - D^T Q^-1(Q(Dx))||_2 <= 0.5 * sqrt(sum(delta_k^2))

when ``D`` is orthonormal and every coefficient is quantized to its nearest
uniform reconstruction level.  It is a correctness/reference layer, not a
production video codec or accelerator kernel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

Vector = tuple[float, ...]
Matrix = tuple[tuple[float, ...], ...]
StepSpec = float | Sequence[float]


@dataclass(frozen=True)
class OrthogonalQuantizationTrace:
    """Complete deterministic receipt for one transform/quantization cycle."""

    input_values: Vector
    transform: Matrix
    coefficient_steps: Vector
    transformed_values: Vector
    quantized_values: tuple[int, ...]
    dequantized_values: Vector
    transform_residual: Vector
    reconstructed_values: Vector
    spatial_residual: Vector
    residual_norm: float
    deterministic_bound: float
    gate_ratio: float
    orthogonality_error: float
    committed: bool


def dct2_orthonormal_basis(size: int) -> Matrix:
    """Return the orthonormal DCT-II basis for a positive dimension."""

    if isinstance(size, bool) or not isinstance(size, int):
        raise TypeError("size must be an integer")
    if size <= 0:
        raise ValueError("size must be positive")

    rows: list[tuple[float, ...]] = []
    for k in range(size):
        alpha = math.sqrt(1.0 / size) if k == 0 else math.sqrt(2.0 / size)
        rows.append(
            tuple(
                alpha * math.cos(math.pi * (n + 0.5) * k / size)
                for n in range(size)
            )
        )
    return tuple(rows)


def orthogonal_quantization_trace(
    values: Sequence[float],
    transform: Sequence[Sequence[float]],
    delta: StepSpec,
    *,
    orthogonality_tolerance: float = 1e-10,
    verification_tolerance: float = 1e-12,
) -> OrthogonalQuantizationTrace:
    """Transform, quantize, reconstruct and verify the deterministic error bound.

    Quantization uses nearest-neighbour rounding with exact half-step ties away
    from zero.  A scalar ``delta`` applies uniformly to all coefficients; a
    sequence enables non-uniform precision.  Non-orthonormal transforms fail
    closed rather than silently widening the admissible error envelope.
    """

    x = _finite_vector(values, "values")
    size = len(x)
    basis = _square_matrix(transform, size)
    steps = _normalise_steps(delta, size)

    if not math.isfinite(orthogonality_tolerance) or orthogonality_tolerance < 0.0:
        raise ValueError("orthogonality_tolerance must be finite and non-negative")
    if not math.isfinite(verification_tolerance) or verification_tolerance < 0.0:
        raise ValueError("verification_tolerance must be finite and non-negative")

    orthogonality_error = _orthogonality_error(basis)
    if orthogonality_error > orthogonality_tolerance:
        raise ValueError(
            "transform is not orthonormal within tolerance; "
            "do not use transpose-as-inverse precision bounds"
        )

    transformed = _matvec(basis, x)
    quantized = tuple(
        _round_half_away_from_zero(value / step)
        for value, step in zip(transformed, steps)
    )
    dequantized = tuple(float(index) * step for index, step in zip(quantized, steps))
    transform_residual = tuple(
        value - reconstructed
        for value, reconstructed in zip(transformed, dequantized)
    )

    # Every nearest-neighbour coefficient residual must satisfy |e_k| <= delta_k/2.
    for residual, step in zip(transform_residual, steps):
        if abs(residual) > (0.5 * step + verification_tolerance):
            raise RuntimeError("coefficient residual exceeds nearest-neighbour bound")

    reconstructed = _matvec(_transpose(basis), dequantized)
    spatial_residual = tuple(value - recovered for value, recovered in zip(x, reconstructed))
    residual_norm = _norm2(spatial_residual)
    deterministic_bound = 0.5 * math.sqrt(sum(step * step for step in steps))
    gate_ratio = residual_norm / deterministic_bound
    committed = residual_norm <= deterministic_bound + verification_tolerance

    if not committed:
        raise RuntimeError(
            "orthogonal quantization residual exceeds deterministic bound; "
            "diagnose transform, payload and precision before widening the gate"
        )

    return OrthogonalQuantizationTrace(
        input_values=x,
        transform=basis,
        coefficient_steps=steps,
        transformed_values=transformed,
        quantized_values=quantized,
        dequantized_values=dequantized,
        transform_residual=transform_residual,
        reconstructed_values=reconstructed,
        spatial_residual=spatial_residual,
        residual_norm=residual_norm,
        deterministic_bound=deterministic_bound,
        gate_ratio=gate_ratio,
        orthogonality_error=orthogonality_error,
        committed=True,
    )


def uniform_error_bound(delta: float, dimension: int) -> float:
    """Return ``delta * sqrt(dimension) / 2`` after validating inputs."""

    if isinstance(dimension, bool) or not isinstance(dimension, int):
        raise TypeError("dimension must be an integer")
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    if isinstance(delta, bool) or not isinstance(delta, (int, float)):
        raise TypeError("delta must be numeric")
    delta_f = float(delta)
    if not math.isfinite(delta_f) or delta_f <= 0.0:
        raise ValueError("delta must be finite and positive")
    return delta_f * math.sqrt(dimension) / 2.0


def _finite_vector(values: Sequence[float], name: str) -> Vector:
    result = tuple(float(value) for value in values)
    if not result:
        raise ValueError(f"{name} must not be empty")
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _square_matrix(transform: Sequence[Sequence[float]], size: int) -> Matrix:
    rows = tuple(tuple(float(value) for value in row) for row in transform)
    if len(rows) != size or any(len(row) != size for row in rows):
        raise ValueError("transform must be square and match the input dimension")
    if not all(math.isfinite(value) for row in rows for value in row):
        raise ValueError("transform must contain only finite values")
    return rows


def _normalise_steps(delta: StepSpec, size: int) -> Vector:
    if isinstance(delta, bool):
        raise TypeError("delta must be numeric or a sequence of numeric steps")
    if isinstance(delta, (int, float)):
        steps = (float(delta),) * size
    else:
        steps = tuple(float(step) for step in delta)
        if len(steps) != size:
            raise ValueError("non-uniform delta must provide one step per coefficient")
    if not all(math.isfinite(step) and step > 0.0 for step in steps):
        raise ValueError("all quantization steps must be finite and positive")
    return steps


def _round_half_away_from_zero(value: float) -> int:
    if value >= 0.0:
        return int(math.floor(value + 0.5))
    return int(math.ceil(value - 0.5))


def _matvec(matrix: Matrix, vector: Vector) -> Vector:
    return tuple(sum(weight * value for weight, value in zip(row, vector)) for row in matrix)


def _transpose(matrix: Matrix) -> Matrix:
    return tuple(tuple(matrix[row][column] for row in range(len(matrix))) for column in range(len(matrix)))


def _orthogonality_error(matrix: Matrix) -> float:
    transpose = _transpose(matrix)
    size = len(matrix)
    maximum = 0.0
    for row in range(size):
        for column in range(size):
            value = sum(transpose[row][k] * matrix[k][column] for k in range(size))
            expected = 1.0 if row == column else 0.0
            maximum = max(maximum, abs(value - expected))
    return maximum


def _norm2(values: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in values))
