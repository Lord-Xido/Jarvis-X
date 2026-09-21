"""Computational latent-geometry verification for the Dr Moagi/Jarvis-X architecture.

This module borrows structural ideas from differential geometry: metric,
transport, curvature-like residuals, and constrained evolution. It does not
claim that cognitive computation obeys general relativity or that these
quantities are physical spacetime tensors.

The layer is deliberately non-authoritative: it produces evidence receipts that
may be consumed by CTR / Pi_Lambda, but it does not commit runtime state.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence, cast

Scalar = int | float
Vector3 = tuple[float, float, float]
Matrix3 = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]


class CognitiveGeometryError(ValueError):
    """Raised when computational geometry inputs violate the declared contract."""


def _finite_scalar(value: Scalar, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CognitiveGeometryError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise CognitiveGeometryError(f"{name} must be finite")
    return result


def _vector3(value: Sequence[Scalar], *, name: str) -> Vector3:
    if len(value) != 3:
        raise CognitiveGeometryError(f"{name} must contain exactly three values")
    return (
        _finite_scalar(value[0], name=f"{name}[0]"),
        _finite_scalar(value[1], name=f"{name}[1]"),
        _finite_scalar(value[2], name=f"{name}[2]"),
    )


def _matrix3(value: Sequence[Sequence[Scalar]], *, name: str) -> Matrix3:
    if len(value) != 3 or any(len(row) != 3 for row in value):
        raise CognitiveGeometryError(f"{name} must be a 3x3 matrix")
    rows = tuple(
        tuple(
            _finite_scalar(value[row][col], name=f"{name}[{row}][{col}]")
            for col in range(3)
        )
        for row in range(3)
    )
    return cast(Matrix3, rows)


def _is_symmetric(matrix: Matrix3, *, tolerance: float = 1e-12) -> bool:
    return all(
        abs(matrix[row][col] - matrix[col][row]) <= tolerance
        for row in range(3)
        for col in range(row + 1, 3)
    )


def _determinant3(matrix: Matrix3) -> float:
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def _validate_positive_definite_metric(metric: Matrix3) -> None:
    if not _is_symmetric(metric):
        raise CognitiveGeometryError("latent metric must be symmetric")
    first_minor = metric[0][0]
    second_minor = metric[0][0] * metric[1][1] - metric[0][1] * metric[1][0]
    third_minor = _determinant3(metric)
    if min(first_minor, second_minor, third_minor) <= 0.0:
        raise CognitiveGeometryError("latent metric must be positive definite")


def _matrix_add(left: Matrix3, right: Matrix3) -> Matrix3:
    return cast(
        Matrix3,
        tuple(
            tuple(left[row][col] + right[row][col] for col in range(3))
            for row in range(3)
        ),
    )


def _matrix_sub(left: Matrix3, right: Matrix3) -> Matrix3:
    return cast(
        Matrix3,
        tuple(
            tuple(left[row][col] - right[row][col] for col in range(3))
            for row in range(3)
        ),
    )


def _matrix_scale(matrix: Matrix3, scalar: float) -> Matrix3:
    return cast(
        Matrix3,
        tuple(
            tuple(scalar * matrix[row][col] for col in range(3))
            for row in range(3)
        ),
    )


def _matrix_vector(matrix: Matrix3, vector: Vector3) -> Vector3:
    return cast(
        Vector3,
        tuple(
            sum(matrix[row][col] * vector[col] for col in range(3))
            for row in range(3)
        ),
    )


def _frobenius(matrix: Matrix3) -> float:
    return math.sqrt(
        sum(matrix[row][col] ** 2 for row in range(3) for col in range(3))
    )


def _vector_distance(left: Vector3, right: Vector3) -> float:
    return math.sqrt(sum((left[index] - right[index]) ** 2 for index in range(3)))


@dataclass(frozen=True, slots=True)
class CognitiveFieldConfig:
    """Coupling and verification limits for the computational geometry layer."""

    lambda_z: float = 0.0
    kappa_z: float = 1.0
    max_geometry_residual: float = 1e-6
    max_holonomy_residual: float = 1e-6
    max_reconstruction_error: float = 1e-6
    max_fixed_point_delta: float = 1e-6

    def __post_init__(self) -> None:
        for name in (
            "lambda_z",
            "kappa_z",
            "max_geometry_residual",
            "max_holonomy_residual",
            "max_reconstruction_error",
            "max_fixed_point_delta",
        ):
            value = _finite_scalar(getattr(self, name), name=name)
            if name.startswith("max_") and value < 0.0:
                raise CognitiveGeometryError(f"{name} must be non-negative")


@dataclass(frozen=True, slots=True)
class InformationalStress:
    """Symmetric computational source tensor; not physical stress-energy."""

    input_density: float
    residual_pressure: float
    memory_pressure: float
    interaction_flux: Vector3 = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        for name in ("input_density", "residual_pressure", "memory_pressure"):
            _finite_scalar(getattr(self, name), name=name)
        object.__setattr__(
            self,
            "interaction_flux",
            _vector3(self.interaction_flux, name="interaction_flux"),
        )

    @property
    def tensor(self) -> Matrix3:
        f01, f02, f12 = self.interaction_flux
        return (
            (float(self.input_density), f01, f02),
            (f01, float(self.residual_pressure), f12),
            (f02, f12, float(self.memory_pressure)),
        )


@dataclass(frozen=True, slots=True)
class LatentGeometry:
    """Computational metric and curvature proxy over a three-axis latent chart."""

    metric: Matrix3
    curvature_proxy: Matrix3

    def __post_init__(self) -> None:
        metric = _matrix3(self.metric, name="metric")
        curvature = _matrix3(self.curvature_proxy, name="curvature_proxy")
        _validate_positive_definite_metric(metric)
        if not _is_symmetric(curvature):
            raise CognitiveGeometryError("curvature_proxy must be symmetric")
        object.__setattr__(self, "metric", metric)
        object.__setattr__(self, "curvature_proxy", curvature)


@dataclass(frozen=True, slots=True)
class CognitiveGeometryReceipt:
    """Evidence that may be supplied to CTR / Pi_Lambda."""

    geometry_residual: float
    holonomy_residual: float
    reconstruction_error: float
    fixed_point_delta: float
    action_proxy: float
    accepted: bool
    claim_status: str = "computational_geometry_only"


def computational_field_residual(
    geometry: LatentGeometry,
    stress: InformationalStress,
    config: CognitiveFieldConfig,
) -> Matrix3:
    """Return the software field residual G_Z + lambda_Z g_Z - kappa_Z T_info."""

    geometry_side = _matrix_add(
        geometry.curvature_proxy,
        _matrix_scale(geometry.metric, config.lambda_z),
    )
    source_side = _matrix_scale(stress.tensor, config.kappa_z)
    return _matrix_sub(geometry_side, source_side)


def holonomy_residual(
    latent: Sequence[Scalar],
    transport_loop: Sequence[Sequence[Sequence[Scalar]]],
) -> float:
    """Measure path-dependent latent drift after one closed transport loop."""

    origin = _vector3(latent, name="latent")
    transported = origin
    for index, transport in enumerate(transport_loop):
        matrix = _matrix3(transport, name=f"transport_loop[{index}]")
        transported = _matrix_vector(matrix, transported)
    return _vector_distance(origin, transported)


class CognitiveFieldVerifier:
    """Produce a bounded geometry receipt without committing authoritative state."""

    def __init__(self, config: CognitiveFieldConfig | None = None) -> None:
        self.config = config or CognitiveFieldConfig()

    def verify(
        self,
        *,
        geometry: LatentGeometry,
        stress: InformationalStress,
        latent: Sequence[Scalar],
        transport_loop: Sequence[Sequence[Sequence[Scalar]]],
        reconstruction_error: Scalar,
        fixed_point_delta: Scalar,
    ) -> CognitiveGeometryReceipt:
        reconstruction = _finite_scalar(
            reconstruction_error,
            name="reconstruction_error",
        )
        fixed_point = _finite_scalar(fixed_point_delta, name="fixed_point_delta")
        if reconstruction < 0.0 or fixed_point < 0.0:
            raise CognitiveGeometryError(
                "reconstruction_error and fixed_point_delta must be non-negative"
            )

        field_residual = computational_field_residual(geometry, stress, self.config)
        geometry_norm = _frobenius(field_residual)
        holonomy = holonomy_residual(latent, transport_loop)
        action_proxy = math.sqrt(
            geometry_norm**2
            + holonomy**2
            + reconstruction**2
            + fixed_point**2
        )
        accepted = (
            geometry_norm <= self.config.max_geometry_residual
            and holonomy <= self.config.max_holonomy_residual
            and reconstruction <= self.config.max_reconstruction_error
            and fixed_point <= self.config.max_fixed_point_delta
        )
        return CognitiveGeometryReceipt(
            geometry_residual=geometry_norm,
            holonomy_residual=holonomy,
            reconstruction_error=reconstruction,
            fixed_point_delta=fixed_point,
            action_proxy=action_proxy,
            accepted=accepted,
        )


__all__ = [
    "CognitiveFieldConfig",
    "CognitiveFieldVerifier",
    "CognitiveGeometryError",
    "CognitiveGeometryReceipt",
    "InformationalStress",
    "LatentGeometry",
    "Matrix3",
    "Vector3",
    "computational_field_residual",
    "holonomy_residual",
]
