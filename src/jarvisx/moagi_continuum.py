"""Dr. Moagi 3D continuum integration kernel.

Implements the discrete operational form

    Psi(t) = exp(-gamma*t) * dV * sum_v Lambda_v^+ Phi_v + Theta_core

where each voxel carries a 3-vector ``Phi_v`` and a 3x3 transform
``Lambda_v``. ``Lambda_v^+`` is represented by a Tikhonov-regularized
left pseudoinverse so singular or nearly singular transforms remain numerically
well-defined without adding a NumPy runtime dependency.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

Vector3 = tuple[float, float, float]
Matrix3 = tuple[Vector3, Vector3, Vector3]


@dataclass(frozen=True)
class ContinuumConfig:
    gamma: float = 0.2
    voxel_volume: float = 1.0
    inverse_epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if self.gamma < 0.0:
            raise ValueError("gamma must be >= 0")
        if self.voxel_volume <= 0.0:
            raise ValueError("voxel_volume must be > 0")
        if self.inverse_epsilon <= 0.0:
            raise ValueError("inverse_epsilon must be > 0")


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


def _transpose(m: Matrix3) -> Matrix3:
    return (
        (m[0][0], m[1][0], m[2][0]),
        (m[0][1], m[1][1], m[2][1]),
        (m[0][2], m[1][2], m[2][2]),
    )


def _matmul(a: Matrix3, b: Matrix3) -> Matrix3:
    bt = _transpose(b)
    return tuple(tuple(_dot(row, col) for col in bt) for row in a)  # type: ignore[return-value]


def _matvec(m: Matrix3, v: Vector3) -> Vector3:
    return tuple(_dot(row, v) for row in m)  # type: ignore[return-value]


def _inverse3(m: Matrix3) -> Matrix3:
    a, b, c = m[0]
    d, e, f = m[1]
    g, h, i = m[2]
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if abs(det) <= 1e-18:
        raise ValueError("matrix is singular")
    inv_det = 1.0 / det
    return (
        ((e * i - f * h) * inv_det, (c * h - b * i) * inv_det, (b * f - c * e) * inv_det),
        ((f * g - d * i) * inv_det, (a * i - c * g) * inv_det, (c * d - a * f) * inv_det),
        ((d * h - e * g) * inv_det, (b * g - a * h) * inv_det, (a * e - b * d) * inv_det),
    )


def regularized_pseudoinverse(m: Matrix3, epsilon: float = 1e-6) -> Matrix3:
    """Return ``(M^T M + epsilon I)^-1 M^T`` for a 3x3 transform."""
    if epsilon <= 0.0:
        raise ValueError("epsilon must be > 0")
    mt = _transpose(m)
    gram = _matmul(mt, m)
    regularized: Matrix3 = (
        (gram[0][0] + epsilon, gram[0][1], gram[0][2]),
        (gram[1][0], gram[1][1] + epsilon, gram[1][2]),
        (gram[2][0], gram[2][1], gram[2][2] + epsilon),
    )
    return _matmul(_inverse3(regularized), mt)


def continuum_step(
    phi_field: Iterable[Vector3],
    lambda_field: Iterable[Matrix3],
    theta_core: Vector3,
    t: float,
    config: ContinuumConfig | None = None,
) -> Vector3:
    """Evaluate one discrete 3D continuum integration step.

    ``phi_field`` and ``lambda_field`` must contain the same number of voxels.
    The tensor/operator product is concretized as ``Lambda_v^+ @ Phi_v``.
    """
    cfg = config or ContinuumConfig()
    if t < 0.0:
        raise ValueError("t must be >= 0")

    phis = tuple(phi_field)
    lambdas = tuple(lambda_field)
    if len(phis) != len(lambdas):
        raise ValueError("phi_field and lambda_field must have equal length")
    if not phis:
        return theta_core

    sx = sy = sz = 0.0
    for phi, transform in zip(phis, lambdas):
        pinv = regularized_pseudoinverse(transform, cfg.inverse_epsilon)
        x, y, z = _matvec(pinv, phi)
        sx += x
        sy += y
        sz += z

    decay = math.exp(-cfg.gamma * t)
    scale = decay * cfg.voxel_volume
    return (
        theta_core[0] + scale * sx,
        theta_core[1] + scale * sy,
        theta_core[2] + scale * sz,
    )


def homogeneous_recurrence(
    psi0: Vector3,
    lambda_field: Sequence[Matrix3],
    theta_core: Vector3,
    steps: int,
    dt: float = 1.0,
    config: ContinuumConfig | None = None,
) -> list[Vector3]:
    """Run the inward recurrence ``Psi_(k+1) = M(Psi_k)``.

    The previous global state is broadcast uniformly across the active voxels,
    normalized by voxel count so the recursive map does not scale merely because
    the active tile contains more voxels.
    """
    if steps < 0:
        raise ValueError("steps must be >= 0")
    if dt <= 0.0:
        raise ValueError("dt must be > 0")
    if not lambda_field:
        return [psi0]

    cfg = config or ContinuumConfig()
    history = [psi0]
    psi = psi0
    n = len(lambda_field)
    for _ in range(steps):
        phi = (psi[0] / n, psi[1] / n, psi[2] / n)
        psi = continuum_step((phi for _ in range(n)), lambda_field, theta_core, dt, cfg)
        history.append(psi)
    return history


__all__ = [
    "ContinuumConfig",
    "Matrix3",
    "Vector3",
    "continuum_step",
    "homogeneous_recurrence",
    "regularized_pseudoinverse",
]
