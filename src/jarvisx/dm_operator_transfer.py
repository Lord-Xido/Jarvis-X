"""Bounded reference implementation for ADR-014.

The module implements the jointly-diagonal/scalar specialization of

    D_M = nu * (Omega/Omega0)^xi
          * (Lambda tensor Theta)^dagger
          * K_Phi(Psi)

on a scalar 3D field.  It is intentionally dependency-free and does not claim
to implement a general dense Moore-Penrose pseudoinverse or measured hardware
latency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

Field3D = tuple[tuple[tuple[float, ...], ...], ...]
Kernel3D = tuple[tuple[tuple[float, ...], ...], ...]


def _shape3(value: Sequence[Sequence[Sequence[float]]], name: str) -> tuple[int, int, int]:
    if not value:
        raise ValueError(f"{name} must be non-empty")
    nx = len(value)
    ny = len(value[0])
    if ny == 0:
        raise ValueError(f"{name} must be non-empty")
    nz = len(value[0][0])
    if nz == 0:
        raise ValueError(f"{name} must be non-empty")
    for plane in value:
        if len(plane) != ny:
            raise ValueError(f"{name} must be rectangular")
        for row in plane:
            if len(row) != nz:
                raise ValueError(f"{name} must be rectangular")
            for sample in row:
                if not math.isfinite(float(sample)):
                    raise ValueError(f"{name} values must be finite")
    return nx, ny, nz


def as_field3d(value: Sequence[Sequence[Sequence[float]]]) -> Field3D:
    _shape3(value, "field")
    return tuple(tuple(tuple(float(v) for v in row) for row in plane) for plane in value)


def as_kernel3d(value: Sequence[Sequence[Sequence[float]]]) -> Kernel3D:
    shape = _shape3(value, "kernel")
    if any(size % 2 == 0 for size in shape):
        raise ValueError("kernel dimensions must be odd")
    return tuple(tuple(tuple(float(v) for v in row) for row in plane) for plane in value)


@dataclass(frozen=True)
class DMOperatorConfig:
    """Numerical contract for the scalar jointly-diagonal ADR-014 reference."""

    nu: float = -0.1
    omega: float = 1.0
    omega0: float = 1.0
    xi: float = 1.0
    lambda_gain: float = 1.0
    theta_gain: float = 1.0
    dt: float = 0.1
    epsilon: float = 1.0e-12

    def __post_init__(self) -> None:
        for name in ("nu", "omega", "omega0", "xi", "lambda_gain", "theta_gain", "dt", "epsilon"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.omega <= 0.0 or self.omega0 <= 0.0:
            raise ValueError("omega and omega0 must be positive for arbitrary real xi")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        if abs(self.constraint_gain) <= self.epsilon:
            raise ValueError("lambda_gain * theta_gain is too close to zero")

    @property
    def constraint_gain(self) -> float:
        return self.lambda_gain * self.theta_gain

    @property
    def memory_gain(self) -> float:
        return (self.omega / self.omega0) ** self.xi

    @property
    def scalar_gain(self) -> float:
        return self.nu * self.memory_gain / self.constraint_gain


def convolve3d_zero(field: Field3D, kernel: Kernel3D) -> Field3D:
    """Return deterministic zero-padded 3D correlation with an odd kernel."""

    nx, ny, nz = _shape3(field, "field")
    kx, ky, kz = _shape3(kernel, "kernel")
    if any(size % 2 == 0 for size in (kx, ky, kz)):
        raise ValueError("kernel dimensions must be odd")
    cx, cy, cz = kx // 2, ky // 2, kz // 2
    output: list[list[list[float]]] = []
    for i in range(nx):
        plane: list[list[float]] = []
        for j in range(ny):
            row: list[float] = []
            for k in range(nz):
                total = 0.0
                for a in range(kx):
                    ii = i + a - cx
                    if not 0 <= ii < nx:
                        continue
                    for b in range(ky):
                        jj = j + b - cy
                        if not 0 <= jj < ny:
                            continue
                        for c in range(kz):
                            kk = k + c - cz
                            if 0 <= kk < nz:
                                total += kernel[a][b][c] * field[ii][jj][kk]
                row.append(total)
            plane.append(row)
        output.append(plane)
    return as_field3d(output)


def scale_field(field: Field3D, gain: float) -> Field3D:
    if not math.isfinite(gain):
        raise ValueError("gain must be finite")
    return tuple(
        tuple(tuple(gain * sample for sample in row) for row in plane)
        for plane in field
    )


def add_fields(left: Field3D, right: Field3D, right_gain: float = 1.0) -> Field3D:
    if _shape3(left, "left") != _shape3(right, "right"):
        raise ValueError("field shapes must match")
    if not math.isfinite(right_gain):
        raise ValueError("right_gain must be finite")
    return tuple(
        tuple(
            tuple(a + right_gain * b for a, b in zip(lrow, rrow))
            for lrow, rrow in zip(lplane, rplane)
        )
        for lplane, rplane in zip(left, right)
    )


def dm_operator(field: Field3D, kernel: Kernel3D, config: DMOperatorConfig) -> Field3D:
    """Evaluate D_M for the scalar jointly-diagonal specialization."""

    transformed = convolve3d_zero(field, kernel)
    return scale_field(transformed, config.scalar_gain)


def step(field: Field3D, kernel: Kernel3D, config: DMOperatorConfig) -> Field3D:
    """Advance Psi_next = Psi + dt * D_M(Psi)."""

    derivative = dm_operator(field, kernel, config)
    return add_fields(field, derivative, config.dt)


def kernel_l1_norm(kernel: Kernel3D) -> float:
    _shape3(kernel, "kernel")
    return sum(abs(value) for plane in kernel for row in plane for value in row)


def operator_gain_bound(kernel: Kernel3D, config: DMOperatorConfig) -> float:
    """Conservative infinity-norm bound for |A| in D_M = A Psi."""

    return abs(config.scalar_gain) * kernel_l1_norm(kernel)


def uniform_mode_eigenvalue(kernel: Kernel3D, config: DMOperatorConfig) -> float:
    """Periodic/interior uniform-mode eigenvalue of the continuous operator."""

    kernel_sum = sum(value for plane in kernel for row in plane for value in row)
    return config.scalar_gain * kernel_sum


def uniform_mode_multiplier(kernel: Kernel3D, config: DMOperatorConfig) -> float:
    """Euler multiplier 1 + dt*a for the uniform translation-invariant mode."""

    return 1.0 + config.dt * uniform_mode_eigenvalue(kernel, config)


def uniform_mode_is_contracting(kernel: Kernel3D, config: DMOperatorConfig) -> bool:
    return abs(uniform_mode_multiplier(kernel, config)) < 1.0


def elasticity_summary(config: DMOperatorConfig) -> dict[str, float]:
    """Return scalar-mode logarithmic sensitivities from ADR-014."""

    return {
        "nu": 1.0,
        "omega": config.xi,
        "lambda_gain": -1.0,
        "theta_gain": -1.0,
        "xi": math.log(config.omega / config.omega0),
    }
