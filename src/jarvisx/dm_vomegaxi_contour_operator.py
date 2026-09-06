"""Numerically bounded reference kernel for the DM-vOmegaXi+ contour operator.

The symbolic law is

    Psi = integral_Theta [ (Phi tensor grad(Lambda)) / Omega**Xi+ ]
                         * exp(DM - v * Omega * Xi+) dv

This module makes the implicit types explicit and provides two execution modes:

* ``causal`` evaluates the archived exponential attenuation literally over an
  interval.  This is the natural interpretation for a forward runtime path.
* ``periodic`` replaces the non-periodic attenuation term with the periodic
  gate ``exp(DM - Omega*Xi+*(1-cos(theta)))`` so that a true closed contour has
  matching values at its endpoints.

All quantities in an exponential are treated as dimensionless normalized
runtime values.  The operator is an internal mathematical kernel; it does not
assert correspondence between an internal state and external reality.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

Vector = tuple[float, ...]
Matrix = tuple[tuple[float, ...], ...]
GateMode = Literal["causal", "periodic"]


@dataclass(frozen=True)
class ContourOperatorConfig:
    """Numerical contract for the contour operator."""

    gate_mode: GateMode = "causal"
    omega_epsilon: float = 1.0e-12
    exponent_limit: float = 700.0
    period: float = 2.0 * math.pi

    def __post_init__(self) -> None:
        if self.gate_mode not in ("causal", "periodic"):
            raise ValueError("gate_mode must be 'causal' or 'periodic'")
        for name in ("omega_epsilon", "exponent_limit", "period"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
            if float(value) <= 0.0:
                raise ValueError(f"{name} must be strictly positive")


@dataclass(frozen=True)
class ContourSample:
    """One typed sample of the DM-vOmegaXi+ integrand."""

    v: float
    phi: Vector
    grad_lambda: Vector
    omega: float
    xi_plus: float
    dm: float

    def __post_init__(self) -> None:
        scalars = {
            "v": self.v,
            "omega": self.omega,
            "xi_plus": self.xi_plus,
            "dm": self.dm,
        }
        for name, value in scalars.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.omega <= 0.0:
            raise ValueError("omega must be strictly positive")
        if self.xi_plus <= 0.0:
            raise ValueError("xi_plus must be strictly positive")
        if not self.phi:
            raise ValueError("phi must not be empty")
        if not self.grad_lambda:
            raise ValueError("grad_lambda must not be empty")
        _validate_vector("phi", self.phi)
        _validate_vector("grad_lambda", self.grad_lambda)


@dataclass(frozen=True)
class WeightSensitivities:
    """Analytic derivatives of log(weight) in causal mode."""

    d_log_weight_d_dm: float
    d_log_weight_d_v: float
    d_log_weight_d_omega: float
    d_log_weight_d_xi_plus: float


class DMvOmegaXiContourOperator:
    """Executable tensor-integral kernel for the archived DM-vOmegaXi+ law."""

    LAW_ID = "DM-vOmegaXi+-contour"

    def __init__(self, config: ContourOperatorConfig | None = None) -> None:
        self.config = config or ContourOperatorConfig()

    @staticmethod
    def outer(phi: Sequence[float], grad_lambda: Sequence[float]) -> Matrix:
        """Return Phi tensor grad(Lambda)."""
        if not phi or not grad_lambda:
            raise ValueError("outer-product vectors must not be empty")
        _validate_vector("phi", phi)
        _validate_vector("grad_lambda", grad_lambda)
        return tuple(
            tuple(float(phi_i) * float(grad_j) for grad_j in grad_lambda)
            for phi_i in phi
        )

    def log_weight(self, sample: ContourSample) -> float:
        """Return log(Omega^-Xi+ * gate) before exponent clipping."""
        omega = max(float(sample.omega), self.config.omega_epsilon)
        xi = float(sample.xi_plus)
        if self.config.gate_mode == "causal":
            gate_exponent = float(sample.dm) - float(sample.v) * omega * xi
        else:
            theta = 2.0 * math.pi * float(sample.v) / self.config.period
            gate_exponent = float(sample.dm) - omega * xi * (1.0 - math.cos(theta))
        return gate_exponent - xi * math.log(omega)

    def weight(self, sample: ContourSample) -> float:
        """Return the bounded scalar weight multiplying the tensor product."""
        log_weight = self.log_weight(sample)
        limit = self.config.exponent_limit
        return math.exp(max(-limit, min(limit, log_weight)))

    def local_derivative(self, sample: ContourSample) -> Matrix:
        """Evaluate dPsi/dv at one sample."""
        tensor = self.outer(sample.phi, sample.grad_lambda)
        scalar = self.weight(sample)
        return _scale_matrix(tensor, scalar)

    def step(self, psi: Matrix, sample: ContourSample, dv: float) -> Matrix:
        """Apply Psi[k+1] = Psi[k] + dv * integrand(sample)."""
        _validate_step(dv)
        derivative = self.local_derivative(sample)
        if not psi:
            psi = _zero_matrix(len(sample.phi), len(sample.grad_lambda))
        _validate_same_shape(psi, derivative)
        return _add_matrix(psi, _scale_matrix(derivative, float(dv)))

    def integrate(self, samples: Sequence[ContourSample], dv: float) -> Matrix:
        """Uniform left-Riemann accumulation over typed samples."""
        if not samples:
            raise ValueError("samples must not be empty")
        _validate_step(dv)
        rows = len(samples[0].phi)
        cols = len(samples[0].grad_lambda)
        psi = _zero_matrix(rows, cols)
        for sample in samples:
            if len(sample.phi) != rows or len(sample.grad_lambda) != cols:
                raise ValueError("all samples must have the same tensor shape")
            psi = self.step(psi, sample, dv)
        return psi

    def causal_sensitivities(self, sample: ContourSample) -> WeightSensitivities:
        """Return analytic sensitivities of log(weight) for the archived gate."""
        if self.config.gate_mode != "causal":
            raise RuntimeError("causal sensitivities require gate_mode='causal'")
        omega = max(float(sample.omega), self.config.omega_epsilon)
        xi = float(sample.xi_plus)
        v = float(sample.v)
        return WeightSensitivities(
            d_log_weight_d_dm=1.0,
            d_log_weight_d_v=-omega * xi,
            d_log_weight_d_omega=-(xi / omega) - v * xi,
            d_log_weight_d_xi_plus=-math.log(omega) - v * omega,
        )

    @staticmethod
    def constant_causal_closed_form(
        *,
        phi: Sequence[float],
        grad_lambda: Sequence[float],
        omega: float,
        xi_plus: float,
        dm: float,
        length: float,
    ) -> Matrix:
        """Exact integral for constant fields over v in [0, length].

        Psi = (Phi tensor grad(Lambda))
              * exp(DM) * (1-exp(-length*Omega*Xi+))
              / (Xi+ * Omega**(Xi+ + 1)).
        """
        for name, value in {
            "omega": omega,
            "xi_plus": xi_plus,
            "dm": dm,
            "length": length,
        }.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if omega <= 0.0:
            raise ValueError("omega must be strictly positive")
        if xi_plus <= 0.0:
            raise ValueError("xi_plus must be strictly positive")
        if length < 0.0:
            raise ValueError("length must be non-negative")

        tensor = DMvOmegaXiContourOperator.outer(phi, grad_lambda)
        decay = -math.expm1(-float(length) * float(omega) * float(xi_plus))
        scale = (
            math.exp(float(dm))
            * decay
            / (float(xi_plus) * float(omega) ** (float(xi_plus) + 1.0))
        )
        return _scale_matrix(tensor, scale)


def _validate_vector(name: str, vector: Sequence[float]) -> None:
    for value in vector:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} entries must be numeric")
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} entries must be finite")


def _validate_step(dv: float) -> None:
    if isinstance(dv, bool) or not isinstance(dv, (int, float)):
        raise TypeError("dv must be numeric")
    if not math.isfinite(float(dv)):
        raise ValueError("dv must be finite")
    if float(dv) <= 0.0:
        raise ValueError("dv must be strictly positive")


def _zero_matrix(rows: int, cols: int) -> Matrix:
    return tuple(tuple(0.0 for _ in range(cols)) for _ in range(rows))


def _scale_matrix(matrix: Matrix, scalar: float) -> Matrix:
    return tuple(tuple(float(value) * scalar for value in row) for row in matrix)


def _add_matrix(left: Matrix, right: Matrix) -> Matrix:
    _validate_same_shape(left, right)
    return tuple(
        tuple(left[i][j] + right[i][j] for j in range(len(left[i])))
        for i in range(len(left))
    )


def _validate_same_shape(left: Matrix, right: Matrix) -> None:
    if len(left) != len(right):
        raise ValueError("matrix row counts must match")
    if any(len(left_row) != len(right_row) for left_row, right_row in zip(left, right)):
        raise ValueError("matrix column counts must match")
