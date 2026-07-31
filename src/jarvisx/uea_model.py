"""Deterministic reference for the Dr. Moagi Unified Autoencoding system.

The state is ``s = [frequency, amplitude, phase]``. The implementation exposes
operation-conditioned reconstruction, diagonal-Gaussian KL regularisation,
fixed-point checks and the stated signal-space gradient flow. It is intended for
small auditable experiments, not as a replacement for an autograd framework.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import cast

Vector3 = tuple[float, float, float]
Matrix3 = tuple[Vector3, Vector3, Vector3]


def _identity() -> Matrix3:
    return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _zeros() -> Matrix3:
    return ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


def _finite_vector(vector: Vector3, name: str) -> None:
    if not all(math.isfinite(value) for value in vector):
        raise ValueError(f"{name} must contain finite values")


def _finite_matrix(matrix: Matrix3, name: str) -> None:
    if not all(math.isfinite(value) for row in matrix for value in row):
        raise ValueError(f"{name} must contain finite values")


def _matvec(matrix: Matrix3, vector: Vector3) -> Vector3:
    values = tuple(
        sum(matrix[row][column] * vector[column] for column in range(3))
        for row in range(3)
    )
    return cast(Vector3, values)


def _add(left: Vector3, right: Vector3) -> Vector3:
    return cast(Vector3, tuple(left[index] + right[index] for index in range(3)))


def _scale(factor: float, vector: Vector3) -> Vector3:
    return cast(Vector3, tuple(factor * value for value in vector))


def _norm(vector: Vector3) -> float:
    return math.sqrt(sum(value * value for value in vector))


def wrap_phase(phase: float) -> float:
    """Return ``phase`` in ``[-pi, pi)``."""

    if not math.isfinite(phase):
        raise ValueError("phase must be finite")
    return (phase + math.pi) % (2.0 * math.pi) - math.pi


def phase_difference(left: float, right: float) -> float:
    """Shortest signed angular difference ``left - right``."""

    return wrap_phase(left - right)


@dataclass(frozen=True)
class Signal3D:
    """Frequency, amplitude and circular phase."""

    frequency: float
    amplitude: float
    phase: float

    def __post_init__(self) -> None:
        _finite_vector(self.as_vector(), "signal")

    @classmethod
    def from_vector(cls, vector: Vector3) -> "Signal3D":
        _finite_vector(vector, "signal vector")
        return cls(vector[0], vector[1], wrap_phase(vector[2]))

    def as_vector(self) -> Vector3:
        return (self.frequency, self.amplitude, self.phase)


@dataclass(frozen=True)
class SignalMetric:
    frequency: float = 1.0
    amplitude: float = 1.0
    phase: float = 1.0

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value) and value > 0.0 for value in self.as_vector()
        ):
            raise ValueError("metric weights must be finite and positive")

    def as_vector(self) -> Vector3:
        return (self.frequency, self.amplitude, self.phase)


def signal_residual(prediction: Signal3D, target: Signal3D) -> Vector3:
    return (
        prediction.frequency - target.frequency,
        prediction.amplitude - target.amplitude,
        phase_difference(prediction.phase, target.phase),
    )


def signal_squared_error(
    prediction: Signal3D,
    target: Signal3D,
    metric: SignalMetric | None = None,
) -> float:
    active_metric = metric or SignalMetric()
    residual = signal_residual(prediction, target)
    return sum(
        weight * value * value
        for weight, value in zip(active_metric.as_vector(), residual)
    )


@dataclass(frozen=True)
class GaussianPosterior:
    mean: Vector3
    log_variance: Vector3

    def __post_init__(self) -> None:
        _finite_vector(self.mean, "posterior mean")
        _finite_vector(self.log_variance, "posterior log variance")

    def sample(self, epsilon: Vector3) -> Vector3:
        _finite_vector(epsilon, "epsilon")
        values = tuple(
            self.mean[index]
            + math.exp(0.5 * self.log_variance[index]) * epsilon[index]
            for index in range(3)
        )
        return cast(Vector3, values)

    @property
    def kl_standard_normal(self) -> float:
        return 0.5 * sum(
            math.exp(log_variance) + mean * mean - 1.0 - log_variance
            for mean, log_variance in zip(self.mean, self.log_variance)
        )


@dataclass(frozen=True)
class LinearGaussianAutoencoder:
    """Small explicit ``q(z|s)`` encoder and linear decoder."""

    mean_matrix: Matrix3 = field(default_factory=_identity)
    mean_bias: Vector3 = (0.0, 0.0, 0.0)
    log_variance_matrix: Matrix3 = field(default_factory=_zeros)
    log_variance_bias: Vector3 = (-4.0, -4.0, -4.0)
    decoder_matrix: Matrix3 = field(default_factory=_identity)
    decoder_bias: Vector3 = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        _finite_matrix(self.mean_matrix, "mean matrix")
        _finite_vector(self.mean_bias, "mean bias")
        _finite_matrix(self.log_variance_matrix, "log-variance matrix")
        _finite_vector(self.log_variance_bias, "log-variance bias")
        _finite_matrix(self.decoder_matrix, "decoder matrix")
        _finite_vector(self.decoder_bias, "decoder bias")

    def posterior(self, signal: Signal3D) -> GaussianPosterior:
        vector = signal.as_vector()
        mean = _add(_matvec(self.mean_matrix, vector), self.mean_bias)
        log_variance = _add(
            _matvec(self.log_variance_matrix, vector),
            self.log_variance_bias,
        )
        if any(abs(value) > 40.0 for value in log_variance):
            raise ValueError("posterior log variance must remain in [-40, 40]")
        return GaussianPosterior(mean, log_variance)

    def reconstruct(self, signal: Signal3D, epsilon: Vector3 | None = None) -> Signal3D:
        posterior = self.posterior(signal)
        latent = posterior.mean if epsilon is None else posterior.sample(epsilon)
        decoded = _add(_matvec(self.decoder_matrix, latent), self.decoder_bias)
        return Signal3D.from_vector(decoded)


@dataclass(frozen=True)
class AffineSignalOperation:
    """Deterministic affine realization of modulation, filtering or noise."""

    name: str
    matrix: Matrix3 = field(default_factory=_identity)
    bias: Vector3 = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("operation name must not be empty")
        _finite_matrix(self.matrix, f"{self.name} matrix")
        _finite_vector(self.bias, f"{self.name} bias")

    def apply(self, signal: Signal3D) -> Signal3D:
        return Signal3D.from_vector(
            _add(_matvec(self.matrix, signal.as_vector()), self.bias)
        )


@dataclass(frozen=True)
class OperationSet:
    modulation: AffineSignalOperation
    filtering: AffineSignalOperation
    noise: AffineSignalOperation

    @classmethod
    def identity(cls) -> "OperationSet":
        return cls(
            AffineSignalOperation("M"),
            AffineSignalOperation("F"),
            AffineSignalOperation("N"),
        )

    @classmethod
    def default(cls) -> "OperationSet":
        return cls(
            AffineSignalOperation(
                "M",
                ((1.0, 0.0, 0.0), (0.0, 1.10, 0.0), (0.0, 0.0, 1.0)),
                (0.0, 0.0, 0.15),
            ),
            AffineSignalOperation(
                "F",
                ((1.0, 0.0, 0.0), (0.0, 0.80, 0.0), (0.0, 0.0, 1.0)),
            ),
            AffineSignalOperation("N", bias=(0.01, -0.02, 0.03)),
        )

    def items(self) -> tuple[AffineSignalOperation, ...]:
        return (self.modulation, self.filtering, self.noise)


@dataclass(frozen=True)
class MoagiCoefficients:
    beta: float = 1.0e-3
    gamma: float = 1.0
    lambda_m: float = 0.0
    lambda_f: float = 0.0
    lambda_n: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.beta) or self.beta < 0.0:
            raise ValueError("beta must be finite and non-negative")
        if not math.isfinite(self.gamma) or self.gamma < 0.0:
            raise ValueError("gamma must be finite and non-negative")
        if not all(
            math.isfinite(value)
            for value in (self.lambda_m, self.lambda_f, self.lambda_n)
        ):
            raise ValueError("operation coefficients must be finite")

    def operation_weights(self) -> Vector3:
        return (self.lambda_m, self.lambda_f, self.lambda_n)


@dataclass(frozen=True)
class LossBreakdown:
    base_reconstruction: float
    operation_reconstruction: tuple[tuple[str, float], ...]
    kl_regularization: float
    total: float

    @property
    def operation_total(self) -> float:
        return sum(value for _, value in self.operation_reconstruction)


@dataclass(frozen=True)
class FixedPointReport:
    base_rms: float
    operation_rms: tuple[tuple[str, float], ...]
    maximum_rms: float
    satisfied: bool
