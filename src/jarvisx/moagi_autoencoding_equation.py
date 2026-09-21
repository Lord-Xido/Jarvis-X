"""Reference objective for Dr. Moagi's auto-encoding equation.

This module turns the submitted mathematical formulation into an executable,
dimensionally consistent research objective while preserving an important
separation of concerns:

* ``encoder`` is a vector-valued map E_theta: R^n -> R^m.
* ``decoder`` is a vector-valued map D_phi: R^m -> R^n.
* ``J`` is the scalar objective that is minimized during training.
* the quantum-inspired contribution is a real expectation value of a Hermitian
  operator, not a complex-valued loss and not a claim of quantum execution.

The deterministic Q16.16 runtime remains a separate deployment substrate.  This
module is intended for research/training-time evaluation and conformance tests.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from typing import Callable, Sequence

Vector = tuple[float, ...]
Matrix = tuple[tuple[float, ...], ...]
ComplexVector = tuple[complex, ...]
ComplexMatrix = tuple[tuple[complex, ...], ...]
Encoder = Callable[[Vector], Sequence[float]]
Decoder = Callable[[Vector], Sequence[float]]


def _finite_vector(values: Sequence[float], *, name: str) -> Vector:
    result = tuple(float(value) for value in values)
    if not result:
        raise ValueError(f"{name} must not be empty")
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _finite_matrix(values: Sequence[Sequence[float]], *, name: str) -> Matrix:
    rows = tuple(_finite_vector(row, name=f"{name} row") for row in values)
    if not rows:
        raise ValueError(f"{name} must not be empty")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError(f"{name} must be rectangular")
    return rows


def _finite_complex_vector(values: Sequence[complex], *, name: str) -> ComplexVector:
    result = tuple(complex(value) for value in values)
    if not result:
        raise ValueError(f"{name} must not be empty")
    if not all(math.isfinite(value.real) and math.isfinite(value.imag) for value in result):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _finite_complex_matrix(values: Sequence[Sequence[complex]], *, name: str) -> ComplexMatrix:
    rows = tuple(_finite_complex_vector(row, name=f"{name} row") for row in values)
    if not rows:
        raise ValueError(f"{name} must not be empty")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError(f"{name} must be rectangular")
    return rows


def _non_negative(value: float, *, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def reconstruction_loss(x: Sequence[float], x_hat: Sequence[float]) -> float:
    """Return 1/2 ||x - x_hat||_2^2."""

    left = _finite_vector(x, name="x")
    right = _finite_vector(x_hat, name="x_hat")
    if len(left) != len(right):
        raise ValueError("x and x_hat must have the same dimension")
    return 0.5 * sum((a - b) ** 2 for a, b in zip(left, right))


def shannon_entropy(probabilities: Sequence[float], *, tolerance: float = 1e-9) -> float:
    """Return Shannon entropy in bits for a normalized discrete distribution."""

    tolerance = _non_negative(tolerance, name="tolerance")
    values = _finite_vector(probabilities, name="probabilities")
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("probabilities must lie within [0, 1]")
    if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError("probabilities must sum to one")
    return -sum(value * math.log2(value) for value in values if value > 0.0)


def contractive_penalty(encoder_jacobian: Sequence[Sequence[float]]) -> float:
    """Return 1/2 ||J_E(x)||_F^2 for the full encoder Jacobian."""

    jacobian = _finite_matrix(encoder_jacobian, name="encoder_jacobian")
    return 0.5 * sum(value * value for row in jacobian for value in row)


def refinement_energy(
    probabilities: Sequence[float] | None = None,
    encoder_jacobian: Sequence[Sequence[float]] | None = None,
    *,
    entropy_weight: float = 1.0,
    jacobian_weight: float = 1.0,
) -> float:
    """Return a real, non-negative self-refinement regularizer.

    Omega(x) = w_H H(p(x)) + w_J/2 ||J_E(x)||_F^2.
    Either component may be omitted explicitly.
    """

    entropy_weight = _non_negative(entropy_weight, name="entropy_weight")
    jacobian_weight = _non_negative(jacobian_weight, name="jacobian_weight")
    entropy = 0.0 if probabilities is None else shannon_entropy(probabilities)
    contractive = 0.0 if encoder_jacobian is None else contractive_penalty(encoder_jacobian)
    return entropy_weight * entropy + jacobian_weight * contractive


def gaussian_basis(x: Sequence[float], mean: Sequence[float], sigma: float) -> float:
    """Return an isotropic n-dimensional normalized Gaussian density value."""

    point = _finite_vector(x, name="x")
    centre = _finite_vector(mean, name="mean")
    if len(point) != len(centre):
        raise ValueError("x and mean must have the same dimension")
    sigma = float(sigma)
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")
    dimension = len(point)
    squared_distance = sum((value - mu) ** 2 for value, mu in zip(point, centre))
    normalizer = (2.0 * math.pi * sigma * sigma) ** (-dimension / 2.0)
    return normalizer * math.exp(-squared_distance / (2.0 * sigma * sigma))


def phase_angle(value: complex) -> float:
    """Return the quadrant-correct phase angle atan2(Im(z), Re(z))."""

    value = complex(value)
    if not math.isfinite(value.real) or not math.isfinite(value.imag):
        raise ValueError("value must be finite")
    return math.atan2(value.imag, value.real)


def wavefunction_components(
    x: Sequence[float],
    means: Sequence[Sequence[float]],
    sigmas: Sequence[float],
    amplitudes: Sequence[complex],
    phases: Sequence[float],
) -> ComplexVector:
    """Construct quantum-inspired basis amplitudes alpha_k phi_k(x) exp(i theta_k)."""

    point = _finite_vector(x, name="x")
    means_tuple = tuple(tuple(float(value) for value in mean) for mean in means)
    sigmas_tuple = tuple(float(value) for value in sigmas)
    amplitudes_tuple = _finite_complex_vector(amplitudes, name="amplitudes")
    phases_tuple = tuple(float(value) for value in phases)
    count = len(means_tuple)
    if not count or not (len(sigmas_tuple) == len(amplitudes_tuple) == len(phases_tuple) == count):
        raise ValueError("means, sigmas, amplitudes, and phases must have equal non-zero length")
    if any(len(mean) != len(point) for mean in means_tuple):
        raise ValueError("every mean must match x dimension")
    if not all(math.isfinite(phase) for phase in phases_tuple):
        raise ValueError("phases must contain only finite values")

    return tuple(
        amplitude * gaussian_basis(point, mean, sigma) * cmath.exp(1j * phase)
        for mean, sigma, amplitude, phase in zip(
            means_tuple, sigmas_tuple, amplitudes_tuple, phases_tuple
        )
    )


def normalize_wavefunction(psi: Sequence[complex]) -> ComplexVector:
    """Normalize a finite complex state so that sum |psi_k|^2 = 1."""

    state = _finite_complex_vector(psi, name="psi")
    norm_squared = sum(abs(value) ** 2 for value in state)
    if norm_squared <= 0.0:
        raise ValueError("psi must have non-zero norm")
    norm = math.sqrt(norm_squared)
    return tuple(value / norm for value in state)


def expectation_value(
    psi: Sequence[complex],
    hamiltonian: Sequence[Sequence[complex]],
    *,
    hermitian_tolerance: float = 1e-10,
) -> float:
    """Return the real normalized expectation <psi|H|psi> for Hermitian H."""

    tolerance = _non_negative(hermitian_tolerance, name="hermitian_tolerance")
    state = normalize_wavefunction(psi)
    matrix = _finite_complex_matrix(hamiltonian, name="hamiltonian")
    dimension = len(state)
    if len(matrix) != dimension or any(len(row) != dimension for row in matrix):
        raise ValueError("hamiltonian must be square with wavefunction dimension")

    for row in range(dimension):
        for col in range(dimension):
            if abs(matrix[row][col] - matrix[col][row].conjugate()) > tolerance:
                raise ValueError("hamiltonian must be Hermitian")

    transformed = tuple(
        sum(matrix[row][col] * state[col] for col in range(dimension))
        for row in range(dimension)
    )
    value = sum(state[row].conjugate() * transformed[row] for row in range(dimension))
    if abs(value.imag) > tolerance:
        raise ValueError("Hermitian expectation must be real within tolerance")
    return float(value.real)


@dataclass(frozen=True)
class MoagiObjectiveConfig:
    """Weights for the corrected scalar Dr. Moagi autoencoding objective."""

    lambda_refinement: float = 0.0
    eta_quantum: float = 0.0
    entropy_weight: float = 1.0
    jacobian_weight: float = 1.0
    hermitian_tolerance: float = 1e-10

    def __post_init__(self) -> None:
        _non_negative(self.lambda_refinement, name="lambda_refinement")
        _non_negative(self.eta_quantum, name="eta_quantum")
        _non_negative(self.entropy_weight, name="entropy_weight")
        _non_negative(self.jacobian_weight, name="jacobian_weight")
        _non_negative(self.hermitian_tolerance, name="hermitian_tolerance")


@dataclass(frozen=True)
class MoagiObjectiveTerms:
    """Auditable decomposition of one objective evaluation."""

    latent: Vector
    reconstruction: Vector
    reconstruction_loss: float
    refinement: float
    quantum_expectation: float
    total: float


def evaluate_autoencoding_objective(
    x: Sequence[float],
    encoder: Encoder,
    decoder: Decoder,
    *,
    probabilities: Sequence[float] | None = None,
    encoder_jacobian: Sequence[Sequence[float]] | None = None,
    wavefunction: Sequence[complex] | None = None,
    hamiltonian: Sequence[Sequence[complex]] | None = None,
    config: MoagiObjectiveConfig | None = None,
) -> MoagiObjectiveTerms:
    """Evaluate the corrected Dr. Moagi scalar objective.

    J(x; theta, phi) = L_rec + lambda * Omega + eta * <psi|H_q|psi>.

    The caller supplies probabilities and the encoder Jacobian because their
    definitions depend on the selected model.  A non-zero quantum coefficient
    requires both a wavefunction and a Hermitian operator.
    """

    cfg = config or MoagiObjectiveConfig()
    point = _finite_vector(x, name="x")
    latent = _finite_vector(encoder(point), name="encoder output")
    reconstruction = _finite_vector(decoder(latent), name="decoder output")
    if len(reconstruction) != len(point):
        raise ValueError("decoder output must match x dimension")

    recon = reconstruction_loss(point, reconstruction)
    omega = refinement_energy(
        probabilities,
        encoder_jacobian,
        entropy_weight=cfg.entropy_weight,
        jacobian_weight=cfg.jacobian_weight,
    )

    if (wavefunction is None) != (hamiltonian is None):
        raise ValueError("wavefunction and hamiltonian must be supplied together")
    if cfg.eta_quantum > 0.0 and wavefunction is None:
        raise ValueError("eta_quantum > 0 requires a wavefunction and hamiltonian")
    quantum = (
        0.0
        if wavefunction is None
        else expectation_value(
            wavefunction,
            hamiltonian or (),
            hermitian_tolerance=cfg.hermitian_tolerance,
        )
    )

    total = recon + cfg.lambda_refinement * omega + cfg.eta_quantum * quantum
    if not math.isfinite(total):
        raise ValueError("objective must be finite and real")
    return MoagiObjectiveTerms(latent, reconstruction, recon, omega, quantum, total)


def gradient_step(
    parameters: Sequence[float], gradient: Sequence[float], learning_rate: float
) -> Vector:
    """Apply theta_{t+1} = theta_t - alpha * grad(theta_t)."""

    params = _finite_vector(parameters, name="parameters")
    grad = _finite_vector(gradient, name="gradient")
    if len(params) != len(grad):
        raise ValueError("parameters and gradient must have the same dimension")
    learning_rate = float(learning_rate)
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    return tuple(value - learning_rate * derivative for value, derivative in zip(params, grad))
