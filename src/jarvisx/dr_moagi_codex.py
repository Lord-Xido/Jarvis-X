"""Executable reference form of the Dr Moagi 3D Codex research loop.

This module binds the conceptual loop

    scene -> encode -> inward fixed point -> latent correction -> smooth
          -> Pi_Lambda -> decode -> computational permeation

while keeping parameter learning in its mathematically correct parameter
space. It is a Layer-5 research operator and does not mutate the canonical
Jarvis-X VM.

The astronomically large ``virtual_depth_label`` is provenance only. Actual
work is bounded by ``max_fixed_point_iterations`` and reported separately.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from typing import Callable, Mapping, Protocol, Sequence

Coordinate = tuple[float, float, float]
Latent = tuple[float, ...]
ScalarField = dict[Coordinate, float]
ComplexField = dict[Coordinate, complex]
SceneLike = Mapping[Coordinate, float]
Condition = object


class Encoder3D(Protocol):
    def __call__(self, scene: SceneLike) -> Sequence[float]: ...


class Decoder3D(Protocol):
    def __call__(self, latent: Sequence[float]) -> Mapping[Coordinate, float]: ...


class InwardOperator(Protocol):
    def __call__(
        self,
        latent: Sequence[float],
        time_index: int,
        condition: Condition | None,
    ) -> Sequence[float]: ...


class SourceMapper(Protocol):
    def __call__(self, latent: Sequence[float]) -> Mapping[Coordinate, float]: ...


EpsilonModel = Callable[[Sequence[float], int, Condition | None], Sequence[float]]


@dataclass(frozen=True)
class DrMoagiCodexConfig:
    """Numerical, projection, recursion, and permeation contract."""

    lambda_max: float = 1.0
    dt: float = 0.1
    smoothing_tau: float = 0.0
    k_epsilon: float = 0.0
    eta_z: float = 0.0
    eta_theta: float = 0.0
    gamma: float = 1.0
    beta: float = 0.0
    wave_number: float = 0.0
    cell_volume: float = 1.0
    green_softening: float = 1.0e-6
    fixed_point_tolerance: float = 1.0e-6
    max_fixed_point_iterations: int = 256
    max_latent_dim: int = 4096
    max_source_cells: int = 100_000
    require_convergence: bool = True
    claimed_contraction: float | None = None
    virtual_depth_label: str = "1000000^1000000"

    def __post_init__(self) -> None:
        positive = (
            "lambda_max",
            "dt",
            "cell_volume",
            "green_softening",
            "fixed_point_tolerance",
        )
        non_negative = ("smoothing_tau",)
        finite = (
            "lambda_max",
            "dt",
            "smoothing_tau",
            "k_epsilon",
            "eta_z",
            "eta_theta",
            "gamma",
            "beta",
            "wave_number",
            "cell_volume",
            "green_softening",
            "fixed_point_tolerance",
        )
        for name in finite:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        for name in positive:
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        for name in non_negative:
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        for name in ("max_fixed_point_iterations", "max_latent_dim", "max_source_cells"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.claimed_contraction is not None:
            value = self.claimed_contraction
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("claimed_contraction must be numeric")
            if not math.isfinite(float(value)) or not 0.0 <= float(value) < 1.0:
                raise ValueError("claimed_contraction must satisfy 0 <= s < 1")
        if not isinstance(self.virtual_depth_label, str) or not self.virtual_depth_label:
            raise ValueError("virtual_depth_label must be a non-empty string")


@dataclass(frozen=True)
class FixedPointResult:
    latent: Latent
    iterations: int
    converged: bool
    final_delta: float
    max_observed_contraction: float | None


@dataclass(frozen=True)
class DrMoagiCodexResult:
    encoded_latent: Latent
    inward_latent: Latent
    raw_latent: Latent
    smoothed_latent: Latent
    projected_latent: Latent
    decoded_scene: ScalarField
    theta_before: Latent | None
    theta_after: Latent | None
    source_charge: ScalarField
    permeation_field: ComplexField
    fixed_point: FixedPointResult
    virtual_depth_label: str


class DiffusionInwardOperator:
    """The user-specified diffusion-like inward map D(Z).

    D(Z) = 1/sqrt(alpha) * [Z - (1-alpha)/sqrt(1-alpha_bar) * epsilon(Z,t,c)]

    The formula is implemented literally. Contractivity is not assumed; if a
    contraction bound is claimed, the outer fixed-point driver can enforce it
    empirically along the executed trajectory.
    """

    def __init__(
        self,
        *,
        alpha: float,
        alpha_bar: float,
        epsilon_model: EpsilonModel,
    ) -> None:
        for name, value in (("alpha", alpha), ("alpha_bar", alpha_bar)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must satisfy 0 < alpha <= 1")
        if not 0.0 <= alpha_bar < 1.0:
            raise ValueError("alpha_bar must satisfy 0 <= alpha_bar < 1")
        self.alpha = float(alpha)
        self.alpha_bar = float(alpha_bar)
        self.epsilon_model = epsilon_model

    def __call__(
        self,
        latent: Sequence[float],
        time_index: int,
        condition: Condition | None,
    ) -> Latent:
        z = _vector("latent", latent)
        epsilon = _vector(
            "epsilon_theta",
            self.epsilon_model(z, time_index, condition),
            dimension=len(z),
        )
        scale = (1.0 - self.alpha) / math.sqrt(1.0 - self.alpha_bar)
        root_alpha = math.sqrt(self.alpha)
        return tuple((value - scale * eps) / root_alpha for value, eps in zip(z, epsilon))


def _coordinate(value: object) -> Coordinate:
    if not isinstance(value, tuple) or len(value) != 3:
        raise TypeError("field coordinates must be 3-tuples")
    coordinate = tuple(float(axis) for axis in value)
    if not all(math.isfinite(axis) for axis in coordinate):
        raise ValueError("field coordinates must be finite")
    return coordinate  # type: ignore[return-value]


def _scalar_field(name: str, field: Mapping[Coordinate, float], max_cells: int) -> ScalarField:
    if len(field) > max_cells:
        raise RuntimeError(f"{name} exceeds source-cell budget")
    result: ScalarField = {}
    for raw_coordinate, raw_value in field.items():
        coordinate = _coordinate(raw_coordinate)
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise TypeError(f"{name} values must be numeric")
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"{name} contains a non-finite value")
        result[coordinate] = value
    return result


def _vector(name: str, values: Sequence[float], dimension: int | None = None) -> Latent:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a numeric sequence")
    result = tuple(float(value) for value in values)
    if not result:
        raise ValueError(f"{name} must not be empty")
    if dimension is not None and len(result) != dimension:
        raise ValueError(f"{name} dimension mismatch: expected {dimension}, got {len(result)}")
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} contains a non-finite value")
    return result


def l2_norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in vector))


def project_l2_ball(vector: Sequence[float], radius: float) -> Latent:
    z = _vector("projection input", vector)
    norm = l2_norm(z)
    if norm <= radius or norm == 0.0:
        return z
    scale = radius / norm
    return tuple(scale * value for value in z)


def smooth_first_order(
    previous: Sequence[float],
    target: Sequence[float],
    *,
    dt: float,
    tau: float,
) -> Latent:
    prev = _vector("previous latent", previous)
    nxt = _vector("smoothing target", target, dimension=len(prev))
    if tau == 0.0:
        return nxt
    weight = dt / (tau + dt)
    return tuple(p + weight * (n - p) for p, n in zip(prev, nxt))


def update_parameters(
    theta: Sequence[float],
    gradient: Sequence[float],
    *,
    eta_theta: float,
) -> Latent:
    current = _vector("Theta", theta)
    grad = _vector("grad_Theta L", gradient, dimension=len(current))
    return tuple(value - eta_theta * delta for value, delta in zip(current, grad))


def fixed_point_recurse(
    initial: Sequence[float],
    operator: InwardOperator,
    *,
    time_index: int,
    condition: Condition | None,
    config: DrMoagiCodexConfig,
) -> FixedPointResult:
    current = _vector("encoded latent", initial)
    if len(current) > config.max_latent_dim:
        raise RuntimeError("latent dimension exceeds configured budget")

    previous_delta: float | None = None
    max_ratio: float | None = None
    final_delta = math.inf

    for iteration in range(1, config.max_fixed_point_iterations + 1):
        candidate = _vector(
            "inward candidate",
            operator(current, time_index, condition),
            dimension=len(current),
        )
        final_delta = l2_norm(tuple(a - b for a, b in zip(candidate, current)))
        if previous_delta is not None and previous_delta > 0.0:
            ratio = final_delta / previous_delta
            max_ratio = ratio if max_ratio is None else max(max_ratio, ratio)
            if (
                config.claimed_contraction is not None
                and ratio > config.claimed_contraction + 1.0e-12
            ):
                raise RuntimeError(
                    "observed fixed-point contraction exceeds claimed_contraction"
                )
        current = candidate
        if final_delta <= config.fixed_point_tolerance:
            return FixedPointResult(
                latent=current,
                iterations=iteration,
                converged=True,
                final_delta=final_delta,
                max_observed_contraction=max_ratio,
            )
        previous_delta = final_delta

    result = FixedPointResult(
        latent=current,
        iterations=config.max_fixed_point_iterations,
        converged=False,
        final_delta=final_delta,
        max_observed_contraction=max_ratio,
    )
    if config.require_convergence:
        raise RuntimeError("fixed-point recursion did not converge within iteration budget")
    return result


def build_permeation_source(
    latent: Sequence[float],
    mapper: SourceMapper,
    *,
    equilibrium: Mapping[Coordinate, float] | None,
    source_gradient: Mapping[Coordinate, float] | None,
    gamma: float,
    beta: float,
    max_cells: int,
) -> ScalarField:
    state = _scalar_field("source map", mapper(latent), max_cells)
    equilibrium_field = (
        _scalar_field("equilibrium source", equilibrium, max_cells) if equilibrium is not None else {}
    )
    gradient_field = (
        _scalar_field("source gradient", source_gradient, max_cells)
        if source_gradient is not None
        else {}
    )
    extra_equilibrium = set(equilibrium_field) - set(state)
    extra_gradient = set(gradient_field) - set(state)
    if extra_equilibrium or extra_gradient:
        raise ValueError("equilibrium and source-gradient support must be subsets of source support")
    return {
        coordinate: gamma * abs(value - equilibrium_field.get(coordinate, 0.0))
        + beta * gradient_field.get(coordinate, 0.0)
        for coordinate, value in state.items()
    }


def helmholtz_permeate(
    source: Mapping[Coordinate, float],
    targets: Sequence[Coordinate],
    *,
    wave_number: float,
    cell_volume: float,
    softening: float,
    max_source_cells: int = 100_000,
) -> ComplexField:
    """Discrete volume quadrature of exp(ikR)/(4*pi*R) * q(r').

    ``softening`` regularizes the Green-kernel singularity when a target lies
    on a source coordinate. This is a computational field, not a claim of
    physical electromagnetic radiation.
    """

    q = _scalar_field("permeation source", source, max_source_cells)
    target_coordinates = [_coordinate(target) for target in targets]
    result: ComplexField = {}
    for target in target_coordinates:
        total = 0.0j
        for origin, charge in q.items():
            dx = target[0] - origin[0]
            dy = target[1] - origin[1]
            dz = target[2] - origin[2]
            distance = max(math.sqrt(dx * dx + dy * dy + dz * dz), softening)
            kernel = cmath.exp(1j * wave_number * distance) / (4.0 * math.pi * distance)
            total += kernel * charge * cell_volume
        result[target] = total
    return result


class DrMoagiCodex:
    """Bounded end-to-end executor for Xi^recur_Phi_3D research semantics."""

    def __init__(
        self,
        *,
        encoder: Encoder3D,
        decoder: Decoder3D,
        inward_operator: InwardOperator,
        source_mapper: SourceMapper,
        config: DrMoagiCodexConfig | None = None,
    ) -> None:
        self.encoder = encoder
        self.decoder = decoder
        self.inward_operator = inward_operator
        self.source_mapper = source_mapper
        self.config = config or DrMoagiCodexConfig()

    def execute(
        self,
        scene: SceneLike,
        *,
        previous_latent: Sequence[float] | None = None,
        prediction: Sequence[float] | None = None,
        epsilon_correction: Sequence[float] | None = None,
        latent_gradient: Sequence[float] | None = None,
        theta: Sequence[float] | None = None,
        theta_gradient: Sequence[float] | None = None,
        condition: Condition | None = None,
        time_index: int = 0,
        equilibrium_source: Mapping[Coordinate, float] | None = None,
        source_gradient: Mapping[Coordinate, float] | None = None,
        permeation_targets: Sequence[Coordinate] | None = None,
    ) -> DrMoagiCodexResult:
        encoded = _vector("encoder output", self.encoder(scene))
        if len(encoded) > self.config.max_latent_dim:
            raise RuntimeError("encoder output exceeds latent-dimension budget")

        fixed_point = fixed_point_recurse(
            encoded,
            self.inward_operator,
            time_index=time_index,
            condition=condition,
            config=self.config,
        )
        inward = fixed_point.latent
        dimension = len(inward)
        zero = (0.0,) * dimension
        p = _vector("prediction", prediction if prediction is not None else zero, dimension)
        eps = _vector(
            "epsilon correction",
            epsilon_correction if epsilon_correction is not None else zero,
            dimension,
        )
        grad_z = _vector(
            "grad_Z L",
            latent_gradient if latent_gradient is not None else zero,
            dimension,
        )

        raw = tuple(
            z + pred - self.config.k_epsilon * err - self.config.eta_z * grad
            for z, pred, err, grad in zip(inward, p, eps, grad_z)
        )
        previous = _vector(
            "previous latent",
            previous_latent if previous_latent is not None else encoded,
            dimension,
        )
        smoothed = smooth_first_order(
            previous,
            raw,
            dt=self.config.dt,
            tau=self.config.smoothing_tau,
        )
        projected = project_l2_ball(smoothed, self.config.lambda_max)

        decoded = _scalar_field(
            "decoded scene",
            self.decoder(projected),
            self.config.max_source_cells,
        )

        if (theta is None) != (theta_gradient is None):
            raise ValueError("theta and theta_gradient must be supplied together")
        theta_before = _vector("Theta", theta) if theta is not None else None
        theta_after = (
            update_parameters(theta_before, theta_gradient, eta_theta=self.config.eta_theta)
            if theta_before is not None and theta_gradient is not None
            else None
        )

        charge = build_permeation_source(
            projected,
            self.source_mapper,
            equilibrium=equilibrium_source,
            source_gradient=source_gradient,
            gamma=self.config.gamma,
            beta=self.config.beta,
            max_cells=self.config.max_source_cells,
        )
        targets = list(permeation_targets) if permeation_targets is not None else list(charge)
        phi = helmholtz_permeate(
            charge,
            targets,
            wave_number=self.config.wave_number,
            cell_volume=self.config.cell_volume,
            softening=self.config.green_softening,
            max_source_cells=self.config.max_source_cells,
        )

        return DrMoagiCodexResult(
            encoded_latent=encoded,
            inward_latent=inward,
            raw_latent=raw,
            smoothed_latent=smoothed,
            projected_latent=projected,
            decoded_scene=decoded,
            theta_before=theta_before,
            theta_after=theta_after,
            source_charge=charge,
            permeation_field=phi,
            fixed_point=fixed_point,
            virtual_depth_label=self.config.virtual_depth_label,
        )
