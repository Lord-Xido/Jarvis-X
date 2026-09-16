"""Bounded 3D permeation field for the Jarvis-X inward architecture.

The permeation phase broadcasts a converged core signature through adjacent
3D state while preserving explicit constitutional gates:

* non-regression: candidate objective may not increase;
* spectral stability: the linearized diffusion/relaxation ceiling stays < 1;
* semantic uncertainty: hbar_semantic remains strictly positive.

The reference update is a diffusion-relaxation equation

    dPsi/dt = alpha * Laplacian(Psi) - kappa * (Psi - Psi_core)

with kappa chosen so the constant spatial mode contracts by the configured
``target_spectral_radius`` per discrete step.  The implementation is backend
agnostic and uses sparse coordinate maps; GPU/Torch backends may implement the
same contract with dense kernels.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

Coordinate3D = tuple[int, int, int]
Vector = tuple[float, ...]
Field3D = Mapping[Coordinate3D, Sequence[float]]

_NEIGHBORS: tuple[Coordinate3D, ...] = (
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
)


@dataclass(frozen=True)
class PermeationConstitution:
    """Stability and epistemic constraints for one permeation process."""

    alpha: float = 0.85
    dt: float = 0.1
    target_spectral_radius: float = 0.85
    semantic_hbar: float = 1.0e-4
    non_regression_tolerance: float = 0.0
    max_steps: int = 8

    def __post_init__(self) -> None:
        if not math.isfinite(self.alpha) or self.alpha < 0.0:
            raise ValueError("alpha must be finite and non-negative")
        if not math.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        if not (0.0 <= self.target_spectral_radius < 1.0):
            raise ValueError("target_spectral_radius must be in [0, 1)")
        if not math.isfinite(self.semantic_hbar) or self.semantic_hbar <= 0.0:
            raise ValueError("semantic_hbar must be finite and strictly positive")
        if (
            not math.isfinite(self.non_regression_tolerance)
            or self.non_regression_tolerance < 0.0
        ):
            raise ValueError("non_regression_tolerance must be finite and non-negative")
        if self.max_steps < 0:
            raise ValueError("max_steps must be non-negative")

        # For the 6-neighbour 3D Laplacian, lambda(L) lies in [-12, 0].
        # The explicit update eigenvalues are rho + alpha*dt*lambda(L).
        # alpha*dt <= rho/6 therefore bounds their magnitude by rho.
        if self.alpha * self.dt > self.target_spectral_radius / 6.0 + 1.0e-15:
            raise ValueError(
                "unstable permeation step: require alpha*dt <= target_spectral_radius/6"
            )

    @property
    def relaxation_gain(self) -> float:
        """Return kappa such that 1 - kappa*dt equals the target radius."""

        return (1.0 - self.target_spectral_radius) / self.dt

    @property
    def theoretical_spectral_ceiling(self) -> float:
        return self.target_spectral_radius


@dataclass(frozen=True)
class PermeationResult:
    field: dict[Coordinate3D, Vector]
    objective_history: tuple[float, ...]
    accepted_steps: int
    rejected_steps: int
    theoretical_spectral_ceiling: float
    semantic_hbar: float


def _validate_field(field: Field3D, core_signature: Sequence[float]) -> int:
    width = len(core_signature)
    if width <= 0:
        raise ValueError("core_signature must contain at least one channel")
    for value in core_signature:
        if not math.isfinite(value):
            raise ValueError("core_signature values must be finite")
    for coordinate, value in field.items():
        if len(coordinate) != 3:
            raise ValueError("field coordinates must be (x, y, z)")
        if len(value) != width:
            raise ValueError("all field vectors must match core_signature width")
        if any(not math.isfinite(component) for component in value):
            raise ValueError("field values must be finite")
    return width


def distance_to_core(field: Field3D, core_signature: Sequence[float]) -> float:
    """Mean squared distance from the field to the broadcast core signature."""

    width = _validate_field(field, core_signature)
    if not field:
        return 0.0
    total = 0.0
    for value in field.values():
        total += sum((value[index] - core_signature[index]) ** 2 for index in range(width))
    return total / (len(field) * width)


def permeation_step(
    field: Field3D,
    core_signature: Sequence[float],
    constitution: PermeationConstitution = PermeationConstitution(),
) -> dict[Coordinate3D, Vector]:
    """Apply one stable 6-neighbour 3D diffusion-relaxation update.

    Missing sparse neighbours use a no-flux (Neumann) boundary: their value is
    treated as the center value, so sparse-domain boundaries do not leak state.
    """

    width = _validate_field(field, core_signature)
    if not field:
        return {}

    alpha_dt = constitution.alpha * constitution.dt
    rho = constitution.target_spectral_radius
    source_weight = 1.0 - rho
    output: dict[Coordinate3D, Vector] = {}

    for coordinate, raw_center in field.items():
        center = tuple(float(value) for value in raw_center)
        laplacian = [0.0] * width
        x, y, z = coordinate
        for dx, dy, dz in _NEIGHBORS:
            neighbour = field.get((x + dx, y + dy, z + dz), center)
            for channel in range(width):
                laplacian[channel] += float(neighbour[channel]) - center[channel]

        output[coordinate] = tuple(
            rho * center[channel]
            + alpha_dt * laplacian[channel]
            + source_weight * float(core_signature[channel])
            for channel in range(width)
        )
    return output


def constitutional_gate(
    baseline_objective: float,
    candidate_objective: float,
    *,
    measured_spectral_radius: float,
    constitution: PermeationConstitution = PermeationConstitution(),
) -> bool:
    """Enforce Delta J <= 0, rho < 1 and hbar_semantic > 0."""

    for name, value in (
        ("baseline_objective", baseline_objective),
        ("candidate_objective", candidate_objective),
        ("measured_spectral_radius", measured_spectral_radius),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")

    non_regression = (
        candidate_objective
        <= baseline_objective + constitution.non_regression_tolerance
    )
    spectrally_stable = measured_spectral_radius < 1.0
    epistemically_open = constitution.semantic_hbar > 0.0
    return non_regression and spectrally_stable and epistemically_open


def permeate(
    field: Field3D,
    core_signature: Sequence[float],
    constitution: PermeationConstitution = PermeationConstitution(),
    *,
    steps: int | None = None,
) -> PermeationResult:
    """Broadcast the core signature while accepting only non-regressive steps."""

    requested_steps = constitution.max_steps if steps is None else steps
    if requested_steps < 0:
        raise ValueError("steps must be non-negative")

    current = {
        coordinate: tuple(float(component) for component in value)
        for coordinate, value in field.items()
    }
    objective = distance_to_core(current, core_signature)
    history = [objective]
    accepted = 0
    rejected = 0

    for _ in range(requested_steps):
        candidate = permeation_step(current, core_signature, constitution)
        candidate_objective = distance_to_core(candidate, core_signature)
        if not constitutional_gate(
            objective,
            candidate_objective,
            measured_spectral_radius=constitution.theoretical_spectral_ceiling,
            constitution=constitution,
        ):
            rejected += 1
            break
        current = candidate
        objective = candidate_objective
        history.append(objective)
        accepted += 1

    return PermeationResult(
        field=current,
        objective_history=tuple(history),
        accepted_steps=accepted,
        rejected_steps=rejected,
        theoretical_spectral_ceiling=constitution.theoretical_spectral_ceiling,
        semantic_hbar=constitution.semantic_hbar,
    )


def saturation_fraction(
    field: Field3D,
    core_signature: Sequence[float],
    *,
    tolerance: float,
) -> float:
    """Fraction of sites whose Euclidean distance to the core is <= tolerance."""

    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative")
    width = _validate_field(field, core_signature)
    if not field:
        return 1.0
    saturated = 0
    for value in field.values():
        distance = math.sqrt(
            sum((value[index] - core_signature[index]) ** 2 for index in range(width))
        )
        if distance <= tolerance:
            saturated += 1
    return saturated / len(field)


def divergence_rms(vector_field: Field3D) -> float:
    """Estimate RMS divergence for a 3-channel vector field on a sparse grid.

    This is a verifier, not a Helmholtz projection.  Callers that require a
    strictly solenoidal field must project it in their numerical backend and
    use this metric as an admission check.
    """

    _validate_field(vector_field, (0.0, 0.0, 0.0))
    if not vector_field:
        return 0.0

    divergence_sq = 0.0
    for (x, y, z), raw_center in vector_field.items():
        center = tuple(float(value) for value in raw_center)
        xp = vector_field.get((x + 1, y, z), center)
        xm = vector_field.get((x - 1, y, z), center)
        yp = vector_field.get((x, y + 1, z), center)
        ym = vector_field.get((x, y - 1, z), center)
        zp = vector_field.get((x, y, z + 1), center)
        zm = vector_field.get((x, y, z - 1), center)
        divergence = (
            0.5 * (float(xp[0]) - float(xm[0]))
            + 0.5 * (float(yp[1]) - float(ym[1]))
            + 0.5 * (float(zp[2]) - float(zm[2]))
        )
        divergence_sq += divergence**2
    return math.sqrt(divergence_sq / len(vector_field))


def solenoidal_within(vector_field: Field3D, *, tolerance: float) -> bool:
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative")
    return divergence_rms(vector_field) <= tolerance
