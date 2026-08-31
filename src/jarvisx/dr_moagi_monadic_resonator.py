"""Executable DM-vOmegaXi+ monadic latent-flow recurrence.

The runtime implements the operational law::

    Z_t     = E_phi(X_t)
    Z_t+1   = Z_t + integral[t,t+1] f_theta(Z(tau), tau) d tau
    X_t+1   = D_psi(Z_t+1)

The integral is evaluated with an explicit bounded numerical solver.  Encoder,
latent dynamics, and decoder remain separate callables so that state encoding,
continuous-time evolution, and reconstruction do not collapse into one opaque
operation.

This is a numerical reference runtime, not a claim that arbitrary learned latent
dynamics are solved exactly.  The report records the integration method,
substeps, derivative evaluations, and the accumulated latent increment.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from typing import Callable, Literal, Sequence

Vector = tuple[float, ...]
Encoder = Callable[[Vector], Sequence[float]]
Decoder = Callable[[Vector], Sequence[float]]
LatentDynamics = Callable[[Vector, float], Sequence[float]]
IntegrationMethod = Literal["euler", "rk4"]


def _vector(
    values: Sequence[float],
    *,
    label: str,
    expected_dim: int | None = None,
    max_abs_value: float | None = None,
) -> Vector:
    result = tuple(float(value) for value in values)
    if not result:
        raise ValueError(f"{label} cannot be empty")
    if expected_dim is not None and len(result) != expected_dim:
        raise ValueError(f"{label} dimension changed from {expected_dim} to {len(result)}")
    for value in result:
        if not math.isfinite(value):
            raise ValueError(f"{label} must contain only finite values")
        if max_abs_value is not None and abs(value) > max_abs_value:
            raise ValueError(
                f"{label} exceeded max_abs_value={max_abs_value}: observed {abs(value)}"
            )
    return result


def _add(left: Vector, right: Vector) -> Vector:
    return tuple(a + b for a, b in zip(left, right))


def _scale(vector: Vector, scalar: float) -> Vector:
    return tuple(scalar * value for value in vector)


def _difference(left: Vector, right: Vector) -> Vector:
    return tuple(a - b for a, b in zip(left, right))


def _l2(vector: Vector) -> float:
    return math.sqrt(sum(value * value for value in vector))


@dataclass(frozen=True)
class ResonatorConfig:
    """Bounded integration policy for one DM-vOmegaXi+ transition."""

    interval: float = 1.0
    substeps: int = 16
    method: IntegrationMethod = "rk4"
    max_abs_value: float = 1.0e12

    def __post_init__(self) -> None:
        if not math.isfinite(self.interval) or self.interval <= 0.0:
            raise ValueError("interval must be finite and positive")
        if isinstance(self.substeps, bool) or not isinstance(self.substeps, int):
            raise TypeError("substeps must be an integer")
        if self.substeps <= 0:
            raise ValueError("substeps must be positive")
        if self.method not in ("euler", "rk4"):
            raise ValueError("method must be 'euler' or 'rk4'")
        if not math.isfinite(self.max_abs_value) or self.max_abs_value <= 0.0:
            raise ValueError("max_abs_value must be finite and positive")


@dataclass(frozen=True)
class ResonatorStepReport:
    """Auditable record of one encode -> integrate -> decode transition."""

    t_start: float
    t_end: float
    input_state: Vector
    latent_start: Vector
    latent_integral: Vector
    latent_end: Vector
    output_state: Vector
    integration_method: str
    integration_substeps: int
    derivative_evaluations: int
    latent_delta_l2: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class MonadicResonator:
    """Composable implementation of the DM-vOmegaXi+ latent-flow law."""

    LAW_ID = "DM-vOmegaXi+-MONADIC-FLOW"
    EQUATION = "X[t+1] = D_psi(E_phi(X[t]) + integral_t^{t+1} f_theta(Z(tau),tau) dtau)"

    def __init__(
        self,
        encoder: Encoder,
        dynamics: LatentDynamics,
        decoder: Decoder,
        config: ResonatorConfig | None = None,
    ) -> None:
        self.encoder = encoder
        self.dynamics = dynamics
        self.decoder = decoder
        self.config = config or ResonatorConfig()

    def _dynamics(self, latent: Vector, time_value: float) -> Vector:
        derivative = self.dynamics(latent, time_value)
        return _vector(
            derivative,
            label="latent derivative",
            expected_dim=len(latent),
            max_abs_value=self.config.max_abs_value,
        )

    def integrate(self, latent_start: Sequence[float], t_start: float) -> tuple[Vector, Vector, int]:
        """Integrate f_theta over one configured interval.

        Returns ``(latent_end, accumulated_integral, derivative_evaluations)``.
        The integral term is represented by ``latent_end - latent_start`` so the
        returned values preserve the governing recurrence directly.
        """

        if not math.isfinite(t_start):
            raise ValueError("t_start must be finite")
        start = _vector(
            latent_start,
            label="latent_start",
            max_abs_value=self.config.max_abs_value,
        )
        current = start
        step_size = self.config.interval / self.config.substeps
        evaluations = 0

        for substep in range(self.config.substeps):
            tau = t_start + substep * step_size
            if self.config.method == "euler":
                k1 = self._dynamics(current, tau)
                current = _add(current, _scale(k1, step_size))
                evaluations += 1
            else:
                k1 = self._dynamics(current, tau)
                k2 = self._dynamics(_add(current, _scale(k1, 0.5 * step_size)), tau + 0.5 * step_size)
                k3 = self._dynamics(_add(current, _scale(k2, 0.5 * step_size)), tau + 0.5 * step_size)
                k4 = self._dynamics(_add(current, _scale(k3, step_size)), tau + step_size)
                weighted = tuple(
                    (a + 2.0 * b + 2.0 * c + d) / 6.0
                    for a, b, c, d in zip(k1, k2, k3, k4)
                )
                current = _add(current, _scale(weighted, step_size))
                evaluations += 4

            current = _vector(
                current,
                label="latent state",
                expected_dim=len(start),
                max_abs_value=self.config.max_abs_value,
            )

        return current, _difference(current, start), evaluations

    def step(self, input_state: Sequence[float], t_start: float = 0.0) -> ResonatorStepReport:
        """Execute one full encode -> latent flow -> decode transition."""

        source = _vector(
            input_state,
            label="input_state",
            max_abs_value=self.config.max_abs_value,
        )
        latent_start = _vector(
            self.encoder(source),
            label="encoder output",
            max_abs_value=self.config.max_abs_value,
        )
        latent_end, latent_integral, evaluations = self.integrate(latent_start, t_start)
        output = _vector(
            self.decoder(latent_end),
            label="decoder output",
            max_abs_value=self.config.max_abs_value,
        )
        return ResonatorStepReport(
            t_start=float(t_start),
            t_end=float(t_start + self.config.interval),
            input_state=source,
            latent_start=latent_start,
            latent_integral=latent_integral,
            latent_end=latent_end,
            output_state=output,
            integration_method=self.config.method,
            integration_substeps=self.config.substeps,
            derivative_evaluations=evaluations,
            latent_delta_l2=_l2(latent_integral),
        )

    def rollout(
        self,
        input_state: Sequence[float],
        *,
        steps: int,
        t_start: float = 0.0,
    ) -> list[ResonatorStepReport]:
        """Recursively feed each decoded state into the next transition."""

        if isinstance(steps, bool) or not isinstance(steps, int):
            raise TypeError("steps must be an integer")
        if steps <= 0:
            raise ValueError("steps must be positive")

        state = _vector(
            input_state,
            label="input_state",
            max_abs_value=self.config.max_abs_value,
        )
        time_value = float(t_start)
        reports: list[ResonatorStepReport] = []
        for _ in range(steps):
            report = self.step(state, time_value)
            reports.append(report)
            state = report.output_state
            time_value = report.t_end
        return reports


def scaled_identity(scale: float) -> Callable[[Vector], Vector]:
    """Return a finite element-wise scaling transform for reference experiments."""

    scale_value = float(scale)
    if not math.isfinite(scale_value):
        raise ValueError("scale must be finite")

    def transform(vector: Vector) -> Vector:
        return tuple(scale_value * value for value in vector)

    return transform


def linear_relaxation(rate: float, forcing: float = 0.0) -> LatentDynamics:
    """Return dZ/dt = -rate * Z + forcing as a deterministic reference field."""

    rate_value = float(rate)
    forcing_value = float(forcing)
    if not math.isfinite(rate_value) or not math.isfinite(forcing_value):
        raise ValueError("rate and forcing must be finite")

    def field(latent: Vector, _time_value: float) -> Vector:
        return tuple(-rate_value * value + forcing_value for value in latent)

    return field


def _parse_state(raw: str) -> Vector:
    try:
        values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("state must be comma-separated numbers") from exc
    if not values or not all(math.isfinite(value) for value in values):
        raise argparse.ArgumentTypeError("state must contain finite comma-separated numbers")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the DM-vOmegaXi+ monadic latent-flow recurrence")
    parser.add_argument("--state", type=_parse_state, default=(1.0, 0.5, -0.25))
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--substeps", type=int, default=16)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--method", choices=("euler", "rk4"), default="rk4")
    parser.add_argument("--rate", type=float, default=0.25)
    parser.add_argument("--forcing", type=float, default=0.0)
    parser.add_argument("--encoder-scale", type=float, default=1.0)
    parser.add_argument("--decoder-scale", type=float, default=1.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ResonatorConfig(
        interval=args.interval,
        substeps=args.substeps,
        method=args.method,
    )
    engine = MonadicResonator(
        encoder=scaled_identity(args.encoder_scale),
        dynamics=linear_relaxation(args.rate, args.forcing),
        decoder=scaled_identity(args.decoder_scale),
        config=config,
    )
    reports = engine.rollout(args.state, steps=args.steps)
    payload = {
        "law_id": engine.LAW_ID,
        "equation": engine.EQUATION,
        "steps": [report.as_dict() for report in reports],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
