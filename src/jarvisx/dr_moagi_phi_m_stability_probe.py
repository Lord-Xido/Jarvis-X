"""Bounded 3D Phi_M fixed-point vs period-two diagnostic (shadow only)."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .dr_moagi_phi_m_3d import PhiMConfig, phi_m, spatial_jacobian


@dataclass(frozen=True)
class StabilityConfig:
    points: int = 64
    steps: int = 200
    seed: int = 221107
    tolerance: float = 1e-6
    times: tuple[float, ...] = (0.0, 0.75, 1.5, 3.0)

    def __post_init__(self) -> None:
        for name, upper in (("points", 256), ("steps", 256)):
            x = getattr(self, name)
            if type(x) is not int or not 2 <= x <= upper:
                raise ValueError(f"{name} must be an integer between 2 and {upper}")
        if type(self.seed) is not int or not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("seed must be uint32")
        if not math.isfinite(self.tolerance) or not 1e-12 <= self.tolerance <= 0.01:
            raise ValueError("tolerance outside [1e-12,0.01]")
        if not 1 <= len(self.times) <= 8 or any(
            not math.isfinite(float(t)) or not 0 <= t <= 10 for t in self.times
        ):
            raise ValueError("times must be finite within [0,10] and length [1,8]")


def probe_stability(config: StabilityConfig = StabilityConfig(),
                    phi_config: PhiMConfig = PhiMConfig()) -> dict[str, object]:
    rng = np.random.default_rng(config.seed)
    initial = rng.uniform(-0.95, 0.95, (config.points, 3))
    outcomes = []
    for t in config.times:
        x = initial.copy()
        initial_norm = float(np.linalg.svd(spatial_jacobian(x, t, phi_config),
                                           compute_uv=False)[:, 0].max())
        previous = None
        one = 0.0
        two = math.inf
        for _ in range(config.steps):
            next_x = phi_m(x, t, phi_config)
            one = float(np.linalg.norm(next_x - x, axis=1).max())
            if previous is not None:
                two = float(np.linalg.norm(next_x - previous, axis=1).max())
            previous, x = x, next_x
        if one <= config.tolerance:
            classification = "sampled_fixed_point_convergence"
        elif two <= config.tolerance:
            classification = "sampled_period_two_convergence"
        else:
            classification = "not_converged_within_budget"
        j = spatial_jacobian(x, t, phi_config)
        y = phi_m(x, t, phi_config)
        j_next = spatial_jacobian(y, t, phi_config)
        two_step = j_next @ j
        two_rho = float(np.abs(np.linalg.eigvals(two_step)).max())
        outcomes.append({
            "time": float(t),
            "initial_sample_max_jacobian_singular_value": initial_norm,
            "terminal_one_step_max_residual": one,
            "terminal_two_step_max_residual": two,
            "terminal_max_two_step_spectral_radius": two_rho,
            "classification": classification,
        })
    return {
        "status": "bounded_sampled_dynamics_only",
        "config": asdict(config),
        "phi_parameters": asdict(phi_config),
        "results": outcomes,
        "warning": "A two-step residual close to zero does not prove fixed-point convergence; sampled local diagnostics do not prove global convergence or lack thereof.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--points", type=int, default=64)
    args = parser.parse_args()
    report = probe_stability(StabilityConfig(points=args.points, steps=args.steps))
    output = json.dumps(report, sort_keys=True, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
