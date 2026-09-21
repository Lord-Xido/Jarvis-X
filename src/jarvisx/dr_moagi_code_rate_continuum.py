"""Finite-N operationalization of the DM-vOmegaXi+ code-rate continuum.

The mathematical continuum uses an N->infinity limit. The executable runtime
evaluates a bounded finite-N approximation from measured wall-clock
source-emission rates. It does not claim infinite physical throughput.

Measured throughput is not differentiable with respect to Python source or
chunk-size choices, so C_opt is implemented by derivative-free measured
selection rather than by pretending time.perf_counter() participates in
autograd.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from typing import Callable, Sequence

from jarvisx.dr_moagi_codegen_engine import (
    AUTOTUNE_CHUNKS,
    DEFAULT_LINES,
    DEFAULT_TEMPLATE,
    TARGET_LINES_PER_SECOND,
    GenerationConfig,
    HighThroughputCodeGenerator,
)

RateMeasure = Callable[[int], float]
Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class ContinuumConfig:
    levels: int = 8
    gamma: float = 0.5
    omega_radians: float = math.pi / 7.0
    initial_radius: float = 1.0
    r0: float = 1.0
    epsilon0: float = 0.25
    epsilon_floor: float = 1.0e-8
    lambda0: float = 1.0e-9
    beta: float = 1.0
    memory_decay: float = 0.9
    min_rate: float = 1.0
    candidates: tuple[int, ...] = AUTOTUNE_CHUNKS

    def validate(self) -> None:
        if self.levels <= 0:
            raise ValueError("levels must be positive")
        if not 0.0 < self.gamma < 1.0:
            raise ValueError("gamma must lie strictly between 0 and 1")
        if self.initial_radius <= 0.0 or self.r0 <= 0.0:
            raise ValueError("radii must be positive")
        if self.epsilon0 <= 0.0 or self.epsilon_floor <= 0.0:
            raise ValueError("epsilon values must be positive")
        if self.epsilon_floor > self.epsilon0:
            raise ValueError("epsilon_floor must not exceed epsilon0")
        if self.lambda0 < 0.0 or self.beta < 0.0:
            raise ValueError("lambda0 and beta must be non-negative")
        if not 0.0 <= self.memory_decay < 1.0:
            raise ValueError("memory_decay must lie in [0, 1)")
        if self.min_rate <= 0.0:
            raise ValueError("min_rate must be positive")
        if not self.candidates or any(value <= 0 for value in self.candidates):
            raise ValueError("candidate chunk sizes must be positive")


@dataclass(frozen=True)
class LevelReceipt:
    level: int
    radius: float
    next_radius: float
    kernel: float
    shell_volume: float
    shell_weight: float
    selected_chunk_lines: int
    committed_chunk_lines: int
    raw_lines_per_second: float
    predicted_lines_per_second: float
    prediction_error: float
    epsilon: float
    accepted: bool
    local_weighted_rate: float
    cumulative_lambda: float
    memory_rate: float
    inverse_rate_objective: float
    operator_objective: float


@dataclass(frozen=True)
class ContinuumReceipt:
    finite_lambda: float
    empirical_tail_bound: float
    empirical_lambda_infinity_upper: float
    max_observed_lines_per_second: float
    accepted_levels: int
    levels: tuple[LevelReceipt, ...]

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def norm3(vector: Vec3) -> float:
    return math.sqrt(sum(component * component for component in vector))


def rotate_contract(vector: Vec3, gamma: float, omega_radians: float) -> Vec3:
    x, y, z = vector
    cosine = math.cos(omega_radians)
    sine = math.sin(omega_radians)
    return (
        gamma * (cosine * x - sine * y),
        gamma * (sine * x + cosine * y),
        gamma * z,
    )


def inward_kernel(radius: float, r0: float) -> float:
    if radius <= 0.0 or r0 <= 0.0:
        raise ValueError("radius and r0 must be positive")
    return math.exp(-radius / r0) / (radius * radius)


def spherical_shell_volume(radius: float, next_radius: float) -> float:
    if radius <= 0.0 or next_radius < 0.0 or next_radius >= radius:
        raise ValueError("expected 0 <= next_radius < radius")
    return (4.0 * math.pi / 3.0) * (radius**3 - next_radius**3)


class CodeRateContinuum:
    """Measured finite-N realization of the inward print-rate equation."""

    def __init__(self, config: ContinuumConfig) -> None:
        config.validate()
        self.config = config

    def run(self, measure: RateMeasure) -> ContinuumReceipt:
        cfg = self.config
        vector: Vec3 = (cfg.initial_radius, 0.0, 0.0)
        epsilon = cfg.epsilon0
        predicted_rate: float | None = None
        memory_rate = 0.0
        cumulative = 0.0
        max_rate = 0.0
        accepted_levels = 0
        committed_chunk = cfg.candidates[0]
        receipts: list[LevelReceipt] = []

        for level in range(cfg.levels):
            measurements: list[tuple[float, float, float, int]] = []
            for chunk_lines in cfg.candidates:
                rate = float(measure(int(chunk_lines)))
                if not math.isfinite(rate) or rate <= 0.0:
                    raise ValueError("measured rate must be finite and positive")
                max_rate = max(max_rate, rate)
                if predicted_rate is None:
                    error = 0.0
                else:
                    scale = max(abs(predicted_rate), cfg.min_rate)
                    error = ((rate - predicted_rate) / scale) ** 2
                objective = (1.0 / max(rate, cfg.min_rate)) + cfg.beta * error
                measurements.append((objective, error, rate, int(chunk_lines)))

            objective, error, raw_rate, selected_chunk = min(
                measurements, key=lambda item: (item[0], -item[2], item[3])
            )
            accepted = error < epsilon
            if accepted:
                committed_chunk = selected_chunk
                accepted_levels += 1
                if predicted_rate is None:
                    predicted_rate = raw_rate
                    memory_rate = raw_rate
                else:
                    alpha = 1.0 - cfg.memory_decay
                    predicted_rate = cfg.memory_decay * predicted_rate + alpha * raw_rate
                    memory_rate = cfg.memory_decay * memory_rate + alpha * raw_rate

            radius = norm3(vector)
            next_vector = rotate_contract(vector, cfg.gamma, cfg.omega_radians)
            next_radius = norm3(next_vector)
            kernel = inward_kernel(radius, cfg.r0)
            shell_volume = spherical_shell_volume(radius, next_radius)
            shell_weight = kernel * shell_volume
            local_weighted_rate = raw_rate * shell_weight if accepted else 0.0
            cumulative += local_weighted_rate
            inverse_rate = 1.0 / max(cumulative, cfg.min_rate)

            receipts.append(
                LevelReceipt(
                    level=level,
                    radius=radius,
                    next_radius=next_radius,
                    kernel=kernel,
                    shell_volume=shell_volume,
                    shell_weight=shell_weight,
                    selected_chunk_lines=selected_chunk,
                    committed_chunk_lines=committed_chunk,
                    raw_lines_per_second=raw_rate,
                    predicted_lines_per_second=raw_rate if predicted_rate is None else predicted_rate,
                    prediction_error=error,
                    epsilon=epsilon,
                    accepted=accepted,
                    local_weighted_rate=local_weighted_rate,
                    cumulative_lambda=cumulative,
                    memory_rate=memory_rate,
                    inverse_rate_objective=inverse_rate,
                    operator_objective=objective,
                )
            )

            exponent = min(cfg.lambda0 * cumulative, 50.0)
            epsilon = max(cfg.epsilon_floor, epsilon * math.exp(-exponent))
            vector = next_vector

        final_radius = norm3(vector)
        shell_constant = (4.0 * math.pi / 3.0) * (1.0 - cfg.gamma**3)
        tail_bound = (
            max_rate
            * shell_constant
            * final_radius
            / max(1.0 - cfg.gamma, 1.0e-12)
        )

        return ContinuumReceipt(
            finite_lambda=cumulative,
            empirical_tail_bound=tail_bound,
            empirical_lambda_infinity_upper=cumulative + tail_bound,
            max_observed_lines_per_second=max_rate,
            accepted_levels=accepted_levels,
            levels=tuple(receipts),
        )


def measure_codegen_continuum(
    generation: GenerationConfig,
    continuum: ContinuumConfig,
    *,
    lines_per_trial: int = 100_000,
) -> ContinuumReceipt:
    generation.validate()
    continuum.validate()
    if lines_per_trial <= 0:
        raise ValueError("lines_per_trial must be positive")

    def measure(chunk_lines: int) -> float:
        trial = GenerationConfig(
            lines=lines_per_trial,
            target_lps=generation.target_lps,
            chunk_lines=chunk_lines,
            template=generation.template,
            digest=False,
        )
        return float(HighThroughputCodeGenerator(trial).benchmark().lines_per_second)

    return CodeRateContinuum(continuum).run(measure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m jarvisx.dr_moagi_code_rate_continuum")
    parser.add_argument("--levels", type=int, default=8)
    parser.add_argument("--lines-per-trial", type=int, default=100_000)
    parser.add_argument("--target-lps", type=float, default=TARGET_LINES_PER_SECOND)
    parser.add_argument("--template", default=DEFAULT_TEMPLATE)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--omega", type=float, default=math.pi / 7.0)
    parser.add_argument("--epsilon", type=float, default=0.25)
    parser.add_argument("--lambda0", type=float, default=1.0e-9)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument(
        "--candidate-chunks", type=int, nargs="+", default=list(AUTOTUNE_CHUNKS)
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    generation = GenerationConfig(
        lines=DEFAULT_LINES,
        target_lps=args.target_lps,
        template=args.template,
    )
    continuum = ContinuumConfig(
        levels=args.levels,
        gamma=args.gamma,
        omega_radians=args.omega,
        epsilon0=args.epsilon,
        lambda0=args.lambda0,
        beta=args.beta,
        candidates=tuple(args.candidate_chunks),
    )
    receipt = measure_codegen_continuum(
        generation, continuum, lines_per_trial=args.lines_per_trial
    )
    print(receipt.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
