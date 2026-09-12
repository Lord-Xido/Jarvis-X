"""Million-fold inward 3D hierarchy and runtime-policy controls.

This module formalizes the canonical 1000^3 -> 100^3 -> 10^3 logical
contraction without claiming a literal 1,000,000x wall-clock speedup.  The
million-fold figure is the reduction in coarse spatial sites between the full
logical field and the 10^3 core.  Fine detail is recovered selectively through
sparse residual refinement.

The governing execution principle is:

    coarse everywhere; fine only where measured error says it matters.

The implementation is intentionally backend-agnostic.  Dense Torch/CUDA tile
execution can consume its active-set and convergence decisions while sparse
transactional runtimes retain authority over admission/commit semantics.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

Coordinate3D = tuple[int, int, int]


@dataclass(frozen=True)
class MillionFoldConfig:
    """Canonical three-level inward hierarchy.

    ``full_side=1000``, ``mid_side=100`` and ``core_side=10`` give:

        1000^3 = 1,000,000,000 sites
         100^3 =     1,000,000 sites
          10^3 =         1,000 sites

    so full/core = 1,000,000.
    """

    full_side: int = 1000
    mid_side: int = 100
    core_side: int = 10
    error_threshold: float = 1.0e-3
    convergence_tol: float = 1.0e-5
    max_active_fraction: float = 1.0e-3

    def __post_init__(self) -> None:
        for name, value in (
            ("full_side", self.full_side),
            ("mid_side", self.mid_side),
            ("core_side", self.core_side),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if not (self.full_side > self.mid_side > self.core_side):
            raise ValueError("expected full_side > mid_side > core_side")
        if self.full_side % self.mid_side or self.mid_side % self.core_side:
            raise ValueError("hierarchy sides must divide exactly")
        if not math.isfinite(self.error_threshold) or self.error_threshold < 0.0:
            raise ValueError("error_threshold must be finite and non-negative")
        if not math.isfinite(self.convergence_tol) or self.convergence_tol < 0.0:
            raise ValueError("convergence_tol must be finite and non-negative")
        if not (0.0 < self.max_active_fraction <= 1.0):
            raise ValueError("max_active_fraction must be in (0, 1]")

    @property
    def full_sites(self) -> int:
        return self.full_side**3

    @property
    def mid_sites(self) -> int:
        return self.mid_side**3

    @property
    def core_sites(self) -> int:
        return self.core_side**3

    @property
    def full_to_mid_reduction(self) -> float:
        return self.full_sites / self.mid_sites

    @property
    def mid_to_core_reduction(self) -> float:
        return self.mid_sites / self.core_sites

    @property
    def coarse_spatial_reduction(self) -> float:
        return self.full_sites / self.core_sites

    @property
    def max_active_sites(self) -> int:
        return max(1, math.ceil(self.full_sites * self.max_active_fraction))


@dataclass(frozen=True)
class ResidualWork:
    """Count the work that survives the coarse hierarchy.

    ``level0_sites`` represents full-resolution residual work and
    ``level1_sites`` represents mid-resolution residual work.  Counts are work
    accounting units; callers may use active voxels, active tiles or another
    consistently measured unit.
    """

    level0_sites: int = 0
    level1_sites: int = 0

    def __post_init__(self) -> None:
        if self.level0_sites < 0 or self.level1_sites < 0:
            raise ValueError("residual work counts must be non-negative")

    @property
    def total(self) -> int:
        return self.level0_sites + self.level1_sites


def effective_work_speedup(
    config: MillionFoldConfig,
    residual: ResidualWork = ResidualWork(),
) -> float:
    """Return full logical work divided by optimized coarse+residual work.

    This is an *effective work reduction*, not a wall-clock benchmark.
    """

    optimized_work = config.core_sites + residual.total
    if optimized_work <= 0:  # Defensive; core_sites is positive by construction.
        raise ValueError("optimized work must be positive")
    return config.full_sites / optimized_work


class ActiveSetController:
    """Select high-error 3D regions for fine residual refinement."""

    def __init__(self, config: MillionFoldConfig | None = None) -> None:
        self.config = config or MillionFoldConfig()

    def select(
        self,
        errors: Mapping[Coordinate3D, float],
        *,
        threshold: float | None = None,
        max_active: int | None = None,
    ) -> tuple[Coordinate3D, ...]:
        limit = self.config.max_active_sites if max_active is None else max_active
        if limit <= 0:
            raise ValueError("max_active must be positive")
        cutoff = self.config.error_threshold if threshold is None else threshold
        if not math.isfinite(cutoff) or cutoff < 0.0:
            raise ValueError("threshold must be finite and non-negative")

        ranked: list[tuple[float, Coordinate3D]] = []
        for coordinate, error in errors.items():
            if len(coordinate) != 3:
                raise ValueError("coordinates must be (x, y, z)")
            if not math.isfinite(error) or error < 0.0:
                raise ValueError("errors must be finite and non-negative")
            if error > cutoff:
                ranked.append((error, coordinate))

        # Stable deterministic selection: highest error first, then coordinate.
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return tuple(coordinate for _error, coordinate in ranked[:limit])


def relative_state_delta(
    previous: Sequence[float],
    current: Sequence[float],
    *,
    epsilon: float = 1.0e-12,
) -> float:
    """Compute ||current-previous||_2 / (||previous||_2 + epsilon)."""

    if len(previous) != len(current):
        raise ValueError("states must have equal length")
    if not previous:
        return 0.0
    if epsilon <= 0.0 or not math.isfinite(epsilon):
        raise ValueError("epsilon must be finite and positive")

    delta_sq = 0.0
    previous_sq = 0.0
    for old, new in zip(previous, current):
        if not math.isfinite(old) or not math.isfinite(new):
            raise ValueError("state values must be finite")
        delta_sq += (new - old) ** 2
        previous_sq += old**2
    return math.sqrt(delta_sq) / (math.sqrt(previous_sq) + epsilon)


def region_converged(
    previous: Sequence[float],
    current: Sequence[float],
    *,
    tolerance: float,
) -> bool:
    if tolerance < 0.0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be finite and non-negative")
    return relative_state_delta(previous, current) <= tolerance


@dataclass(frozen=True)
class RuntimeCost:
    """Measured runtime and quality telemetry for one execution policy."""

    latency_s: float
    bandwidth_bytes: float
    memory_bytes: float
    energy_j: float
    quality_loss: float

    def __post_init__(self) -> None:
        for name, value in (
            ("latency_s", self.latency_s),
            ("bandwidth_bytes", self.bandwidth_bytes),
            ("memory_bytes", self.memory_bytes),
            ("energy_j", self.energy_j),
            ("quality_loss", self.quality_loss),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class RuntimeWeights:
    latency: float = 1.0
    bandwidth: float = 0.0
    memory: float = 0.0
    energy: float = 0.0
    quality: float = 1.0

    def __post_init__(self) -> None:
        for value in (
            self.latency,
            self.bandwidth,
            self.memory,
            self.energy,
            self.quality,
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("runtime weights must be finite and non-negative")


def runtime_objective(cost: RuntimeCost, weights: RuntimeWeights = RuntimeWeights()) -> float:
    """Scalar objective for measured runtime-policy selection."""

    return (
        weights.latency * cost.latency_s
        + weights.bandwidth * cost.bandwidth_bytes
        + weights.memory * cost.memory_bytes
        + weights.energy * cost.energy_j
        + weights.quality * cost.quality_loss
    )


def accept_runtime_candidate(
    baseline: RuntimeCost,
    candidate: RuntimeCost,
    *,
    max_quality_loss: float,
    weights: RuntimeWeights = RuntimeWeights(),
) -> bool:
    """Accept a self-optimization only if it is measured better and safe.

    The candidate must stay inside the explicit quality bound and strictly
    improve the configured scalar objective.  This prevents the meta-loop from
    treating a theoretical spatial reduction as an achieved runtime speedup.
    """

    if max_quality_loss < 0.0 or not math.isfinite(max_quality_loss):
        raise ValueError("max_quality_loss must be finite and non-negative")
    if candidate.quality_loss > max_quality_loss:
        return False
    return runtime_objective(candidate, weights) < runtime_objective(baseline, weights)


@dataclass(frozen=True)
class RuntimePolicy:
    """Backend-neutral execution controls optimized by the outer meta-loop."""

    tile_side: int = 16
    precision_bits: int = 16
    recursion_depth: int = 2
    sparsity_threshold: float = 1.0e-3
    fused_operators: bool = True

    def __post_init__(self) -> None:
        if self.tile_side <= 0:
            raise ValueError("tile_side must be positive")
        if self.precision_bits not in {4, 8, 16, 32, 64}:
            raise ValueError("precision_bits must be one of 4, 8, 16, 32, 64")
        if self.recursion_depth < 0:
            raise ValueError("recursion_depth must be non-negative")
        if not math.isfinite(self.sparsity_threshold) or self.sparsity_threshold < 0.0:
            raise ValueError("sparsity_threshold must be finite and non-negative")


def count_active_errors(errors: Iterable[float], threshold: float) -> int:
    """Count residual elements whose measured error warrants fine refinement."""

    if threshold < 0.0 or not math.isfinite(threshold):
        raise ValueError("threshold must be finite and non-negative")
    count = 0
    for error in errors:
        if not math.isfinite(error) or error < 0.0:
            raise ValueError("errors must be finite and non-negative")
        if error > threshold:
            count += 1
    return count
