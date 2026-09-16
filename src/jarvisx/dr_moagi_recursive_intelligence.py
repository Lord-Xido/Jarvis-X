"""Recursive 3D multimodal intelligence orchestration layer.

This module composes the bounded multimodal codec and recursive inward-folded
swarm into a frequency-modulated execution model.  It keeps three quantities
separate:

1. physical/modelled codec work,
2. represented information enabled by abstraction, and
3. sparse-work reduction enabled by inward folding.

All throughput values in this module are analytical/modelled quantities.  They
are not wall-clock benchmark results.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from typing import Optional, Sequence

from .dr_moagi_inward_swarm import (
    InwardSwarmConfig,
    InwardSwarmMetrics,
    VirtualExavoxelInwardSwarm,
)


@dataclass
class RecursiveIntelligenceConfig:
    """Frequency, information-mass, and abstraction configuration."""

    modeled_codec_capacity_hz: float = 1.0e12
    min_local_frequency_hz: float = 1.0
    max_local_frequency_hz: float = 144.0
    frequency_kappa: float = 0.025
    information_mass_bytes_per_event: float = 512.0
    abstraction_gain: float = 1_000.0
    execution_efficiency: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.modeled_codec_capacity_hz) or self.modeled_codec_capacity_hz <= 0:
            raise ValueError("modeled_codec_capacity_hz must be finite and positive")
        if not math.isfinite(self.min_local_frequency_hz) or self.min_local_frequency_hz < 0:
            raise ValueError("min_local_frequency_hz must be finite and non-negative")
        if (
            not math.isfinite(self.max_local_frequency_hz)
            or self.max_local_frequency_hz <= self.min_local_frequency_hz
        ):
            raise ValueError("max_local_frequency_hz must exceed min_local_frequency_hz")
        if not math.isfinite(self.frequency_kappa) or self.frequency_kappa <= 0:
            raise ValueError("frequency_kappa must be finite and positive")
        if (
            not math.isfinite(self.information_mass_bytes_per_event)
            or self.information_mass_bytes_per_event <= 0
        ):
            raise ValueError("information_mass_bytes_per_event must be finite and positive")
        if not math.isfinite(self.abstraction_gain) or self.abstraction_gain < 1:
            raise ValueError("abstraction_gain must be finite and >= 1")
        if (
            not math.isfinite(self.execution_efficiency)
            or not 0 < self.execution_efficiency <= 1
        ):
            raise ValueError("execution_efficiency must be in (0, 1]")


@dataclass
class RecursiveIntelligenceMetrics:
    """One verified cycle of the recursive intelligence orchestration layer."""

    cycle: int
    residual_rms: float
    residual_threshold: float
    active_refinement_fraction: float
    modeled_work_reduction: float
    max_modeled_work_reduction: float
    allocated_local_frequency_hz: float
    normalized_frequency: float
    modeled_codec_capacity_hz: float
    modeled_scheduled_event_rate_hz: float
    information_mass_bytes_per_event: float
    abstraction_gain: float
    execution_efficiency: float
    modeled_physical_bytes_per_second: float
    modeled_represented_bytes_per_second: float
    modeled_reallocated_ceiling_bytes_per_second: float
    fixed_point_delta: float
    converged: bool
    geometry_verified: bool
    work_model_verified: bool
    frequency_verified: bool
    throughput_verified: bool
    operational_mechanics_verified: bool


def frequency_modulation(
    residual: float,
    threshold: float,
    min_frequency_hz: float,
    max_frequency_hz: float,
    kappa: float,
) -> float:
    """Map residual magnitude to a bounded local execution frequency.

    f(r) = f_min + (f_max - f_min) * sigmoid((r - tau) / kappa)

    The sigmoid makes the schedule continuous while preserving monotonicity:
    larger residuals receive at least as much update frequency as smaller ones.
    """

    values = (residual, threshold, min_frequency_hz, max_frequency_hz, kappa)
    if not all(math.isfinite(v) for v in values):
        raise ValueError("frequency inputs must be finite")
    if residual < 0 or threshold < 0:
        raise ValueError("residual and threshold must be non-negative")
    if min_frequency_hz < 0 or max_frequency_hz <= min_frequency_hz:
        raise ValueError("invalid frequency bounds")
    if kappa <= 0:
        raise ValueError("kappa must be positive")

    x = max(-60.0, min(60.0, (residual - threshold) / kappa))
    sigma = 1.0 / (1.0 + math.exp(-x))
    return min_frequency_hz + (max_frequency_hz - min_frequency_hz) * sigma


def modeled_information_throughput(
    event_rate_hz: float,
    information_mass_bytes_per_event: float,
    abstraction_gain: float,
    execution_efficiency: float,
) -> tuple[float, float]:
    """Return modeled physical and represented information rates in bytes/s."""

    values = (
        event_rate_hz,
        information_mass_bytes_per_event,
        abstraction_gain,
        execution_efficiency,
    )
    if not all(math.isfinite(v) for v in values):
        raise ValueError("throughput inputs must be finite")
    if event_rate_hz < 0:
        raise ValueError("event_rate_hz must be non-negative")
    if information_mass_bytes_per_event <= 0:
        raise ValueError("information_mass_bytes_per_event must be positive")
    if abstraction_gain < 1:
        raise ValueError("abstraction_gain must be >= 1")
    if not 0 < execution_efficiency <= 1:
        raise ValueError("execution_efficiency must be in (0, 1]")

    physical = event_rate_hz * information_mass_bytes_per_event * execution_efficiency
    represented = physical * abstraction_gain
    return physical, represented


def verify_frequency_allocation(
    residual: float,
    threshold: float,
    config: RecursiveIntelligenceConfig,
    frequency_hz: float,
) -> bool:
    """Verify bounds and exact agreement with the frequency law."""

    expected = frequency_modulation(
        residual,
        threshold,
        config.min_local_frequency_hz,
        config.max_local_frequency_hz,
        config.frequency_kappa,
    )
    return (
        math.isfinite(frequency_hz)
        and config.min_local_frequency_hz <= frequency_hz <= config.max_local_frequency_hz
        and math.isclose(frequency_hz, expected, rel_tol=1e-12, abs_tol=1e-12)
    )


def verify_throughput(
    metrics: RecursiveIntelligenceMetrics,
) -> bool:
    """Check dimensional identities for modeled physical/represented throughput."""

    cfg_values = (
        metrics.modeled_codec_capacity_hz,
        metrics.modeled_scheduled_event_rate_hz,
        metrics.information_mass_bytes_per_event,
        metrics.abstraction_gain,
        metrics.execution_efficiency,
        metrics.modeled_physical_bytes_per_second,
        metrics.modeled_represented_bytes_per_second,
        metrics.modeled_reallocated_ceiling_bytes_per_second,
    )
    if not all(math.isfinite(v) for v in cfg_values):
        return False

    expected_physical = (
        metrics.modeled_scheduled_event_rate_hz
        * metrics.information_mass_bytes_per_event
        * metrics.execution_efficiency
    )
    expected_represented = expected_physical * metrics.abstraction_gain
    expected_ceiling = (
        metrics.modeled_codec_capacity_hz
        * metrics.normalized_frequency
        * metrics.information_mass_bytes_per_event
        * metrics.abstraction_gain
        * metrics.modeled_work_reduction
        * metrics.execution_efficiency
    )
    return (
        math.isclose(
            metrics.modeled_physical_bytes_per_second,
            expected_physical,
            rel_tol=1e-12,
            abs_tol=1e-9,
        )
        and math.isclose(
            metrics.modeled_represented_bytes_per_second,
            expected_represented,
            rel_tol=1e-12,
            abs_tol=1e-9,
        )
        and math.isclose(
            metrics.modeled_reallocated_ceiling_bytes_per_second,
            expected_ceiling,
            rel_tol=1e-12,
            abs_tol=1e-9,
        )
    )


class Recursive3DMultimodalIntelligenceEngine:
    """End-to-end bounded orchestration of the inward 3D multimodal swarm.

    The implementation deliberately treats large event-rate and abstraction
    quantities as *modeled capacity/equivalent-work metrics*.  Actual hardware
    throughput must be established with profilers and wall-clock benchmarks.
    """

    def __init__(
        self,
        config: Optional[RecursiveIntelligenceConfig] = None,
        swarm: Optional[VirtualExavoxelInwardSwarm] = None,
        swarm_config: Optional[InwardSwarmConfig] = None,
    ) -> None:
        self.config = config or RecursiveIntelligenceConfig()
        if swarm is not None and swarm_config is not None:
            raise ValueError("provide swarm or swarm_config, not both")
        self.swarm = swarm or VirtualExavoxelInwardSwarm(config=swarm_config)

    def _orchestrate(self, inward: InwardSwarmMetrics) -> RecursiveIntelligenceMetrics:
        cfg = self.config
        frequency = frequency_modulation(
            inward.proxy_residual_rms,
            inward.residual_threshold,
            cfg.min_local_frequency_hz,
            cfg.max_local_frequency_hz,
            cfg.frequency_kappa,
        )
        normalized_frequency = frequency / cfg.max_local_frequency_hz

        # The sparse work fraction is the reciprocal of the inward work reduction.
        # This preserves a strict separation between available logical capacity and
        # the fraction of that capacity physically scheduled in this cycle.
        work_fraction = 1.0 / inward.modeled_work_reduction
        scheduled_event_rate = cfg.modeled_codec_capacity_hz * normalized_frequency * work_fraction

        physical, represented = modeled_information_throughput(
            scheduled_event_rate,
            cfg.information_mass_bytes_per_event,
            cfg.abstraction_gain,
            cfg.execution_efficiency,
        )

        # If saved sparse capacity were perfectly reusable for independent work,
        # this is the analytical upper ceiling. It is not a benchmark result.
        reallocated_ceiling = (
            cfg.modeled_codec_capacity_hz
            * normalized_frequency
            * cfg.information_mass_bytes_per_event
            * cfg.abstraction_gain
            * inward.modeled_work_reduction
            * cfg.execution_efficiency
        )

        frequency_ok = verify_frequency_allocation(
            inward.proxy_residual_rms,
            inward.residual_threshold,
            cfg,
            frequency,
        )

        result = RecursiveIntelligenceMetrics(
            cycle=inward.cycle,
            residual_rms=inward.proxy_residual_rms,
            residual_threshold=inward.residual_threshold,
            active_refinement_fraction=inward.active_refinement_fraction,
            modeled_work_reduction=inward.modeled_work_reduction,
            max_modeled_work_reduction=inward.max_modeled_work_reduction,
            allocated_local_frequency_hz=frequency,
            normalized_frequency=normalized_frequency,
            modeled_codec_capacity_hz=cfg.modeled_codec_capacity_hz,
            modeled_scheduled_event_rate_hz=scheduled_event_rate,
            information_mass_bytes_per_event=cfg.information_mass_bytes_per_event,
            abstraction_gain=cfg.abstraction_gain,
            execution_efficiency=cfg.execution_efficiency,
            modeled_physical_bytes_per_second=physical,
            modeled_represented_bytes_per_second=represented,
            modeled_reallocated_ceiling_bytes_per_second=reallocated_ceiling,
            fixed_point_delta=inward.fixed_point_delta,
            converged=inward.converged,
            geometry_verified=inward.geometry_verified,
            work_model_verified=inward.work_model_verified,
            frequency_verified=frequency_ok,
            throughput_verified=False,
            operational_mechanics_verified=False,
        )
        result.throughput_verified = verify_throughput(result)
        result.operational_mechanics_verified = (
            inward.operational_mechanics_verified
            and result.frequency_verified
            and result.throughput_verified
        )
        return result

    def step(
        self,
        virtual_ops: Optional[int] = None,
        auto_optimize: bool = True,
    ) -> RecursiveIntelligenceMetrics:
        inward = self.swarm.step(virtual_ops=virtual_ops, auto_optimize=auto_optimize)
        return self._orchestrate(inward)

    def run(
        self,
        cycles: int,
        deterministic_virtual_stride: Optional[int] = None,
        auto_optimize: bool = True,
    ) -> list[RecursiveIntelligenceMetrics]:
        if cycles < 1:
            raise ValueError("cycles must be >= 1")
        if deterministic_virtual_stride is not None and deterministic_virtual_stride < 0:
            raise ValueError("deterministic_virtual_stride must be >= 0")
        out: list[RecursiveIntelligenceMetrics] = []
        for i in range(cycles):
            virtual_ops = None
            if deterministic_virtual_stride is not None:
                virtual_ops = i * deterministic_virtual_stride
            out.append(self.step(virtual_ops=virtual_ops, auto_optimize=auto_optimize))
        return out


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Recursive 3D multimodal intelligence engine")
    p.add_argument("--cycles", type=int, default=8)
    p.add_argument("--capacity-hz", type=float, default=1.0e12)
    p.add_argument("--min-frequency", type=float, default=1.0)
    p.add_argument("--max-frequency", type=float, default=144.0)
    p.add_argument("--information-mass-bytes", type=float, default=512.0)
    p.add_argument("--abstraction-gain", type=float, default=1_000.0)
    p.add_argument("--efficiency", type=float, default=1.0)
    p.add_argument("--proxy-agents", type=int, default=2_000)
    p.add_argument("--fold-edge", type=int, default=10)
    p.add_argument("--fold-levels", type=int, default=2)
    p.add_argument("--threshold", type=float, default=0.12)
    p.add_argument("--deterministic-stride", type=int, default=1_000_000)
    p.add_argument("--no-auto-optimize", action="store_true")
    p.add_argument("--json", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    config = RecursiveIntelligenceConfig(
        modeled_codec_capacity_hz=args.capacity_hz,
        min_local_frequency_hz=args.min_frequency,
        max_local_frequency_hz=args.max_frequency,
        information_mass_bytes_per_event=args.information_mass_bytes,
        abstraction_gain=args.abstraction_gain,
        execution_efficiency=args.efficiency,
    )
    swarm_config = InwardSwarmConfig(
        proxy_agents=args.proxy_agents,
        fold_edge=args.fold_edge,
        fold_levels=args.fold_levels,
        residual_threshold=args.threshold,
    )
    engine = Recursive3DMultimodalIntelligenceEngine(config=config, swarm_config=swarm_config)
    metrics = engine.run(
        args.cycles,
        deterministic_virtual_stride=args.deterministic_stride,
        auto_optimize=not args.no_auto_optimize,
    )

    if args.json:
        print(json.dumps([asdict(m) for m in metrics], indent=2, sort_keys=True))
    else:
        last = metrics[-1]
        print(
            "cycle={cycle} active={active:.6f} f={frequency:.3f}Hz "
            "work_reduction={reduction:.3f}x scheduled={scheduled:.6e}/s "
            "physical={physical:.6e}B/s represented={represented:.6e}B/s "
            "verified={verified}".format(
                cycle=last.cycle,
                active=last.active_refinement_fraction,
                frequency=last.allocated_local_frequency_hz,
                reduction=last.modeled_work_reduction,
                scheduled=last.modeled_scheduled_event_rate_hz,
                physical=last.modeled_physical_bytes_per_second,
                represented=last.modeled_represented_bytes_per_second,
                verified=last.operational_mechanics_verified,
            )
        )
        print("note: throughput values are analytical/modelled, not wall-clock benchmarks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
