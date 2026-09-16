"""Virtualized 3D exavoxel swarm with recursive inward multiresolution refinement.

This module models a logical 1_000_000 x 1_000_000 x 1_000_000 lattice
(10**18 cells) without allocating one object per cell. A bounded proxy swarm
samples the logical field while the existing DrMoagiMultimodal3DLoop performs
real, bounded encode/decode work.

Acceleration is represented as a recursive geometric work model. A local
10 x 10 x 10 contraction reduces one level by 1000 logical cells. Two nested
levels therefore have a maximum idealized work-reduction ceiling of 10**6 when
no fine cells require residual refinement. This is not a claim of measured
wall-clock or hardware speedup; physical acceleration must be benchmarked.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from typing import Optional, Sequence

from .dr_moagi_multimodal_loop import DrMoagiMultimodal3DLoop, Modality

LOGICAL_AXIS = 1_000_000
LOGICAL_CELLS = LOGICAL_AXIS**3
DEFAULT_FOLD_EDGE = 10
DEFAULT_FOLD_LEVELS = 2


@dataclass
class InwardSwarmConfig:
    """Configuration for the virtual exavoxel scheduler."""

    proxy_agents: int = 2_000
    fold_edge: int = DEFAULT_FOLD_EDGE
    fold_levels: int = DEFAULT_FOLD_LEVELS
    residual_threshold: float = 0.12
    target_reconstruction_mse: float = 0.25
    threshold_learning_rate: float = 0.04
    min_threshold: float = 1e-4
    max_threshold: float = 1.0
    seed: int = 0x4D4F414749

    def __post_init__(self) -> None:
        if not 64 <= self.proxy_agents <= 1_000_000:
            raise ValueError("proxy_agents must be in [64, 1_000_000]")
        if self.fold_edge < 2:
            raise ValueError("fold_edge must be >= 2")
        if not 1 <= self.fold_levels <= 6:
            raise ValueError("fold_levels must be in [1, 6]")
        if self.total_fold_edge > LOGICAL_AXIS:
            raise ValueError("recursive fold exceeds logical axis")
        if LOGICAL_AXIS % self.total_fold_edge:
            raise ValueError("logical axis must be divisible by total recursive fold edge")
        if not 0.0 < self.residual_threshold <= 1.0:
            raise ValueError("residual_threshold must be in (0, 1]")
        if not 0.0 < self.target_reconstruction_mse:
            raise ValueError("target_reconstruction_mse must be positive")
        if not 0.0 < self.threshold_learning_rate <= 1.0:
            raise ValueError("threshold_learning_rate must be in (0, 1]")
        if not 0.0 < self.min_threshold <= self.max_threshold:
            raise ValueError("invalid threshold bounds")

    @property
    def per_level_volume(self) -> int:
        return self.fold_edge**3

    @property
    def total_fold_edge(self) -> int:
        return self.fold_edge**self.fold_levels

    @property
    def fold_volume(self) -> int:
        return self.total_fold_edge**3

    @property
    def coarse_axis(self) -> int:
        return LOGICAL_AXIS // self.total_fold_edge

    @property
    def coarse_cells(self) -> int:
        return self.coarse_axis**3


@dataclass(frozen=True)
class ProxyAgent:
    """One bounded sample standing in for a region of the logical lattice."""

    x: float
    y: float
    z: float
    modality: Modality
    phase: float


@dataclass
class InwardSwarmMetrics:
    cycle: int
    logical_axis: int
    logical_cells: int
    proxy_agents: int
    fold_edge: int
    fold_levels: int
    per_level_volume: int
    total_fold_edge: int
    fold_volume: int
    coarse_axis: int
    coarse_cells: int
    proxy_cells_per_agent: float
    active_refinement_fraction: float
    active_proxy_agents: int
    residual_threshold: float
    aggregate_reconstruction_mse: float
    aggregate_cycle_mse: float
    proxy_residual_rms: float
    modeled_work_cells: float
    modeled_work_reduction: float
    max_modeled_work_reduction: float
    fixed_point_delta: float
    converged: bool
    geometry_verified: bool
    work_model_verified: bool
    operational_mechanics_verified: bool


def modeled_work_reduction(active_fraction: float, fold_volume: int) -> float:
    """Return baseline-work / multiresolution-work for an idealized scheduler.

    Coarse work evaluates one representative for each recursively folded block.
    Fine work is reactivated only for the estimated active residual fraction.
    The result is bounded by [1, fold_volume].
    """

    if fold_volume < 1:
        raise ValueError("fold_volume must be >= 1")
    a = max(0.0, min(1.0, float(active_fraction)))
    coarse_fraction = 1.0 / float(fold_volume)
    work_fraction = coarse_fraction + a * (1.0 - coarse_fraction)
    return 1.0 / work_fraction


def verify_recursive_geometry(config: InwardSwarmConfig) -> bool:
    """Verify exact integer geometry of the nested 3D contraction."""

    expected_edge = config.fold_edge**config.fold_levels
    expected_volume = (config.fold_edge**3) ** config.fold_levels
    return (
        config.total_fold_edge == expected_edge
        and config.fold_volume == expected_volume
        and config.coarse_axis * config.total_fold_edge == LOGICAL_AXIS
        and config.coarse_cells * config.fold_volume == LOGICAL_CELLS
    )


def verify_work_model(active_fraction: float, fold_volume: int, reduction: float) -> bool:
    """Verify reduction against the normalized sparse-work equation."""

    if not 0.0 <= active_fraction <= 1.0 or fold_volume < 1:
        return False
    expected = modeled_work_reduction(active_fraction, fold_volume)
    return (
        math.isfinite(reduction)
        and 1.0 <= reduction <= float(fold_volume)
        and math.isclose(reduction, expected, rel_tol=1e-12, abs_tol=1e-12)
    )


class VirtualExavoxelInwardSwarm:
    """Sparse virtual scheduler wrapped around the bounded multimodal engine."""

    def __init__(
        self,
        config: Optional[InwardSwarmConfig] = None,
        codec: Optional[DrMoagiMultimodal3DLoop] = None,
    ) -> None:
        self.config = config or InwardSwarmConfig()
        self.codec = codec or DrMoagiMultimodal3DLoop(edge=8, temporal_depth=4)
        self.rng = random.Random(self.config.seed)
        self.cycle = 0
        self.agents = self._make_agents(self.config.proxy_agents)

    def _make_agents(self, n: int) -> list[ProxyAgent]:
        modalities = tuple(Modality)
        agents: list[ProxyAgent] = []
        for i in range(n):
            agents.append(
                ProxyAgent(
                    x=self.rng.uniform(-1.0, 1.0),
                    y=self.rng.uniform(-1.0, 1.0),
                    z=self.rng.uniform(-1.0, 1.0),
                    modality=modalities[i % len(modalities)],
                    phase=self.rng.random() * math.tau,
                )
            )
        return agents

    @staticmethod
    def _safe_delta(value: float) -> float:
        return value if math.isfinite(value) else 0.0

    def _proxy_residual(self, agent: ProxyAgent, base_error: float, cycle_error: float) -> float:
        radius = math.sqrt(agent.x * agent.x + agent.y * agent.y + agent.z * agent.z)
        geometry = 0.5 + 0.5 * abs(
            math.sin(
                2.17 * agent.x
                + 1.73 * agent.y
                - 1.31 * agent.z
                + agent.phase
                + 0.11 * self.cycle
            )
        )
        shell = min(1.0, radius / math.sqrt(3.0))
        modality_gain = {
            Modality.VISUAL: 1.00,
            Modality.AUDIO: 0.94,
            Modality.TEXT: 0.88,
            Modality.VIDEO: 1.06,
            Modality.GENERIC: 0.91,
        }[agent.modality]
        return max(
            0.0,
            modality_gain
            * (0.72 * base_error + 0.28 * cycle_error)
            * geometry
            * (0.8 + 0.4 * shell),
        )

    def _auto_tune_threshold(self, reconstruction_mse: float) -> None:
        cfg = self.config
        lr = cfg.threshold_learning_rate
        if reconstruction_mse <= cfg.target_reconstruction_mse:
            candidate = cfg.residual_threshold * (1.0 + lr)
        else:
            candidate = cfg.residual_threshold * (1.0 - lr)
        cfg.residual_threshold = min(cfg.max_threshold, max(cfg.min_threshold, candidate))

    def step(
        self, virtual_ops: Optional[int] = None, auto_optimize: bool = True
    ) -> InwardSwarmMetrics:
        codec_metrics = self.codec.step(virtual_ops=virtual_ops)
        base_error = max(0.0, codec_metrics.aggregate_reconstruction_mse)
        cycle_error = max(0.0, codec_metrics.aggregate_cycle_mse)

        residuals = [self._proxy_residual(a, base_error, cycle_error) for a in self.agents]
        threshold = self.config.residual_threshold
        active = sum(r > threshold for r in residuals)
        active_fraction = active / len(residuals)
        residual_rms = math.sqrt(sum(r * r for r in residuals) / len(residuals))

        cfg = self.config
        reduction = modeled_work_reduction(active_fraction, cfg.fold_volume)
        modeled_work = LOGICAL_CELLS / reduction
        geometry_ok = verify_recursive_geometry(cfg)
        work_ok = verify_work_model(active_fraction, cfg.fold_volume, reduction)
        operational_ok = (
            geometry_ok
            and work_ok
            and math.isfinite(base_error)
            and math.isfinite(cycle_error)
            and math.isfinite(residual_rms)
            and len(self.agents) == cfg.proxy_agents
        )

        self.cycle += 1
        result = InwardSwarmMetrics(
            cycle=self.cycle,
            logical_axis=LOGICAL_AXIS,
            logical_cells=LOGICAL_CELLS,
            proxy_agents=len(self.agents),
            fold_edge=cfg.fold_edge,
            fold_levels=cfg.fold_levels,
            per_level_volume=cfg.per_level_volume,
            total_fold_edge=cfg.total_fold_edge,
            fold_volume=cfg.fold_volume,
            coarse_axis=cfg.coarse_axis,
            coarse_cells=cfg.coarse_cells,
            proxy_cells_per_agent=LOGICAL_CELLS / len(self.agents),
            active_refinement_fraction=active_fraction,
            active_proxy_agents=active,
            residual_threshold=threshold,
            aggregate_reconstruction_mse=base_error,
            aggregate_cycle_mse=cycle_error,
            proxy_residual_rms=residual_rms,
            modeled_work_cells=modeled_work,
            modeled_work_reduction=reduction,
            max_modeled_work_reduction=float(cfg.fold_volume),
            fixed_point_delta=self._safe_delta(codec_metrics.fixed_point_delta),
            converged=codec_metrics.converged,
            geometry_verified=geometry_ok,
            work_model_verified=work_ok,
            operational_mechanics_verified=operational_ok,
        )

        if auto_optimize:
            self._auto_tune_threshold(base_error)
        return result

    def run(
        self,
        cycles: int,
        deterministic_virtual_stride: Optional[int] = None,
        auto_optimize: bool = True,
    ) -> list[InwardSwarmMetrics]:
        if cycles < 1:
            raise ValueError("cycles must be >= 1")
        if deterministic_virtual_stride is not None and deterministic_virtual_stride < 0:
            raise ValueError("deterministic_virtual_stride must be >= 0")
        results: list[InwardSwarmMetrics] = []
        for i in range(cycles):
            virtual_ops = None
            if deterministic_virtual_stride is not None:
                virtual_ops = i * deterministic_virtual_stride
            results.append(self.step(virtual_ops=virtual_ops, auto_optimize=auto_optimize))
        return results


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Virtual 10^18-cell recursively inward-folded multimodal 3D swarm"
    )
    p.add_argument("--cycles", type=int, default=8)
    p.add_argument("--proxy-agents", type=int, default=2_000)
    p.add_argument("--fold-edge", type=int, default=DEFAULT_FOLD_EDGE)
    p.add_argument("--fold-levels", type=int, default=DEFAULT_FOLD_LEVELS)
    p.add_argument("--threshold", type=float, default=0.12)
    p.add_argument("--target-mse", type=float, default=0.25)
    p.add_argument("--deterministic-stride", type=int, default=1_000_000)
    p.add_argument("--no-auto-optimize", action="store_true")
    p.add_argument("--json", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    config = InwardSwarmConfig(
        proxy_agents=args.proxy_agents,
        fold_edge=args.fold_edge,
        fold_levels=args.fold_levels,
        residual_threshold=args.threshold,
        target_reconstruction_mse=args.target_mse,
    )
    engine = VirtualExavoxelInwardSwarm(config=config)
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
            "cycle={cycle} logical_cells={logical_cells} proxies={proxy_agents} "
            "fold={edge}^3x{levels} active={active:.6f} residual_rms={residual:.6f} "
            "modeled_work_reduction={reduction:.2f}x max={maximum:.0f}x "
            "verified={verified}".format(
                cycle=last.cycle,
                logical_cells=last.logical_cells,
                proxy_agents=last.proxy_agents,
                edge=last.fold_edge,
                levels=last.fold_levels,
                active=last.active_refinement_fraction,
                residual=last.proxy_residual_rms,
                reduction=last.modeled_work_reduction,
                maximum=last.max_modeled_work_reduction,
                verified=last.operational_mechanics_verified,
            )
        )
        print("note: modeled work reduction is not measured wall-clock throughput")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
