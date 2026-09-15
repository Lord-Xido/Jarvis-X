"""Recursive 3D geometry optimization for the sparse Dr Moagi runtime.

This module implements a bounded, testable interpretation of recursive
self-optimization.  It searches *runtime configuration geometry* rather than
rewriting Python source code.

The 3D meta-state is

    g = (spatial, representation, dynamics) in [-1, 1]^3

and decodes into the existing ``DrMoagi3D1000xEngine`` knobs:

* spatial -> active sparse tile budget;
* representation -> latent rank and deterministic encoder basis seed;
* dynamics -> Omega decay and residual-correction gain.

Every candidate executes the full encode -> Omega -> decode -> residual ->
correction -> scheduling loop over a deterministic multimodal benchmark suite.
Promotion occurs only after an admissibility gate and a measured objective
improvement; otherwise the incumbent is retained (rollback).

No state-of-the-art claim is emitted without a supplied same-workload external
baseline.  Indefinite reiteration means repeated search/measure/commit-or-
rollback cycles, not infinite performance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .dr_moagi_3d_1000x import DrMoagi3D1000xEngine, EngineConfig

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional accel environments
    np = None  # type: ignore[assignment]


MIN_ACTIVE_TILES = 64
MAX_ACTIVE_TILES = 1000
MIN_LATENT_DIM = 2
MAX_LATENT_DIM = 8
EPS = 1.0e-12


def _require_numpy() -> Any:
    if np is None:
        raise RuntimeError(
            "recursive geometry optimization requires NumPy; install with "
            "`python -m pip install -e '.[accel]'`"
        )
    return np


@dataclass(frozen=True, slots=True)
class Geometry3D:
    spatial: float = 0.0
    representation: float = 0.0
    dynamics: float = 0.0

    def clipped(self) -> "Geometry3D":
        backend = _require_numpy()
        return Geometry3D(
            float(backend.clip(self.spatial, -1.0, 1.0)),
            float(backend.clip(self.representation, -1.0, 1.0)),
            float(backend.clip(self.dynamics, -1.0, 1.0)),
        )

    def array(self) -> Any:
        backend = _require_numpy()
        return backend.asarray(
            [self.spatial, self.representation, self.dynamics], dtype=backend.float64
        )


@dataclass(frozen=True, slots=True)
class ExternalBaseline:
    """Matched-workload thresholds used before any beyond-SOTA claim is allowed."""

    name: str
    max_residual_mse: float
    max_median_latency_ms: float
    max_active_voxels: int | None = None
    minimum_work_reduction: float | None = None

    @classmethod
    def from_json(cls, path: Path) -> "ExternalBaseline":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            name=str(payload["name"]),
            max_residual_mse=float(payload["max_residual_mse"]),
            max_median_latency_ms=float(payload["max_median_latency_ms"]),
            max_active_voxels=(
                None
                if payload.get("max_active_voxels") is None
                else int(payload["max_active_voxels"])
            ),
            minimum_work_reduction=(
                None
                if payload.get("minimum_work_reduction") is None
                else float(payload["minimum_work_reduction"])
            ),
        )


@dataclass(slots=True)
class CandidateMetrics:
    geometry: Geometry3D
    config: EngineConfig
    residual_mse: float
    median_latency_ms: float
    p95_latency_ms: float
    active_voxels: int
    work_reduction: float
    merit: float = math.inf
    admissible: bool = False
    beyond_external_baseline: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "geometry": asdict(self.geometry),
            "config": asdict(self.config),
            "residual_mse": self.residual_mse,
            "median_latency_ms": self.median_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "active_voxels": self.active_voxels,
            "work_reduction": self.work_reduction,
            "merit": self.merit,
            "admissible": self.admissible,
            "beyond_external_baseline": self.beyond_external_baseline,
        }


@dataclass(slots=True)
class SearchEvent:
    generation: int
    radius: float
    accepted: bool
    reason: str
    incumbent_before: CandidateMetrics
    candidate_best: CandidateMetrics
    incumbent_after: CandidateMetrics

    def as_dict(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "radius": self.radius,
            "accepted": self.accepted,
            "reason": self.reason,
            "incumbent_before": self.incumbent_before.as_dict(),
            "candidate_best": self.candidate_best.as_dict(),
            "incumbent_after": self.incumbent_after.as_dict(),
        }


class GeometryCodec:
    """Decode a continuous 3D meta-geometry into bounded runtime parameters."""

    @staticmethod
    def _unit(value: float) -> float:
        return 0.5 * (max(-1.0, min(1.0, float(value))) + 1.0)

    @classmethod
    def decode(cls, geometry: Geometry3D) -> EngineConfig:
        g = geometry.clipped()
        spatial = cls._unit(g.spatial)
        representation = cls._unit(g.representation)
        dynamics = cls._unit(g.dynamics)

        # Geometric movement corresponds to multiplicative active-work changes.
        log_min = math.log(MIN_ACTIVE_TILES)
        log_max = math.log(MAX_ACTIVE_TILES)
        active_tiles = int(
            round(math.exp(log_min + spatial * (log_max - log_min)))
        )
        active_tiles = max(MIN_ACTIVE_TILES, min(MAX_ACTIVE_TILES, active_tiles))

        latent_dim = int(
            round(
                MIN_LATENT_DIM
                + representation * (MAX_LATENT_DIM - MIN_LATENT_DIM)
            )
        )
        latent_dim = max(MIN_LATENT_DIM, min(MAX_LATENT_DIM, latent_dim))

        omega_decay = 0.60 + 0.39 * dynamics
        correction_gain = 0.05 + 0.40 * (1.0 - dynamics)

        # The sparse engine's orthogonal basis is deterministic from seed.  Map
        # geometry into seed space so representation geometry also explores the
        # encoder manifold without changing the engine's source contract.
        digest = hashlib.blake2s(
            f"{g.spatial:.8f}:{g.representation:.8f}:{g.dynamics:.8f}".encode(),
            digest_size=4,
        ).digest()
        seed = int.from_bytes(digest, "big")

        return EngineConfig(
            cube_edge=1000,
            tile_edge=10,
            active_fraction=0.001,
            latent_dim=latent_dim,
            omega_decay=omega_decay,
            residual_threshold=1.0e-3,
            correction_gain=correction_gain,
            max_active_tiles=active_tiles,
            seed=seed,
        )



def build_multimodal_suite() -> dict[str, bytes]:
    """Create deterministic, dependency-light representative byte workloads."""

    backend = _require_numpy()
    rng = backend.random.default_rng(20260915)
    yy, xx = backend.mgrid[0:96, 0:96]

    image = (
        127.5 + 75.0 * backend.sin(xx / 8.0) + 35.0 * backend.cos(yy / 11.0)
    ).clip(0, 255).astype(backend.uint8).tobytes()

    t = backend.linspace(0.0, 1.0, 12000, endpoint=False)
    audio = (
        127.5
        + 55.0 * backend.sin(2.0 * backend.pi * 220.0 * t)
        + 20.0 * backend.sin(2.0 * backend.pi * 440.0 * t)
    ).clip(0, 255).astype(backend.uint8).tobytes()

    video = b"".join(
        (
            127.5
            + 60.0 * backend.sin((xx + frame * 4) / 9.0)
            + 35.0 * backend.cos((yy - frame * 3) / 13.0)
        ).clip(0, 255).astype(backend.uint8).tobytes()
        for frame in range(8)
    )

    points = rng.normal(size=(2048, 3)).astype(backend.float32)
    radius = backend.linalg.norm(points, axis=1, keepdims=True)
    points /= backend.maximum(radius, 1.0e-6)

    return {
        "text": (
            b"Dr Moagi sparse recursive engine encode latent memory decode "
            b"residual verify correct schedule recur " * 24
        ),
        "code": (
            b"def encode(x,w): return x@w\ndef decode(z,w): return z@w.T\n" * 48
        ),
        "image": image,
        "audio": audio,
        "video": video,
        "3d": points.tobytes(),
        "binary": rng.integers(0, 256, size=16384, dtype=backend.uint8).tobytes(),
    }


class GeometryEvaluator:
    """Evaluate one geometry using the existing authoritative sparse engine."""

    def __init__(
        self,
        suite: dict[str, bytes] | None = None,
        *,
        warmup_cycles: int = 1,
        measured_cycles: int = 2,
        external_baseline: ExternalBaseline | None = None,
    ) -> None:
        self.suite = suite or build_multimodal_suite()
        self.warmup_cycles = max(0, int(warmup_cycles))
        self.measured_cycles = max(1, int(measured_cycles))
        self.external_baseline = external_baseline
        self.reference: CandidateMetrics | None = None

    def evaluate(self, geometry: Geometry3D) -> CandidateMetrics:
        backend = _require_numpy()
        config = GeometryCodec.decode(geometry)
        residuals: list[float] = []
        latencies: list[float] = []
        active_voxels = 0
        work_reduction = 0.0

        for payload in self.suite.values():
            engine = DrMoagi3D1000xEngine(config)
            engine.ingest_bytes(payload)
            for _ in range(self.warmup_cycles):
                engine.cycle()
            measured = [engine.cycle() for _ in range(self.measured_cycles)]
            residuals.append(float(measured[-1].residual_mse))
            latencies.extend(float(item.total_ms) for item in measured)
            active_voxels = int(measured[-1].active_voxels)
            work_reduction = float(measured[-1].work_reduction_factor)

        result = CandidateMetrics(
            geometry=geometry.clipped(),
            config=config,
            residual_mse=float(statistics.fmean(residuals)),
            median_latency_ms=float(statistics.median(latencies)),
            p95_latency_ms=float(backend.percentile(latencies, 95)),
            active_voxels=active_voxels,
            work_reduction=work_reduction,
        )

        if self.reference is None:
            self.reference = result
        reference = self.reference
        result.merit = self._merit(result, reference)
        result.admissible = self._admissible(result, reference)
        result.beyond_external_baseline = self._beats_external(result)
        return result

    @staticmethod
    def _merit(candidate: CandidateMetrics, reference: CandidateMetrics) -> float:
        # Stable internal objective: quality dominates, active work is explicit.
        # Latency is measured and externally gateable but is excluded from this
        # deterministic internal ranking to avoid promoting timer noise in CI.
        quality = candidate.residual_mse / max(reference.residual_mse, EPS)
        work = candidate.active_voxels / max(reference.active_voxels, 1)
        return float(0.72 * quality + 0.28 * work)

    @staticmethod
    def _admissible(candidate: CandidateMetrics, reference: CandidateMetrics) -> bool:
        values = (
            candidate.residual_mse,
            candidate.median_latency_ms,
            candidate.p95_latency_ms,
            candidate.work_reduction,
        )
        if not all(math.isfinite(value) for value in values):
            return False
        if candidate.residual_mse > reference.residual_mse * 1.25:
            return False
        return True

    def _beats_external(self, candidate: CandidateMetrics) -> bool:
        baseline = self.external_baseline
        if baseline is None:
            return False
        if candidate.residual_mse > baseline.max_residual_mse:
            return False
        if candidate.median_latency_ms > baseline.max_median_latency_ms:
            return False
        if (
            baseline.max_active_voxels is not None
            and candidate.active_voxels > baseline.max_active_voxels
        ):
            return False
        if (
            baseline.minimum_work_reduction is not None
            and candidate.work_reduction < baseline.minimum_work_reduction
        ):
            return False
        return True


class RecursiveGeometryOptimizer:
    """Recursive shell search with explicit commit/rollback semantics."""

    def __init__(
        self,
        evaluator: GeometryEvaluator,
        *,
        initial_geometry: Geometry3D = Geometry3D(),
        initial_radius: float = 0.50,
        minimum_radius: float = 0.01,
        commit_margin: float = 0.005,
    ) -> None:
        if initial_radius <= 0.0 or minimum_radius <= 0.0:
            raise ValueError("search radii must be positive")
        self.evaluator = evaluator
        self.initial_radius = float(initial_radius)
        self.minimum_radius = float(minimum_radius)
        self.commit_margin = float(commit_margin)
        self.radius = self.initial_radius
        self.generation = 0
        self.incumbent = evaluator.evaluate(initial_geometry)
        self.best_ever = self.incumbent
        self.events: list[SearchEvent] = []

    @staticmethod
    def shell(center: Geometry3D, radius: float) -> tuple[Geometry3D, ...]:
        backend = _require_numpy()
        origin = center.array()
        candidates: dict[tuple[float, float, float], Geometry3D] = {}

        for dx in (-1.0, 0.0, 1.0):
            for dy in (-1.0, 0.0, 1.0):
                for dz in (-1.0, 0.0, 1.0):
                    if dx == dy == dz == 0.0:
                        continue
                    direction = backend.asarray([dx, dy, dz], dtype=backend.float64)
                    direction /= max(float(backend.linalg.norm(direction)), EPS)
                    geometry = Geometry3D(*(origin + radius * direction)).clipped()
                    key = tuple(round(float(value), 8) for value in geometry.array())
                    candidates[key] = geometry

        # Non-Cartesian helical directions reduce axis-locking.
        for index in range(12):
            phi = 2.0 * math.pi * index / 12.0
            direction = backend.asarray(
                [math.cos(phi), math.sin(phi), math.sin(2.0 * phi)],
                dtype=backend.float64,
            )
            direction /= max(float(backend.linalg.norm(direction)), EPS)
            geometry = Geometry3D(*(origin + radius * direction)).clipped()
            key = tuple(round(float(value), 8) for value in geometry.array())
            candidates[key] = geometry

        return tuple(candidates.values())

    def step(self) -> SearchEvent:
        before = self.incumbent
        candidates = [self.evaluator.evaluate(item) for item in self.shell(before.geometry, self.radius)]
        admissible = [item for item in candidates if item.admissible]
        best = min(admissible, key=lambda item: item.merit) if admissible else before

        required = before.merit * (1.0 - self.commit_margin)
        accepted = bool(best is not before and best.merit < required)
        if accepted:
            self.incumbent = best
            if best.merit < self.best_ever.merit:
                self.best_ever = best
            self.radius = min(0.75, self.radius * 1.08)
            reason = "commit: measured admissible merit improvement"
        else:
            self.radius *= 0.55
            reason = "rollback: no admissible improvement above commit margin"

        self.generation += 1
        event = SearchEvent(
            generation=self.generation,
            radius=self.radius,
            accepted=accepted,
            reason=reason,
            incumbent_before=before,
            candidate_best=best,
            incumbent_after=self.incumbent,
        )
        self.events.append(event)
        return event

    def optimize(
        self,
        *,
        generations: int = 6,
        forever: bool = False,
        plateau_limit: int = 4,
    ) -> CandidateMetrics:
        rejected_streak = 0
        try:
            while forever or self.generation < generations:
                event = self.step()
                rejected_streak = 0 if event.accepted else rejected_streak + 1
                if not forever and (
                    self.radius < self.minimum_radius or rejected_streak >= plateau_limit
                ):
                    break
                if forever and self.radius < self.minimum_radius:
                    self.radius = max(self.minimum_radius, self.initial_radius * 0.35)
                    rejected_streak = 0
        except KeyboardInterrupt:
            pass
        return self.best_ever

    def report(self) -> dict[str, Any]:
        baseline = self.evaluator.reference
        if baseline is None:  # pragma: no cover - constructor evaluates baseline
            raise RuntimeError("optimizer has no baseline")
        return {
            "protocol": "jarvisx.dr-moagi-recursive-geometry.v1",
            "claims": {
                "recursive_self_optimization": True,
                "source_self_modification": False,
                "indefinite_reiteration_supported": True,
                "infinite_performance_claim": False,
                "beyond_sota_requires_external_matched_baseline": True,
                "external_sota_baseline_supplied": self.evaluator.external_baseline is not None,
                "external_sota_gate_passed": self.best_ever.beyond_external_baseline,
            },
            "loop": (
                "g_t -> decode(g_t) -> E -> Z -> Omega -> D -> residual -> "
                "benchmark -> admissibility -> commit/rollback -> g_(t+1)"
            ),
            "baseline": baseline.as_dict(),
            "best": self.best_ever.as_dict(),
            "generations": self.generation,
            "accepted_generations": sum(event.accepted for event in self.events),
            "events": [event.as_dict() for event in self.events],
        }



def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dr Moagi recursive 3D geometry optimizer")
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--forever", action="store_true")
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("dr_moagi_recursive_geometry.json"))
    return parser


def main() -> int:
    args = _parser().parse_args()
    external = None if args.baseline is None else ExternalBaseline.from_json(args.baseline)
    evaluator = GeometryEvaluator(external_baseline=external)
    optimizer = RecursiveGeometryOptimizer(evaluator)
    optimizer.optimize(generations=max(0, args.generations), forever=args.forever)
    report = optimizer.report()
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


__all__ = [
    "CandidateMetrics",
    "ExternalBaseline",
    "Geometry3D",
    "GeometryCodec",
    "GeometryEvaluator",
    "RecursiveGeometryOptimizer",
    "SearchEvent",
    "build_multimodal_suite",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
