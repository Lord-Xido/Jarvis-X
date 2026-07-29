"""Deterministic architecture governor for the Jarvis-X meta-volume renderer.

This is a control-plane prototype, not a NeRF/3DGS renderer or a benchmark claim.
It evolves bounded depth, width, sample counts, and pruning masks, then commits
only verified non-regressive structural changes.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from typing import Iterable, Optional, Sequence, Tuple


def _mean(values: Sequence[float]) -> float:
    return sum(values) / max(1, len(values))


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


@dataclass(frozen=True)
class MetaVolumeConfig:
    region_count: int = 64
    parameter_count: int = 256
    min_depth: int = 4
    max_depth: int = 12
    min_width: int = 32
    max_width: int = 256
    min_steps: int = 4
    max_steps: int = 128
    learning_rate: float = 0.22
    lambda_compute: float = 0.035
    lambda_memory: float = 0.025
    minimum_active_ratio: float = 0.20
    mask_threshold: float = 0.50
    commit_tolerance: float = 1.0e-10
    max_memory_mb: float = 24576.0
    target_frame_ms: float = 16.667

    def validate(self) -> None:
        scalars = (
            self.learning_rate,
            self.lambda_compute,
            self.lambda_memory,
            self.minimum_active_ratio,
            self.mask_threshold,
            self.commit_tolerance,
            self.max_memory_mb,
            self.target_frame_ms,
        )
        if not all(math.isfinite(float(v)) for v in scalars):
            raise ValueError("configuration must be finite")
        if self.region_count < 1 or self.parameter_count < 1:
            raise ValueError("region_count and parameter_count must be positive")
        if not 1 <= self.min_depth <= self.max_depth:
            raise ValueError("invalid depth bounds")
        if (
            not 1 <= self.min_width <= self.max_width
            or self.min_width % 16
            or self.max_width % 16
        ):
            raise ValueError("width bounds must be ordered multiples of 16")
        if not 1 <= self.min_steps <= self.max_steps:
            raise ValueError("invalid step bounds")
        if not 0 < self.learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1]")
        if self.lambda_compute < 0 or self.lambda_memory < 0:
            raise ValueError("cost penalties must be non-negative")
        if not 0 < self.minimum_active_ratio <= 1:
            raise ValueError("minimum_active_ratio must be in (0, 1]")
        if not 0 < self.mask_threshold < 1:
            raise ValueError("mask_threshold must be in (0, 1)")
        if (
            self.commit_tolerance < 0
            or self.max_memory_mb <= 0
            or self.target_frame_ms <= 0
        ):
            raise ValueError("invalid transaction or hardware budget")


@dataclass(frozen=True)
class FrameSignals:
    error: Tuple[float, ...]
    edge_density: Tuple[float, ...]
    occupancy: Tuple[float, ...]

    def validate(self, count: int) -> None:
        if not (
            len(self.error)
            == len(self.edge_density)
            == len(self.occupancy)
            == count
        ):
            raise ValueError("signal vectors must match region_count")
        for vector in (self.error, self.edge_density, self.occupancy):
            if not all(math.isfinite(v) and 0 <= v <= 1 for v in vector):
                raise ValueError("signals must be finite values in [0, 1]")

    def complexity(self) -> Tuple[float, ...]:
        return tuple(
            _clamp(0.55 * e + 0.25 * g + 0.20 * o, 0, 1)
            for e, g, o in zip(self.error, self.edge_density, self.occupancy)
        )


@dataclass(frozen=True)
class HardwareTelemetry:
    frame_ms: float
    flops: float
    memory_mb: float
    sm_cycles: float = 0.0

    def validate(self) -> None:
        if not all(math.isfinite(v) and v >= 0 for v in asdict(self).values()):
            raise ValueError("hardware telemetry must be finite and non-negative")


@dataclass(frozen=True)
class ArchitectureDNA:
    depth: Tuple[int, ...]
    width: Tuple[int, ...]
    kernel: Tuple[int, ...]
    revision: int = 0


@dataclass(frozen=True)
class MetaGradients:
    depth: Tuple[float, ...]
    width: Tuple[float, ...]
    steps: Tuple[float, ...]
    mask: Tuple[float, ...]


@dataclass(frozen=True)
class MetaEvaluation:
    render_loss: float
    compute_cost: float
    memory_cost: float
    objective: float
    psnr_proxy_db: float
    active_ratio: float
    average_depth: float
    average_width: float
    average_steps: float


@dataclass(frozen=True)
class StructuralInstruction:
    opcode: str
    index: int
    old_value: float
    new_value: float


@dataclass(frozen=True)
class ParetoPoint:
    name: str
    psnr_db: float
    flops: float
    memory_mb: float


@dataclass(frozen=True)
class MetaVolumeState:
    architecture: ArchitectureDNA
    step_map: Tuple[int, ...]
    mask_scores: Tuple[float, ...]
    mask: Tuple[bool, ...]
    meta_gradients: MetaGradients
    cycle: int
    journal_hash: str
    last_committed: bool
    rollback_reason: Optional[str] = None

    def layer_manifest(self) -> Tuple[dict, ...]:
        fixed = (
            (0, "input_pixel_buffer", [512, 512, 3]),
            (1, "encoded_latent_volume", [64, 64, 256]),
            (2, "decoded_pixel_buffer", [512, 512, 3]),
            (3, "error_volume", [512, 512, 3]),
            (4, "gradient_volume", [512, 512, 256]),
            (5, "bytecode_and_weights", [128, 128, 64]),
            (6, "shader_registers", [256, 256, 16]),
            (7, "rendered_self_image", [512, 512, 4]),
        )
        layers = [
            {"layer": i, "name": name, "shape": shape}
            for i, name, shape in fixed
        ]
        layers.extend(
            [
                {
                    "layer": 8,
                    "name": "architecture_dna",
                    "logical_entries": len(self.architecture.depth),
                },
                {
                    "layer": 9,
                    "name": "step_allocation_map",
                    "logical_entries": len(self.step_map),
                },
                {
                    "layer": 10,
                    "name": "sparsity_mask",
                    "logical_entries": len(self.mask),
                },
                {
                    "layer": 11,
                    "name": "meta_gradient_accumulator",
                    "logical_entries": sum(
                        len(v)
                        for v in (
                            self.meta_gradients.depth,
                            self.meta_gradients.width,
                            self.meta_gradients.steps,
                            self.meta_gradients.mask,
                        )
                    ),
                },
            ]
        )
        return tuple(layers)


@dataclass(frozen=True)
class EvolutionResult:
    committed: bool
    state: MetaVolumeState
    baseline: MetaEvaluation
    candidate: MetaEvaluation
    instructions: Tuple[StructuralInstruction, ...]


class SelfEvolutionaryMetaVolume:
    """Projected meta-update with deterministic transaction semantics."""

    def __init__(self, config: Optional[MetaVolumeConfig] = None) -> None:
        self.config = config or MetaVolumeConfig()
        self.config.validate()
        rc, pc = self.config.region_count, self.config.parameter_count
        zeros_r, zeros_m = (0.0,) * rc, (0.0,) * pc
        scores = (0.75,) * pc
        self.state = MetaVolumeState(
            ArchitectureDNA((8,) * rc, (128,) * rc, (3,) * rc),
            (96,) * rc,
            scores,
            tuple(v >= self.config.mask_threshold for v in scores),
            MetaGradients(zeros_r, zeros_r, zeros_r, zeros_m),
            0,
            "0" * 64,
            True,
        )

    @staticmethod
    def _active_ratio(mask: Sequence[bool]) -> float:
        return sum(mask) / len(mask)

    def evaluate(
        self,
        state: MetaVolumeState,
        signals: FrameSignals,
        hw: HardwareTelemetry,
    ) -> MetaEvaluation:
        signals.validate(self.config.region_count)
        hw.validate()
        c, active = self.config, self._active_ratio(state.mask)
        capacity = []
        for d, w, s in zip(
            state.architecture.depth,
            state.architecture.width,
            state.step_map,
        ):
            dn = (d - c.min_depth) / max(1, c.max_depth - c.min_depth)
            wn = (w - c.min_width) / max(1, c.max_width - c.min_width)
            sn = (s - c.min_steps) / max(1, c.max_steps - c.min_steps)
            capacity.append(
                _clamp(
                    0.30 * dn + 0.20 * wn + 0.35 * sn + 0.15 * active,
                    0,
                    1,
                )
            )
        residual = tuple(
            e * (1 - 0.78 * cap)
            + 0.08 * g * (1 - cap)
            + 0.04 * o * (1 - cap)
            for e, g, o, cap in zip(
                signals.error,
                signals.edge_density,
                signals.occupancy,
                capacity,
            )
        )
        loss = _mean(tuple(v * v for v in residual))
        architecture_compute = _mean(
            tuple(
                (d / c.max_depth)
                * (w / c.max_width)
                * (s / c.max_steps)
                * active
                for d, w, s in zip(
                    state.architecture.depth,
                    state.architecture.width,
                    state.step_map,
                )
            )
        )
        compute = (
            architecture_compute
            + 0.05 * hw.frame_ms / c.target_frame_ms
            + 0.01 * hw.flops / 1e12
            + 0.005 * hw.sm_cycles / 1e9
        )
        memory = (
            active
            * _mean(tuple(float(w) for w in state.architecture.width))
            / c.max_width
            + 0.05 * hw.memory_mb / c.max_memory_mb
        )
        objective = loss + c.lambda_compute * compute + c.lambda_memory * memory
        return MetaEvaluation(
            loss,
            compute,
            memory,
            objective,
            -10 * math.log10(max(loss, 1e-12)),
            active,
            _mean(tuple(float(v) for v in state.architecture.depth)),
            _mean(tuple(float(v) for v in state.architecture.width)),
            _mean(tuple(float(v) for v in state.step_map)),
        )

    def _targets(
        self,
        signals: FrameSignals,
        hw: HardwareTelemetry,
    ) -> MetaGradients:
        c, complexity = self.config, signals.complexity()
        frame_pressure = max(0.0, hw.frame_ms / c.target_frame_ms - 1.0)
        memory_pressure = max(0.0, hw.memory_mb / c.max_memory_mb - 0.75)
        depth = tuple(
            c.min_depth
            + x * (c.max_depth - c.min_depth)
            - 2 * frame_pressure
            for x in complexity
        )
        width = tuple(
            c.min_width
            + x * (c.max_width - c.min_width)
            - 32 * memory_pressure
            - 16 * frame_pressure
            for x in complexity
        )
        steps = tuple(
            c.min_steps
            + x * (c.max_steps - c.min_steps)
            - 24 * frame_pressure
            for x in complexity
        )
        avg = _mean(complexity)
        mask = tuple(
            max(
                c.minimum_active_ratio,
                0.15 + 0.55 * complexity[i % c.region_count] + 0.30 * avg,
            )
            - 0.35 * memory_pressure
            for i in range(c.parameter_count)
        )
        return MetaGradients(depth, width, steps, mask)

    def _propose(self, targets: MetaGradients) -> MetaVolumeState:
        c, old, lr = self.config, self.state, self.config.learning_rate
        depth = tuple(
            int(round(_clamp(v + lr * (t - v), c.min_depth, c.max_depth)))
            for v, t in zip(old.architecture.depth, targets.depth)
        )
        width = tuple(
            max(
                c.min_width,
                min(
                    c.max_width,
                    int(round((v + lr * (t - v)) / 16) * 16),
                ),
            )
            for v, t in zip(old.architecture.width, targets.width)
        )
        steps = tuple(
            int(round(_clamp(v + lr * (t - v), c.min_steps, c.max_steps)))
            for v, t in zip(old.step_map, targets.steps)
        )
        scores = tuple(
            _clamp(v + lr * (t - v), 0, 1)
            for v, t in zip(old.mask_scores, targets.mask)
        )
        mask = [v >= c.mask_threshold for v in scores]
        floor = max(1, math.ceil(c.minimum_active_ratio * len(mask)))
        if sum(mask) < floor:
            for i in sorted(
                range(len(scores)), key=lambda j: (-scores[j], j)
            )[:floor]:
                mask[i] = True
        return MetaVolumeState(
            ArchitectureDNA(
                depth,
                width,
                old.architecture.kernel,
                old.architecture.revision + 1,
            ),
            steps,
            scores,
            tuple(mask),
            targets,
            old.cycle + 1,
            old.journal_hash,
            False,
        )

    def _verify(self, state: MetaVolumeState) -> Optional[str]:
        c = self.config
        if (
            len(state.architecture.depth) != c.region_count
            or len(state.architecture.width) != c.region_count
            or len(state.architecture.kernel) != c.region_count
            or len(state.step_map) != c.region_count
        ):
            return "regional map size mismatch"
        if (
            len(state.mask) != c.parameter_count
            or len(state.mask_scores) != c.parameter_count
        ):
            return "mask size mismatch"
        if not all(
            c.min_depth <= v <= c.max_depth for v in state.architecture.depth
        ):
            return "depth bound exceeded"
        if not all(
            c.min_width <= v <= c.max_width and v % 16 == 0
            for v in state.architecture.width
        ):
            return "width bound exceeded"
        if not all(c.min_steps <= v <= c.max_steps for v in state.step_map):
            return "step bound exceeded"
        if not all(v in (1, 3, 5) for v in state.architecture.kernel):
            return "unsupported kernel size"
        if not all(
            math.isfinite(v) and 0 <= v <= 1 for v in state.mask_scores
        ):
            return "invalid mask score"
        if self._active_ratio(state.mask) < c.minimum_active_ratio:
            return "minimum active ratio violated"
        return None

    @staticmethod
    def _instructions(
        before: MetaVolumeState,
        after: MetaVolumeState,
    ) -> Tuple[StructuralInstruction, ...]:
        out = []
        groups = (
            (
                "SET_DEPTH",
                before.architecture.depth,
                after.architecture.depth,
            ),
            (
                "SET_WIDTH",
                before.architecture.width,
                after.architecture.width,
            ),
            ("SET_STEPS", before.step_map, after.step_map),
            ("SET_MASK", before.mask, after.mask),
        )
        for opcode, left, right in groups:
            for i, (old, new) in enumerate(zip(left, right)):
                if old != new:
                    out.append(
                        StructuralInstruction(
                            opcode,
                            i,
                            float(old),
                            float(new),
                        )
                    )
        return tuple(out)

    def _seal(
        self,
        state: MetaVolumeState,
        evaluation: MetaEvaluation,
        instructions: Sequence[StructuralInstruction],
    ) -> str:
        payload = {
            "cycle": state.cycle,
            "architecture": asdict(state.architecture),
            "step_map": state.step_map,
            "mask_scores": tuple(round(v, 15) for v in state.mask_scores),
            "mask": state.mask,
            "evaluation": asdict(evaluation),
            "instructions": [asdict(v) for v in instructions],
        }
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(
            bytes.fromhex(self.state.journal_hash) + canonical
        ).hexdigest()

    def evolve(
        self,
        signals: FrameSignals,
        hw: HardwareTelemetry,
    ) -> EvolutionResult:
        signals.validate(self.config.region_count)
        hw.validate()
        baseline = self.evaluate(self.state, signals, hw)
        candidate_state = self._propose(self._targets(signals, hw))
        reason = self._verify(candidate_state)
        if reason:
            self.state = replace(
                self.state,
                last_committed=False,
                rollback_reason=reason,
            )
            return EvolutionResult(False, self.state, baseline, baseline, ())
        candidate = self.evaluate(candidate_state, signals, hw)
        if candidate.objective > baseline.objective + self.config.commit_tolerance:
            self.state = replace(
                self.state,
                last_committed=False,
                rollback_reason="candidate objective did not improve",
            )
            return EvolutionResult(False, self.state, baseline, candidate, ())
        instructions = self._instructions(self.state, candidate_state)
        self.state = replace(
            candidate_state,
            journal_hash=self._seal(candidate_state, candidate, instructions),
            last_committed=True,
            rollback_reason=None,
        )
        return EvolutionResult(
            True,
            self.state,
            baseline,
            candidate,
            instructions,
        )

    def snapshot(self) -> dict:
        return {
            "cycle": self.state.cycle,
            "journal_hash": self.state.journal_hash,
            "last_committed": self.state.last_committed,
            "rollback_reason": self.state.rollback_reason,
            "architecture_revision": self.state.architecture.revision,
            "average_depth": _mean(
                tuple(float(v) for v in self.state.architecture.depth)
            ),
            "average_width": _mean(
                tuple(float(v) for v in self.state.architecture.width)
            ),
            "average_steps": _mean(
                tuple(float(v) for v in self.state.step_map)
            ),
            "active_ratio": self._active_ratio(self.state.mask),
            "layers": self.state.layer_manifest(),
        }


def pareto_front(points: Iterable[ParetoPoint]) -> Tuple[ParetoPoint, ...]:
    values = tuple(points)
    for point in values:
        if (
            not all(
                math.isfinite(v)
                for v in (point.psnr_db, point.flops, point.memory_mb)
            )
            or point.flops < 0
            or point.memory_mb < 0
        ):
            raise ValueError(
                "Pareto values must be finite and costs non-negative"
            )
    front = []
    for candidate in values:
        dominated = any(
            other != candidate
            and other.psnr_db >= candidate.psnr_db
            and other.flops <= candidate.flops
            and other.memory_mb <= candidate.memory_mb
            and (
                other.psnr_db > candidate.psnr_db
                or other.flops < candidate.flops
                or other.memory_mb < candidate.memory_mb
            )
            for other in values
        )
        if not dominated:
            front.append(candidate)
    return tuple(
        sorted(
            front,
            key=lambda v: (-v.psnr_db, v.flops, v.memory_mb, v.name),
        )
    )
