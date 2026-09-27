"""Spatial bytecode control plane for the bounded Jarvis-X inward 3D runtime.

This module closes the gap between the repository's geometric ANN bytecode and
its bounded meta-optimization contracts.  A frame is executed as a sequence of
64-bit instructions addressed by an explicit (x, y, z) execution coordinate.
The numerical work is delegated to :class:`GeometricBytecodeANN`; the spatial
layer adds execution provenance, residual localization, deterministic shadow
replay, and guarded configuration promotion.

The optimizer is deliberately bounded.  It searches runtime configuration
neighbours, never rewrites Python source or host code, and never treats internal
score improvements as evidence of state-of-the-art performance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from dataclasses import asdict, dataclass, replace
from enum import IntEnum
from typing import Sequence

from .dr_moagi_geometric_bytecode_ann import (
    ANNConfig,
    CycleReceipt,
    GeometricBytecodeANN,
    Instruction,
    Op,
)


class SpatialOp(IntEnum):
    """Version-1 macro ISA for the inward 3D control plane."""

    INGEST = 0x01
    DIFFUSE6 = 0x02
    ENCODE3D = 0x03
    PHI3D_ITER = 0x04
    OMEGA_MEMORY = 0x05
    DECODE3D = 0x06
    RESIDUAL = 0x07
    CORRECT = 0x08
    ADAPT = 0x09
    GEOM_MAP = 0x0C
    RECUR = 0x0B
    OUTPUT_BUS = 0x0D
    STATE_SNAPSHOT = 0xA0
    FP_CHECK = 0xA2
    ERROR_FIELD = 0xA3
    SHADOW_FORK = 0xB3
    BENCH_RUN = 0xB4
    COMPARE_MULTI = 0xB5
    PROMOTE = 0xB6
    ROLLBACK = 0xB7
    HALT = 0xFF


@dataclass(frozen=True)
class SpatialInstruction3D:
    """A 64-bit instruction: OP8 | FLAGS8 | X12 | Y12 | Z12 | ARG12."""

    opcode: SpatialOp
    x: int = 0
    y: int = 0
    z: int = 0
    arg: int = 0
    flags: int = 0

    def __post_init__(self) -> None:
        if not 0 <= self.flags <= 0xFF:
            raise ValueError("flags must fit in 8 bits")
        for name in ("x", "y", "z", "arg"):
            value = getattr(self, name)
            if not 0 <= value <= 0xFFF:
                raise ValueError(f"{name} must fit in 12 bits")

    @property
    def pointer(self) -> tuple[int, int, int]:
        return self.x, self.y, self.z

    def pack(self) -> int:
        return (
            (int(self.opcode) << 56)
            | (self.flags << 48)
            | (self.x << 36)
            | (self.y << 24)
            | (self.z << 12)
            | self.arg
        )

    @classmethod
    def unpack(cls, word: int) -> "SpatialInstruction3D":
        if not 0 <= word <= 0xFFFFFFFFFFFFFFFF:
            raise ValueError("instruction word must fit in 64 bits")
        return cls(
            opcode=SpatialOp((word >> 56) & 0xFF),
            flags=(word >> 48) & 0xFF,
            x=(word >> 36) & 0xFFF,
            y=(word >> 24) & 0xFFF,
            z=(word >> 12) & 0xFFF,
            arg=word & 0xFFF,
        )


@dataclass(frozen=True, order=True)
class CandidateVector3D:
    """Signed displacement in refinement, memory and correction policy space."""

    refinement: int = 0
    memory: int = 0
    correction: int = 0

    def __post_init__(self) -> None:
        for name in ("refinement", "memory", "correction"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value not in (-1, 0, 1):
                raise ValueError(f"{name} must be one of -1, 0, +1")

    @property
    def manhattan(self) -> int:
        return abs(self.refinement) + abs(self.memory) + abs(self.correction)


@dataclass(frozen=True)
class Inward3DVMConfig:
    """Bounded physical execution and meta-search policy."""

    side: int = 4
    channels: int = 1
    latent_dim: int = 8
    diffusion_alpha: float = 0.18
    latent_relaxation: float = 0.55
    memory_rho: float = 0.88
    correction_gamma: float = 0.65
    learning_rate: float = 0.001
    fixed_point_tolerance: float = 1.0e-6
    max_refine_steps: int = 64
    active_threshold: float = 1.0e-8
    seed: int = 7
    meta_interval: int = 2
    shadow_cycles: int = 2
    max_shadow_candidates: int = 6
    min_score_improvement: float = 0.005
    max_mse_regression: float = 0.05
    max_error_voxels: int = 8

    def __post_init__(self) -> None:
        if self.side**3 * self.channels != 64:
            raise ValueError("the v1 bus contract requires exactly 64 scalar input channels")
        if self.meta_interval <= 0 or self.shadow_cycles <= 0:
            raise ValueError("meta_interval and shadow_cycles must be positive")
        if not 1 <= self.max_shadow_candidates <= 26:
            raise ValueError("max_shadow_candidates must be in [1, 26]")
        if not 0.0 <= self.min_score_improvement <= 1.0:
            raise ValueError("min_score_improvement must be in [0, 1]")
        if not 0.0 <= self.max_mse_regression <= 1.0:
            raise ValueError("max_mse_regression must be in [0, 1]")
        if self.max_error_voxels < 0:
            raise ValueError("max_error_voxels must be non-negative")
        self.ann_config()

    def ann_config(self) -> ANNConfig:
        return ANNConfig(
            side=self.side,
            channels=self.channels,
            latent_dim=self.latent_dim,
            diffusion_alpha=self.diffusion_alpha,
            latent_relaxation=self.latent_relaxation,
            memory_rho=self.memory_rho,
            correction_gamma=self.correction_gamma,
            learning_rate=self.learning_rate,
            fixed_point_tolerance=self.fixed_point_tolerance,
            max_refine_steps=self.max_refine_steps,
            active_threshold=self.active_threshold,
            seed=self.seed,
        )


@dataclass(frozen=True)
class ErrorVoxel3D:
    x: int
    y: int
    z: int
    index: int
    absolute_error: float


@dataclass(frozen=True)
class ExecutionTrace:
    step: int
    opcode: str
    word_hex: str
    pointer: tuple[int, int, int]
    route_to: tuple[int, int, int] | None
    cycle: int
    fixed_point_residual: float | None
    mse: float | None


@dataclass(frozen=True)
class FrameReport:
    cycle: int
    mse: float
    fixed_point_residual: float
    refine_steps: int
    converged: bool
    active_voxels: int
    spectral_bound: float
    convergence_radius: float
    state_hash: str
    error_field: tuple[ErrorVoxel3D, ...]

    def as_dict(self) -> dict[str, object]:
        out = asdict(self)
        out["error_field"] = [asdict(item) for item in self.error_field]
        return out


@dataclass(frozen=True)
class CandidateMetrics:
    score: float
    mse: float
    fixed_point_residual: float
    refine_steps: float
    active_voxels: float
    spectral_bound: float
    finite: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateResult:
    vector: CandidateVector3D
    config: ANNConfig
    metrics: CandidateMetrics

    def as_dict(self) -> dict[str, object]:
        return {
            "vector": asdict(self.vector),
            "config": asdict(self.config),
            "metrics": self.metrics.as_dict(),
        }


@dataclass(frozen=True)
class MetaReport:
    epoch: int
    baseline: CandidateResult
    best: CandidateResult
    promoted: bool
    relative_improvement: float
    evaluated_candidates: int
    authoritative_state_unchanged: bool
    claim_status: str = "internal_bounded_optimization_only_external_sota_unverified"

    def as_dict(self) -> dict[str, object]:
        return {
            "epoch": self.epoch,
            "baseline": self.baseline.as_dict(),
            "best": self.best.as_dict(),
            "promoted": self.promoted,
            "relative_improvement": self.relative_improvement,
            "evaluated_candidates": self.evaluated_candidates,
            "authoritative_state_unchanged": self.authoritative_state_unchanged,
            "claim_status": self.claim_status,
        }


@dataclass(frozen=True)
class AutonomicReport:
    frames: tuple[FrameReport, ...]
    meta: tuple[MetaReport, ...]
    trace_steps: int

    def as_dict(self) -> dict[str, object]:
        return {
            "frames": [item.as_dict() for item in self.frames],
            "meta": [item.as_dict() for item in self.meta],
            "trace_steps": self.trace_steps,
        }


def _q16(value: float) -> int:
    value = max(0.0, min(1.0, value))
    return int(round(value * 0xFFFF))


def _relative_improvement(baseline: float, candidate: float) -> float:
    denominator = max(abs(baseline), 1.0e-12)
    return (baseline - candidate) / denominator


def demo_bus(width: int = 8) -> list[float]:
    """Return a deterministic 8x8 signal plane in [-1, 1]."""

    if width != 8:
        raise ValueError("the v1 demo bus is fixed at 8x8")
    out: list[float] = []
    center = (width - 1) / 2.0
    for y in range(width):
        for x in range(width):
            dx = (x - center) / center
            dy = (y - center) / center
            radius = math.sqrt(dx * dx + dy * dy)
            out.append(
                math.sin(math.pi * dx)
                * math.cos(math.pi * dy)
                * math.exp(-0.5 * radius)
            )
    return out


class InwardSelfOptimizing3DVM:
    """Execute spatial bytecode and turn its bounded runtime policy inward."""

    def __init__(self, config: Inward3DVMConfig | None = None) -> None:
        self.config = config or Inward3DVMConfig()
        self.engine = GeometricBytecodeANN(self.config.ann_config())
        self.source_bus: list[float] | None = None
        self.trace: list[ExecutionTrace] = []
        self.error_field: tuple[ErrorVoxel3D, ...] = ()
        self.meta_epoch = 0
        self.program = self.default_program()

    @staticmethod
    def default_program() -> tuple[SpatialInstruction3D, ...]:
        return (
            SpatialInstruction3D(SpatialOp.STATE_SNAPSHOT, z=0),
            SpatialInstruction3D(SpatialOp.INGEST, z=10),
            SpatialInstruction3D(SpatialOp.DIFFUSE6, z=120),
            SpatialInstruction3D(SpatialOp.ENCODE3D, z=240),
            SpatialInstruction3D(SpatialOp.PHI3D_ITER, z=500),
            SpatialInstruction3D(SpatialOp.FP_CHECK, z=540),
            SpatialInstruction3D(SpatialOp.OMEGA_MEMORY, z=580),
            SpatialInstruction3D(SpatialOp.DECODE3D, z=720),
            SpatialInstruction3D(SpatialOp.RESIDUAL, z=820),
            SpatialInstruction3D(SpatialOp.ERROR_FIELD, z=840),
            SpatialInstruction3D(SpatialOp.CORRECT, z=860),
            SpatialInstruction3D(SpatialOp.ADAPT, z=880),
            SpatialInstruction3D(SpatialOp.GEOM_MAP, z=900),
            SpatialInstruction3D(SpatialOp.RECUR, z=930),
            SpatialInstruction3D(SpatialOp.OUTPUT_BUS, z=980),
            SpatialInstruction3D(SpatialOp.HALT, z=999),
        )

    def load_bus(self, values: Sequence[float]) -> None:
        if len(values) != 64:
            raise ValueError("expected exactly 64 values from the 8x8 input bus")
        finite = [float(value) for value in values]
        if not all(math.isfinite(value) for value in finite):
            raise ValueError("bus values must be finite")
        self.source_bus = finite[:]
        self.engine.load(finite)
        self.trace.clear()
        self.error_field = ()
        self.meta_epoch = 0

    def disassemble(self) -> list[str]:
        return [
            (
                f"{index:02d} 0x{ins.pack():016X} "
                f"@{ins.x:03X},{ins.y:03X},{ins.z:03X} {ins.opcode.name}"
            )
            for index, ins in enumerate(self.program)
        ]

    def program_binary(self) -> bytes:
        return b"".join(struct.pack(">Q", ins.pack()) for ins in self.program)

    @staticmethod
    def contraction_bound(config: ANNConfig) -> float:
        """Upper bound induced by the ANN's normalized latent row sum (0.72)."""

        return (1.0 - config.latent_relaxation) + config.latent_relaxation * 0.72

    def _state_hash(self) -> str:
        state = self.engine.state
        payload = {
            "state": {
                "x": state.x,
                "x_mixed": state.x_mixed,
                "z": state.z,
                "omega": state.omega,
                "x_hat": state.x_hat,
                "residual": state.residual,
                "corrected": state.corrected,
                "cycle": state.cycle,
                "fixed_point_residual": state.fixed_point_residual,
                "physical_refine_steps": state.physical_refine_steps,
                "converged": state.converged,
            },
            "weights": {
                "enc": self.engine.w_enc,
                "latent": self.engine.w_latent,
                "dec": self.engine.w_dec,
                "b_enc": self.engine.b_enc,
                "b_dec": self.engine.b_dec,
            },
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _mse(self) -> float | None:
        residual = self.engine.state.residual
        if not residual:
            return None
        return sum(value * value for value in residual) / len(residual)

    def _append_trace(
        self,
        instruction: SpatialInstruction3D,
        route_to: tuple[int, int, int] | None,
    ) -> None:
        residual = self.engine.state.fixed_point_residual
        self.trace.append(
            ExecutionTrace(
                step=len(self.trace),
                opcode=instruction.opcode.name,
                word_hex=f"0x{instruction.pack():016X}",
                pointer=instruction.pointer,
                route_to=route_to,
                cycle=self.engine.state.cycle,
                fixed_point_residual=(residual if math.isfinite(residual) else None),
                mse=self._mse(),
            )
        )

    def _localize_error(self) -> tuple[ErrorVoxel3D, ...]:
        side = self.engine.config.side
        ranked = sorted(
            ((abs(value), index) for index, value in enumerate(self.engine.state.residual)),
            reverse=True,
        )
        out: list[ErrorVoxel3D] = []
        for error, index in ranked[: self.config.max_error_voxels]:
            voxel = index // self.engine.config.channels
            x = voxel % side
            y = (voxel // side) % side
            z = voxel // (side * side)
            out.append(ErrorVoxel3D(x, y, z, index, error))
        return tuple(out)

    def _execute_core(self, instruction: SpatialInstruction3D) -> None:
        mapping = {
            SpatialOp.INGEST: Instruction(Op.INGEST),
            SpatialOp.DIFFUSE6: Instruction(
                Op.DIFFUSE6, immediate=_q16(self.engine.config.diffusion_alpha)
            ),
            SpatialOp.ENCODE3D: Instruction(Op.ENCODE),
            SpatialOp.PHI3D_ITER: Instruction(Op.LATENT_REFINE),
            SpatialOp.OMEGA_MEMORY: Instruction(
                Op.MEMORY, immediate=_q16(self.engine.config.memory_rho)
            ),
            SpatialOp.DECODE3D: Instruction(Op.DECODE),
            SpatialOp.RESIDUAL: Instruction(Op.RESIDUAL),
            SpatialOp.CORRECT: Instruction(
                Op.CORRECT, immediate=_q16(self.engine.config.correction_gamma)
            ),
            SpatialOp.ADAPT: Instruction(Op.ADAPT),
            SpatialOp.GEOM_MAP: Instruction(Op.GEOM_MAP),
            SpatialOp.RECUR: Instruction(Op.RECUR),
            SpatialOp.HALT: Instruction(Op.HALT),
        }
        inner = mapping.get(instruction.opcode)
        if inner is not None:
            self.engine.step_instruction(inner)
        elif instruction.opcode is SpatialOp.ERROR_FIELD:
            self.error_field = self._localize_error()
        elif instruction.opcode in (
            SpatialOp.STATE_SNAPSHOT,
            SpatialOp.FP_CHECK,
            SpatialOp.OUTPUT_BUS,
        ):
            return
        else:
            raise ValueError(f"opcode {instruction.opcode.name} is not valid in the frame program")

    def run_frame(self) -> FrameReport:
        if self.source_bus is None:
            raise RuntimeError("load the 8x8 bus before execution")
        self.engine.halted = False
        snapshot_hash = self._state_hash()
        for index, instruction in enumerate(self.program):
            self._execute_core(instruction)
            if index + 1 < len(self.program):
                route = self.program[index + 1].pointer
            else:
                route = None
            self._append_trace(instruction, route)
            if self.engine.halted:
                break
        if not self.engine.receipts:
            raise RuntimeError("frame program did not commit a RECUR instruction")
        receipt: CycleReceipt = self.engine.receipts[-1]
        radius = min(
            1.0,
            receipt.fixed_point_residual
            / max(self.engine.config.fixed_point_tolerance, 1.0e-12),
        )
        state_hash = self._state_hash()
        if state_hash == snapshot_hash:
            raise RuntimeError("frame did not advance authoritative state")
        return FrameReport(
            cycle=receipt.cycle,
            mse=receipt.mse,
            fixed_point_residual=receipt.fixed_point_residual,
            refine_steps=receipt.physical_refine_steps,
            converged=receipt.converged,
            active_voxels=receipt.active_voxels,
            spectral_bound=self.contraction_bound(self.engine.config),
            convergence_radius=radius,
            state_hash=state_hash,
            error_field=self.error_field,
        )

    def _candidate_config(self, vector: CandidateVector3D) -> ANNConfig:
        base = self.engine.config
        relaxation = base.latent_relaxation
        steps = base.max_refine_steps
        if vector.refinement < 0:
            relaxation = max(0.10, relaxation * 0.90)
            steps = max(4, int(round(steps * 0.75)))
        elif vector.refinement > 0:
            relaxation = min(0.95, relaxation * 1.10)
            steps = min(256, max(steps + 1, int(round(steps * 1.25))))

        rho = base.memory_rho
        if vector.memory < 0:
            rho = max(0.0, rho - 0.03)
        elif vector.memory > 0:
            rho = min(0.98, rho + 0.03)

        correction = base.correction_gamma
        diffusion = base.diffusion_alpha
        if vector.correction < 0:
            correction = max(0.0, correction - 0.05)
            diffusion = max(0.0, diffusion - 0.02)
        elif vector.correction > 0:
            correction = min(1.0, correction + 0.05)
            diffusion = min(1.0, diffusion + 0.02)

        return replace(
            base,
            latent_relaxation=relaxation,
            max_refine_steps=steps,
            memory_rho=rho,
            correction_gamma=correction,
            diffusion_alpha=diffusion,
        )

    @staticmethod
    def _vectors() -> list[CandidateVector3D]:
        values = (-1, 0, 1)
        vectors = [
            CandidateVector3D(x, y, z)
            for x in values
            for y in values
            for z in values
            if (x, y, z) != (0, 0, 0)
        ]
        return sorted(vectors, key=lambda item: (item.manhattan, item))

    def _evaluate(self, config: ANNConfig) -> CandidateMetrics:
        if self.source_bus is None:
            raise RuntimeError("load the 8x8 bus before optimization")
        shadow = GeometricBytecodeANN(config)
        shadow.load(self.source_bus)
        receipts = shadow.run(self.config.shadow_cycles)
        count = len(receipts)
        mse = sum(item.mse for item in receipts) / count
        fixed = sum(item.fixed_point_residual for item in receipts) / count
        steps = sum(item.physical_refine_steps for item in receipts) / count
        active = sum(item.active_voxels for item in receipts) / count
        bound = self.contraction_bound(config)
        finite = all(math.isfinite(value) for value in (mse, fixed, steps, active, bound))
        score = 8.0 * mse + 2.0 * fixed + 0.002 * steps + 0.0005 * active
        if not finite or bound >= 1.0:
            score += 1_000.0
        return CandidateMetrics(score, mse, fixed, steps, active, bound, finite)

    def _append_meta_op(
        self,
        op: SpatialOp,
        z: int,
        *,
        route_to: tuple[int, int, int] | None,
    ) -> None:
        self._append_trace(SpatialInstruction3D(op, z=z), route_to)

    def turn_inward(self) -> MetaReport:
        """Shadow-evaluate bounded policy neighbours and atomically promote one."""

        if self.source_bus is None:
            raise RuntimeError("load the 8x8 bus before optimization")
        before_hash = self._state_hash()
        self._append_meta_op(SpatialOp.SHADOW_FORK, 940, route_to=(0, 0, 950))

        baseline = CandidateResult(
            CandidateVector3D(),
            self.engine.config,
            self._evaluate(self.engine.config),
        )
        candidates: list[CandidateResult] = []
        for vector in self._vectors()[: self.config.max_shadow_candidates]:
            cfg = self._candidate_config(vector)
            candidates.append(CandidateResult(vector, cfg, self._evaluate(cfg)))
        self._append_meta_op(SpatialOp.BENCH_RUN, 950, route_to=(0, 0, 960))

        best = min((baseline, *candidates), key=lambda item: (item.metrics.score, item.vector))
        relative = _relative_improvement(baseline.metrics.score, best.metrics.score)
        self._append_meta_op(SpatialOp.COMPARE_MULTI, 960, route_to=(0, 0, 970))

        unchanged = self._state_hash() == before_hash
        regression_limit = (
            baseline.metrics.mse * (1.0 + self.config.max_mse_regression) + 1.0e-12
        )
        promoted = (
            unchanged
            and best.vector != CandidateVector3D()
            and best.metrics.finite
            and best.metrics.spectral_bound < 1.0
            and relative >= self.config.min_score_improvement
            and best.metrics.mse <= regression_limit
        )
        if promoted:
            self.engine.config = best.config
            terminal_op = SpatialOp.PROMOTE
        else:
            terminal_op = SpatialOp.ROLLBACK
        self._append_meta_op(terminal_op, 970, route_to=(0, 0, 0))

        self.meta_epoch += 1
        return MetaReport(
            epoch=self.meta_epoch,
            baseline=baseline,
            best=best,
            promoted=promoted,
            relative_improvement=relative,
            evaluated_candidates=1 + len(candidates),
            authoritative_state_unchanged=unchanged,
        )

    def run_autonomic(self, cycles: int) -> AutonomicReport:
        if isinstance(cycles, bool) or not isinstance(cycles, int) or cycles <= 0:
            raise ValueError("cycles must be a positive integer")
        frames: list[FrameReport] = []
        meta: list[MetaReport] = []
        for _ in range(cycles):
            frames.append(self.run_frame())
            if frames[-1].cycle % self.config.meta_interval == 0:
                meta.append(self.turn_inward())
        return AutonomicReport(tuple(frames), tuple(meta), len(self.trace))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Jarvis-X inward self-optimizing 3D VM"
    )
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--meta-interval", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--disassemble", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = Inward3DVMConfig(seed=args.seed, meta_interval=args.meta_interval)
    vm = InwardSelfOptimizing3DVM(config)
    vm.load_bus(demo_bus())
    if args.disassemble:
        print("\n".join(vm.disassemble()))
    report = vm.run_autonomic(args.cycles)
    payload = {
        "isa": "jarvisx-inward3d-v1",
        "program_bytes": len(vm.program_binary()),
        "active_config": asdict(vm.engine.config),
        "report": report.as_dict(),
        "claim_status": "external_sota_unverified",
    }
    if args.json_output:
        print(json.dumps(payload, sort_keys=True))
    else:
        last = report.frames[-1]
        print(
            "inward3d "
            f"cycles={len(report.frames)} mse={last.mse:.8f} "
            f"fp={last.fixed_point_residual:.3e} radius={last.convergence_radius:.3e} "
            f"meta_epochs={len(report.meta)}"
        )
    return 0


__all__ = [
    "AutonomicReport",
    "CandidateMetrics",
    "CandidateResult",
    "CandidateVector3D",
    "ErrorVoxel3D",
    "ExecutionTrace",
    "FrameReport",
    "Inward3DVMConfig",
    "InwardSelfOptimizing3DVM",
    "MetaReport",
    "SpatialInstruction3D",
    "SpatialOp",
    "demo_bus",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
