"""Deterministic kinetic bytecode-wave reference runtime.

This module models bytecode as a bounded wave moving through a sparse, symbolic
3D address fabric. It is a research-layer simulator: it does not allocate the
declared virtual extent and it cannot perform authoritative Jarvis-X commits or
external side effects.

The runtime deliberately separates three things:

* virtual scale: represented symbolically by ``PowerExtent``;
* kinetic execution: packets advance one pipeline stage per tick;
* physical work: only referenced/resident regions occupy runtime collections.

The executable path is candidate-first. A packet may materialize and execute,
but its candidate value is published only after projection and verification.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable, Iterable


class PipelineStage(str, Enum):
    FETCH = "fetch"
    DECODE = "decode"
    RESOLVE = "resolve"
    ACTIVATE = "activate"
    MATERIALIZE = "materialize"
    EXECUTE = "execute"
    PROJECT = "project"
    VERIFY = "verify"
    ENCODE = "encode"
    COMMIT = "commit"
    EVICT = "evict"
    COMPLETE = "complete"
    ROLLBACK = "rollback"


PIPELINE: tuple[PipelineStage, ...] = (
    PipelineStage.FETCH,
    PipelineStage.DECODE,
    PipelineStage.RESOLVE,
    PipelineStage.ACTIVATE,
    PipelineStage.MATERIALIZE,
    PipelineStage.EXECUTE,
    PipelineStage.PROJECT,
    PipelineStage.VERIFY,
    PipelineStage.ENCODE,
    PipelineStage.COMMIT,
    PipelineStage.EVICT,
    PipelineStage.COMPLETE,
)


class KineticOp(str, Enum):
    """Research macro-operations lowered onto the kinetic pipeline."""

    G3D = "G3D"
    DELTA = "DELTA"
    ENCODE = "ENCODE"
    DECODE = "DECODE"


@dataclass(frozen=True)
class PowerExtent:
    """Symbolic axis extent ``base ** exponent`` without materializing the integer."""

    base: int
    exponent: int

    def __post_init__(self) -> None:
        if isinstance(self.base, bool) or not isinstance(self.base, int) or self.base < 2:
            raise ValueError("base must be an integer >= 2")
        if (
            isinstance(self.exponent, bool)
            or not isinstance(self.exponent, int)
            or self.exponent <= 0
        ):
            raise ValueError("exponent must be a positive integer")

    @property
    def log10_axis(self) -> float:
        return self.exponent * math.log10(self.base)

    @property
    def log10_volume(self) -> float:
        return 3.0 * self.log10_axis

    @property
    def approximate_axis_bits(self) -> int:
        return math.ceil(self.exponent * math.log2(self.base))


@dataclass(frozen=True)
class RegionDescriptor:
    """Finite descriptor for one sparse region inside a symbolic 3D fabric."""

    ref: int
    level: int = 0
    tile_count_hint: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.ref, bool) or not isinstance(self.ref, int) or not 0 <= self.ref < 2**64:
            raise ValueError("ref must be an unsigned 64-bit integer")
        if isinstance(self.level, bool) or not isinstance(self.level, int) or self.level < 0:
            raise ValueError("level must be a non-negative integer")
        if (
            isinstance(self.tile_count_hint, bool)
            or not isinstance(self.tile_count_hint, int)
            or self.tile_count_hint <= 0
        ):
            raise ValueError("tile_count_hint must be a positive integer")


@dataclass(frozen=True)
class KineticInstruction:
    """One bounded research instruction.

    ``observation`` is the immutable per-instruction anchor. ``prediction`` and
    ``omega`` are finite scalar projections used by the reference G3D
    recurrence. Production runtimes may substitute typed tensors provided they
    preserve the same candidate-first boundary.
    """

    op: KineticOp
    region_ref: int
    observation: float = 0.0
    prediction: float = 0.0
    omega: float = 0.0
    immediate: float = 0.0
    verification_score: float = 1.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.region_ref, bool)
            or not isinstance(self.region_ref, int)
            or not 0 <= self.region_ref < 2**64
        ):
            raise ValueError("region_ref must be an unsigned 64-bit integer")
        for name in ("observation", "prediction", "omega", "immediate", "verification_score"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if not 0.0 <= self.verification_score <= 1.0:
            raise ValueError("verification_score must be within [0, 1]")


@dataclass(frozen=True)
class KineticConfig:
    """Resource, activation, projection and verification bounds."""

    max_inflight: int = 64
    max_resident_regions: int = 8
    activation_injection: float = 1.0
    activation_retention: float = 0.75
    activation_threshold: float = 0.5
    projection_limit: float = 1_000_000.0
    verification_threshold: float = 0.90
    quantization_digits: int = 6

    def __post_init__(self) -> None:
        for name in ("max_inflight", "max_resident_regions"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("activation_injection", "activation_retention", "activation_threshold"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and within [0, 1]")
        limit = float(self.projection_limit)
        if not math.isfinite(limit) or limit <= 0.0:
            raise ValueError("projection_limit must be finite and positive")
        threshold = float(self.verification_threshold)
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("verification_threshold must be finite and within [0, 1]")
        if (
            isinstance(self.quantization_digits, bool)
            or not isinstance(self.quantization_digits, int)
            or not 0 <= self.quantization_digits <= 12
        ):
            raise ValueError("quantization_digits must be an integer within [0, 12]")


@dataclass(frozen=True)
class WavePacket:
    packet_id: int
    instruction_index: int
    stage: PipelineStage
    region_ref: int
    candidate: float | None = None
    projected: float | None = None
    encoded: float | None = None
    verified: bool | None = None
    stall_count: int = 0


@dataclass(frozen=True)
class KineticSnapshot:
    clock: int
    pc: int
    inflight: int
    resident_regions: tuple[int, ...]
    active_regions: tuple[int, ...]
    committed_regions: tuple[int, ...]
    commits: int
    rollbacks: int
    stalls: int
    stage_counts: tuple[tuple[str, int], ...]


PacketValidator = Callable[[WavePacket, KineticInstruction], bool]


def default_validator(packet: WavePacket, instruction: KineticInstruction) -> bool:
    """Accept finite projected candidates whose declared score is checked by runtime."""

    del instruction
    return packet.projected is not None and math.isfinite(packet.projected)


class KineticBytecodeRuntime:
    """Bounded sparse pipeline for kinetic bytecode-wave conformance fixtures."""

    def __init__(
        self,
        *,
        space: PowerExtent,
        regions: Iterable[RegionDescriptor],
        program: Iterable[KineticInstruction],
        config: KineticConfig | None = None,
        validator: PacketValidator | None = None,
    ) -> None:
        self.space = space
        self.config = config or KineticConfig()
        self.validator = validator or default_validator
        self.regions = {region.ref: region for region in regions}
        self.program = tuple(program)
        if len(self.regions) == 0:
            raise ValueError("at least one region descriptor is required")
        if len(self.program) == 0:
            raise ValueError("program must contain at least one instruction")
        missing = sorted({instr.region_ref for instr in self.program} - self.regions.keys())
        if missing:
            raise ValueError(f"program references unknown regions: {missing}")

        self.clock = 0
        self.pc = 0
        self._next_packet_id = 0
        self.inflight: list[WavePacket] = []
        self.activity: dict[int, float] = {}
        self.resident: set[int] = set()
        self.committed: dict[int, float] = {}
        self.commits = 0
        self.rollbacks = 0
        self.stalls = 0
        self.trace: list[tuple[int, int, PipelineStage]] = []

    def _decay_activity(self) -> None:
        retention = self.config.activation_retention
        threshold = self.config.activation_threshold
        next_activity: dict[int, float] = {}
        for ref, value in self.activity.items():
            decayed = value * retention
            if decayed >= threshold:
                next_activity[ref] = decayed
        self.activity = next_activity

    def _inject(self) -> None:
        if self.pc >= len(self.program) or len(self.inflight) >= self.config.max_inflight:
            return
        instruction = self.program[self.pc]
        packet = WavePacket(
            packet_id=self._next_packet_id,
            instruction_index=self.pc,
            stage=PipelineStage.FETCH,
            region_ref=instruction.region_ref,
        )
        self._next_packet_id += 1
        self.pc += 1
        self.inflight.append(packet)

    @staticmethod
    def _next_stage(stage: PipelineStage) -> PipelineStage:
        index = PIPELINE.index(stage)
        if index + 1 >= len(PIPELINE):
            return PipelineStage.COMPLETE
        return PIPELINE[index + 1]

    def _candidate(self, instruction: KineticInstruction) -> float:
        current = self.committed.get(instruction.region_ref, instruction.observation)
        if instruction.op is KineticOp.G3D:
            error = instruction.observation - instruction.prediction
            return (
                current
                + instruction.prediction
                - error
                + instruction.omega
                + instruction.immediate
            )
        if instruction.op is KineticOp.DELTA:
            return current + instruction.immediate
        if instruction.op is KineticOp.ENCODE:
            return math.tanh(current + instruction.immediate)
        if instruction.op is KineticOp.DECODE:
            return current + 0.5 * current + instruction.immediate
        raise RuntimeError(f"unsupported kinetic op: {instruction.op}")

    def _advance_packet(self, packet: WavePacket) -> WavePacket | None:
        instruction = self.program[packet.instruction_index]
        stage = packet.stage
        self.trace.append((self.clock, packet.packet_id, stage))

        if stage is PipelineStage.RESOLVE:
            if packet.region_ref not in self.regions:
                self.rollbacks += 1
                return replace(packet, stage=PipelineStage.ROLLBACK)

        elif stage is PipelineStage.ACTIVATE:
            prior = self.activity.get(packet.region_ref, 0.0)
            injected = min(1.0, prior + self.config.activation_injection)
            self.activity[packet.region_ref] = injected

        elif stage is PipelineStage.MATERIALIZE:
            if packet.region_ref not in self.resident:
                if len(self.resident) >= self.config.max_resident_regions:
                    self.stalls += 1
                    return replace(packet, stall_count=packet.stall_count + 1)
                self.resident.add(packet.region_ref)

        elif stage is PipelineStage.EXECUTE:
            candidate = self._candidate(instruction)
            if not math.isfinite(candidate):
                self.rollbacks += 1
                return replace(packet, stage=PipelineStage.ROLLBACK, candidate=candidate)
            packet = replace(packet, candidate=candidate)

        elif stage is PipelineStage.PROJECT:
            if packet.candidate is None:
                raise RuntimeError("candidate missing before projection")
            limit = self.config.projection_limit
            projected = max(-limit, min(limit, packet.candidate))
            packet = replace(packet, projected=projected)

        elif stage is PipelineStage.VERIFY:
            score_ok = instruction.verification_score >= self.config.verification_threshold
            validator_ok = self.validator(packet, instruction)
            if not score_ok or not validator_ok:
                self.rollbacks += 1
                return replace(packet, stage=PipelineStage.ROLLBACK, verified=False)
            packet = replace(packet, verified=True)

        elif stage is PipelineStage.ENCODE:
            if packet.projected is None or packet.verified is not True:
                raise RuntimeError("verified projected candidate required before encoding")
            encoded = round(packet.projected, self.config.quantization_digits)
            packet = replace(packet, encoded=encoded)

        elif stage is PipelineStage.COMMIT:
            if packet.encoded is None or packet.verified is not True:
                raise RuntimeError("encoded verified candidate required before commit")
            self.committed[packet.region_ref] = packet.encoded
            self.commits += 1

        elif stage is PipelineStage.EVICT:
            self.resident.discard(packet.region_ref)

        elif stage is PipelineStage.COMPLETE:
            return None

        elif stage is PipelineStage.ROLLBACK:
            self.resident.discard(packet.region_ref)
            return None

        next_stage = self._next_stage(packet.stage)
        return replace(packet, stage=next_stage)

    def tick(self) -> KineticSnapshot:
        """Advance every in-flight packet by at most one stage and inject one packet."""

        self.clock += 1
        self._decay_activity()
        advanced: list[WavePacket] = []
        for packet in self.inflight:
            updated = self._advance_packet(packet)
            if updated is not None:
                advanced.append(updated)
        self.inflight = advanced
        self._inject()
        return self.snapshot()

    def run(self, *, max_ticks: int = 10_000) -> KineticSnapshot:
        """Run until the program and in-flight wavefront drain."""

        if isinstance(max_ticks, bool) or not isinstance(max_ticks, int) or max_ticks <= 0:
            raise ValueError("max_ticks must be a positive integer")
        for _ in range(max_ticks):
            snapshot = self.tick()
            if self.pc >= len(self.program) and not self.inflight:
                return snapshot
        raise RuntimeError("kinetic runtime did not quiesce within max_ticks")

    def snapshot(self) -> KineticSnapshot:
        stage_counts = Counter(packet.stage.value for packet in self.inflight)
        active = tuple(
            sorted(
                ref
                for ref, value in self.activity.items()
                if value >= self.config.activation_threshold
            )
        )
        return KineticSnapshot(
            clock=self.clock,
            pc=self.pc,
            inflight=len(self.inflight),
            resident_regions=tuple(sorted(self.resident)),
            active_regions=active,
            committed_regions=tuple(sorted(self.committed)),
            commits=self.commits,
            rollbacks=self.rollbacks,
            stalls=self.stalls,
            stage_counts=tuple(sorted(stage_counts.items())),
        )


def canonical_million_power_space() -> PowerExtent:
    """Return one axis of ``1_000_000 ** 1_000_000`` symbolically."""

    return PowerExtent(base=1_000_000, exponent=1_000_000)
