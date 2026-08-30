"""Bounded cyclic execution for the Dr Moagi 3D auto-encoding bytecode runtime.

The outer DM3D loop format contains a verified inner DM3D program plus finite loop
metadata. It does not add arbitrary jumps or host-code execution. Each cycle runs the
existing finite VM, captures selected 3D volume states, feeds a designated output
volume back as the next input, and stops at an explicit cycle or work budget.
"""

from __future__ import annotations

import struct
from dataclasses import asdict, dataclass, field
from hashlib import sha256

from .dr_moagi_pdf_bytecode import (
    ProgramLimits,
    VmMetrics,
    Volume3D,
    canonical_autoencoder_program,
    execute_program,
    parse_program,
)

AUTO_PROGRAM_MAGIC = b"DM3DLP1\0"
AUTO_PROGRAM_VERSION = 1
AUTO_PROGRAM_HEADER = struct.Struct("<8sHHIIIII")
AUTO_PROGRAM_DIGEST_BYTES = 32
AUTO_FLAG_STOP_ON_FIXED = 0x0001
AUTO_KNOWN_FLAGS = AUTO_FLAG_STOP_ON_FIXED
DEFAULT_FRAME_REGISTERS = (0, 1, 2, 3, 4)
STAGE_NAMES = {
    0: "input",
    1: "encoded",
    2: "refined",
    3: "decoded",
    4: "reencoded",
}


@dataclass(frozen=True)
class AutoLoopLimits:
    """Resource bounds applied across every VM cycle."""

    max_cycles: int = 256
    max_total_physical_steps: int = 200_000_000
    max_frames: int = 4096

    def validate(self) -> None:
        if self.max_cycles <= 0:
            raise ValueError("max_cycles must be positive")
        if self.max_total_physical_steps <= 0:
            raise ValueError("max_total_physical_steps must be positive")
        if self.max_frames <= 0:
            raise ValueError("max_frames must be positive")


@dataclass(frozen=True)
class AutoLoopProgram:
    """Decoded outer loop bytecode and its verified inner DM3D program."""

    cycles: int
    feedback_register: int
    convergence_scalar_register: int
    frame_register_mask: int
    stop_on_fixed_point: bool
    inner_program: bytes

    @property
    def frame_registers(self) -> tuple[int, ...]:
        return tuple(register for register in range(32) if self.frame_register_mask & (1 << register))


@dataclass(frozen=True)
class VolumeFrame:
    """One captured volumetric stage suitable for a downstream renderer."""

    cycle: int
    register: int
    stage: str
    volume: Volume3D

    def report(self) -> dict[str, object]:
        values = self.volume.values
        digest = sha256()
        digest.update(struct.pack("<III", self.volume.nx, self.volume.ny, self.volume.nz))
        for value in values:
            digest.update(struct.pack("<d", value))
        return {
            "cycle": self.cycle,
            "register": self.register,
            "stage": self.stage,
            "shape": [self.volume.nx, self.volume.ny, self.volume.nz],
            "minimum": min(values),
            "maximum": max(values),
            "mean": sum(values) / len(values),
            "sha256": digest.hexdigest(),
        }


@dataclass(frozen=True)
class CycleTelemetry:
    cycle: int
    reconstruction_mse: float | None
    cycle_mse: float | None
    omega: float | None
    fixed_point_pass: bool
    physical_steps: int


@dataclass
class AutoLoopResult:
    cycles_executed: int
    stopped_on_fixed_point: bool
    final_volume: Volume3D
    metrics: VmMetrics
    cycles: list[CycleTelemetry] = field(default_factory=list)
    frames: list[VolumeFrame] = field(default_factory=list)

    def report(self) -> dict[str, object]:
        metrics = {
            **asdict(self.metrics),
            "physical_steps_per_second": self.metrics.physical_steps_per_second,
        }
        return {
            "cycles_executed": self.cycles_executed,
            "stopped_on_fixed_point": self.stopped_on_fixed_point,
            "final_shape": [
                self.final_volume.nx,
                self.final_volume.ny,
                self.final_volume.nz,
            ],
            "metrics": metrics,
            "cycles": [asdict(item) for item in self.cycles],
            "frames": [frame.report() for frame in self.frames],
        }


def serialize_auto_loop_program(
    inner_program: bytes,
    *,
    cycles: int,
    feedback_register: int = 3,
    convergence_scalar_register: int = 13,
    frame_registers: tuple[int, ...] = DEFAULT_FRAME_REGISTERS,
    stop_on_fixed_point: bool = False,
) -> bytes:
    """Wrap a verified DM3D program in finite cyclic execution metadata."""

    parse_program(inner_program)
    if cycles <= 0:
        raise ValueError("cycles must be positive")
    _validate_register(feedback_register)
    _validate_register(convergence_scalar_register)
    frame_mask = _frame_mask(frame_registers)
    flags = AUTO_FLAG_STOP_ON_FIXED if stop_on_fixed_point else 0
    header = AUTO_PROGRAM_HEADER.pack(
        AUTO_PROGRAM_MAGIC,
        AUTO_PROGRAM_VERSION,
        flags,
        cycles,
        feedback_register,
        convergence_scalar_register,
        frame_mask,
        len(inner_program),
    )
    body = header + inner_program
    return body + sha256(body).digest()


def parse_auto_loop_program(
    payload: bytes,
    *,
    loop_limits: AutoLoopLimits | None = None,
    vm_limits: ProgramLimits | None = None,
) -> AutoLoopProgram:
    """Validate and decode DM3D cyclic bytecode."""

    loop_limits = loop_limits or AutoLoopLimits()
    vm_limits = vm_limits or ProgramLimits()
    loop_limits.validate()
    vm_limits.validate()
    minimum = AUTO_PROGRAM_HEADER.size + AUTO_PROGRAM_DIGEST_BYTES
    if len(payload) <= minimum:
        raise ValueError("DM3D auto-loop program is truncated")

    body = payload[:-AUTO_PROGRAM_DIGEST_BYTES]
    digest = payload[-AUTO_PROGRAM_DIGEST_BYTES:]
    if sha256(body).digest() != digest:
        raise ValueError("DM3D auto-loop SHA-256 mismatch")

    (
        magic,
        version,
        flags,
        cycles,
        feedback_register,
        convergence_scalar_register,
        frame_mask,
        inner_length,
    ) = AUTO_PROGRAM_HEADER.unpack_from(body, 0)
    if magic != AUTO_PROGRAM_MAGIC:
        raise ValueError("invalid DM3D auto-loop magic")
    if version != AUTO_PROGRAM_VERSION:
        raise ValueError("unsupported DM3D auto-loop version")
    if flags & ~AUTO_KNOWN_FLAGS:
        raise ValueError("unknown DM3D auto-loop flags")
    if cycles == 0 or cycles > loop_limits.max_cycles:
        raise ValueError("DM3D auto-loop cycle count exceeds configured limit")
    _validate_register(feedback_register)
    _validate_register(convergence_scalar_register)

    inner = body[AUTO_PROGRAM_HEADER.size :]
    if len(inner) != inner_length:
        raise ValueError("DM3D auto-loop inner program length mismatch")
    parse_program(inner, vm_limits)

    frame_registers = tuple(register for register in range(32) if frame_mask & (1 << register))
    if cycles * len(frame_registers) > loop_limits.max_frames:
        raise ValueError("DM3D auto-loop frame count exceeds configured limit")

    return AutoLoopProgram(
        cycles=cycles,
        feedback_register=feedback_register,
        convergence_scalar_register=convergence_scalar_register,
        frame_register_mask=frame_mask,
        stop_on_fixed_point=bool(flags & AUTO_FLAG_STOP_ON_FIXED),
        inner_program=inner,
    )


def canonical_animation_loop_program(
    *,
    cycles: int = 8,
    pool: int = 2,
    refinement_passes: int = 6,
    stop_on_fixed_point: bool = False,
) -> bytes:
    """Build the canonical X -> E -> R^K -> D -> E -> feedback animation loop."""

    inner = canonical_autoencoder_program(pool=pool, refinement_passes=refinement_passes)
    return serialize_auto_loop_program(
        inner,
        cycles=cycles,
        feedback_register=3,
        convergence_scalar_register=13,
        frame_registers=DEFAULT_FRAME_REGISTERS,
        stop_on_fixed_point=stop_on_fixed_point,
    )


def execute_auto_loop(
    payload: bytes,
    initial_volume: Volume3D,
    *,
    loop_limits: AutoLoopLimits | None = None,
    vm_limits: ProgramLimits | None = None,
) -> AutoLoopResult:
    """Execute a finite auto-encoding/decoding feedback loop end to end."""

    loop_limits = loop_limits or AutoLoopLimits()
    vm_limits = vm_limits or ProgramLimits()
    program = parse_auto_loop_program(payload, loop_limits=loop_limits, vm_limits=vm_limits)
    current = initial_volume.copy()
    cumulative = VmMetrics()
    frames: list[VolumeFrame] = []
    telemetry: list[CycleTelemetry] = []
    stopped = False

    for cycle in range(program.cycles):
        remaining = loop_limits.max_total_physical_steps - cumulative.physical_steps
        if remaining <= 0:
            raise RuntimeError("DM3D auto-loop physical-step budget exceeded")
        cycle_limits = ProgramLimits(
            max_instructions=vm_limits.max_instructions,
            max_voxels=vm_limits.max_voxels,
            max_refinement_passes=vm_limits.max_refinement_passes,
            max_physical_steps=min(vm_limits.max_physical_steps, remaining),
        )
        result = execute_program(program.inner_program, current, cycle_limits)
        _accumulate_metrics(cumulative, result.metrics)
        if cumulative.physical_steps > loop_limits.max_total_physical_steps:
            raise RuntimeError("DM3D auto-loop physical-step budget exceeded")

        for register in program.frame_registers:
            volume = result.volumes.get(register)
            if volume is not None:
                frames.append(
                    VolumeFrame(
                        cycle=cycle,
                        register=register,
                        stage=STAGE_NAMES.get(register, f"volume-r{register}"),
                        volume=volume.copy(),
                    )
                )
                if len(frames) > loop_limits.max_frames:
                    raise RuntimeError("DM3D auto-loop frame budget exceeded")

        fixed = result.scalars.get(program.convergence_scalar_register, 0.0) >= 1.0
        telemetry.append(
            CycleTelemetry(
                cycle=cycle,
                reconstruction_mse=result.scalars.get(10),
                cycle_mse=result.scalars.get(11),
                omega=result.scalars.get(12),
                fixed_point_pass=fixed,
                physical_steps=result.metrics.physical_steps,
            )
        )

        try:
            current = result.volumes[program.feedback_register].copy()
        except KeyError as exc:
            raise RuntimeError(
                f"DM3D feedback volume r{program.feedback_register} is unavailable"
            ) from exc

        if program.stop_on_fixed_point and fixed:
            stopped = True
            break

    return AutoLoopResult(
        cycles_executed=len(telemetry),
        stopped_on_fixed_point=stopped,
        final_volume=current,
        metrics=cumulative,
        cycles=telemetry,
        frames=frames,
    )


def _frame_mask(registers: tuple[int, ...]) -> int:
    if len(set(registers)) != len(registers):
        raise ValueError("frame registers must be unique")
    mask = 0
    for register in registers:
        _validate_register(register)
        mask |= 1 << register
    return mask


def _validate_register(register: int) -> None:
    if not 0 <= register < 32:
        raise ValueError("DM3D auto-loop register must lie in [0, 31]")


def _accumulate_metrics(total: VmMetrics, current: VmMetrics) -> None:
    total.instructions_executed += current.instructions_executed
    total.physical_steps += current.physical_steps
    total.voxel_reads += current.voxel_reads
    total.voxel_writes += current.voxel_writes
    total.neighbor_reads += current.neighbor_reads
    total.refinement_updates += current.refinement_updates
    total.elapsed_seconds += current.elapsed_seconds


__all__ = [
    "AutoLoopLimits",
    "AutoLoopProgram",
    "AutoLoopResult",
    "CycleTelemetry",
    "VolumeFrame",
    "canonical_animation_loop_program",
    "execute_auto_loop",
    "parse_auto_loop_program",
    "serialize_auto_loop_program",
]
