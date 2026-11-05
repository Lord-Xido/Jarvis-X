"""Deterministic DM-vOmegaXi+ spatial-kernel reference.

The module turns the symbolic three-axis Dr Moagi spatial equation into a bounded,
auditable bytecode machine.  It deliberately distinguishes a logical ROM capacity from
the tiny physical program image used by the reference implementation.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass, replace
from enum import IntEnum
from typing import Iterable, Sequence, Tuple

Vector3 = Tuple[float, float, float]


class SpatialOpcode(IntEnum):
    """Eight-byte micro-operations executed by :class:`OME6400SpatialKernel`."""

    ENCODE = 0x10
    EVOLVE = 0x20
    PROJECT = 0x30
    DECODE = 0x40
    ROTATE = 0x50
    PULSE = 0x60
    SEAL = 0x70
    HALT = 0xFF


@dataclass(frozen=True)
class SpatialInstruction:
    """Fixed-width 64-bit instruction: opcode/flags/operand/immediate."""

    opcode: SpatialOpcode
    flags: int = 0
    operand: int = 0
    immediate: int = 0

    WIDTH = 8
    _STRUCT = struct.Struct(">BBHI")

    def __post_init__(self) -> None:
        if not 0 <= self.flags <= 0xFF:
            raise ValueError("flags must fit in 8 bits")
        if not 0 <= self.operand <= 0xFFFF:
            raise ValueError("operand must fit in 16 bits")
        if not 0 <= self.immediate <= 0xFFFFFFFF:
            raise ValueError("immediate must fit in 32 bits")

    def to_bytes(self) -> bytes:
        return self._STRUCT.pack(int(self.opcode), self.flags, self.operand, self.immediate)

    @classmethod
    def from_bytes(cls, payload: bytes) -> "SpatialInstruction":
        if len(payload) != cls.WIDTH:
            raise ValueError("spatial instruction must be exactly 8 bytes")
        opcode, flags, operand, immediate = cls._STRUCT.unpack(payload)
        try:
            decoded = SpatialOpcode(opcode)
        except ValueError as exc:
            raise ValueError(f"unsupported spatial opcode 0x{opcode:02X}") from exc
        return cls(decoded, flags, operand, immediate)


def default_program() -> Tuple[SpatialInstruction, ...]:
    return tuple(
        SpatialInstruction(opcode)
        for opcode in (
            SpatialOpcode.ENCODE,
            SpatialOpcode.EVOLVE,
            SpatialOpcode.PROJECT,
            SpatialOpcode.DECODE,
            SpatialOpcode.ROTATE,
            SpatialOpcode.PULSE,
            SpatialOpcode.SEAL,
            SpatialOpcode.HALT,
        )
    )


def assemble_program(program: Iterable[SpatialInstruction]) -> bytes:
    instructions = tuple(program)
    if not instructions:
        raise ValueError("program cannot be empty")
    return b"".join(instruction.to_bytes() for instruction in instructions)


def decode_program(payload: bytes) -> Tuple[SpatialInstruction, ...]:
    if not payload or len(payload) % SpatialInstruction.WIDTH:
        raise ValueError("bytecode length must be a non-zero multiple of 8")
    return tuple(
        SpatialInstruction.from_bytes(payload[offset : offset + SpatialInstruction.WIDTH])
        for offset in range(0, len(payload), SpatialInstruction.WIDTH)
    )


@dataclass(frozen=True)
class SpatialKernelConfig:
    """Numerical and execution bounds for the SK-3D reference machine."""

    phi: float = 1.618033988749895
    recursion_base: float = 2.0
    projection_limit: float = 4.0
    velocity: float = 0.75
    dt: float = 1.0 / 60.0
    kappa: float = 0.10
    eta: float = 0.05
    zeta: float = 0.05
    omega_decay: float = 0.98
    omega_gain: float = 0.05
    pulse_base: float = 1.0
    pulse_gain: float = 0.25
    max_program_instructions: int = 64
    max_cycles: int = 4096
    virtual_rom_bytes: int = 6_400_000_000

    def __post_init__(self) -> None:
        for name in (
            "phi",
            "recursion_base",
            "projection_limit",
            "velocity",
            "dt",
            "kappa",
            "eta",
            "zeta",
            "omega_decay",
            "omega_gain",
            "pulse_base",
            "pulse_gain",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")

        if self.phi <= 0.0:
            raise ValueError("phi must be positive")
        if self.recursion_base <= 0.0:
            raise ValueError("recursion_base must be positive")
        if self.projection_limit <= 0.0:
            raise ValueError("projection_limit must be positive")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.kappa < 0.0 or self.eta < 0.0 or self.zeta < 0.0:
            raise ValueError("kappa, eta and zeta must be non-negative")
        if not 0.0 <= self.omega_decay <= 1.0:
            raise ValueError("omega_decay must be in [0, 1]")
        if self.omega_gain < 0.0:
            raise ValueError("omega_gain must be non-negative")
        if self.pulse_base <= 0.0 or self.pulse_gain < 0.0:
            raise ValueError("pulse_base must be positive and pulse_gain non-negative")

        for name in ("max_program_instructions", "max_cycles", "virtual_rom_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class SpatialInputs:
    """External terms supplied to one bounded Dr Moagi recurrence."""

    observation: float = 0.0
    intent: float | None = None
    prediction: float = 0.0
    refinement: float = 0.0
    grad_theta: float = 0.0
    grad_h: float = 0.0

    def validate(self) -> None:
        for name in ("observation", "prediction", "refinement", "grad_theta", "grad_h"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.intent is not None:
            if isinstance(self.intent, bool) or not isinstance(self.intent, (int, float)):
                raise TypeError("intent must be numeric when supplied")
            if not math.isfinite(float(self.intent)):
                raise ValueError("intent must be finite")


@dataclass(frozen=True)
class SpatialState:
    """Persistent state of the symbolic cognitive/recursive kernel."""

    psi: float = 0.0
    theta: float = 1.0
    xi: float = 0.0
    omega: float = 0.0
    cycle: int = 0
    rotation_angle: float = 0.0
    pulse: float = 1.0
    residual: float = 0.0
    digest: str = ""


@dataclass(frozen=True)
class SpatialFrame:
    """Decoded render-neutral manifold for a single committed cycle."""

    cycle: int
    axes: Vector3
    singularity: float
    vertices: Tuple[Vector3, Vector3, Vector3, Vector3]
    rotation_angle: float
    pulse: float
    residual: float
    digest: str


@dataclass
class _WorkingCycle:
    state: SpatialState
    encoded: float = 0.0
    anticipation: float = 0.0
    candidate_xi: float = 0.0
    axes: Vector3 = (0.0, 0.0, 1.0)
    singularity: float = 0.0
    vertices: Tuple[Vector3, Vector3, Vector3, Vector3] = (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )
    halted: bool = False


class OME6400SpatialKernel:
    """Bounded bytecode executor for the DM-vOmegaXi+ SK-3D equation.

    One bytecode program execution is one complete spatial cycle.  All work occurs on a
    shadow state and is committed only after HALT, so malformed programs or non-finite
    transitions cannot partially mutate authoritative state.
    """

    def __init__(
        self,
        config: SpatialKernelConfig | None = None,
        state: SpatialState | None = None,
        program: Sequence[SpatialInstruction] | None = None,
    ) -> None:
        self.config = config or SpatialKernelConfig()
        self._program = tuple(program or default_program())
        self._validate_program(self._program)
        initial = state or SpatialState(pulse=self.config.pulse_base)
        self._validate_state(initial)
        self._state = self._seal(initial)

    @property
    def state(self) -> SpatialState:
        return self._state

    @property
    def program(self) -> Tuple[SpatialInstruction, ...]:
        return self._program

    @property
    def rom_image(self) -> bytes:
        return assemble_program(self._program)

    @property
    def physical_rom_bytes(self) -> int:
        return len(self.rom_image)

    @property
    def virtual_rom_bytes(self) -> int:
        return self.config.virtual_rom_bytes

    def geometry(self, state: SpatialState | None = None) -> Tuple[Vector3, float]:
        current = state or self._state
        recursion = self.config.recursion_base ** current.omega
        axes = (
            self.config.phi * current.xi,
            current.psi * current.theta,
            recursion,
        )
        singularity = (current.psi * self.config.phi) / recursion
        return axes, singularity

    def execute_cycle(self, inputs: SpatialInputs | None = None) -> SpatialFrame:
        if self._state.cycle >= self.config.max_cycles:
            raise RuntimeError("maximum cycle budget reached")
        supplied = inputs or SpatialInputs()
        supplied.validate()
        working = _WorkingCycle(state=self._state)

        for instruction in self._program:
            self._dispatch(instruction, supplied, working)
            if working.halted:
                break

        if not working.halted:
            raise RuntimeError("program terminated without HALT")
        committed = replace(working.state, cycle=self._state.cycle + 1)
        self._validate_state(committed)
        committed = self._seal(committed)
        self._state = committed

        axes, singularity = self.geometry(committed)
        base_vertices = self._tetrahedron(committed, axes[2])
        vertices = (
            self._rotate_and_scale(base_vertices[0], committed.rotation_angle, committed.pulse),
            self._rotate_and_scale(base_vertices[1], committed.rotation_angle, committed.pulse),
            self._rotate_and_scale(base_vertices[2], committed.rotation_angle, committed.pulse),
            self._rotate_and_scale(base_vertices[3], committed.rotation_angle, committed.pulse),
        )
        return SpatialFrame(
            cycle=committed.cycle,
            axes=axes,
            singularity=singularity,
            vertices=vertices,
            rotation_angle=committed.rotation_angle,
            pulse=committed.pulse,
            residual=committed.residual,
            digest=committed.digest,
        )

    def run(self, cycles: int, inputs: SpatialInputs | None = None) -> SpatialFrame:
        if isinstance(cycles, bool) or not isinstance(cycles, int):
            raise TypeError("cycles must be an integer")
        if cycles <= 0:
            raise ValueError("cycles must be positive")
        if self._state.cycle + cycles > self.config.max_cycles:
            raise RuntimeError("requested cycles exceed maximum cycle budget")
        frame: SpatialFrame | None = None
        for _ in range(cycles):
            frame = self.execute_cycle(inputs)
        assert frame is not None
        return frame

    def _dispatch(
        self,
        instruction: SpatialInstruction,
        inputs: SpatialInputs,
        working: _WorkingCycle,
    ) -> None:
        opcode = instruction.opcode
        if opcode is SpatialOpcode.ENCODE:
            psi = working.state.psi if inputs.intent is None else self._project(float(inputs.intent))
            encoded = math.tanh(psi * self.config.phi + float(inputs.observation))
            anticipation = math.tanh(float(inputs.prediction) + encoded + 0.5 * working.state.xi)
            working.state = replace(working.state, psi=psi)
            working.encoded = encoded
            working.anticipation = anticipation
        elif opcode is SpatialOpcode.EVOLVE:
            _, decoded = self.geometry(working.state)
            residual = float(inputs.observation) - decoded
            omega = self._project(
                self.config.omega_decay * working.state.omega
                + self.config.omega_gain * residual
            )
            theta = self._project(
                working.state.theta - self.config.eta * float(inputs.grad_theta)
            )
            working.candidate_xi = (
                working.state.xi
                + working.anticipation
                - residual
                + omega
                + self.config.kappa * float(inputs.refinement)
                - self.config.eta * float(inputs.grad_theta)
                - self.config.zeta * float(inputs.grad_h)
            )
            working.state = replace(
                working.state,
                theta=theta,
                omega=omega,
                residual=residual,
            )
        elif opcode is SpatialOpcode.PROJECT:
            working.state = replace(working.state, xi=self._project(working.candidate_xi))
        elif opcode is SpatialOpcode.DECODE:
            working.axes, working.singularity = self.geometry(working.state)
            working.vertices = self._tetrahedron(working.state, working.axes[2])
        elif opcode is SpatialOpcode.ROTATE:
            angular_velocity = self.config.velocity * (working.state.omega * working.state.xi)
            angle = (working.state.rotation_angle + angular_velocity * self.config.dt) % (
                2.0 * math.pi
            )
            working.state = replace(working.state, rotation_angle=angle)
        elif opcode is SpatialOpcode.PULSE:
            pulse = self.config.pulse_base + self.config.pulse_gain * abs(working.state.residual)
            working.state = replace(working.state, pulse=pulse)
        elif opcode is SpatialOpcode.SEAL:
            working.state = self._seal(working.state)
        elif opcode is SpatialOpcode.HALT:
            working.halted = True
        else:  # pragma: no cover - IntEnum validation makes this unreachable.
            raise RuntimeError(f"unhandled opcode {opcode!r}")

        self._validate_state(working.state)

    def _validate_program(self, program: Sequence[SpatialInstruction]) -> None:
        if not program:
            raise ValueError("program cannot be empty")
        if len(program) > self.config.max_program_instructions:
            raise ValueError("program exceeds instruction budget")
        if SpatialOpcode.HALT not in tuple(instruction.opcode for instruction in program):
            raise ValueError("program must contain HALT")

    def _validate_state(self, state: SpatialState) -> None:
        if isinstance(state.cycle, bool) or not isinstance(state.cycle, int) or state.cycle < 0:
            raise ValueError("cycle must be a non-negative integer")
        for name in ("psi", "theta", "xi", "omega", "rotation_angle", "pulse", "residual"):
            value = getattr(state, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"state {name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"state {name} must be finite")
        for name in ("psi", "theta", "xi", "omega"):
            if abs(float(getattr(state, name))) > self.config.projection_limit + 1.0e-12:
                raise ValueError(f"state {name} lies outside the projection manifold")
        if state.pulse <= 0.0:
            raise ValueError("state pulse must be positive")

    def _project(self, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("projection candidate must be finite")
        limit = self.config.projection_limit
        return min(limit, max(-limit, value))

    def _tetrahedron(self, state: SpatialState, recursion: float) -> Tuple[Vector3, Vector3, Vector3, Vector3]:
        return (
            (0.0, state.psi, 0.0),
            (self.config.phi, 0.0, 0.0),
            (0.0, 0.0, recursion),
            (-state.omega, -state.theta, -state.xi),
        )

    @staticmethod
    def _rotate_and_scale(vertex: Vector3, angle: float, scale: float) -> Vector3:
        # Rodrigues rotation around normalized axis (1,1,1)/sqrt(3).
        x, y, z = vertex
        ux = uy = uz = 1.0 / math.sqrt(3.0)
        cosine = math.cos(angle)
        sine = math.sin(angle)
        dot = ux * x + uy * y + uz * z
        cross = (uy * z - uz * y, uz * x - ux * z, ux * y - uy * x)
        rotated = (
            x * cosine + cross[0] * sine + ux * dot * (1.0 - cosine),
            y * cosine + cross[1] * sine + uy * dot * (1.0 - cosine),
            z * cosine + cross[2] * sine + uz * dot * (1.0 - cosine),
        )
        return tuple(scale * component for component in rotated)  # type: ignore[return-value]

    def _seal(self, state: SpatialState) -> SpatialState:
        digest = hashlib.sha256()
        digest.update(b"jarvisx-dm-vomegaxi-sk3d-v1\0")
        digest.update(struct.pack(">Q", state.cycle))
        for value in (
            self.config.phi,
            self.config.recursion_base,
            self.config.projection_limit,
            state.psi,
            state.theta,
            state.xi,
            state.omega,
            state.rotation_angle,
            state.pulse,
            state.residual,
        ):
            digest.update(struct.pack(">d", float(value)))
        digest.update(self.rom_image)
        return replace(state, digest=digest.hexdigest())
