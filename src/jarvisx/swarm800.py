"""Operational reference model for the Jarvis-X 800-instance swarm.

This module turns the corrected arithmetic and hierarchy into a deterministic,
standard-library-only simulator. It models the control plane, SVI encoding,
fixed-point arithmetic, safe mutation, fusion cadence, and sealed invariants;
it does not pretend to be a WebGPU or distributed-training backend.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import struct
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

Q_FRAC_BITS = 8
Q_SCALE = 1 << Q_FRAC_BITS
Q_MIN = -(1 << 15)
Q_MAX = (1 << 15) - 1
SVI_BYTES = 16
ROM_BYTES = 4096
ROM_INSTRUCTIONS = ROM_BYTES // SVI_BYTES
AGE_MAX = (1 << 12) - 1
COORD_MIN = -(1 << 11)
COORD_MAX = (1 << 11) - 1

INSTANCE_COUNT = 800
INSTANCES_PER_ZONE = 10
ZONES_PER_REGION = 10
REGION_COUNT = 8
ZONE_COUNT = REGION_COUNT * ZONES_PER_REGION

PIPELINE_US = {
    "fetch": 1.0,
    "encode": 5.0,
    "decode": 5.0,
    "execute": 2.0,
    "backprop": 10.0,
}
PIPELINE_LATENCY_US = sum(PIPELINE_US.values())
PIPELINE_BOTTLENECK_US = max(PIPELINE_US.values())


class SwarmInvariantError(RuntimeError):
    """Raised when a sealed runtime invariant is violated."""


def _sat16(value: int) -> int:
    return max(Q_MIN, min(Q_MAX, int(value)))


def _trunc_div(numerator: int, denominator: int) -> int:
    if denominator == 0:
        raise ZeroDivisionError("Q8.8 division by zero")
    sign = -1 if (numerator < 0) ^ (denominator < 0) else 1
    return sign * (abs(numerator) // abs(denominator))


def q_from_float(value: float) -> int:
    """Encode a real value as saturated signed Q8.8."""
    return _sat16(round(float(value) * Q_SCALE))


def q_to_float(value: int) -> float:
    return int(value) / Q_SCALE


def q_add(a: int, b: int) -> int:
    return _sat16(int(a) + int(b))


def q_sub(a: int, b: int) -> int:
    return _sat16(int(a) - int(b))


def q_mul(a: int, b: int) -> int:
    """Q8.8 multiply with a wide intermediate and nearest rounding."""
    product = int(a) * int(b)
    magnitude = (abs(product) + (Q_SCALE // 2)) >> Q_FRAC_BITS
    return _sat16(-magnitude if product < 0 else magnitude)


def q_div(a: int, b: int) -> int:
    """Q8.8 divide with truncation toward zero."""
    return _sat16(_trunc_div(int(a) << Q_FRAC_BITS, int(b)))


def _encode_signed_12(value: int) -> int:
    if not COORD_MIN <= value <= COORD_MAX:
        raise ValueError("12-bit signed coordinate out of range")
    return value & 0xFFF


def _decode_signed_12(value: int) -> int:
    value &= 0xFFF
    return value - 0x1000 if value & 0x800 else value


@dataclass(frozen=True)
class SVI:
    """Corrected 128-bit Spatial Virtual Instruction.

    Layout, least significant bit first:
      opcode: 8, flags: 8, x/y/z: 12 each, operand: 32,
      edge_fingerprint: 32, age_timer: 12.
    """

    opcode: int = 0
    flags: int = 0
    x: int = 0
    y: int = 0
    z: int = 0
    operand: int = 0
    edge_fingerprint: int = 0
    age_timer: int = 0

    def validate(self) -> None:
        if not 0 <= self.opcode <= 0xFF:
            raise ValueError("opcode must fit 8 bits")
        if not 0 <= self.flags <= 0xFF:
            raise ValueError("flags must fit 8 bits")
        for value in (self.x, self.y, self.z):
            if not COORD_MIN <= value <= COORD_MAX:
                raise ValueError("coordinate must fit signed 12 bits")
        if not 0 <= self.operand <= 0xFFFFFFFF:
            raise ValueError("operand must fit 32 bits")
        if not 0 <= self.edge_fingerprint <= 0xFFFFFFFF:
            raise ValueError("edge_fingerprint must fit 32 bits")
        if not 0 <= self.age_timer <= AGE_MAX:
            raise ValueError("age_timer must fit 12 bits")

    def pack_int(self) -> int:
        self.validate()
        word = self.opcode
        word |= self.flags << 8
        word |= _encode_signed_12(self.x) << 16
        word |= _encode_signed_12(self.y) << 28
        word |= _encode_signed_12(self.z) << 40
        word |= self.operand << 52
        word |= self.edge_fingerprint << 84
        word |= self.age_timer << 116
        if word.bit_length() > 128:
            raise SwarmInvariantError("SVI exceeded 128 bits")
        return word

    def to_bytes(self) -> bytes:
        return self.pack_int().to_bytes(SVI_BYTES, "little", signed=False)

    @classmethod
    def unpack_int(cls, word: int) -> "SVI":
        if word < 0 or word.bit_length() > 128:
            raise ValueError("SVI word must be an unsigned 128-bit integer")
        return cls(
            opcode=word & 0xFF,
            flags=(word >> 8) & 0xFF,
            x=_decode_signed_12((word >> 16) & 0xFFF),
            y=_decode_signed_12((word >> 28) & 0xFFF),
            z=_decode_signed_12((word >> 40) & 0xFFF),
            operand=(word >> 52) & 0xFFFFFFFF,
            edge_fingerprint=(word >> 84) & 0xFFFFFFFF,
            age_timer=(word >> 116) & 0xFFF,
        )

    @classmethod
    def from_bytes(cls, payload: bytes) -> "SVI":
        if len(payload) != SVI_BYTES:
            raise ValueError("SVI payload must be exactly 16 bytes")
        return cls.unpack_int(int.from_bytes(payload, "little", signed=False))

    def aged(self) -> "SVI":
        values = asdict(self)
        values["age_timer"] = min(AGE_MAX, self.age_timer + 1)
        return SVI(**values)


def pc_to_xyz(pc_byte: int) -> Tuple[int, int, int]:
    """Map a byte PC in a 4 KiB ROM to the corrected 8x8x4 grid."""
    if pc_byte % SVI_BYTES != 0:
        raise ValueError("PC must be 16-byte aligned")
    index = pc_byte // SVI_BYTES
    if not 0 <= index < ROM_INSTRUCTIONS:
        raise ValueError("PC outside 4 KiB ROM")
    return index & 7, (index >> 3) & 7, (index >> 6) & 3


def xyz_to_pc(x: int, y: int, z: int) -> int:
    if not (0 <= x < 8 and 0 <= y < 8 and 0 <= z < 4):
        raise ValueError("ROM coordinate outside 8x8x4 geometry")
    return SVI_BYTES * (x + 8 * y + 64 * z)


def next_pc(pc_byte: int, branch: bool = False, instruction_offset: int = 0) -> int:
    current_index = pc_byte // SVI_BYTES
    target = current_index + 1 + (instruction_offset if branch else 0)
    return (target % ROM_INSTRUCTIONS) * SVI_BYTES


def edge_fingerprint(x: int, y: int, z: int, operand: int) -> int:
    """Produce a 32-bit fingerprint; unlike a 36-bit Morton code, collisions are allowed."""
    payload = struct.pack("<hhhI", x, y, z, operand & 0xFFFFFFFF)
    return int.from_bytes(hashlib.blake2s(payload, digest_size=4).digest(), "little")


class SVICodec:
    """Exact 128-dimensional Q8.8 bit-latent codec for the reference runtime.

    Every instruction bit is represented as +1 or -1 in Q8.8. This keeps the
    simulator deterministic and exact while leaving a clean interface for a
    learned 16,384-to-128 encoder backend.
    """

    latent_dim = 128

    @staticmethod
    def encode(instruction: SVI) -> Tuple[int, ...]:
        word = instruction.pack_int()
        return tuple(Q_SCALE if (word >> bit) & 1 else -Q_SCALE for bit in range(128))

    @staticmethod
    def decode(latent: Sequence[int]) -> SVI:
        if len(latent) != 128:
            raise ValueError("latent vector must have 128 elements")
        word = 0
        for bit, value in enumerate(latent):
            if int(value) >= 0:
                word |= 1 << bit
        return SVI.unpack_int(word)


@dataclass(frozen=True)
class Metrics:
    byte_fidelity: float
    spatial_fidelity: float
    latency_us: float
    throughput_per_second: float
    psnr: float
    latent_energy: float
    meta_loss: float
    fitness: float


def _psnr(original: bytes, reconstructed: bytes) -> float:
    if len(original) != len(reconstructed):
        return 0.0
    mse = sum((a - b) ** 2 for a, b in zip(original, reconstructed)) / len(original)
    if mse == 0:
        return 99.0
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def evaluate_instruction(original: SVI, reconstructed: SVI, latent: Sequence[int]) -> Metrics:
    before = original.to_bytes()
    after = reconstructed.to_bytes()
    equal_bits = 128 - (original.pack_int() ^ reconstructed.pack_int()).bit_count()
    byte_fidelity = equal_bits / 128.0
    spatial_fidelity = (
        int(original.x == reconstructed.x)
        + int(original.y == reconstructed.y)
        + int(original.z == reconstructed.z)
    ) / 3.0
    latency = PIPELINE_LATENCY_US
    throughput = 1_000_000.0 / PIPELINE_BOTTLENECK_US
    latent_energy = sum(q_to_float(v) ** 2 for v in latent) / max(1, len(latent))
    meta_loss = (
        0.7 * (1.0 - byte_fidelity)
        + 0.2 * (1.0 - spatial_fidelity)
        + 0.09 * (latency / PIPELINE_LATENCY_US)
        + 0.01 * latent_energy
    )
    fitness = (
        1.0 / (1.0 + latency / PIPELINE_LATENCY_US)
        + 0.02 * _psnr(before, after)
        - 0.1 * latent_energy
    )
    return Metrics(
        byte_fidelity=byte_fidelity,
        spatial_fidelity=spatial_fidelity,
        latency_us=latency,
        throughput_per_second=throughput,
        psnr=_psnr(before, after),
        latent_energy=latent_energy,
        meta_loss=meta_loss,
        fitness=fitness,
    )


class SharedROM:
    """Immutable 4 KiB base ROM with per-instance copy-on-write patches."""

    def __init__(self, instructions: Sequence[SVI]):
        if len(instructions) != ROM_INSTRUCTIONS:
            raise ValueError("base ROM must contain exactly 256 SVIs")
        self.instructions = tuple(instructions)
        self.manifest_hash = hashlib.sha256(
            b"".join(instruction.to_bytes() for instruction in self.instructions)
        ).hexdigest()

    @classmethod
    def deterministic(cls, seed: int = 0xDEADBEEF) -> "SharedROM":
        rng = random.Random(seed)
        instructions: List[SVI] = []
        for index in range(ROM_INSTRUCTIONS):
            x, y, z = pc_to_xyz(index * SVI_BYTES)
            operand = rng.getrandbits(32)
            instructions.append(
                SVI(
                    opcode=index & 0xFF,
                    flags=0,
                    x=x,
                    y=y,
                    z=z,
                    operand=operand,
                    edge_fingerprint=edge_fingerprint(x, y, z, operand),
                    age_timer=0,
                )
            )
        return cls(instructions)

    def fetch(self, pc_byte: int, patches: Dict[int, SVI]) -> SVI:
        index = pc_byte // SVI_BYTES
        if not 0 <= index < ROM_INSTRUCTIONS:
            raise SwarmInvariantError("fetch outside ROM")
        return patches.get(index, self.instructions[index])


@dataclass
class MutationResult:
    accepted: bool
    index: int
    flipped_bits: Tuple[int, int, int]
    old_fitness: float
    new_fitness: float


@dataclass
class InstanceTelemetry:
    instance_id: int
    cycle: int
    meta_loss: float
    best_loss: float
    fitness: float
    latency_us: float
    accepted_mutations: int
    pc_byte: int
    hash_head: str


@dataclass
class SwarmInstance:
    instance_id: int
    base_rom: SharedROM
    seed: int
    pc_byte: int = 0
    registers: List[int] = field(default_factory=lambda: [0] * 16)
    patches: Dict[int, SVI] = field(default_factory=dict)
    cycle: int = 0
    accepted_mutations: int = 0
    best_loss: float = float("inf")
    hash_head: str = "0" * 64

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def _execute(self, instruction: SVI) -> None:
        operand = instruction.operand
        rd = operand & 0xF
        rs1 = (operand >> 4) & 0xF
        rs2 = (operand >> 8) & 0xF
        a = self.registers[rs1]
        b = self.registers[rs2]
        operation = instruction.opcode % 10
        if operation == 0:
            out = q_add(a, b)
        elif operation == 1:
            out = q_sub(a, b)
        elif operation == 2:
            out = q_mul(a, b)
        elif operation == 3:
            out = q_div(a, b) if b else Q_MAX
        elif operation == 4:
            out = _sat16((a & 0xFFFF) & (b & 0xFFFF))
        elif operation == 5:
            out = _sat16((a & 0xFFFF) | (b & 0xFFFF))
        elif operation == 6:
            out = _sat16((a & 0xFFFF) ^ (b & 0xFFFF))
        elif operation == 7:
            out = _sat16(a << (b & 0xF))
        elif operation == 8:
            out = _sat16((a & 0xFFFF) >> (b & 0xF))
        else:
            out = _sat16(a >> (b & 0xF))
        self.registers[rd] = out

        branch = bool(instruction.flags & 0x1) and out != 0
        offset_raw = (operand >> 16) & 0xFFFF
        offset = offset_raw - 0x10000 if offset_raw & 0x8000 else offset_raw
        self.pc_byte = next_pc(self.pc_byte, branch=branch, instruction_offset=offset)

    @staticmethod
    def _shadow_cost(instruction: SVI) -> float:
        return (
            instruction.edge_fingerprint.bit_count() / 32.0
            + instruction.operand.bit_count() / 64.0
            + instruction.age_timer / AGE_MAX
        )

    def _attempt_mutation(self, index: int, current: SVI, baseline: Metrics) -> MutationResult:
        # Only operand and edge-fingerprint bits are mutable: packed bits 52..115.
        bits = tuple(sorted(self._rng.sample(range(52, 116), 3)))
        candidate_word = current.pack_int()
        for bit in bits:
            candidate_word ^= 1 << bit
        candidate = SVI.unpack_int(candidate_word)
        candidate.validate()

        old_fitness = baseline.fitness - 0.01 * self._shadow_cost(current)
        new_fitness = baseline.fitness - 0.01 * self._shadow_cost(candidate)
        accepted = new_fitness > old_fitness * 1.001
        if accepted:
            self.patches[index] = candidate
            self.accepted_mutations += 1
        return MutationResult(accepted, index, bits, old_fitness, new_fitness)

    def step(self, mutate: bool = True) -> InstanceTelemetry:
        index = self.pc_byte // SVI_BYTES
        instruction = self.base_rom.fetch(self.pc_byte, self.patches)
        latent = SVICodec.encode(instruction)
        reconstructed = SVICodec.decode(latent)
        metrics = evaluate_instruction(instruction, reconstructed, latent)

        self._execute(reconstructed)
        aged = reconstructed.aged()
        self.patches[index] = aged
        if mutate:
            self._attempt_mutation(index, aged, metrics)

        self.cycle += 1
        self.best_loss = min(self.best_loss, metrics.meta_loss)
        event = {
            "instance": self.instance_id,
            "cycle": self.cycle,
            "pc": self.pc_byte,
            "loss": round(metrics.meta_loss, 12),
            "patches": len(self.patches),
        }
        self.hash_head = hashlib.sha256(
            bytes.fromhex(self.hash_head) + json.dumps(event, sort_keys=True).encode("utf-8")
        ).hexdigest()
        self.validate()
        return InstanceTelemetry(
            instance_id=self.instance_id,
            cycle=self.cycle,
            meta_loss=metrics.meta_loss,
            best_loss=self.best_loss,
            fitness=metrics.fitness,
            latency_us=metrics.latency_us,
            accepted_mutations=self.accepted_mutations,
            pc_byte=self.pc_byte,
            hash_head=self.hash_head,
        )

    def validate(self) -> None:
        if self.pc_byte % SVI_BYTES != 0 or not 0 <= self.pc_byte < ROM_BYTES:
            raise SwarmInvariantError("invalid PC")
        if len(self.registers) != 16 or any(not Q_MIN <= value <= Q_MAX for value in self.registers):
            raise SwarmInvariantError("invalid Q8.8 register state")
        for index, instruction in self.patches.items():
            if not 0 <= index < ROM_INSTRUCTIONS:
                raise SwarmInvariantError("patch index outside ROM")
            instruction.validate()


@dataclass(frozen=True)
class FusionState:
    count: int
    cycle: int
    mean_loss: float
    best_loss: float
    mean_fitness: float
    p95_latency_us: float
    accepted_mutations: int


def fuse_telemetry(items: Sequence[InstanceTelemetry]) -> FusionState:
    if not items:
        raise ValueError("cannot fuse an empty telemetry set")
    ordered_latency = sorted(item.latency_us for item in items)
    p95_index = min(len(ordered_latency) - 1, math.ceil(0.95 * len(ordered_latency)) - 1)
    return FusionState(
        count=len(items),
        cycle=max(item.cycle for item in items),
        mean_loss=sum(item.meta_loss for item in items) / len(items),
        best_loss=min(item.best_loss for item in items),
        mean_fitness=sum(item.fitness for item in items) / len(items),
        p95_latency_us=ordered_latency[p95_index],
        accepted_mutations=sum(item.accepted_mutations for item in items),
    )


@dataclass
class SwarmReport:
    cycle: int
    instance_count: int
    zone_count: int
    region_count: int
    current: FusionState
    zone_fusions: int
    region_fusions: int
    global_fusions: int
    global_best_loss: float
    manifest_hash: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class Swarm800:
    """Hierarchical 800-instance reference swarm.

    Fusion cadence:
      - zone: every 10 cycles
      - region: every 100 cycles
      - global checkpoint: every 1000 cycles
    """

    def __init__(self, seed: int = 0xDEADBEEF):
        self.seed = seed
        self.base_rom = SharedROM.deterministic(seed)
        self.instances = [
            SwarmInstance(instance_id=i, base_rom=self.base_rom, seed=seed ^ (i * 0x9E3779B1))
            for i in range(INSTANCE_COUNT)
        ]
        self.cycle = 0
        self.zone_states: Dict[int, FusionState] = {}
        self.region_states: Dict[int, FusionState] = {}
        self.global_state: Optional[FusionState] = None
        self.global_best_loss = float("inf")
        self._last_telemetry: List[InstanceTelemetry] = []
        self.validate_topology()

    def validate_topology(self) -> None:
        if len(self.instances) != INSTANCE_COUNT:
            raise SwarmInvariantError("canonical swarm must contain exactly 800 instances")
        if INSTANCE_COUNT != REGION_COUNT * ZONES_PER_REGION * INSTANCES_PER_ZONE:
            raise SwarmInvariantError("hierarchy product does not equal 800")
        if any(instance.base_rom.manifest_hash != self.base_rom.manifest_hash for instance in self.instances):
            raise SwarmInvariantError("sealed ROM manifest differs across instances")

    def _zone_slice(self, zone_id: int) -> slice:
        start = zone_id * INSTANCES_PER_ZONE
        return slice(start, start + INSTANCES_PER_ZONE)

    def step(self, mutate: bool = True) -> SwarmReport:
        telemetry = [instance.step(mutate=mutate) for instance in self.instances]
        self._last_telemetry = telemetry
        self.cycle += 1
        current = fuse_telemetry(telemetry)
        self.global_best_loss = min(self.global_best_loss, current.best_loss)

        if self.cycle % 10 == 0:
            for zone_id in range(ZONE_COUNT):
                self.zone_states[zone_id] = fuse_telemetry(telemetry[self._zone_slice(zone_id)])

        if self.cycle % 100 == 0:
            region_width = ZONES_PER_REGION * INSTANCES_PER_ZONE
            for region_id in range(REGION_COUNT):
                start = region_id * region_width
                self.region_states[region_id] = fuse_telemetry(telemetry[start : start + region_width])

        if self.cycle % 1000 == 0:
            self.global_state = current

        self.validate_topology()
        return SwarmReport(
            cycle=self.cycle,
            instance_count=INSTANCE_COUNT,
            zone_count=ZONE_COUNT,
            region_count=REGION_COUNT,
            current=current,
            zone_fusions=len(self.zone_states),
            region_fusions=len(self.region_states),
            global_fusions=1 if self.global_state is not None else 0,
            global_best_loss=self.global_best_loss,
            manifest_hash=self.base_rom.manifest_hash,
        )

    def run(self, cycles: int, mutate: bool = True) -> SwarmReport:
        if cycles < 1:
            raise ValueError("cycles must be positive")
        report: Optional[SwarmReport] = None
        for _ in range(cycles):
            report = self.step(mutate=mutate)
        assert report is not None
        return report


def run_swarm(cycles: int = 1, seed: int = 0xDEADBEEF, mutate: bool = True) -> Dict[str, object]:
    """Convenience entry point used by the CLI and smoke tests."""
    swarm = Swarm800(seed=seed)
    return swarm.run(cycles=cycles, mutate=mutate).to_dict()
