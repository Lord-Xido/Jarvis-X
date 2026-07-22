"""Deterministic operational model of the Jarvis-X 800-instance swarm."""
from __future__ import annotations

import hashlib
import json
import math
import random
import struct
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

Q_SCALE = 256
Q_MIN, Q_MAX = -32768, 32767
SVI_BYTES, ROM_BYTES = 16, 4096
ROM_INSTRUCTIONS = ROM_BYTES // SVI_BYTES
AGE_MAX = 4095
COORD_MIN, COORD_MAX = -2048, 2047
INSTANCE_COUNT, INSTANCES_PER_ZONE = 800, 10
ZONES_PER_REGION, REGION_COUNT = 10, 8
ZONE_COUNT = REGION_COUNT * ZONES_PER_REGION
PIPELINE_US = {"fetch": 1.0, "encode": 5.0, "decode": 5.0, "execute": 2.0, "backprop": 10.0}
PIPELINE_LATENCY_US = sum(PIPELINE_US.values())
PIPELINE_BOTTLENECK_US = max(PIPELINE_US.values())


class SwarmInvariantError(RuntimeError):
    pass


def _sat16(v: int) -> int:
    return max(Q_MIN, min(Q_MAX, int(v)))


def _signed16(v: int) -> int:
    v &= 0xFFFF
    return v - 0x10000 if v & 0x8000 else v


def _popcount(v: int) -> int:
    return bin(abs(int(v))).count("1")


def _trunc_div(a: int, b: int) -> int:
    if b == 0:
        raise ZeroDivisionError("Q8.8 division by zero")
    sign = -1 if (a < 0) ^ (b < 0) else 1
    return sign * (abs(a) // abs(b))


def q_from_float(v: float) -> int:
    return _sat16(round(float(v) * Q_SCALE))


def q_to_float(v: int) -> float:
    return int(v) / Q_SCALE


def q_add(a: int, b: int) -> int:
    return _sat16(a + b)


def q_sub(a: int, b: int) -> int:
    return _sat16(a - b)


def q_mul(a: int, b: int) -> int:
    product = int(a) * int(b)
    out = (abs(product) + 128) >> 8
    return _sat16(-out if product < 0 else out)


def q_div(a: int, b: int) -> int:
    return _sat16(_trunc_div(int(a) << 8, int(b)))


def _enc12(v: int) -> int:
    if not COORD_MIN <= v <= COORD_MAX:
        raise ValueError("coordinate outside signed 12-bit range")
    return v & 0xFFF


def _dec12(v: int) -> int:
    v &= 0xFFF
    return v - 0x1000 if v & 0x800 else v


@dataclass(frozen=True)
class SVI:
    opcode: int = 0
    flags: int = 0
    x: int = 0
    y: int = 0
    z: int = 0
    operand: int = 0
    edge_fingerprint: int = 0
    age_timer: int = 0

    def validate(self) -> None:
        if not 0 <= self.opcode <= 0xFF or not 0 <= self.flags <= 0xFF:
            raise ValueError("opcode/flags outside 8-bit range")
        if any(not COORD_MIN <= v <= COORD_MAX for v in (self.x, self.y, self.z)):
            raise ValueError("coordinate outside signed 12-bit range")
        if not 0 <= self.operand <= 0xFFFFFFFF:
            raise ValueError("operand outside 32-bit range")
        if not 0 <= self.edge_fingerprint <= 0xFFFFFFFF:
            raise ValueError("edge fingerprint outside 32-bit range")
        if not 0 <= self.age_timer <= AGE_MAX:
            raise ValueError("age outside 12-bit range")

    def pack_int(self) -> int:
        self.validate()
        word = self.opcode | (self.flags << 8)
        word |= _enc12(self.x) << 16
        word |= _enc12(self.y) << 28
        word |= _enc12(self.z) << 40
        word |= self.operand << 52
        word |= self.edge_fingerprint << 84
        word |= self.age_timer << 116
        if word.bit_length() > 128:
            raise SwarmInvariantError("SVI exceeds 128 bits")
        return word

    def to_bytes(self) -> bytes:
        return self.pack_int().to_bytes(16, "little")

    @classmethod
    def unpack_int(cls, word: int) -> "SVI":
        if word < 0 or word.bit_length() > 128:
            raise ValueError("word is not unsigned 128-bit")
        return cls(
            word & 0xFF,
            (word >> 8) & 0xFF,
            _dec12(word >> 16),
            _dec12(word >> 28),
            _dec12(word >> 40),
            (word >> 52) & 0xFFFFFFFF,
            (word >> 84) & 0xFFFFFFFF,
            (word >> 116) & 0xFFF,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "SVI":
        if len(data) != 16:
            raise ValueError("SVI must be 16 bytes")
        return cls.unpack_int(int.from_bytes(data, "little"))

    def aged(self) -> "SVI":
        values = asdict(self)
        values["age_timer"] = min(AGE_MAX, self.age_timer + 1)
        return SVI(**values)


def pc_to_xyz(pc: int) -> Tuple[int, int, int]:
    if pc % 16:
        raise ValueError("PC must be 16-byte aligned")
    n = pc // 16
    if not 0 <= n < 256:
        raise ValueError("PC outside 4 KiB ROM")
    return n & 7, (n >> 3) & 7, (n >> 6) & 3


def xyz_to_pc(x: int, y: int, z: int) -> int:
    if not (0 <= x < 8 and 0 <= y < 8 and 0 <= z < 4):
        raise ValueError("coordinate outside 8x8x4 ROM")
    return 16 * (x + 8 * y + 64 * z)


def next_pc(pc: int, branch: bool = False, instruction_offset: int = 0) -> int:
    return ((pc // 16 + 1 + (instruction_offset if branch else 0)) % 256) * 16


def edge_fingerprint(x: int, y: int, z: int, operand: int) -> int:
    data = struct.pack("<hhhI", x, y, z, operand & 0xFFFFFFFF)
    return int.from_bytes(hashlib.blake2s(data, digest_size=4).digest(), "little")


class SVICodec:
    latent_dim = 128

    @staticmethod
    def encode(i: SVI) -> Tuple[int, ...]:
        word = i.pack_int()
        return tuple(Q_SCALE if (word >> b) & 1 else -Q_SCALE for b in range(128))

    @staticmethod
    def decode(z: Sequence[int]) -> SVI:
        if len(z) != 128:
            raise ValueError("latent must have 128 values")
        word = sum((1 << b) for b, v in enumerate(z) if int(v) >= 0)
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


def _psnr(a: bytes, b: bytes) -> float:
    mse = sum((x - y) ** 2 for x, y in zip(a, b)) / len(a)
    return 99.0 if mse == 0 else 20 * math.log10(255 / math.sqrt(mse))


def evaluate_instruction(a: SVI, b: SVI, z: Sequence[int]) -> Metrics:
    fidelity = (128 - _popcount(a.pack_int() ^ b.pack_int())) / 128
    spatial = sum((a.x == b.x, a.y == b.y, a.z == b.z)) / 3
    energy = sum(q_to_float(v) ** 2 for v in z) / len(z)
    psnr = _psnr(a.to_bytes(), b.to_bytes())
    loss = 0.7 * (1 - fidelity) + 0.2 * (1 - spatial) + 0.09 + 0.01 * energy
    fitness = 0.5 + 0.02 * psnr - 0.1 * energy
    return Metrics(
        fidelity,
        spatial,
        PIPELINE_LATENCY_US,
        1_000_000 / PIPELINE_BOTTLENECK_US,
        psnr,
        energy,
        loss,
        fitness,
    )


class SharedROM:
    def __init__(self, instructions: Sequence[SVI]):
        if len(instructions) != 256:
            raise ValueError("ROM must contain 256 instructions")
        self.instructions = tuple(instructions)
        self.manifest_hash = hashlib.sha256(
            b"".join(i.to_bytes() for i in instructions)
        ).hexdigest()

    @classmethod
    def deterministic(cls, seed: int = 0xDEADBEEF) -> "SharedROM":
        rng = random.Random(seed)
        rom = []
        for n in range(256):
            x, y, z = pc_to_xyz(n * 16)
            operand = rng.getrandbits(32)
            rom.append(SVI(n, 0, x, y, z, operand, edge_fingerprint(x, y, z, operand), 0))
        return cls(rom)

    def fetch(self, pc: int, patches: Dict[int, SVI]) -> SVI:
        n = pc // 16
        if not 0 <= n < 256:
            raise SwarmInvariantError("fetch outside ROM")
        return patches.get(n, self.instructions[n])


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

    def _execute(self, i: SVI) -> None:
        rd = i.operand & 15
        rs1 = (i.operand >> 4) & 15
        rs2 = (i.operand >> 8) & 15
        a, b, op = self.registers[rs1], self.registers[rs2], i.opcode % 10
        operations = {
            0: lambda: q_add(a, b),
            1: lambda: q_sub(a, b),
            2: lambda: q_mul(a, b),
            3: lambda: q_div(a, b) if b else Q_MAX,
            4: lambda: _signed16(a & b),
            5: lambda: _signed16(a | b),
            6: lambda: _signed16(a ^ b),
            7: lambda: _signed16(a << (b & 15)),
            8: lambda: _signed16((a & 0xFFFF) >> (b & 15)),
            9: lambda: _sat16(a >> (b & 15)),
        }
        out = operations[op]()
        self.registers[rd] = out
        raw = (i.operand >> 16) & 0xFFFF
        offset = raw - 0x10000 if raw & 0x8000 else raw
        self.pc_byte = next_pc(self.pc_byte, bool(i.flags & 1) and out != 0, offset)

    @staticmethod
    def _shadow_cost(i: SVI) -> float:
        return (
            _popcount(i.edge_fingerprint) / 32
            + _popcount(i.operand) / 64
            + i.age_timer / AGE_MAX
        )

    def _attempt_mutation(self, n: int, current: SVI, baseline: Metrics) -> MutationResult:
        bits = tuple(sorted(self._rng.sample(range(52, 116), 3)))
        word = current.pack_int()
        for bit in bits:
            word ^= 1 << bit
        candidate = SVI.unpack_int(word)
        candidate.validate()
        old = baseline.fitness - 0.05 * self._shadow_cost(current)
        new = baseline.fitness - 0.05 * self._shadow_cost(candidate)
        accepted = new > old * 1.001
        if accepted:
            self.patches[n] = candidate
            self.accepted_mutations += 1
        return MutationResult(accepted, n, bits, old, new)

    def step(self, mutate: bool = True) -> InstanceTelemetry:
        n = self.pc_byte // 16
        instruction = self.base_rom.fetch(self.pc_byte, self.patches)
        latent = SVICodec.encode(instruction)
        reconstructed = SVICodec.decode(latent)
        metrics = evaluate_instruction(instruction, reconstructed, latent)
        self._execute(reconstructed)
        aged = reconstructed.aged()
        self.patches[n] = aged
        if mutate:
            self._attempt_mutation(n, aged, metrics)
        self.cycle += 1
        self.best_loss = min(self.best_loss, metrics.meta_loss)
        event = json.dumps(
            {"i": self.instance_id, "c": self.cycle, "pc": self.pc_byte,
             "loss": round(metrics.meta_loss, 12)},
            sort_keys=True,
        ).encode()
        self.hash_head = hashlib.sha256(bytes.fromhex(self.hash_head) + event).hexdigest()
        self.validate()
        return InstanceTelemetry(
            self.instance_id,
            self.cycle,
            metrics.meta_loss,
            self.best_loss,
            metrics.fitness,
            metrics.latency_us,
            self.accepted_mutations,
            self.pc_byte,
            self.hash_head,
        )

    def validate(self) -> None:
        if self.pc_byte % 16 or not 0 <= self.pc_byte < 4096:
            raise SwarmInvariantError("invalid PC")
        if len(self.registers) != 16 or any(not Q_MIN <= v <= Q_MAX for v in self.registers):
            raise SwarmInvariantError("invalid register state")
        for n, instruction in self.patches.items():
            if not 0 <= n < 256:
                raise SwarmInvariantError("invalid patch index")
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
        raise ValueError("cannot fuse empty telemetry")
    latencies = sorted(i.latency_us for i in items)
    p95 = latencies[min(len(latencies) - 1, math.ceil(0.95 * len(latencies)) - 1)]
    return FusionState(
        len(items),
        max(i.cycle for i in items),
        sum(i.meta_loss for i in items) / len(items),
        min(i.best_loss for i in items),
        sum(i.fitness for i in items) / len(items),
        p95,
        sum(i.accepted_mutations for i in items),
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
    def __init__(self, seed: int = 0xDEADBEEF):
        self.base_rom = SharedROM.deterministic(seed)
        self.instances = [
            SwarmInstance(i, self.base_rom, seed ^ (i * 0x9E3779B1))
            for i in range(INSTANCE_COUNT)
        ]
        self.cycle = 0
        self.zone_states: Dict[int, FusionState] = {}
        self.region_states: Dict[int, FusionState] = {}
        self.global_state: Optional[FusionState] = None
        self.global_best_loss = float("inf")
        self.validate_topology()

    def validate_topology(self) -> None:
        if len(self.instances) != 800 or 800 != 8 * 10 * 10:
            raise SwarmInvariantError("invalid 800-instance hierarchy")
        if any(i.base_rom.manifest_hash != self.base_rom.manifest_hash for i in self.instances):
            raise SwarmInvariantError("base ROM manifest mismatch")

    def step(self, mutate: bool = True) -> SwarmReport:
        telemetry = [i.step(mutate) for i in self.instances]
        self.cycle += 1
        current = fuse_telemetry(telemetry)
        self.global_best_loss = min(self.global_best_loss, current.best_loss)
        if self.cycle % 10 == 0:
            for zone in range(80):
                start = zone * 10
                self.zone_states[zone] = fuse_telemetry(telemetry[start:start + 10])
        if self.cycle % 100 == 0:
            for region in range(8):
                start = region * 100
                self.region_states[region] = fuse_telemetry(telemetry[start:start + 100])
        if self.cycle % 1000 == 0:
            self.global_state = current
        self.validate_topology()
        return SwarmReport(
            self.cycle,
            800,
            80,
            8,
            current,
            len(self.zone_states),
            len(self.region_states),
            int(self.global_state is not None),
            self.global_best_loss,
            self.base_rom.manifest_hash,
        )

    def run(self, cycles: int, mutate: bool = True) -> SwarmReport:
        if cycles < 1:
            raise ValueError("cycles must be positive")
        report = None
        for _ in range(cycles):
            report = self.step(mutate)
        assert report is not None
        return report


def run_swarm(cycles: int = 1, seed: int = 0xDEADBEEF, mutate: bool = True) -> Dict[str, object]:
    return Swarm800(seed).run(cycles, mutate).to_dict()
