"""Sparse 3D 1 MiB-cubed auto-encoding bytecode ROM.

The logical ROM geometry is ``2**20 x 2**20 x 2**20`` byte cells, i.e.
``2**60`` bytes = 1 EiB.  The implementation never allocates that address
space physically.  It exposes a deterministic sparse ROM, 60-bit Morton
addressing, a compact 64-bit bytecode ISA, and a bounded inward autoencoder
reference loop.

The 1000x figure in this module refers to the spatial site reduction of one
``10 x 10 x 10 -> 1`` contraction stage.  It is not a wall-clock speed claim.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import struct
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Dict, Sequence

AXIS = 1 << 20
LOGICAL_BYTES = AXIS**3
ADDRESS_BITS = 60
PAGE_SIZE = 4096
PAGE_MASK = PAGE_SIZE - 1
BLOCK_EDGE = 10
BLOCK_VOLUME = BLOCK_EDGE**3
MAGIC = b"M3DROM1\x00"
VERSION = 1


def _part1by2(value: int) -> int:
    """Spread the low 20 source bits with two zero bits between them."""

    value &= 0xFFFFF
    out = 0
    for bit in range(20):
        out |= ((value >> bit) & 1) << (3 * bit)
    return out


def morton3_encode(x: int, y: int, z: int) -> int:
    """Map one 3D ROM coordinate to a reversible 60-bit Morton address."""

    if not (0 <= x < AXIS and 0 <= y < AXIS and 0 <= z < AXIS):
        raise ValueError("coordinate outside 1 MiB-cubed logical ROM")
    return _part1by2(x) | (_part1by2(y) << 1) | (_part1by2(z) << 2)


def morton3_decode(code: int) -> tuple[int, int, int]:
    """Invert :func:`morton3_encode`."""

    if not 0 <= code < (1 << ADDRESS_BITS):
        raise ValueError("address outside 60-bit logical ROM")
    x = y = z = 0
    for bit in range(20):
        x |= ((code >> (3 * bit)) & 1) << bit
        y |= ((code >> (3 * bit + 1)) & 1) << bit
        z |= ((code >> (3 * bit + 2)) & 1) << bit
    return x, y, z


class Op(IntEnum):
    """Reference ROM instruction set."""

    NOP = 0x00
    INGEST_3D = 0x01
    LOAD_TILE = 0x02
    ENCODE_3D = 0x03
    FOLD_IN_10 = 0x04
    LATENT_STEP = 0x05
    FIXPOINT_CHK = 0x06
    DECODE_3D = 0x07
    RESIDUAL_ADD = 0x08
    CONTRAST = 0x09
    OMEGA_UPDATE = 0x0A
    THETA_UPDATE = 0x0B
    RETILE = 0x0C
    EMIT_BYTE = 0x0D
    JUMP = 0x0E
    JNZ = 0x0F
    HALT = 0xFF


@dataclass(frozen=True)
class Instruction:
    """One 64-bit instruction: opcode:8, a:8, b:16, c:32."""

    op: Op
    a: int = 0
    b: int = 0
    c: int = 0

    def pack(self) -> bytes:
        return struct.pack(
            ">BBHI", int(self.op) & 0xFF, self.a & 0xFF, self.b & 0xFFFF, self.c & 0xFFFFFFFF
        )

    @staticmethod
    def unpack(blob: bytes) -> "Instruction":
        if len(blob) != 8:
            raise ValueError("one ROM instruction is exactly 8 bytes")
        op, a, b, c = struct.unpack(">BBHI", blob)
        return Instruction(Op(op), a, b, c)


BOOT_PROGRAM = (
    Instruction(Op.INGEST_3D),
    Instruction(Op.LOAD_TILE),
    Instruction(Op.ENCODE_3D),
    Instruction(Op.FOLD_IN_10),
    Instruction(Op.LATENT_STEP),
    Instruction(Op.FIXPOINT_CHK),
    Instruction(Op.DECODE_3D),
    Instruction(Op.RESIDUAL_ADD),
    Instruction(Op.CONTRAST),
    Instruction(Op.OMEGA_UPDATE),
    Instruction(Op.THETA_UPDATE),
    Instruction(Op.RETILE),
    Instruction(Op.EMIT_BYTE),
    Instruction(Op.HALT),
)


@dataclass
class VirtualROM:
    """Logical 1 EiB read-only ROM backed by deterministic sparse pages."""

    seed: bytes = b"MOAGI-3D-ROM-v1"
    explicit_pages: Dict[int, bytes] = field(default_factory=dict)

    def _synthesize_page(self, page_index: int) -> bytes:
        if page_index < 0 or page_index >= LOGICAL_BYTES // PAGE_SIZE:
            raise ValueError("page outside logical ROM")
        root = hashlib.blake2b(
            self.seed + page_index.to_bytes(8, "big"), digest_size=32
        ).digest()
        out = bytearray()
        counter = 0
        while len(out) < PAGE_SIZE:
            out.extend(
                hashlib.blake2b(root + counter.to_bytes(8, "big"), digest_size=64).digest()
            )
            counter += 1
        return bytes(out[:PAGE_SIZE])

    def read_linear(self, address: int) -> int:
        if not 0 <= address < LOGICAL_BYTES:
            raise ValueError("linear address out of range")
        page_index = address // PAGE_SIZE
        offset = address & PAGE_MASK
        page = self.explicit_pages.get(page_index)
        if page is None:
            page = self._synthesize_page(page_index)
        if len(page) != PAGE_SIZE:
            raise ValueError("explicit ROM pages must be PAGE_SIZE bytes")
        return page[offset]

    def read3(self, x: int, y: int, z: int) -> int:
        return self.read_linear(morton3_encode(x, y, z))

    def read_block(self, x0: int, y0: int, z0: int, edge: int = BLOCK_EDGE) -> bytes:
        if edge < 1:
            raise ValueError("edge must be positive")
        if not (0 <= x0 < AXIS and 0 <= y0 < AXIS and 0 <= z0 < AXIS):
            raise ValueError("block origin outside logical ROM")
        out = bytearray()
        for dz in range(edge):
            for dy in range(edge):
                for dx in range(edge):
                    x = min(AXIS - 1, x0 + dx)
                    y = min(AXIS - 1, y0 + dy)
                    z = min(AXIS - 1, z0 + dz)
                    out.append(self.read3(x, y, z))
        return bytes(out)

    def install_boot_program(self) -> None:
        boot = b"".join(instruction.pack() for instruction in BOOT_PROGRAM)
        page = bytearray(self._synthesize_page(0))
        page[: len(boot)] = boot
        self.explicit_pages[0] = bytes(page)


@dataclass
class LatentState:
    z: list[float]
    omega: list[float]
    theta: list[float]
    error: float = 1.0
    iterations: int = 0


class InwardAutoEncoder:
    """Deterministic bounded reference for inward encode/refine/decode."""

    def __init__(self, latent_dim: int = 32) -> None:
        if latent_dim < 2:
            raise ValueError("latent_dim must be >= 2")
        self.latent_dim = latent_dim

    def _features(self, block: bytes) -> list[float]:
        if not block:
            return [0.0] * self.latent_dim
        count = len(block)
        mean = sum(block) / (255.0 * count)
        variance = sum(((value / 255.0) - mean) ** 2 for value in block) / count
        features = [mean, variance]
        for channel in range(self.latent_dim - 2):
            frequency = 1.0 + channel % 11
            phase = (channel * 0.61803398875) % 1.0
            accumulator = 0.0
            for index, value in enumerate(block):
                accumulator += (value / 255.0) * math.sin(
                    (index + 1) * frequency * 0.013 + phase
                )
            features.append(math.tanh(accumulator / max(1.0, count * 0.15)))
        return features

    def encode(self, block: bytes, previous: LatentState | None = None) -> LatentState:
        features = self._features(block)
        if previous is None:
            omega = [0.0] * self.latent_dim
            theta = [0.12 + 0.01 * ((i % 7) - 3) for i in range(self.latent_dim)]
        else:
            omega = list(previous.omega)
            theta = list(previous.theta)
        z = [
            math.tanh(features[i] + 0.15 * omega[i] + 0.05 * theta[i])
            for i in range(self.latent_dim)
        ]
        return LatentState(z=z, omega=omega, theta=theta)

    @staticmethod
    def fold_in(z: Sequence[float]) -> list[float]:
        if not z:
            return []
        count = len(z)
        return [
            math.tanh(0.2 * z[(i - 1) % count] + 0.6 * z[i] + 0.2 * z[(i + 1) % count])
            for i in range(count)
        ]

    def refine_fixed_point(
        self, state: LatentState, max_iter: int = 64, eps: float = 1e-6
    ) -> LatentState:
        if max_iter < 1:
            raise ValueError("max_iter must be >= 1")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        z = list(state.z)
        ratio = float("inf")
        for iteration in range(1, max_iter + 1):
            candidate = self.fold_in(z)
            for i in range(self.latent_dim):
                candidate[i] = math.tanh(
                    candidate[i] + 0.03 * state.omega[i] + 0.01 * state.theta[i]
                )
            numerator = math.sqrt(sum((a - b) ** 2 for a, b in zip(candidate, z)))
            denominator = math.sqrt(sum(value * value for value in z)) + 1e-12
            ratio = numerator / denominator
            z = candidate
            if ratio < eps:
                return LatentState(
                    z=z,
                    omega=list(state.omega),
                    theta=list(state.theta),
                    error=ratio,
                    iterations=iteration,
                )
        return LatentState(
            z=z,
            omega=list(state.omega),
            theta=list(state.theta),
            error=ratio,
            iterations=max_iter,
        )

    def decode(self, state: LatentState, length: int) -> bytes:
        if length < 0:
            raise ValueError("length must be non-negative")
        out = bytearray(length)
        for index in range(length):
            a = state.z[index % self.latent_dim]
            b = state.z[(index * 7 + 3) % self.latent_dim]
            c = state.z[(index * 13 + 5) % self.latent_dim]
            value = 0.5 + 0.22 * a + 0.17 * b + 0.11 * c
            out[index] = round(255 * max(0.0, min(1.0, value)))
        return bytes(out)

    def learn_from_residual(
        self,
        expected: bytes,
        predicted: bytes,
        state: LatentState,
        rho: float = 0.97,
        learning_rate: float = 0.02,
    ) -> LatentState:
        if len(expected) != len(predicted):
            raise ValueError("expected and predicted lengths differ")
        if not expected:
            return state
        channels = [0.0] * self.latent_dim
        counts = [0] * self.latent_dim
        absolute_error = 0.0
        for index, (left, right) in enumerate(zip(expected, predicted)):
            error = (left - right) / 255.0
            channel = index % self.latent_dim
            channels[channel] += error
            counts[channel] += 1
            absolute_error += abs(error)
        for channel, count in enumerate(counts):
            if count:
                channels[channel] /= count
        omega = [rho * state.omega[i] + (1.0 - rho) * channels[i] for i in range(self.latent_dim)]
        theta = [
            max(-1.0, min(1.0, state.theta[i] + learning_rate * channels[i]))
            for i in range(self.latent_dim)
        ]
        return LatentState(
            z=list(state.z),
            omega=omega,
            theta=theta,
            error=absolute_error / len(expected),
            iterations=state.iterations,
        )


class ROMVM:
    """Small auditable VM for the ROM bootstrap sequence."""

    def __init__(self, rom: VirtualROM, autoencoder: InwardAutoEncoder) -> None:
        self.rom = rom
        self.autoencoder = autoencoder
        self.pc = 0
        self.halted = False
        self.current_xyz = (0, 0, 0)
        self.last_block = b""
        self.prediction = b""
        self.state = LatentState(
            z=[0.0] * autoencoder.latent_dim,
            omega=[0.0] * autoencoder.latent_dim,
            theta=[0.1] * autoencoder.latent_dim,
        )

    def fetch(self) -> Instruction:
        blob = bytes(self.rom.read_linear(self.pc + offset) for offset in range(8))
        self.pc += 8
        return Instruction.unpack(blob)

    def execute(self, instruction: Instruction) -> None:
        op = instruction.op
        if op == Op.NOP:
            return
        if op == Op.INGEST_3D:
            self.current_xyz = (
                instruction.c & 0xFFFFF,
                ((instruction.c >> 4) ^ instruction.b) & 0xFFFFF,
                ((instruction.c >> 8) ^ instruction.a) & 0xFFFFF,
            )
        elif op == Op.LOAD_TILE:
            self.last_block = self.rom.read_block(*self.current_xyz, edge=BLOCK_EDGE)
        elif op == Op.ENCODE_3D:
            self.state = self.autoencoder.encode(self.last_block, self.state)
        elif op == Op.FOLD_IN_10:
            self.state.z = self.autoencoder.fold_in(self.state.z)
        elif op == Op.LATENT_STEP:
            self.state = self.autoencoder.refine_fixed_point(self.state, max_iter=8, eps=1e-5)
        elif op == Op.FIXPOINT_CHK:
            return
        elif op == Op.DECODE_3D:
            self.prediction = self.autoencoder.decode(self.state, len(self.last_block))
        elif op == Op.RESIDUAL_ADD:
            return
        elif op == Op.CONTRAST:
            if self.last_block and self.prediction:
                self.state = self.autoencoder.learn_from_residual(
                    self.last_block, self.prediction, self.state
                )
        elif op in (Op.OMEGA_UPDATE, Op.THETA_UPDATE, Op.RETILE, Op.EMIT_BYTE):
            return
        elif op == Op.JUMP:
            self.pc = instruction.c
        elif op == Op.JNZ:
            if self.state.error > 1e-3:
                self.pc = instruction.c
        elif op == Op.HALT:
            self.halted = True
        else:  # pragma: no cover - exhaustive IntEnum guard
            raise RuntimeError(f"unhandled opcode {op}")

    def run(self, max_steps: int = 1000) -> int:
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        steps = 0
        while not self.halted and steps < max_steps:
            self.execute(self.fetch())
            steps += 1
        return steps


def descriptor_bytes(seed: bytes = b"MOAGI-3D-ROM-v1") -> bytes:
    """Return the compact descriptor for the logical 1 EiB ROM."""

    boot = b"".join(instruction.pack() for instruction in BOOT_PROGRAM)
    header = struct.pack(
        ">8sIQQIIII32s",
        MAGIC,
        VERSION,
        AXIS,
        LOGICAL_BYTES,
        ADDRESS_BITS,
        PAGE_SIZE,
        BLOCK_EDGE,
        len(boot),
        hashlib.sha256(seed).digest(),
    )
    return header + boot


def write_descriptor(path: str | Path, seed: bytes = b"MOAGI-3D-ROM-v1") -> Path:
    target = Path(path)
    target.write_bytes(descriptor_bytes(seed))
    return target


def self_test() -> dict[str, float | int | bool]:
    """Run a bounded deterministic end-to-end ROM smoke test."""

    samples = (
        (0, 0, 0),
        (1, 2, 3),
        (12345, 54321, 999999),
        (AXIS - 1, AXIS - 1, AXIS - 1),
    )
    for xyz in samples:
        assert morton3_decode(morton3_encode(*xyz)) == xyz

    rom = VirtualROM()
    rom.install_boot_program()
    autoencoder = InwardAutoEncoder(latent_dim=32)
    vm = ROMVM(rom, autoencoder)
    steps = vm.run(max_steps=len(BOOT_PROGRAM) + 8)
    return {
        "logical_bytes": LOGICAL_BYTES,
        "logical_eib": LOGICAL_BYTES / (1 << 60),
        "block_volume": BLOCK_VOLUME,
        "vm_steps": steps,
        "vm_halted": vm.halted,
        "reconstruction_mae": vm.state.error,
    }


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--self-test", action="store_true")
    command.add_argument("--write-rom-descriptor", metavar="PATH")
    return command


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.write_rom_descriptor:
        path = write_descriptor(args.write_rom_descriptor)
        print(path)
    if args.self_test or not args.write_rom_descriptor:
        for key, value in self_test().items():
            print(f"{key}={value}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
