#!/usr/bin/env python3
"""Jarvis X 3D Auto-Encoding/Decoding Bytecode ROM.

A self-contained educational VM with:
- fixed 32-bit instructions,
- a 256-word ROM,
- 256-byte data memory,
- eight 32-bit registers,
- a reversible compressed snapshot codec,
- a deterministic 3D ROM coordinate map,
- transactional restore and journal hashing.

The latent representation is a compressed state snapshot. It is reversible; a
cryptographic digest is used only for integrity, not as a decoder.
"""

from __future__ import annotations

import hashlib
import re
import struct
import zlib
from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable, Sequence

ROM_WORDS = 256
DATA_BYTES = 256
REGISTER_COUNT = 8
STACK_DEPTH = 16
GRID_SIDE = 8  # 8^3 = 512 addressable spatial cells
SNAPSHOT_MAGIC = b"JX3D"
SNAPSHOT_VERSION = 1


class Opcode(IntEnum):
    NOP = 0x01
    LOADI = 0x02
    LOAD = 0x03
    STORE = 0x04
    ADD = 0x05
    SUB = 0x06
    MUL = 0x07
    DIV = 0x08
    JMP = 0x09
    JIF = 0x0A
    CALL = 0x0B
    RET = 0x0C
    HALT = 0x0D
    ENCODE = 0x0E
    DECODE = 0x0F
    PRINT = 0x10


@dataclass(frozen=True)
class Instruction:
    opcode: Opcode
    op1: int = 0
    op2: int = 0
    flags: int = 0

    def pack(self) -> int:
        for value in (self.op1, self.op2, self.flags):
            if not 0 <= value <= 0xFF:
                raise ValueError(f"instruction field out of range: {value}")
        return (
            (int(self.opcode) << 24)
            | (self.op1 << 16)
            | (self.op2 << 8)
            | self.flags
        )

    @classmethod
    def unpack(cls, word: int) -> "Instruction":
        opcode = Opcode((word >> 24) & 0xFF)
        return cls(opcode, (word >> 16) & 0xFF, (word >> 8) & 0xFF, word & 0xFF)


@dataclass(frozen=True)
class JournalEntry:
    cycle: int
    pc: int
    state_digest: str
    latent_digest: str
    chain_digest: str


class Assembler:
    """Two-pass assembler for the Jarvis X instruction set."""

    _register = re.compile(r"^R([0-7])$", re.IGNORECASE)

    def assemble(self, source: str) -> list[int]:
        statements = list(self._statements(source))
        labels: dict[str, int] = {}
        pc = 0
        for label, _tokens in statements:
            if label:
                if label in labels:
                    raise ValueError(f"duplicate label: {label}")
                labels[label] = pc
            if _tokens:
                pc += 1
        if pc > ROM_WORDS:
            raise ValueError(f"program has {pc} instructions; maximum is {ROM_WORDS}")

        words: list[int] = []
        for _label, tokens in statements:
            if not tokens:
                continue
            words.append(self._encode(tokens, labels).pack())
        return words

    def _statements(self, source: str) -> Iterable[tuple[str | None, list[str]]]:
        for line_number, raw in enumerate(source.splitlines(), start=1):
            line = raw.split(";", 1)[0].strip()
            if not line:
                continue
            label = None
            if ":" in line:
                prefix, remainder = line.split(":", 1)
                label = prefix.strip()
                if not label or not re.match(r"^[A-Za-z_]\w*$", label):
                    raise ValueError(f"line {line_number}: invalid label")
                line = remainder.strip()
            tokens = [token for token in re.split(r"[\s,]+", line) if token]
            yield label, tokens

    def _encode(self, tokens: Sequence[str], labels: dict[str, int]) -> Instruction:
        mnemonic = tokens[0].upper()
        try:
            opcode = Opcode[mnemonic]
        except KeyError as exc:
            raise ValueError(f"unknown opcode: {mnemonic}") from exc

        args = list(tokens[1:])
        if opcode in {Opcode.NOP, Opcode.RET, Opcode.HALT, Opcode.ENCODE, Opcode.DECODE}:
            self._arity(mnemonic, args, 0)
            return Instruction(opcode)
        if opcode == Opcode.LOADI:
            self._arity(mnemonic, args, 2)
            return Instruction(opcode, self._reg(args[0]), self._byte(args[1]))
        if opcode in {Opcode.LOAD, Opcode.STORE}:
            self._arity(mnemonic, args, 2)
            return Instruction(opcode, self._reg(args[0]), self._byte(args[1]))
        if opcode in {Opcode.ADD, Opcode.SUB, Opcode.MUL, Opcode.DIV}:
            self._arity(mnemonic, args, 3)
            return Instruction(opcode, self._reg(args[0]), self._reg(args[1]), self._reg(args[2]))
        if opcode in {Opcode.JMP, Opcode.JIF, Opcode.CALL}:
            expected = 2 if opcode == Opcode.JIF else 1
            self._arity(mnemonic, args, expected)
            target_token = args[-1]
            target = labels.get(target_token, self._integer(target_token))
            if not 0 <= target < ROM_WORDS:
                raise ValueError(f"jump target out of range: {target}")
            condition_register = self._reg(args[0]) if opcode == Opcode.JIF else 0
            return Instruction(opcode, condition_register, target)
        if opcode == Opcode.PRINT:
            self._arity(mnemonic, args, 1)
            return Instruction(opcode, self._reg(args[0]))
        raise AssertionError(f"unhandled opcode: {opcode}")

    @staticmethod
    def _arity(name: str, args: Sequence[str], expected: int) -> None:
        if len(args) != expected:
            raise ValueError(f"{name} expects {expected} operands; received {len(args)}")

    def _reg(self, token: str) -> int:
        match = self._register.match(token)
        if not match:
            raise ValueError(f"invalid register: {token}")
        return int(match.group(1))

    def _byte(self, token: str) -> int:
        value = self._integer(token)
        if not 0 <= value <= 0xFF:
            raise ValueError(f"byte operand out of range: {value}")
        return value

    @staticmethod
    def _integer(token: str) -> int:
        try:
            return int(token, 0)
        except ValueError:
            return -1


class StateCodec:
    """Reversible state encoder/decoder with integrity verification."""

    @staticmethod
    def encode(registers: Sequence[int], memory: bytes, pc: int, stack: Sequence[int]) -> bytes:
        if len(registers) != REGISTER_COUNT or len(memory) != DATA_BYTES:
            raise ValueError("invalid VM state shape")
        if len(stack) > STACK_DEPTH:
            raise ValueError("stack overflow in snapshot")
        payload = bytearray()
        payload += SNAPSHOT_MAGIC
        payload += struct.pack(">BBH", SNAPSHOT_VERSION, len(stack), pc)
        payload += struct.pack(">8i", *registers)
        payload += memory
        payload += bytes(stack)
        digest = hashlib.sha256(payload).digest()
        return zlib.compress(bytes(payload) + digest, level=9)

    @staticmethod
    def decode(latent: bytes) -> tuple[list[int], bytearray, int, list[int]]:
        packed = zlib.decompress(latent)
        minimum = 4 + 4 + 32 + DATA_BYTES + 32
        if len(packed) < minimum:
            raise ValueError("truncated latent snapshot")
        payload, recorded_digest = packed[:-32], packed[-32:]
        if hashlib.sha256(payload).digest() != recorded_digest:
            raise ValueError("latent snapshot integrity failure")
        if payload[:4] != SNAPSHOT_MAGIC:
            raise ValueError("invalid snapshot magic")
        version, stack_size, pc = struct.unpack(">BBH", payload[4:8])
        if version != SNAPSHOT_VERSION or stack_size > STACK_DEPTH:
            raise ValueError("unsupported snapshot format")
        registers = list(struct.unpack(">8i", payload[8:40]))
        memory = bytearray(payload[40 : 40 + DATA_BYTES])
        stack_start = 40 + DATA_BYTES
        stack = list(payload[stack_start : stack_start + stack_size])
        return registers, memory, pc, stack


class VirtualMachine:
    def __init__(self, rom: Sequence[int]) -> None:
        if len(rom) > ROM_WORDS:
            raise ValueError("ROM exceeds 256 words")
        self.rom = list(rom)
        self.registers = [0] * REGISTER_COUNT
        self.memory = bytearray(DATA_BYTES)
        self.stack: list[int] = []
        self.pc = 0
        self.clock = 0
        self.halted = False
        self.latent = b""
        self.output: list[int] = []
        self.journal: list[JournalEntry] = []
        self._chain = bytes(32)

    @staticmethod
    def address_to_xyz(address: int) -> tuple[int, int, int]:
        if not 0 <= address < GRID_SIDE**3:
            raise ValueError("3D address out of range")
        z, remainder = divmod(address, GRID_SIDE * GRID_SIDE)
        y, x = divmod(remainder, GRID_SIDE)
        return x, y, z

    def step(self) -> bool:
        if self.halted or not 0 <= self.pc < len(self.rom):
            self.halted = True
            return False
        instruction = Instruction.unpack(self.rom[self.pc])
        self.pc += 1
        op, a, b, c = instruction.opcode, instruction.op1, instruction.op2, instruction.flags

        if op == Opcode.NOP:
            pass
        elif op == Opcode.LOADI:
            self.registers[a] = b
        elif op == Opcode.LOAD:
            self.registers[a] = self.memory[b]
        elif op == Opcode.STORE:
            self.memory[b] = self.registers[a] & 0xFF
        elif op in {Opcode.ADD, Opcode.SUB, Opcode.MUL, Opcode.DIV}:
            left, right = self.registers[b], self.registers[c]
            if op == Opcode.ADD:
                self.registers[a] = left + right
            elif op == Opcode.SUB:
                self.registers[a] = left - right
            elif op == Opcode.MUL:
                self.registers[a] = left * right
            else:
                self.registers[a] = 0 if right == 0 else left // right
        elif op == Opcode.JMP:
            self.pc = b
        elif op == Opcode.JIF:
            if self.registers[a] != 0:
                self.pc = b
        elif op == Opcode.CALL:
            if len(self.stack) >= STACK_DEPTH:
                raise RuntimeError("call stack overflow")
            self.stack.append(self.pc)
            self.pc = b
        elif op == Opcode.RET:
            if not self.stack:
                raise RuntimeError("call stack underflow")
            self.pc = self.stack.pop()
        elif op == Opcode.ENCODE:
            self.latent = StateCodec.encode(self.registers, bytes(self.memory), self.pc, self.stack)
        elif op == Opcode.DECODE:
            if not self.latent:
                raise RuntimeError("DECODE executed before ENCODE")
            self.registers, self.memory, self.pc, self.stack = StateCodec.decode(self.latent)
        elif op == Opcode.PRINT:
            value = self.registers[a]
            self.output.append(value)
            print(f"[VM] R{a} = {value}")
        elif op == Opcode.HALT:
            self.halted = True
        else:
            raise RuntimeError(f"unsupported opcode: {op}")

        self.clock += 1
        self._append_journal()
        return not self.halted

    def run(self, max_steps: int = 1000) -> int:
        steps = 0
        while steps < max_steps and self.step():
            steps += 1
        return steps + (1 if self.halted and self.clock > steps else 0)

    def _append_journal(self) -> None:
        state = StateCodec.encode(self.registers, bytes(self.memory), self.pc, self.stack)
        state_digest = hashlib.sha256(state).hexdigest()
        latent_digest = hashlib.sha256(self.latent).hexdigest() if self.latent else "0" * 64
        record = struct.pack(">QH", self.clock, self.pc) + bytes.fromhex(state_digest) + bytes.fromhex(latent_digest)
        self._chain = hashlib.sha256(self._chain + record).digest()
        self.journal.append(JournalEntry(self.clock, self.pc, state_digest, latent_digest, self._chain.hex()))


DEMO_PROGRAM = """
    LOADI R0, 10
    LOADI R1, 20
    ADD   R2, R0, R1
    STORE R2, 0x2A
    ENCODE
    LOADI R2, 99
    DECODE
    PRINT R2
    HALT
"""


def main() -> None:
    rom = Assembler().assemble(DEMO_PROGRAM)
    vm = VirtualMachine(rom)
    print("=== Jarvis X 3D Bytecode ROM ===")
    print(f"ROM words: {len(rom)}")
    for address, word in enumerate(rom):
        print(f"{address:03d} {vm.address_to_xyz(address)} 0x{word:08X}")
    vm.run()
    assert vm.output == [30]
    assert vm.memory[0x2A] == 30
    print(f"Latent bytes: {len(vm.latent)}")
    print(f"Journal head: {vm.journal[-1].chain_digest}")
    print("Permeation complete: state round-trip verified.")


if __name__ == "__main__":
    main()
