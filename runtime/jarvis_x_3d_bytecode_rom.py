#!/usr/bin/env python3
"""Jarvis X: reversible 3D bytecode ROM and virtual machine.

The 32-bit word format is [opcode][a][b][c]. ROM addresses are mapped into an
8x8x8 coordinate lattice. ENCODE creates a compressed, integrity-protected VM
snapshot; DECODE restores data state without rewinding instruction flow.
"""
from __future__ import annotations

import hashlib
import re
import struct
import zlib
from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable, Sequence

ROM_WORDS, DATA_BYTES, REG_COUNT, STACK_DEPTH, GRID_SIDE = 256, 256, 8, 16, 8
MAGIC, VERSION = b"JX3D", 1


class Op(IntEnum):
    NOP=0x01; LOADI=0x02; LOAD=0x03; STORE=0x04
    ADD=0x05; SUB=0x06; MUL=0x07; DIV=0x08
    JMP=0x09; JIF=0x0A; CALL=0x0B; RET=0x0C
    HALT=0x0D; ENCODE=0x0E; DECODE=0x0F; PRINT=0x10


@dataclass(frozen=True)
class Instruction:
    op: Op
    a: int = 0
    b: int = 0
    c: int = 0

    def pack(self) -> int:
        if any(not 0 <= x <= 255 for x in (self.a, self.b, self.c)):
            raise ValueError("instruction field outside one-byte range")
        return (int(self.op) << 24) | (self.a << 16) | (self.b << 8) | self.c

    @classmethod
    def unpack(cls, word: int) -> "Instruction":
        return cls(Op((word >> 24) & 255), (word >> 16) & 255,
                   (word >> 8) & 255, word & 255)


@dataclass(frozen=True)
class JournalEntry:
    clock: int
    pc: int
    state_hash: str
    latent_hash: str
    chain_hash: str


class Assembler:
    register_pattern = re.compile(r"^R([0-7])$", re.I)

    def assemble(self, source: str) -> list[int]:
        rows = list(self._rows(source))
        labels: dict[str, int] = {}
        pc = 0
        for label, tokens in rows:
            if label:
                if label in labels:
                    raise ValueError(f"duplicate label: {label}")
                labels[label] = pc
            pc += bool(tokens)
        if pc > ROM_WORDS:
            raise ValueError("program exceeds ROM capacity")
        return [self._instruction(tokens, labels).pack()
                for _, tokens in rows if tokens]

    def _rows(self, source: str) -> Iterable[tuple[str | None, list[str]]]:
        for number, raw in enumerate(source.splitlines(), 1):
            line = raw.split(";", 1)[0].strip()
            if not line:
                continue
            label = None
            if ":" in line:
                label, line = (part.strip() for part in line.split(":", 1))
                if not re.match(r"^[A-Za-z_]\w*$", label):
                    raise ValueError(f"line {number}: invalid label")
            yield label, [x for x in re.split(r"[\s,]+", line) if x]

    def _instruction(self, tokens: Sequence[str], labels: dict[str, int]) -> Instruction:
        try:
            op = Op[tokens[0].upper()]
        except KeyError as exc:
            raise ValueError(f"unknown opcode: {tokens[0]}") from exc
        args = list(tokens[1:])
        zero = {Op.NOP, Op.RET, Op.HALT, Op.ENCODE, Op.DECODE}
        if op in zero:
            self._arity(op, args, 0); return Instruction(op)
        if op == Op.LOADI:
            self._arity(op, args, 2); return Instruction(op, self._reg(args[0]), self._byte(args[1]))
        if op in {Op.LOAD, Op.STORE}:
            self._arity(op, args, 2); return Instruction(op, self._reg(args[0]), self._byte(args[1]))
        if op in {Op.ADD, Op.SUB, Op.MUL, Op.DIV}:
            self._arity(op, args, 3)
            return Instruction(op, self._reg(args[0]), self._reg(args[1]), self._reg(args[2]))
        if op in {Op.JMP, Op.CALL}:
            self._arity(op, args, 1); return Instruction(op, 0, self._target(args[0], labels))
        if op == Op.JIF:
            self._arity(op, args, 2); return Instruction(op, self._reg(args[0]), self._target(args[1], labels))
        if op == Op.PRINT:
            self._arity(op, args, 1); return Instruction(op, self._reg(args[0]))
        raise AssertionError(op)

    @staticmethod
    def _arity(op: Op, args: Sequence[str], n: int) -> None:
        if len(args) != n:
            raise ValueError(f"{op.name} expects {n} operands, got {len(args)}")

    def _reg(self, token: str) -> int:
        match = self.register_pattern.match(token)
        if not match:
            raise ValueError(f"invalid register: {token}")
        return int(match.group(1))

    @staticmethod
    def _integer(token: str) -> int:
        try:
            return int(token, 0)
        except ValueError:
            return -1

    def _byte(self, token: str) -> int:
        value = self._integer(token)
        if not 0 <= value <= 255:
            raise ValueError(f"byte value out of range: {token}")
        return value

    def _target(self, token: str, labels: dict[str, int]) -> int:
        value = labels[token] if token in labels else self._integer(token)
        if not 0 <= value < ROM_WORDS:
            raise ValueError(f"target out of range: {token}")
        return value


class StateCodec:
    """Lossless state codec. SHA-256 authenticates; zlib compresses."""

    @staticmethod
    def encode(regs: Sequence[int], memory: bytes, pc: int, stack: Sequence[int]) -> bytes:
        if len(regs) != REG_COUNT or len(memory) != DATA_BYTES or len(stack) > STACK_DEPTH:
            raise ValueError("invalid state dimensions")
        payload = (MAGIC + struct.pack(">BBH", VERSION, len(stack), pc)
                   + struct.pack(">8i", *regs) + memory + bytes(stack))
        return zlib.compress(payload + hashlib.sha256(payload).digest(), 9)

    @staticmethod
    def decode(latent: bytes) -> tuple[list[int], bytearray, int, list[int]]:
        packed = zlib.decompress(latent)
        payload, digest = packed[:-32], packed[-32:]
        if len(payload) < 296 or payload[:4] != MAGIC:
            raise ValueError("invalid or truncated snapshot")
        if hashlib.sha256(payload).digest() != digest:
            raise ValueError("snapshot integrity failure")
        version, stack_size, pc = struct.unpack(">BBH", payload[4:8])
        if version != VERSION or stack_size > STACK_DEPTH:
            raise ValueError("unsupported snapshot")
        regs = list(struct.unpack(">8i", payload[8:40]))
        memory = bytearray(payload[40:296])
        stack = list(payload[296:296 + stack_size])
        return regs, memory, pc, stack


class VM:
    def __init__(self, rom: Sequence[int]):
        if len(rom) > ROM_WORDS:
            raise ValueError("ROM overflow")
        self.rom = list(rom)
        self.regs = [0] * REG_COUNT
        self.memory = bytearray(DATA_BYTES)
        self.stack: list[int] = []
        self.pc = self.clock = 0
        self.halted = False
        self.latent = b""
        self.output: list[int] = []
        self.journal: list[JournalEntry] = []
        self._chain = bytes(32)

    @staticmethod
    def xyz(address: int) -> tuple[int, int, int]:
        if not 0 <= address < GRID_SIDE ** 3:
            raise ValueError("spatial address out of range")
        z, rem = divmod(address, GRID_SIDE * GRID_SIDE)
        y, x = divmod(rem, GRID_SIDE)
        return x, y, z

    def step(self) -> bool:
        if self.halted or not 0 <= self.pc < len(self.rom):
            self.halted = True
            return False
        ins = Instruction.unpack(self.rom[self.pc]); self.pc += 1
        op, a, b, c = ins.op, ins.a, ins.b, ins.c
        if op == Op.NOP: pass
        elif op == Op.LOADI: self.regs[a] = b
        elif op == Op.LOAD: self.regs[a] = self.memory[b]
        elif op == Op.STORE: self.memory[b] = self.regs[a] & 255
        elif op == Op.ADD: self.regs[a] = self.regs[b] + self.regs[c]
        elif op == Op.SUB: self.regs[a] = self.regs[b] - self.regs[c]
        elif op == Op.MUL: self.regs[a] = self.regs[b] * self.regs[c]
        elif op == Op.DIV: self.regs[a] = 0 if self.regs[c] == 0 else self.regs[b] // self.regs[c]
        elif op == Op.JMP: self.pc = b
        elif op == Op.JIF:
            if self.regs[a] != 0: self.pc = b
        elif op == Op.CALL:
            if len(self.stack) >= STACK_DEPTH: raise RuntimeError("stack overflow")
            self.stack.append(self.pc); self.pc = b
        elif op == Op.RET:
            if not self.stack: raise RuntimeError("stack underflow")
            self.pc = self.stack.pop()
        elif op == Op.ENCODE:
            self.latent = StateCodec.encode(self.regs, bytes(self.memory), self.pc, self.stack)
        elif op == Op.DECODE:
            if not self.latent: raise RuntimeError("DECODE before ENCODE")
            restored_regs, restored_memory, _saved_pc, restored_stack = StateCodec.decode(self.latent)
            self.regs, self.memory, self.stack = restored_regs, restored_memory, restored_stack
        elif op == Op.PRINT:
            self.output.append(self.regs[a]); print(f"[VM] R{a} = {self.regs[a]}")
        elif op == Op.HALT: self.halted = True
        self.clock += 1
        self._journal()
        return not self.halted

    def run(self, max_steps: int = 1000) -> int:
        steps = 0
        while steps < max_steps:
            steps += 1
            if not self.step(): break
        return steps

    def _journal(self) -> None:
        state = StateCodec.encode(self.regs, bytes(self.memory), self.pc, self.stack)
        state_hash = hashlib.sha256(state).hexdigest()
        latent_hash = hashlib.sha256(self.latent).hexdigest() if self.latent else "0" * 64
        record = struct.pack(">QH", self.clock, self.pc) + bytes.fromhex(state_hash + latent_hash)
        self._chain = hashlib.sha256(self._chain + record).digest()
        self.journal.append(JournalEntry(self.clock, self.pc, state_hash, latent_hash, self._chain.hex()))


DEMO = """
    LOADI R0, 10
    LOADI R1, 20
    ADD R2, R0, R1
    STORE R2, 0x2A
    ENCODE
    LOADI R2, 99
    DECODE
    PRINT R2
    HALT
"""


def main() -> None:
    rom = Assembler().assemble(DEMO)
    vm = VM(rom)
    print("=== Jarvis X 3D Bytecode ROM ===")
    for address, word in enumerate(rom):
        print(f"{address:03d} {vm.xyz(address)} 0x{word:08X}")
    steps = vm.run()
    assert vm.output == [30] and vm.memory[0x2A] == 30 and vm.halted
    print(f"Verified {steps} cycles; latent={len(vm.latent)} bytes")
    print(f"Journal head: {vm.journal[-1].chain_hash}")
    print("Permeation complete: reversible state round-trip verified.")


if __name__ == "__main__":
    main()
