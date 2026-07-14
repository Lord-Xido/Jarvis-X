"""Opcode policy gate. Semantic policy belongs above the bytecode layer."""

from .opcodes import OPCODE_NAMES


class LambdaShield:
    def __init__(self, allowed=None):
        self.allowed = set(OPCODE_NAMES if allowed is None else allowed)
        self.blocked = set()

    def block(self, opcode):
        self.blocked.add(int(opcode))

    def allow(self, instr):
        return instr.opcode in self.allowed and instr.opcode not in self.blocked
