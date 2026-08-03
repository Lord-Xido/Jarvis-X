from __future__ import annotations

from .registers import REGISTER_NAMES

REG_NAMES: list[str] = list(REGISTER_NAMES)
SUPPORTED_OPCODES = frozenset({0x01, 0x03, 0x04, 0x0A})


class Executor:
    def __init__(self, registers):
        self.regs = registers

    @staticmethod
    def _register_name(index: int) -> str:
        if not 0 <= index < len(REG_NAMES):
            raise RuntimeError(f"register index {index} is outside the register file")
        return REG_NAMES[index]

    def execute(self, instr):
        if instr.opcode == 0x01:  # SET
            self.regs[self._register_name(instr.dst)] = instr.imm

        elif instr.opcode == 0x03:  # ADD
            a = self.regs[self._register_name(instr.src1)]
            b = self.regs[self._register_name(instr.src2)]
            self.regs[self._register_name(instr.dst)] = a + b

        elif instr.opcode == 0x04:  # SUB
            a = self.regs[self._register_name(instr.src1)]
            b = self.regs[self._register_name(instr.src2)]
            self.regs[self._register_name(instr.dst)] = a - b

        elif instr.opcode == 0x0A:  # HALT
            return False

        else:
            raise RuntimeError(f"Unsupported opcode 0x{instr.opcode:02X}")

        return True
