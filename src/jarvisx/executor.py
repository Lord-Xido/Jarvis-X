from dataclasses import dataclass
from typing import Optional


REG_NAMES = [
    "Ξ", "Ψ", "Φ", "Λ", "Ω", "Θ", "𝒮", "Π",
    "A", "B", "C", "D", "IP", "SP", "FLAGS", "TMP",
]

FLAG_ZERO = 0x01
FLAG_NEGATIVE = 0x02


@dataclass(frozen=True)
class ExecutionResult:
    continue_running: bool = True
    next_ip: Optional[int] = None


class Executor:
    def __init__(self, registers, memory):
        self.regs = registers
        self.memory = memory

    def _reg(self, index):
        try:
            return REG_NAMES[int(index)]
        except (IndexError, ValueError) as exc:
            raise RuntimeError("Invalid register index: {}".format(index)) from exc

    def _set_flags(self, value):
        flags = 0
        if int(value) == 0:
            flags |= FLAG_ZERO
        if int(value) < 0:
            flags |= FLAG_NEGATIVE
        self.regs["FLAGS"] = flags

    def _binary(self, instr, operation):
        a = self.regs[self._reg(instr.src1)]
        b = self.regs[self._reg(instr.src2)]
        result = int(operation(a, b))
        self.regs[self._reg(instr.dst)] = result
        self._set_flags(result)

    def execute(self, instr, current_ip=0):
        opcode = instr.opcode

        if opcode == 0x01:  # SET
            self.regs[self._reg(instr.dst)] = instr.imm
            self._set_flags(instr.imm)
        elif opcode == 0x02:  # MOV
            value = self.regs[self._reg(instr.src1)]
            self.regs[self._reg(instr.dst)] = value
            self._set_flags(value)
        elif opcode == 0x03:  # ADD
            self._binary(instr, lambda a, b: a + b)
        elif opcode == 0x04:  # SUB
            self._binary(instr, lambda a, b: a - b)
        elif opcode == 0x05:  # LOAD
            value = self.memory.load_int(instr.imm)
            self.regs[self._reg(instr.dst)] = value
            self._set_flags(value)
        elif opcode == 0x06:  # STORE
            self.memory.store_int(instr.imm, self.regs[self._reg(instr.src1)])
        elif opcode == 0x07:  # CMP
            left = self.regs[self._reg(instr.src1)]
            right = self.regs[self._reg(instr.src2)]
            self._set_flags(left - right)
        elif opcode == 0x08:  # JMP
            return ExecutionResult(next_ip=instr.imm)
        elif opcode == 0x09:  # JZ
            if self.regs["FLAGS"] & FLAG_ZERO:
                return ExecutionResult(next_ip=instr.imm)
        elif opcode == 0x0A:  # HALT
            return ExecutionResult(continue_running=False, next_ip=current_ip + 1)
        elif opcode == 0x0B:  # JNZ
            if not self.regs["FLAGS"] & FLAG_ZERO:
                return ExecutionResult(next_ip=instr.imm)
        elif opcode == 0x0C:  # MUL
            self._binary(instr, lambda a, b: a * b)
        elif opcode == 0x0D:  # XOR
            self._binary(instr, lambda a, b: a ^ b)
        elif opcode == 0x0E:  # AND
            self._binary(instr, lambda a, b: a & b)
        elif opcode == 0x0F:  # OR
            self._binary(instr, lambda a, b: a | b)
        else:
            raise RuntimeError("Unsupported opcode: 0x{:02X}".format(opcode))

        return ExecutionResult(next_ip=current_ip + 1)
