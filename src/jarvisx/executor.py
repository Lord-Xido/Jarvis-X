REG_NAMES = [
    "Ξ","Ψ","Φ","Λ","Ω","Θ","𝒮","Π",
    "A","B","C","D","IP","SP","FLAGS","TMP"
]

class Executor:
    def __init__(self, registers):
        self.regs = registers

    def execute(self, instr):
        if instr.opcode == 0x01:  # SET
            self.regs[REG_NAMES[instr.dst]] = instr.imm

        elif instr.opcode == 0x03:  # ADD
            a = self.regs[REG_NAMES[instr.src1]]
            b = self.regs[REG_NAMES[instr.src2]]
            self.regs[REG_NAMES[instr.dst]] = a + b

        elif instr.opcode == 0x04:  # SUB
            a = self.regs[REG_NAMES[instr.src1]]
            b = self.regs[REG_NAMES[instr.src2]]
            self.regs[REG_NAMES[instr.dst]] = a - b

        elif instr.opcode == 0x0A:  # HALT
            return False

        return True
