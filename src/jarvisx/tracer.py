class Tracer:
    def __init__(self):
        self.log = []

    def record(self, instr, regs):
        self.log.append((instr.opcode, regs.copy()))
