class Tracer:
    def __init__(self):
        self.log = []

    def record(self, instr, regs):
        self.log.append((int(instr.opcode), regs.copy()))

    def checkpoint(self):
        return len(self.log)

    def restore(self, checkpoint):
        del self.log[int(checkpoint) :]
