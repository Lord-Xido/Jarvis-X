class Tracer:
    def __init__(self):
        self.log = []

    def record(self, instr, regs):
        self.log.append((instr.opcode, regs.copy()))

    def record_event(self, event, state):
        self.log.append((str(event), dict(state)))
