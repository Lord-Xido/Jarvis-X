class LambdaShield:
    def __init__(self):
        self.blocked = set()

    def block(self, opcode):
        self.blocked.add(opcode)

    def allow(self, instr):
        return instr.opcode not in self.blocked
