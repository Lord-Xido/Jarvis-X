class LambdaShield:
    def __init__(self):
        self.blocked = set()
        self.blocked_actions = set()

    def block(self, opcode):
        self.blocked.add(opcode)

    def block_action(self, action):
        self.blocked_actions.add(str(action))

    def allow(self, instr):
        return instr.opcode not in self.blocked

    def allow_action(self, action):
        return str(action) not in self.blocked_actions
