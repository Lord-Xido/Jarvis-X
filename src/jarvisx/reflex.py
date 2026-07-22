class ReflexEngine:
    def __init__(self):
        self.last_delta = 0

    def stabilize(self, regs):
        psi = regs["Ψ"]
        phi = regs["Φ"]
        self.last_delta = int(0.1 * (psi - phi))
        return self.last_delta
