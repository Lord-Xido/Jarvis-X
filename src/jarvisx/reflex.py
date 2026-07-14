class ReflexEngine:
    def __init__(self, enabled=False):
        self.enabled = bool(enabled)

    def stabilize(self, regs):
        if not self.enabled:
            return
        psi = regs["Ψ"]
        phi = regs["Φ"]
        delta = int(0.1 * (psi - phi))
        regs["Φ"] += delta
