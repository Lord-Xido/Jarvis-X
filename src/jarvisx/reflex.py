class ReflexEngine:
    def stabilize(self, regs):
        psi = regs["Ψ"]
        phi = regs["Φ"]
        delta = int(0.1 * (psi - phi))
        regs["Φ"] += delta
