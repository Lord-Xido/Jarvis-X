REFLEX_ENABLE_FLAG = 0x01


class ReflexEngine:
    def stabilize(self, regs):
        if not (regs["FLAGS"] & REFLEX_ENABLE_FLAG):
            return
        psi = regs["Ψ"]
        phi = regs["Φ"]
        delta = int(0.1 * (psi - phi))
        regs["Φ"] += delta
