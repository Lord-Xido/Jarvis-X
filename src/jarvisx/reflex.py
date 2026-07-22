class ReflexEngine:
    """Evaluate or apply bounded Psi/Phi stabilization.

    Reflex evaluation is separated from mutation so deterministic instruction
    results are not changed implicitly. Active stabilization remains available
    through the explicit ``apply=True`` mode.
    """

    def __init__(self):
        self.last_delta = 0

    def stabilize(self, regs, apply=False):
        psi = regs["Ψ"]
        phi = regs["Φ"]
        delta = int(0.1 * (psi - phi))
        self.last_delta = delta

        if apply:
            regs["Φ"] += delta

        return delta
