class ReflexEngine:
    """Evaluate or apply bounded Psi/Phi stabilization.

    Reflex evaluation is separated from mutation so a deterministic VM can
    observe the correction signal without silently changing instruction
    operands. Active stabilization remains available as an explicit mode.
    """

    def __init__(self):
        self.last_delta = 0

    def stabilize(self, regs, apply=True):
        psi = regs["Ψ"]
        phi = regs["Φ"]
        delta = int(0.1 * (psi - phi))
        self.last_delta = delta

        if apply:
            regs["Φ"] += delta

        return delta
