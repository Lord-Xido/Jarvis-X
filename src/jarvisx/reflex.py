class ReflexEngine:
    """Optional bounded reflex controller.

    Reflex adaptation is disabled by default so ordinary VM instructions retain
    deterministic arithmetic semantics. Workloads that need the feedback loop
    may enable it explicitly.
    """

    def __init__(self, enabled=False, gain=0.1):
        if not 0.0 <= gain <= 1.0:
            raise ValueError("reflex gain must be in [0, 1]")
        self.enabled = bool(enabled)
        self.gain = float(gain)

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def stabilize(self, regs):
        if not self.enabled:
            return 0
        psi = regs["Ψ"]
        phi = regs["Φ"]
        delta = int(self.gain * (psi - phi))
        regs["Φ"] += delta
        return delta
