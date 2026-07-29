class ReflexEngine:
    """Optional bounded proportional correction for the dedicated state pair.

    Reflex correction is disabled by default so ordinary VM instructions retain
    exact register semantics. Systems that intentionally use the Ψ/Φ feedback
    pair must opt in explicitly.
    """

    def __init__(self, gain=0.1, enabled=False):
        if gain < 0.0:
            raise ValueError("gain must be non-negative")
        self.gain = gain
        self.enabled = enabled

    def stabilize(self, regs):
        if not self.enabled:
            return 0

        psi = regs["Ψ"]
        phi = regs["Φ"]
        correction = int(self.gain * (psi - phi))
        regs["Φ"] += correction
        return correction
