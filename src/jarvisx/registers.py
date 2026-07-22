REG_NAMES = [
    "Ξ", "Ψ", "Φ", "Λ", "Ω", "Θ", "𝒮", "Π",
    "A", "B", "C", "D", "IP", "SP", "FLAGS", "TMP",
]


class Registers:
    def __init__(self):
        self._regs = {name: 0 for name in REG_NAMES}

    def __getitem__(self, key):
        return self._regs[key]

    def __setitem__(self, key, value):
        if key not in self._regs:
            raise KeyError(f"unknown register: {key}")
        self._regs[key] = int(value)

    def reset(self):
        for key in self._regs:
            self._regs[key] = 0

    def snapshot(self):
        return dict(self._regs)
