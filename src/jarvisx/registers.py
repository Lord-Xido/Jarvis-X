class Registers:
    def __init__(self):
        self._regs = {
            "Ξ": 0, "Ψ": 0, "Φ": 0, "Λ": 0, "Ω": 0,
            "Θ": 0, "𝒮": 0, "Π": 0,
            "A": 0, "B": 0, "C": 0, "D": 0,
            "IP": 0, "SP": 0, "FLAGS": 0, "TMP": 0,
        }

    def __getitem__(self, key):
        return self._regs[key]

    def __setitem__(self, key, value):
        if key not in self._regs:
            raise KeyError("Unknown register: {}".format(key))
        self._regs[key] = int(value)

    def snapshot(self):
        return dict(self._regs)

    def restore(self, snapshot):
        if set(snapshot) != set(self._regs):
            raise ValueError("Register snapshot shape does not match register bank")
        self._regs = {name: int(value) for name, value in snapshot.items()}
