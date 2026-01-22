class Sandbox:
    def __init__(self, max_cycles=10000):
        self.max_cycles = max_cycles

    def enforce(self, cycles):
        if cycles > self.max_cycles:
            raise RuntimeError("Sandbox limit exceeded")
