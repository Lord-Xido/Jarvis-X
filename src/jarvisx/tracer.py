class Tracer:
    def __init__(self, max_entries=10000):
        self.max_entries = int(max_entries)
        self.log = []

    def clear(self):
        self.log.clear()

    def record(self, instr, regs, ann=None):
        entry = {"opcode": instr.opcode, "registers": regs.copy()}
        if ann is not None:
            entry["ann30d"] = ann
        self.log.append(entry)
        if len(self.log) > self.max_entries:
            del self.log[: len(self.log) - self.max_entries]
