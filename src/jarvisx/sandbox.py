class Sandbox:
    def __init__(self, max_cycles=10000, max_program_words=10000, max_active_cells=100000):
        if min(max_cycles, max_program_words, max_active_cells) <= 0:
            raise ValueError("sandbox limits must be positive")
        self.max_cycles = int(max_cycles)
        self.max_program_words = int(max_program_words)
        self.max_active_cells = int(max_active_cells)

    def validate_program(self, program):
        if not program:
            raise ValueError("program must not be empty")
        if len(program) > self.max_program_words:
            raise RuntimeError("program word quota exceeded")

    def enforce(self, cycles, active_cells=0):
        if cycles > self.max_cycles:
            raise RuntimeError("sandbox cycle limit exceeded")
        if active_cells > self.max_active_cells:
            raise RuntimeError("sandbox active-cell limit exceeded")
