class Memory:
    def __init__(self, size=4096):
        if size <= 0:
            raise ValueError("memory size must be positive")
        self.data = bytearray(size)

    def _bounds(self, address, size):
        if address < 0 or size < 0 or address + size > len(self.data):
            raise IndexError("memory access outside allocated range")

    def load(self, address, size):
        self._bounds(int(address), int(size))
        return bytes(self.data[address:address + size])

    def store(self, address, values):
        payload = bytes(values)
        self._bounds(int(address), len(payload))
        self.data[address:address + len(payload)] = payload
