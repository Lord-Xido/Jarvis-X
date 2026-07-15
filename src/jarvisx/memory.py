class Memory:
    def __init__(self, size=4096):
        if size <= 0:
            raise ValueError("Memory size must be positive")
        self.data = bytearray(size)

    def __len__(self):
        return len(self.data)

    def _check_range(self, address, size):
        address = int(address)
        size = int(size)
        if address < 0 or size < 0 or address + size > len(self.data):
            raise RuntimeError(
                "Memory access out of bounds: address={}, size={}, capacity={}".format(
                    address, size, len(self.data)
                )
            )
        return address, size

    def load(self, address, size):
        address, size = self._check_range(address, size)
        return bytes(self.data[address : address + size])

    def store(self, address, values):
        payload = bytes(values)
        address, size = self._check_range(address, len(payload))
        self.data[address : address + size] = payload

    def load_int(self, address, width=8, signed=True):
        return int.from_bytes(self.load(address, width), "little", signed=signed)

    def store_int(self, address, value, width=8):
        bits = width * 8
        encoded = int(value) & ((1 << bits) - 1)
        self.store(address, encoded.to_bytes(width, "little", signed=False))

    def snapshot(self):
        return bytes(self.data)

    def restore(self, snapshot):
        payload = bytes(snapshot)
        if len(payload) != len(self.data):
            raise ValueError("Memory snapshot size does not match memory capacity")
        self.data[:] = payload
