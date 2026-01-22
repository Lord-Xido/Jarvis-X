class Memory:
    def __init__(self, size=4096):
        self.data = bytearray(size)

    def load(self, address, size):
        return self.data[address:address+size]

    def store(self, address, values):
        self.data[address:address+len(values)] = values
