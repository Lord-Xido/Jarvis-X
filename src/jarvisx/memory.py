from __future__ import annotations

from collections.abc import Iterable


class Memory:
    def __init__(self, size: int = 4096) -> None:
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise ValueError("memory size must be a positive integer")
        self.data = bytearray(size)

    def _checked_end(self, address: int, size: int) -> int:
        if not isinstance(address, int) or isinstance(address, bool):
            raise TypeError("memory address must be an integer")
        if not isinstance(size, int) or isinstance(size, bool):
            raise TypeError("memory size must be an integer")
        if address < 0 or size < 0 or address + size > len(self.data):
            raise IndexError("memory access is outside the allocated range")
        return address + size

    def load(self, address: int, size: int) -> bytearray:
        end = self._checked_end(address, size)
        return self.data[address:end]

    def store(self, address: int, values: Iterable[int]) -> None:
        payload = bytes(values)
        end = self._checked_end(address, len(payload))
        self.data[address:end] = payload

    def snapshot(self) -> bytes:
        return bytes(self.data)

    def restore(self, snapshot: bytes | bytearray) -> None:
        if len(snapshot) != len(self.data):
            raise ValueError("memory snapshot size mismatch")
        self.data[:] = snapshot

    def reset(self) -> None:
        self.data[:] = b"\x00" * len(self.data)
