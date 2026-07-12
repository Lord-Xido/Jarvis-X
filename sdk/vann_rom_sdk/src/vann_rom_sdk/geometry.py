from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True, order=True)
class Address3D:
    """Virtual 3D address used by the sparse ROM and program counter."""

    x: int
    y: int
    z: int
    level: int = 0
    space: int = 0

    def __post_init__(self) -> None:
        if min(self.x, self.y, self.z, self.level, self.space) < 0:
            raise ValueError("3D addresses must be non-negative")

    def neighbor(self, dx: int = 0, dy: int = 0, dz: int = 0) -> "Address3D":
        nx, ny, nz = self.x + dx, self.y + dy, self.z + dz
        if min(nx, ny, nz) < 0:
            raise ValueError("neighbor would leave the non-negative address space")
        return Address3D(nx, ny, nz, self.level, self.space)

    def morton(self) -> int:
        return morton3d_encode(self.x, self.y, self.z)


def _split_by_3(value: int) -> int:
    value &= 0x1FFFFF
    value = (value | value << 32) & 0x1F00000000FFFF
    value = (value | value << 16) & 0x1F0000FF0000FF
    value = (value | value << 8) & 0x100F00F00F00F00F
    value = (value | value << 4) & 0x10C30C30C30C30C3
    value = (value | value << 2) & 0x1249249249249249
    return value


def _compact_by_3(value: int) -> int:
    value &= 0x1249249249249249
    value = (value ^ (value >> 2)) & 0x10C30C30C30C30C3
    value = (value ^ (value >> 4)) & 0x100F00F00F00F00F
    value = (value ^ (value >> 8)) & 0x1F0000FF0000FF
    value = (value ^ (value >> 16)) & 0x1F00000000FFFF
    value = (value ^ (value >> 32)) & 0x1FFFFF
    return value


def morton3d_encode(x: int, y: int, z: int) -> int:
    """Interleave 21 bits from each coordinate into a 63-bit Morton key."""

    if min(x, y, z) < 0 or max(x, y, z) >= 2**21:
        raise ValueError("Morton coordinates must be in [0, 2**21)")
    return _split_by_3(x) | (_split_by_3(y) << 1) | (_split_by_3(z) << 2)


def morton3d_decode(code: int) -> tuple[int, int, int]:
    if code < 0 or code >= 2**63:
        raise ValueError("Morton code must be a non-negative 63-bit integer")
    return _compact_by_3(code), _compact_by_3(code >> 1), _compact_by_3(code >> 2)


class GeometricProgramLayout:
    """Maps sequential instruction indices onto a finite XYZ execution lattice."""

    def __init__(self, width: int = 16, height: int = 16) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        self.width = width
        self.height = height
        self.plane = width * height

    def address_for_index(self, index: int) -> Address3D:
        if index < 0:
            raise ValueError("index must be non-negative")
        z, rem = divmod(index, self.plane)
        y, x = divmod(rem, self.width)
        return Address3D(x, y, z)

    def index_for_address(self, address: Address3D) -> int:
        if address.x >= self.width or address.y >= self.height:
            raise ValueError("address lies outside this program layout")
        return address.z * self.plane + address.y * self.width + address.x

    def next_address(self, address: Address3D) -> Address3D:
        return self.address_for_index(self.index_for_address(address) + 1)
