"""Deterministic Bit Matrix 3D codec-runtime.

The runtime provides a lossless, renderer-independent transport from bytes/text
into a bounded 3D Boolean lattice and back again.

Canonical layout::

    [0..31]             unsigned 32-bit big-endian payload byte length
    [32..32+n*8-1]      payload bits, big-endian within each byte
    [remaining cells]   zero padding

The lattice is traversed X-fastest, then Y, then Z::

    index = z * (x_size * y_size) + y * x_size + x

The implementation deliberately separates the authoritative information state
from visualization.  A renderer may rotate, scale, colour or animate active
voxels without modifying the encoded lattice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

_HEADER_BITS = 32
_MAX_PAYLOAD_BYTES = (1 << _HEADER_BITS) - 1


@dataclass(frozen=True, slots=True)
class Dimensions3D:
    """Positive logical dimensions of a Bit Matrix lattice."""

    x: int
    y: int
    z: int

    def __post_init__(self) -> None:
        for name, value in (("x", self.x), ("y", self.y), ("z", self.z)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")

    @property
    def cells(self) -> int:
        return self.x * self.y * self.z

    @property
    def payload_capacity_bytes(self) -> int:
        if self.cells < _HEADER_BITS:
            return 0
        return (self.cells - _HEADER_BITS) // 8


@dataclass(frozen=True, slots=True)
class Coordinate3D:
    """Integer coordinate in a bounded Bit Matrix lattice."""

    x: int
    y: int
    z: int


@dataclass(frozen=True, slots=True)
class BitMatrix3D:
    """Immutable canonical 3D Boolean lattice.

    ``bits`` is stored linearly using the canonical X -> Y -> Z traversal.
    The immutable tuple keeps the codec state deterministic and safe to share
    with renderers or downstream runtime stages.
    """

    dimensions: Dimensions3D
    bits: tuple[bool, ...]

    def __post_init__(self) -> None:
        if len(self.bits) != self.dimensions.cells:
            raise ValueError(
                "bit count does not match dimensions: "
                f"{len(self.bits)} != {self.dimensions.cells}"
            )
        if any(type(bit) is not bool for bit in self.bits):
            raise TypeError("bits must contain bool values only")

    @property
    def active_count(self) -> int:
        return sum(self.bits)

    def bit_at(self, coordinate: Coordinate3D) -> bool:
        return self.bits[coordinate_to_index(coordinate, self.dimensions)]

    def active_coordinates(self) -> Iterator[Coordinate3D]:
        for index, bit in enumerate(self.bits):
            if bit:
                yield index_to_coordinate(index, self.dimensions)


@dataclass(frozen=True, slots=True)
class DecodeResult:
    """Decoded payload plus structural metadata."""

    payload: bytes
    declared_length: int
    payload_capacity_bytes: int
    padding_bits: int


def _validate_index(index: int, dimensions: Dimensions3D) -> None:
    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError("index must be an integer")
    if not 0 <= index < dimensions.cells:
        raise IndexError("index outside lattice")


def coordinate_to_index(coordinate: Coordinate3D, dimensions: Dimensions3D) -> int:
    """Map ``(x, y, z)`` to the canonical X-fastest linear address."""

    if not isinstance(coordinate, Coordinate3D):
        raise TypeError("coordinate must be Coordinate3D")
    for name, value, bound in (
        ("x", coordinate.x, dimensions.x),
        ("y", coordinate.y, dimensions.y),
        ("z", coordinate.z, dimensions.z),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"coordinate {name} must be an integer")
        if not 0 <= value < bound:
            raise IndexError(f"coordinate {name} outside lattice")
    return coordinate.z * dimensions.x * dimensions.y + coordinate.y * dimensions.x + coordinate.x


def index_to_coordinate(index: int, dimensions: Dimensions3D) -> Coordinate3D:
    """Invert the canonical linear address transform exactly."""

    _validate_index(index, dimensions)
    plane = dimensions.x * dimensions.y
    z, within_plane = divmod(index, plane)
    y, x = divmod(within_plane, dimensions.x)
    return Coordinate3D(x=x, y=y, z=z)


def _bits_from_uint32(value: int) -> list[bool]:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("header value must be an integer")
    if not 0 <= value <= _MAX_PAYLOAD_BYTES:
        raise ValueError("header value does not fit uint32")
    return [bool((value >> shift) & 1) for shift in range(31, -1, -1)]


def _uint32_from_bits(bits: Sequence[bool]) -> int:
    if len(bits) != _HEADER_BITS:
        raise ValueError("uint32 header must contain exactly 32 bits")
    value = 0
    for bit in bits:
        if type(bit) is not bool:
            raise TypeError("header bits must contain bool values only")
        value = (value << 1) | int(bit)
    return value


def _bits_from_bytes(payload: bytes) -> list[bool]:
    output: list[bool] = []
    for byte in payload:
        output.extend(bool((byte >> shift) & 1) for shift in range(7, -1, -1))
    return output


def _bytes_from_bits(bits: Sequence[bool]) -> bytes:
    if len(bits) % 8:
        raise ValueError("payload bit count must be divisible by 8")
    output = bytearray()
    for offset in range(0, len(bits), 8):
        value = 0
        for bit in bits[offset : offset + 8]:
            if type(bit) is not bool:
                raise TypeError("payload bits must contain bool values only")
            value = (value << 1) | int(bit)
        output.append(value)
    return bytes(output)


def encode_bytes(payload: bytes, dimensions: Dimensions3D) -> BitMatrix3D:
    """Encode bytes into a canonical fixed-size Bit Matrix 3D lattice."""

    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    if dimensions.cells < _HEADER_BITS:
        raise ValueError("lattice requires at least 32 cells for the length header")
    if len(payload) > _MAX_PAYLOAD_BYTES:
        raise ValueError("payload exceeds uint32 length header")
    if len(payload) > dimensions.payload_capacity_bytes:
        raise ValueError(
            "payload exceeds lattice capacity: "
            f"{len(payload)} > {dimensions.payload_capacity_bytes} bytes"
        )

    logical_bits = _bits_from_uint32(len(payload))
    logical_bits.extend(_bits_from_bytes(payload))
    logical_bits.extend([False] * (dimensions.cells - len(logical_bits)))
    return BitMatrix3D(dimensions=dimensions, bits=tuple(logical_bits))


def encode_text(text: str, dimensions: Dimensions3D, *, encoding: str = "utf-8") -> BitMatrix3D:
    """Encode text using UTF-8 by default, then map it into the 3D lattice."""

    if not isinstance(text, str):
        raise TypeError("text must be str")
    return encode_bytes(text.encode(encoding), dimensions)


def decode_bytes(matrix: BitMatrix3D, *, require_zero_padding: bool = True) -> DecodeResult:
    """Decode and structurally validate a canonical Bit Matrix 3D lattice."""

    if not isinstance(matrix, BitMatrix3D):
        raise TypeError("matrix must be BitMatrix3D")
    if matrix.dimensions.cells < _HEADER_BITS:
        raise ValueError("lattice is too small to contain a header")

    declared_length = _uint32_from_bits(matrix.bits[:_HEADER_BITS])
    capacity = matrix.dimensions.payload_capacity_bytes
    if declared_length > capacity:
        raise ValueError(
            "declared payload length exceeds lattice capacity: "
            f"{declared_length} > {capacity} bytes"
        )

    payload_end = _HEADER_BITS + declared_length * 8
    payload = _bytes_from_bits(matrix.bits[_HEADER_BITS:payload_end])
    padding = matrix.bits[payload_end:]
    if require_zero_padding and any(padding):
        raise ValueError("non-zero bits found in canonical padding region")

    return DecodeResult(
        payload=payload,
        declared_length=declared_length,
        payload_capacity_bytes=capacity,
        padding_bits=len(padding),
    )


def decode_text(
    matrix: BitMatrix3D,
    *,
    encoding: str = "utf-8",
    require_zero_padding: bool = True,
) -> str:
    """Decode a text payload from the lattice."""

    result = decode_bytes(matrix, require_zero_padding=require_zero_padding)
    return result.payload.decode(encoding)


def from_active_coordinates(
    dimensions: Dimensions3D,
    coordinates: Iterable[Coordinate3D],
) -> BitMatrix3D:
    """Materialize a matrix from sparse active coordinates.

    This is the renderer/storage bridge: only active voxels need to be carried
    by a sparse transport, while reconstruction restores the authoritative
    dense logical lattice before decoding.
    """

    bits = [False] * dimensions.cells
    for coordinate in coordinates:
        index = coordinate_to_index(coordinate, dimensions)
        bits[index] = True
    return BitMatrix3D(dimensions=dimensions, bits=tuple(bits))


def normalized_scene_coordinate(
    coordinate: Coordinate3D,
    dimensions: Dimensions3D,
    *,
    scale: float = 1.0,
) -> tuple[float, float, float]:
    """Map a logical voxel to a centred renderer-independent scene position."""

    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        raise TypeError("scale must be numeric")
    coordinate_to_index(coordinate, dimensions)  # validates coordinate
    return (
        (coordinate.x - (dimensions.x - 1) / 2.0) * float(scale),
        (coordinate.y - (dimensions.y - 1) / 2.0) * float(scale),
        (coordinate.z - (dimensions.z - 1) / 2.0) * float(scale),
    )


def verify_round_trip(payload: bytes, dimensions: Dimensions3D) -> bool:
    """Return True only when encode -> sparse transport -> decode is exact."""

    encoded = encode_bytes(payload, dimensions)
    sparse_copy = from_active_coordinates(dimensions, encoded.active_coordinates())
    decoded = decode_bytes(sparse_copy)
    return decoded.payload == payload


__all__ = [
    "BitMatrix3D",
    "Coordinate3D",
    "DecodeResult",
    "Dimensions3D",
    "coordinate_to_index",
    "decode_bytes",
    "decode_text",
    "encode_bytes",
    "encode_text",
    "from_active_coordinates",
    "index_to_coordinate",
    "normalized_scene_coordinate",
    "verify_round_trip",
]
