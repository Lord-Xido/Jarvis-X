"""Deterministic 3D byte-ROM storage for Jarvis-X bytecode.

The module deliberately uses a lossless byte representation rather than an
untrained latent model. Jarvis-X instructions are packed as unsigned 64-bit
words, wrapped in a versioned bytecode image, distributed across a fixed 3D
ROM geometry, and protected by SHA-256.
"""

from __future__ import annotations

import hashlib
import os
import struct
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

from .assembler import Assembler, OPCODES
from .core import CodexVM
from .parser import Parser

PathLike = Union[str, os.PathLike]

_ROM_MAGIC = b"JXROM3D\0"
_ROM_VERSION = 1
_ROM_FLAG_COMPRESSED = 0x01
_ROM_KNOWN_FLAGS = _ROM_FLAG_COMPRESSED
_ROM_HEADER = struct.Struct(">8sBBH4HIQQ32s")

_BYTECODE_MAGIC = b"JXBC"
_BYTECODE_VERSION = 1
_BYTECODE_HEADER = struct.Struct(">4sB3xI")
_WORD = struct.Struct(">Q")
_MAX_WORD = (1 << 64) - 1
_IMMEDIATE_MASK = 0xFFFF << 8


class ROMError(Exception):
    """Base exception for the Chrysalis ROM subsystem."""


class ROMFormatError(ROMError):
    """Raised when a serialized ROM or bytecode image is malformed."""


class ROMCapacityError(ROMError):
    """Raised when a payload does not fit the requested geometry."""


class ROMIntegrityError(ROMError):
    """Raised when payload length or SHA-256 verification fails."""


@dataclass(frozen=True)
class ROMGeometry:
    """Physical layout of the engine-by-3D-cell ROM array."""

    engines: int
    x: int
    y: int
    z: int
    cell_bytes: int

    def __post_init__(self) -> None:
        for name, value, maximum in (
            ("engines", self.engines, 0xFFFF),
            ("x", self.x, 0xFFFF),
            ("y", self.y, 0xFFFF),
            ("z", self.z, 0xFFFF),
            ("cell_bytes", self.cell_bytes, 0xFFFFFFFF),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError("{} must be an integer".format(name))
            if value < 1 or value > maximum:
                raise ValueError("{} must be in the range 1..{}".format(name, maximum))

    @property
    def cells_per_engine(self) -> int:
        return self.x * self.y * self.z

    @property
    def cell_count(self) -> int:
        return self.engines * self.cells_per_engine

    @property
    def capacity_bytes(self) -> int:
        return self.cell_count * self.cell_bytes

    def cell_index(self, engine: int, x: int, y: int, z: int) -> int:
        bounds = (
            ("engine", engine, self.engines),
            ("x", x, self.x),
            ("y", y, self.y),
            ("z", z, self.z),
        )
        for name, value, upper in bounds:
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError("{} coordinate must be an integer".format(name))
            if value < 0 or value >= upper:
                raise IndexError("{} coordinate {} outside 0..{}".format(name, value, upper - 1))
        return (((engine * self.x) + x) * self.y + y) * self.z + z


class ChrysalisROM:
    """Immutable, checksum-protected byte payload distributed over a 3D grid."""

    def __init__(
        self,
        geometry: ROMGeometry,
        body: bytes,
        stored_size: int,
        raw_size: int,
        digest: bytes,
        compressed: bool = False,
    ) -> None:
        if len(body) != geometry.capacity_bytes:
            raise ROMFormatError(
                "ROM body is {} bytes; geometry requires {}".format(
                    len(body), geometry.capacity_bytes
                )
            )
        if stored_size < 0 or stored_size > len(body):
            raise ROMFormatError("stored payload size exceeds ROM capacity")
        if raw_size < 0:
            raise ROMFormatError("raw payload size cannot be negative")
        if len(digest) != hashlib.sha256().digest_size:
            raise ROMFormatError("SHA-256 digest must be 32 bytes")

        self.geometry = geometry
        self._body = bytes(body)
        self.stored_size = stored_size
        self.raw_size = raw_size
        self.digest = bytes(digest)
        self.compressed = bool(compressed)

    @classmethod
    def from_payload(
        cls,
        payload: bytes,
        engines: int = 1,
        grid_shape: Tuple[int, int, int] = (4, 4, 4),
        cell_bytes: Optional[int] = None,
        compress: bool = False,
    ) -> "ChrysalisROM":
        raw = bytes(payload)
        x, y, z = _validate_grid_shape(grid_shape)
        cell_count = engines * x * y * z
        if cell_count < 1:
            raise ValueError("ROM must contain at least one cell")

        stored = raw
        compressed = False
        if compress and raw:
            candidate = zlib.compress(raw, level=9)
            if len(candidate) < len(raw):
                stored = candidate
                compressed = True

        if cell_bytes is None:
            cell_bytes = max(1, (len(stored) + cell_count - 1) // cell_count)

        geometry = ROMGeometry(engines, x, y, z, cell_bytes)
        if len(stored) > geometry.capacity_bytes:
            raise ROMCapacityError(
                "payload requires {} bytes but geometry provides {}".format(
                    len(stored), geometry.capacity_bytes
                )
            )

        body = stored + (b"\x00" * (geometry.capacity_bytes - len(stored)))
        return cls(
            geometry=geometry,
            body=body,
            stored_size=len(stored),
            raw_size=len(raw),
            digest=hashlib.sha256(raw).digest(),
            compressed=compressed,
        )

    @classmethod
    def deserialize(cls, data: bytes) -> "ChrysalisROM":
        serialized = bytes(data)
        if len(serialized) < _ROM_HEADER.size:
            raise ROMFormatError("serialized ROM is shorter than its header")

        (
            magic,
            version,
            flags,
            reserved,
            engines,
            x,
            y,
            z,
            cell_bytes,
            stored_size,
            raw_size,
            digest,
        ) = _ROM_HEADER.unpack_from(serialized)

        if magic != _ROM_MAGIC:
            raise ROMFormatError("invalid ROM magic")
        if version != _ROM_VERSION:
            raise ROMFormatError("unsupported ROM version {}".format(version))
        if reserved != 0:
            raise ROMFormatError("reserved ROM header bits must be zero")
        if flags & ~_ROM_KNOWN_FLAGS:
            raise ROMFormatError("ROM uses unknown feature flags")

        geometry = ROMGeometry(engines, x, y, z, cell_bytes)
        expected_size = _ROM_HEADER.size + geometry.capacity_bytes
        if len(serialized) != expected_size:
            raise ROMFormatError(
                "serialized ROM is {} bytes; expected {}".format(len(serialized), expected_size)
            )

        return cls(
            geometry=geometry,
            body=serialized[_ROM_HEADER.size :],
            stored_size=stored_size,
            raw_size=raw_size,
            digest=digest,
            compressed=bool(flags & _ROM_FLAG_COMPRESSED),
        )

    @classmethod
    def read(cls, path: PathLike) -> "ChrysalisROM":
        return cls.deserialize(Path(path).read_bytes())

    def serialize(self) -> bytes:
        flags = _ROM_FLAG_COMPRESSED if self.compressed else 0
        header = _ROM_HEADER.pack(
            _ROM_MAGIC,
            _ROM_VERSION,
            flags,
            0,
            self.geometry.engines,
            self.geometry.x,
            self.geometry.y,
            self.geometry.z,
            self.geometry.cell_bytes,
            self.stored_size,
            self.raw_size,
            self.digest,
        )
        return header + self._body

    def write(self, path: PathLike) -> None:
        """Atomically persist the complete fixed-capacity ROM image."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=destination.name + ".",
            suffix=".tmp",
            dir=str(destination.parent),
            delete=False,
        )
        temporary_name = handle.name
        try:
            with handle:
                handle.write(self.serialize())
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, str(destination))
        except Exception:
            try:
                os.unlink(temporary_name)
            except OSError:
                pass
            raise

    def payload(self, verify: bool = True) -> bytes:
        stored = self._body[: self.stored_size]
        if self.compressed:
            try:
                raw = zlib.decompress(stored)
            except zlib.error as exc:
                raise ROMIntegrityError("compressed payload is corrupt") from exc
        else:
            raw = stored

        if verify:
            if any(self._body[self.stored_size :]):
                raise ROMIntegrityError("ROM padding is non-zero")
            if len(raw) != self.raw_size:
                raise ROMIntegrityError(
                    "decoded payload is {} bytes; expected {}".format(len(raw), self.raw_size)
                )
            actual = hashlib.sha256(raw).digest()
            if actual != self.digest:
                raise ROMIntegrityError("payload SHA-256 mismatch")
        return raw

    def verify(self) -> bool:
        self.payload(verify=True)
        return True

    def cell(self, engine: int, x: int, y: int, z: int) -> bytes:
        index = self.geometry.cell_index(engine, x, y, z)
        start = index * self.geometry.cell_bytes
        end = start + self.geometry.cell_bytes
        return self._body[start:end]

    def address_for_stored_offset(self, offset: int) -> Tuple[int, int, int, int, int]:
        if offset < 0 or offset >= self.stored_size:
            raise IndexError("stored payload offset outside 0..{}".format(self.stored_size - 1))
        cell_index, within_cell = divmod(offset, self.geometry.cell_bytes)
        engine, remainder = divmod(cell_index, self.geometry.cells_per_engine)
        x, remainder = divmod(remainder, self.geometry.y * self.geometry.z)
        y, z = divmod(remainder, self.geometry.z)
        return engine, x, y, z, within_cell

    def statistics(self) -> Dict[str, object]:
        return {
            "format": "JXROM3D",
            "version": _ROM_VERSION,
            "engines": self.geometry.engines,
            "grid_shape": [self.geometry.x, self.geometry.y, self.geometry.z],
            "cell_bytes": self.geometry.cell_bytes,
            "cell_count": self.geometry.cell_count,
            "capacity_bytes": self.geometry.capacity_bytes,
            "stored_bytes": self.stored_size,
            "raw_bytes": self.raw_size,
            "compressed": self.compressed,
            "utilization": (
                float(self.stored_size) / float(self.geometry.capacity_bytes)
                if self.geometry.capacity_bytes
                else 0.0
            ),
            "sha256": self.digest.hex(),
        }


def _validate_grid_shape(grid_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
    if len(grid_shape) != 3:
        raise ValueError("grid shape must contain exactly X, Y, and Z")
    values = tuple(grid_shape)
    for value in values:
        if not isinstance(value, int) or isinstance(value, bool) or value < 1 or value > 0xFFFF:
            raise ValueError("grid dimensions must be integers in the range 1..65535")
    return values[0], values[1], values[2]


def pack_bytecode(words: Iterable[int]) -> bytes:
    materialized = list(words)
    if len(materialized) > 0xFFFFFFFF:
        raise ValueError("bytecode image contains too many words")

    encoded = bytearray(_BYTECODE_HEADER.pack(_BYTECODE_MAGIC, _BYTECODE_VERSION, len(materialized)))
    for index, word in enumerate(materialized):
        if not isinstance(word, int) or isinstance(word, bool):
            raise TypeError("bytecode word {} is not an integer".format(index))
        if word < 0 or word > _MAX_WORD:
            raise ValueError("bytecode word {} does not fit in 64 bits".format(index))
        encoded.extend(_WORD.pack(word))
    return bytes(encoded)


def unpack_bytecode(image: bytes) -> List[int]:
    data = bytes(image)
    if len(data) < _BYTECODE_HEADER.size:
        raise ROMFormatError("bytecode image is shorter than its header")

    magic, version, count = _BYTECODE_HEADER.unpack_from(data)
    if magic != _BYTECODE_MAGIC:
        raise ROMFormatError("invalid Jarvis-X bytecode image magic")
    if version != _BYTECODE_VERSION:
        raise ROMFormatError("unsupported Jarvis-X bytecode image version {}".format(version))

    expected = _BYTECODE_HEADER.size + (count * _WORD.size)
    if len(data) != expected:
        raise ROMFormatError(
            "bytecode image is {} bytes; expected {} for {} words".format(
                len(data), expected, count
            )
        )
    return [
        _WORD.unpack_from(data, _BYTECODE_HEADER.size + index * _WORD.size)[0]
        for index in range(count)
    ]


def assemble_source(source: str) -> List[int]:
    return Assembler().assemble(Parser().parse(source))


def rom_from_source(
    source: str,
    engines: int = 1,
    grid_shape: Tuple[int, int, int] = (4, 4, 4),
    cell_bytes: Optional[int] = None,
    compress: bool = False,
) -> ChrysalisROM:
    image = pack_bytecode(assemble_source(source))
    return ChrysalisROM.from_payload(
        image,
        engines=engines,
        grid_shape=grid_shape,
        cell_bytes=cell_bytes,
        compress=compress,
    )


def words_from_rom(rom: ChrysalisROM) -> List[int]:
    return unpack_bytecode(rom.payload(verify=True))


def run_rom(rom: ChrysalisROM) -> CodexVM:
    words = words_from_rom(rom)
    if not words:
        raise ROMFormatError("cannot execute an empty bytecode image")
    vm = CodexVM()
    vm.load(words)
    vm.run()
    return vm


def mutate_immediate(rom: ChrysalisROM, word_index: int, delta: int) -> ChrysalisROM:
    """Create a verified candidate ROM by changing one SET immediate only.

    The opcode, destination register, source registers, grid dimensions, and
    cell capacity remain unchanged. This is a bounded structural mutation, not
    an arbitrary bit flip.
    """

    words = words_from_rom(rom)
    if word_index < 0 or word_index >= len(words):
        raise IndexError("word index outside 0..{}".format(len(words) - 1))
    if not isinstance(delta, int) or isinstance(delta, bool):
        raise TypeError("delta must be an integer")

    word = words[word_index]
    opcode = (word >> 56) & 0xFF
    if opcode != OPCODES["SET"]:
        raise ValueError("bounded mutation only supports SET instructions")

    current = (word >> 8) & 0xFFFF
    candidate = current + delta
    if candidate < 0 or candidate > 0xFFFF:
        raise ValueError("mutated immediate must remain in the range 0..65535")

    words[word_index] = (word & ~_IMMEDIATE_MASK) | (candidate << 8)
    image = pack_bytecode(words)
    geometry = rom.geometry
    return ChrysalisROM.from_payload(
        image,
        engines=geometry.engines,
        grid_shape=(geometry.x, geometry.y, geometry.z),
        cell_bytes=geometry.cell_bytes,
        compress=rom.compressed,
    )
