"""Deterministic sparse-block encoding and compression."""

import base64
import hashlib
import json
import zlib
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, Tuple

from .geometry import Coordinate3D


@dataclass
class SparseBlock:
    """A block whose logical cells are allocated only when written."""

    index: Coordinate3D
    shape: Coordinate3D
    cell_bytes: int
    cells: Dict[Coordinate3D, bytes] = field(default_factory=dict)
    reads: int = 0
    writes: int = 0

    def _validate_offset(self, offset: Coordinate3D) -> None:
        if len(offset) != 3:
            raise ValueError("offset must contain exactly three values")
        for axis, limit in zip(offset, self.shape):
            if axis < 0 or axis >= limit:
                raise IndexError(
                    "offset {} lies outside block shape {}".format(
                        offset, self.shape
                    )
                )

    def read(self, offset: Coordinate3D) -> bytes:
        self._validate_offset(offset)
        self.reads += 1
        return self.cells.get(offset, b"")

    def write(self, offset: Coordinate3D, value: bytes) -> None:
        self._validate_offset(offset)
        if not isinstance(value, bytes):
            raise TypeError("value must be bytes")
        if len(value) > self.cell_bytes:
            raise ValueError(
                "payload has {} bytes but cell capacity is {}".format(
                    len(value), self.cell_bytes
                )
            )
        self.writes += 1
        if value:
            self.cells[offset] = value
        else:
            self.cells.pop(offset, None)

    @property
    def allocated_cells(self) -> int:
        return len(self.cells)

    @property
    def physical_payload_bytes(self) -> int:
        return sum(len(value) for value in self.cells.values())

    @property
    def is_empty(self) -> bool:
        return not self.cells

    def iter_cells(self) -> Iterator[Tuple[Coordinate3D, bytes]]:
        for offset in sorted(self.cells):
            yield offset, self.cells[offset]

    def canonical_document(self) -> Dict[str, object]:
        return {
            "cell_bytes": self.cell_bytes,
            "cells": [
                {
                    "offset": list(offset),
                    "payload": base64.b64encode(value).decode("ascii"),
                }
                for offset, value in self.iter_cells()
            ],
            "index": list(self.index),
            "reads": self.reads,
            "shape": list(self.shape),
            "writes": self.writes,
        }

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.canonical_document(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")

    @classmethod
    def from_canonical_bytes(cls, payload: bytes) -> "SparseBlock":
        document = json.loads(payload.decode("utf-8"))
        block = cls(
            index=tuple(document["index"]),
            shape=tuple(document["shape"]),
            cell_bytes=int(document["cell_bytes"]),
            reads=int(document.get("reads", 0)),
            writes=int(document.get("writes", 0)),
        )
        for cell in document["cells"]:
            offset = tuple(cell["offset"])
            value = base64.b64decode(cell["payload"].encode("ascii"))
            block.cells[offset] = value
        return block


@dataclass(frozen=True)
class EncodedBlock:
    """One compressed block with enough metadata for verified restoration."""

    index: Coordinate3D
    level: int
    raw_bytes: int
    compressed_bytes: int
    checksum: str
    payload: bytes
    codec: str = "zlib-sparse-v1"

    def to_document(self) -> Dict[str, object]:
        return {
            "checksum": self.checksum,
            "codec": self.codec,
            "compressed_bytes": self.compressed_bytes,
            "index": list(self.index),
            "level": self.level,
            "payload": base64.b64encode(self.payload).decode("ascii"),
            "raw_bytes": self.raw_bytes,
        }

    @classmethod
    def from_document(cls, document: Dict[str, object]) -> "EncodedBlock":
        return cls(
            index=tuple(document["index"]),  # type: ignore[arg-type]
            level=int(document["level"]),
            raw_bytes=int(document["raw_bytes"]),
            compressed_bytes=int(document["compressed_bytes"]),
            checksum=str(document["checksum"]),
            codec=str(document["codec"]),
            payload=base64.b64decode(
                str(document["payload"]).encode("ascii")
            ),
        )


class ZlibSparseBlockCodec:
    """Lossless reference codec used until learned 3D codecs are attached."""

    name = "zlib-sparse-v1"

    @staticmethod
    def encode(block: SparseBlock, level: int = 6) -> EncodedBlock:
        if level < 1 or level > 9:
            raise ValueError("zlib compression level must be in [1, 9]")
        raw = block.canonical_bytes()
        payload = zlib.compress(raw, level)
        return EncodedBlock(
            index=block.index,
            level=level,
            raw_bytes=len(raw),
            compressed_bytes=len(payload),
            checksum=hashlib.sha256(raw).hexdigest(),
            payload=payload,
        )

    @staticmethod
    def decode(encoded: EncodedBlock) -> SparseBlock:
        if encoded.codec != ZlibSparseBlockCodec.name:
            raise ValueError("unsupported codec: {}".format(encoded.codec))
        raw = zlib.decompress(encoded.payload)
        checksum = hashlib.sha256(raw).hexdigest()
        if checksum != encoded.checksum:
            raise ValueError("block checksum mismatch")
        block = SparseBlock.from_canonical_bytes(raw)
        if block.index != encoded.index:
            raise ValueError("encoded block index does not match decoded block")
        return block

    @classmethod
    def optimize(
        cls, block: SparseBlock, levels: Iterable[int] = range(1, 10)
    ) -> EncodedBlock:
        candidates = [cls.encode(block, level) for level in levels]
        if not candidates:
            raise ValueError("at least one compression level is required")
        return min(
            candidates,
            key=lambda item: (item.compressed_bytes, item.level),
        )
