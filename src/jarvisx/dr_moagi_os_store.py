"""Exact sparse transport/persistence packet for the Dr Moagi 3D OS.

The in-memory OS state is a sparse mapping of integer 3D coordinates to finite
floating-point values.  This module provides a deterministic byte representation
for that state without allocating the logical ``side ** 3`` lattice.

Records are Morton ordered for spatial locality, delta-coded by Morton key, store
values as IEEE-754 float64 bytes, and are compressed with DEFLATE.  The codec is
lossless for Python floats: decoding reproduces the exact float64 values and
coordinates that were encoded.
"""

from __future__ import annotations

import hashlib
import math
import struct
import zlib
from dataclasses import dataclass
from typing import Mapping

from .dr_moagi_field_runtime import Coordinate, SparseField
from .dr_moagi_frontier import morton3_decode, morton3_encode

_MAGIC = b"DMOS2"
_RAW_RECORD_BYTES = 32  # three logical int64 coordinates + one float64 reference footprint


def _encode_varint(value: int) -> bytes:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("varint value must be a non-negative integer")
    output = bytearray()
    current = value
    while True:
        byte = current & 0x7F
        current >>= 7
        if current:
            output.append(byte | 0x80)
        else:
            output.append(byte)
            return bytes(output)


def _decode_varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    cursor = offset
    while cursor < len(data):
        byte = data[cursor]
        cursor += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, cursor
        shift += 7
        if shift > 70:
            raise ValueError("varint exceeds supported coordinate range")
    raise ValueError("truncated varint")


@dataclass(frozen=True)
class SparseStatePacket3D:
    """Compressed exact sparse-state envelope."""

    side: int
    active_cells: int
    payload: bytes
    checksum_sha256: str

    @property
    def encoded_bytes(self) -> int:
        return len(self.payload)

    @property
    def reference_bytes(self) -> int:
        return self.active_cells * _RAW_RECORD_BYTES

    @property
    def compression_ratio(self) -> float:
        if self.encoded_bytes == 0:
            return math.inf
        return self.reference_bytes / self.encoded_bytes

    def as_dict(self) -> dict[str, object]:
        return {
            "format": "DMOS2",
            "side": self.side,
            "active_cells": self.active_cells,
            "encoded_bytes": self.encoded_bytes,
            "reference_bytes": self.reference_bytes,
            "compression_ratio": self.compression_ratio,
            "checksum_sha256": self.checksum_sha256,
        }


class SparseStateCodec3D:
    """Lossless Morton-delta sparse field packet codec."""

    def encode(self, field: Mapping[Coordinate, float], *, side: int) -> SparseStatePacket3D:
        self._validate_side(side)
        ordered: list[tuple[int, float]] = []
        for coordinate, raw_value in field.items():
            x, y, z = self._validate_coordinate(coordinate, side)
            value = self._finite(raw_value)
            ordered.append((morton3_encode(x, y, z), value))
        ordered.sort(key=lambda item: item[0])

        raw = bytearray(_MAGIC)
        raw.extend(_encode_varint(side))
        raw.extend(_encode_varint(len(ordered)))
        previous = 0
        for index, (code, value) in enumerate(ordered):
            delta = code if index == 0 else code - previous
            if index > 0 and delta <= 0:
                raise ValueError("Morton coordinates must be unique")
            raw.extend(_encode_varint(delta))
            raw.extend(struct.pack(">d", value))
            previous = code

        payload = zlib.compress(bytes(raw), level=9)
        return SparseStatePacket3D(
            side=side,
            active_cells=len(ordered),
            payload=payload,
            checksum_sha256=hashlib.sha256(payload).hexdigest(),
        )

    def decode(self, packet: SparseStatePacket3D) -> SparseField:
        if hashlib.sha256(packet.payload).hexdigest() != packet.checksum_sha256:
            raise ValueError("sparse-state packet checksum mismatch")
        parsed, field = self.decode_payload(packet.payload)
        if parsed.side != packet.side:
            raise ValueError("sparse-state packet side mismatch")
        if parsed.active_cells != packet.active_cells:
            raise ValueError("sparse-state packet active-cell count mismatch")
        return field

    def decode_payload(
        self,
        payload: bytes,
        *,
        expected_checksum: str | None = None,
    ) -> tuple[SparseStatePacket3D, SparseField]:
        if not isinstance(payload, bytes):
            raise TypeError("payload must be bytes")
        checksum = hashlib.sha256(payload).hexdigest()
        if expected_checksum is not None and checksum != expected_checksum:
            raise ValueError("sparse-state packet checksum mismatch")
        try:
            raw = zlib.decompress(payload)
        except zlib.error as exc:
            raise ValueError("invalid compressed sparse-state packet") from exc
        if not raw.startswith(_MAGIC):
            raise ValueError("invalid sparse-state packet magic")

        cursor = len(_MAGIC)
        side, cursor = _decode_varint(raw, cursor)
        self._validate_side(side)
        count, cursor = _decode_varint(raw, cursor)
        field: SparseField = {}
        code = 0
        for index in range(count):
            delta, cursor = _decode_varint(raw, cursor)
            if index > 0 and delta <= 0:
                raise ValueError("sparse-state Morton stream is not strictly ordered")
            code = delta if index == 0 else code + delta
            if cursor + 8 > len(raw):
                raise ValueError("truncated sparse-state float64 value")
            value = struct.unpack(">d", raw[cursor : cursor + 8])[0]
            cursor += 8
            if not math.isfinite(value):
                raise ValueError("sparse-state packet contains non-finite value")
            coordinate = morton3_decode(code)
            self._validate_coordinate(coordinate, side)
            if coordinate in field:
                raise ValueError("duplicate coordinate in sparse-state packet")
            field[coordinate] = value

        if cursor != len(raw):
            raise ValueError("sparse-state packet contains trailing bytes")
        if len(field) != count:
            raise ValueError("sparse-state packet active-cell count mismatch")

        packet = SparseStatePacket3D(
            side=side,
            active_cells=count,
            payload=payload,
            checksum_sha256=checksum,
        )
        return packet, field

    @staticmethod
    def _validate_side(side: int) -> None:
        if isinstance(side, bool) or not isinstance(side, int) or side <= 0:
            raise ValueError("side must be a positive integer")

    @staticmethod
    def _validate_coordinate(coordinate: Coordinate, side: int) -> Coordinate:
        if (
            not isinstance(coordinate, tuple)
            or len(coordinate) != 3
            or any(isinstance(axis, bool) or not isinstance(axis, int) for axis in coordinate)
        ):
            raise TypeError("coordinates must be integer (x, y, z) tuples")
        if any(axis < 0 or axis >= side for axis in coordinate):
            raise ValueError("coordinate outside logical lattice")
        return coordinate

    @staticmethod
    def _finite(value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("state values must be numeric")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("state values must be finite")
        return result


__all__ = ["SparseStateCodec3D", "SparseStatePacket3D"]
