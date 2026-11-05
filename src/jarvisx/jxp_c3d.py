"""JXP-C3D sparse 3D cloud media-transfer reference runtime.

The logical address space contains 1000^3 independently addressable cells.  The
reference implementation never allocates that dense volume: only verified,
committed cells occupy host memory.

The codec is deliberately a deterministic reference transform, not a claim of a
trained production ANN.  It exposes the same encode -> inward refine -> decode
-> residual -> verify boundary that a learned codec can implement later.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Iterable, Optional, Tuple


AXIS = 1000
LOGICAL_CELLS = AXIS**3
_MAGIC = b"JXPC3D1\0"
_MAX_HEADER = 64 * 1024


class IntegrityError(ValueError):
    """Raised when a JXP-C3D frame or reconstruction fails integrity checks."""


class Modality(IntEnum):
    BINARY = 0
    TEXT = 1
    IMAGE = 2
    AUDIO = 3
    VIDEO = 4
    TENSOR = 5
    BYTECODE = 6


class Opcode(IntEnum):
    INGEST = 1
    MAP3D = 2
    ENCODE = 3
    FOLD = 4
    DECODE = 5
    RESIDUAL = 6
    VERIFY = 7
    COMMIT = 8
    STREAM = 9


@dataclass(frozen=True, order=True)
class SpatialAddress:
    x: int
    y: int
    z: int

    def __post_init__(self) -> None:
        for name, value in (("x", self.x), ("y", self.y), ("z", self.z)):
            if not isinstance(value, int):
                raise TypeError(f"{name} must be an int")
            if not 0 <= value < AXIS:
                raise ValueError(f"{name} must be in [0, {AXIS})")

    @property
    def linear(self) -> int:
        return self.x + AXIS * (self.y + AXIS * self.z)

    @property
    def morton30(self) -> int:
        """Interleave ten bits from each coordinate into a 30-bit Morton key."""

        code = 0
        for bit in range(10):
            code |= ((self.x >> bit) & 1) << (3 * bit)
            code |= ((self.y >> bit) & 1) << (3 * bit + 1)
            code |= ((self.z >> bit) & 1) << (3 * bit + 2)
        return code

    @classmethod
    def from_morton30(cls, code: int) -> "SpatialAddress":
        if not isinstance(code, int) or not 0 <= code < (1 << 30):
            raise ValueError("Morton code must fit in 30 bits")
        xyz = [0, 0, 0]
        for bit in range(10):
            for axis in range(3):
                xyz[axis] |= ((code >> (3 * bit + axis)) & 1) << bit
        return cls(*xyz)


def route_coordinate(stream_id: str, sequence: int) -> SpatialAddress:
    """Map a stream/sequence pair deterministically into the 1000^3 namespace."""

    if sequence < 0:
        raise ValueError("sequence must be non-negative")
    seed = f"{stream_id}:{sequence}".encode("utf-8")
    index = int.from_bytes(hashlib.sha256(seed).digest()[:8], "big") % LOGICAL_CELLS
    x = index % AXIS
    y = (index // AXIS) % AXIS
    z = index // (AXIS * AXIS)
    return SpatialAddress(x, y, z)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _spread(values: Iterable[int]) -> int:
    values = tuple(values)
    return 0 if not values else max(values) - min(values)


class ReferenceInwardCodec:
    """Deterministic codec with bounded inward refinement and exact residuals.

    Initial latent values are byte high-nibbles.  Refinement contracts those
    values toward their centroid.  Exact source reconstruction is retained in a
    signed 16-bit residual channel.  This is intentionally correctness-first:
    its residual can be larger than the source and is not a compression claim.
    """

    def encode(
        self,
        data: bytes,
        *,
        iterations: int = 4,
        rho: float = 0.5,
    ) -> Tuple[bytes, bytes, int, int]:
        if iterations < 0:
            raise ValueError("iterations must be non-negative")
        if not 0.0 <= rho < 1.0:
            raise ValueError("rho must satisfy 0 <= rho < 1")

        latent = [byte >> 4 for byte in data]
        spread_before = _spread(latent)

        for _ in range(iterations):
            if not latent:
                break
            centre = sum(latent) / len(latent)
            latent = [
                max(0, min(15, int(round(centre + rho * (value - centre)))))
                for value in latent
            ]

        spread_after = _spread(latent)
        base = [value << 4 for value in latent]
        residual_values = [source - prediction for source, prediction in zip(data, base)]
        residual = b"".join(struct.pack(">h", value) for value in residual_values)
        return bytes(latent), residual, spread_before, spread_after

    def decode(self, latent: bytes, residual: bytes) -> bytes:
        if len(residual) != 2 * len(latent):
            raise IntegrityError("residual length must be exactly twice latent length")
        output = bytearray(len(latent))
        for i, value in enumerate(latent):
            if value > 15:
                raise IntegrityError("latent value exceeds 4-bit reference domain")
            delta = struct.unpack_from(">h", residual, 2 * i)[0]
            reconstructed = (value << 4) + delta
            if not 0 <= reconstructed <= 255:
                raise IntegrityError("residual reconstructs a value outside byte range")
            output[i] = reconstructed
        return bytes(output)


@dataclass(frozen=True)
class JXPC3DFrame:
    stream_id: str
    sequence: int
    address: SpatialAddress
    modality: Modality
    opcode: Opcode
    iteration: int
    latent: bytes
    residual: bytes
    source_sha256: str

    def pack(self) -> bytes:
        payload_sha256 = _sha256(self.latent + self.residual)
        header = {
            "version": 1,
            "stream_id": self.stream_id,
            "sequence": self.sequence,
            "coord": [self.address.x, self.address.y, self.address.z],
            "morton30": self.address.morton30,
            "modality": int(self.modality),
            "opcode": int(self.opcode),
            "iteration": self.iteration,
            "latent_length": len(self.latent),
            "residual_length": len(self.residual),
            "source_sha256": self.source_sha256,
            "payload_sha256": payload_sha256,
        }
        raw_header = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if len(raw_header) > _MAX_HEADER:
            raise ValueError("JXP-C3D header exceeds maximum size")
        return _MAGIC + struct.pack(">I", len(raw_header)) + raw_header + self.latent + self.residual

    @classmethod
    def unpack(cls, wire: bytes) -> "JXPC3DFrame":
        prefix = len(_MAGIC) + 4
        if len(wire) < prefix or wire[: len(_MAGIC)] != _MAGIC:
            raise IntegrityError("invalid JXP-C3D magic")
        header_length = struct.unpack_from(">I", wire, len(_MAGIC))[0]
        if header_length > _MAX_HEADER or len(wire) < prefix + header_length:
            raise IntegrityError("invalid JXP-C3D header length")

        raw_header = wire[prefix : prefix + header_length]
        try:
            header = json.loads(raw_header.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise IntegrityError("invalid JXP-C3D JSON header") from exc

        latent_length = int(header["latent_length"])
        residual_length = int(header["residual_length"])
        payload_start = prefix + header_length
        payload_end = payload_start + latent_length + residual_length
        if payload_end != len(wire):
            raise IntegrityError("JXP-C3D payload length mismatch")

        latent = wire[payload_start : payload_start + latent_length]
        residual = wire[payload_start + latent_length : payload_end]
        if _sha256(latent + residual) != header["payload_sha256"]:
            raise IntegrityError("JXP-C3D payload hash mismatch")

        address = SpatialAddress(*header["coord"])
        if address.morton30 != int(header["morton30"]):
            raise IntegrityError("coordinate/Morton mismatch")

        return cls(
            stream_id=str(header["stream_id"]),
            sequence=int(header["sequence"]),
            address=address,
            modality=Modality(int(header["modality"])),
            opcode=Opcode(int(header["opcode"])),
            iteration=int(header["iteration"]),
            latent=latent,
            residual=residual,
            source_sha256=str(header["source_sha256"]),
        )


@dataclass(frozen=True)
class VoxelRecord:
    frame: JXPC3DFrame
    wire_bytes: int


class SparseC3DStore:
    """Content-bearing active set for the 1000^3 logical cloud manifold."""

    logical_cells = LOGICAL_CELLS

    def __init__(self) -> None:
        self._cells: Dict[int, VoxelRecord] = {}

    @property
    def active_cells(self) -> int:
        return len(self._cells)

    @property
    def resident_wire_bytes(self) -> int:
        return sum(record.wire_bytes for record in self._cells.values())

    def commit(self, frame: JXPC3DFrame, wire_bytes: int) -> None:
        self._cells[frame.address.morton30] = VoxelRecord(frame=frame, wire_bytes=wire_bytes)

    def get(self, address: SpatialAddress) -> Optional[VoxelRecord]:
        return self._cells.get(address.morton30)


@dataclass(frozen=True)
class TransferReceipt:
    address: SpatialAddress
    morton30: int
    source_bytes: int
    wire_bytes: int
    active_cells: int
    resident_wire_bytes: int
    spread_before: int
    spread_after: int
    iterations: int
    verified: bool
    source_sha256: str

    def as_dict(self) -> dict:
        return {
            "coord": [self.address.x, self.address.y, self.address.z],
            "morton30": self.morton30,
            "source_bytes": self.source_bytes,
            "wire_bytes": self.wire_bytes,
            "active_cells": self.active_cells,
            "resident_wire_bytes": self.resident_wire_bytes,
            "spread_before": self.spread_before,
            "spread_after": self.spread_after,
            "iterations": self.iterations,
            "verified": self.verified,
            "source_sha256": self.source_sha256,
            "logical_cells": LOGICAL_CELLS,
        }


class JXPC3DRuntime:
    """Reference transaction path for sparse 3D media transfer."""

    def __init__(
        self,
        *,
        store: Optional[SparseC3DStore] = None,
        codec: Optional[ReferenceInwardCodec] = None,
    ) -> None:
        self.store = store or SparseC3DStore()
        self.codec = codec or ReferenceInwardCodec()

    def transfer(
        self,
        data: bytes,
        *,
        stream_id: str,
        sequence: int = 0,
        address: Optional[SpatialAddress] = None,
        modality: Modality = Modality.BINARY,
        iterations: int = 4,
        rho: float = 0.5,
    ) -> TransferReceipt:
        if not isinstance(data, bytes):
            raise TypeError("data must be bytes")

        address = address or route_coordinate(stream_id, sequence)
        latent, residual, spread_before, spread_after = self.codec.encode(
            data, iterations=iterations, rho=rho
        )
        source_sha256 = _sha256(data)

        candidate = JXPC3DFrame(
            stream_id=stream_id,
            sequence=sequence,
            address=address,
            modality=modality,
            opcode=Opcode.COMMIT,
            iteration=iterations,
            latent=latent,
            residual=residual,
            source_sha256=source_sha256,
        )

        wire = candidate.pack()
        parsed = JXPC3DFrame.unpack(wire)
        reconstructed = self.codec.decode(parsed.latent, parsed.residual)
        verified = _sha256(reconstructed) == parsed.source_sha256
        if not verified:
            raise IntegrityError("candidate failed source reconstruction verification")

        # Candidate-first transaction boundary: commit only after byte-exact verification.
        self.store.commit(parsed, len(wire))

        return TransferReceipt(
            address=address,
            morton30=address.morton30,
            source_bytes=len(data),
            wire_bytes=len(wire),
            active_cells=self.store.active_cells,
            resident_wire_bytes=self.store.resident_wire_bytes,
            spread_before=spread_before,
            spread_after=spread_after,
            iterations=iterations,
            verified=True,
            source_sha256=source_sha256,
        )

    def reconstruct(self, address: SpatialAddress) -> bytes:
        record = self.store.get(address)
        if record is None:
            raise KeyError(address)
        reconstructed = self.codec.decode(record.frame.latent, record.frame.residual)
        if _sha256(reconstructed) != record.frame.source_sha256:
            raise IntegrityError("stored voxel failed source hash verification")
        return reconstructed


def main() -> int:
    runtime = JXPC3DRuntime()
    sample = bytes(range(256)) * 4
    receipt = runtime.transfer(
        sample,
        stream_id="jarvis-x-demo",
        modality=Modality.BINARY,
        iterations=4,
        rho=0.5,
    )
    print(json.dumps(receipt.as_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
