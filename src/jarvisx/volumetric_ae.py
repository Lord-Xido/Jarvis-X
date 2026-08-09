from __future__ import annotations

import hashlib
import json
import math
import struct
import time
import zlib
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterator, Tuple

MAGIC = b"JX3DVAE1"
FORMAT_VERSION = 1
_HEADER_LEN = struct.Struct(">Q")
_CHUNK_LEN = struct.Struct(">I")


class ArtifactError(ValueError):
    """Raised when a volumetric latent artifact is malformed or fails verification."""


@dataclass(frozen=True)
class VirtualVolumeSpec:
    """Logical 3D tensor substrate; capacity is virtual and is never eagerly allocated."""

    capacity_gib: int = 6400
    cell_bits: int = 32
    chunk_bytes: int = 1 << 20

    def __post_init__(self) -> None:
        if self.capacity_gib <= 0:
            raise ValueError("capacity_gib must be positive")
        if self.cell_bits <= 0 or self.cell_bits % 8:
            raise ValueError("cell_bits must be a positive multiple of 8")
        if self.chunk_bytes <= 0:
            raise ValueError("chunk_bytes must be positive")

    @property
    def capacity_bytes(self) -> int:
        return self.capacity_gib * (1024**3)

    @property
    def bytes_per_cell(self) -> int:
        return self.cell_bits // 8

    @property
    def total_cells(self) -> int:
        return self.capacity_bytes // self.bytes_per_cell

    @property
    def cubic_side_cells(self) -> int:
        side = max(1, int(round(self.total_cells ** (1.0 / 3.0))))
        while side**3 < self.total_cells:
            side += 1
        while (side - 1) ** 3 >= self.total_cells:
            side -= 1
        return side

    def linear_cell_to_xyz(self, cell_index: int) -> Tuple[int, int, int]:
        if not 0 <= cell_index < self.total_cells:
            raise ValueError("cell index is outside the virtual substrate")
        side = self.cubic_side_cells
        plane = side * side
        z, rem = divmod(cell_index, plane)
        y, x = divmod(rem, side)
        return x, y, z

    def metrics(self) -> Dict[str, Any]:
        side = self.cubic_side_cells
        return {
            "capacity_gib": self.capacity_gib,
            "capacity_bytes": self.capacity_bytes,
            "cell_bits": self.cell_bits,
            "bytes_per_cell": self.bytes_per_cell,
            "total_cells": self.total_cells,
            "logical_cube_side_cells": side,
            "logical_cube_cells": side**3,
            "chunk_bytes": self.chunk_bytes,
            "allocation_mode": "sparse_virtual",
            "resident_bytes_at_idle": 0,
        }


@dataclass(frozen=True)
class EncodingReceipt:
    operation: str
    format_version: int
    payload_bytes: int
    artifact_bytes: int
    chunk_count: int
    compression_ratio: float
    payload_sha256: str
    artifact_sha256: str
    virtual_capacity_gib: int
    cell_bits: int
    logical_cube_side_cells: int
    first_chunk_xyz: Tuple[int, int, int] | None
    last_chunk_xyz: Tuple[int, int, int] | None
    latency_ms: float
    status: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DecodeReceipt:
    operation: str
    payload_bytes: int
    chunk_count: int
    payload_sha256: str
    verified: bool
    latency_ms: float
    status: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Universal3DAutoEncoder:
    """Reversible chunked encoder mapped onto a sparse virtual 3D Q16.16 substrate.

    This is an operational storage/transport codec, not a trained neural network. The
    virtual 6400-GiB field describes the address space; only active payload chunks are
    materialized in memory or artifacts.
    """

    def __init__(self, spec: VirtualVolumeSpec | None = None, *, compression_level: int = 9):
        self.spec = spec or VirtualVolumeSpec()
        if not 0 <= compression_level <= 9:
            raise ValueError("compression_level must be between 0 and 9")
        self.compression_level = compression_level

    def _chunks(self, data: bytes) -> Iterator[bytes]:
        size = self.spec.chunk_bytes
        for offset in range(0, len(data), size):
            yield data[offset : offset + size]

    def _chunk_cell_index(self, chunk_index: int) -> int:
        cells_per_chunk = math.ceil(self.spec.chunk_bytes / self.spec.bytes_per_cell)
        cell_index = chunk_index * cells_per_chunk
        if cell_index >= self.spec.total_cells:
            raise ValueError("payload exceeds the configured virtual substrate")
        return cell_index

    def encode(self, payload: bytes) -> tuple[bytes, EncodingReceipt]:
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise TypeError("payload must be bytes-like")
        payload = bytes(payload)
        if len(payload) > self.spec.capacity_bytes:
            raise ValueError("payload exceeds the configured virtual capacity")

        started = time.perf_counter()
        payload_hash = hashlib.sha256(payload).hexdigest()
        chunks = list(self._chunks(payload))

        chunk_meta = []
        encoded_chunks = []
        for index, raw in enumerate(chunks):
            cell_index = self._chunk_cell_index(index)
            xyz = self.spec.linear_cell_to_xyz(cell_index)
            compressed = zlib.compress(raw, self.compression_level)
            encoded_chunks.append(compressed)
            chunk_meta.append(
                {
                    "index": index,
                    "xyz": xyz,
                    "raw_bytes": len(raw),
                    "encoded_bytes": len(compressed),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )

        header = {
            "format": "jarvisx-3d-volumetric-latent",
            "version": FORMAT_VERSION,
            "codec": "zlib",
            "compression_level": self.compression_level,
            "payload_bytes": len(payload),
            "payload_sha256": payload_hash,
            "virtual_volume": self.spec.metrics(),
            "chunks": chunk_meta,
        }
        header_bytes = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")

        artifact = bytearray(MAGIC)
        artifact.extend(_HEADER_LEN.pack(len(header_bytes)))
        artifact.extend(header_bytes)
        for compressed in encoded_chunks:
            artifact.extend(_CHUNK_LEN.pack(len(compressed)))
            artifact.extend(compressed)
        artifact_bytes = bytes(artifact)

        artifact_hash = hashlib.sha256(artifact_bytes).hexdigest()
        ratio = (len(payload) / len(artifact_bytes)) if artifact_bytes else 0.0
        first_xyz = tuple(chunk_meta[0]["xyz"]) if chunk_meta else None
        last_xyz = tuple(chunk_meta[-1]["xyz"]) if chunk_meta else None
        receipt = EncodingReceipt(
            operation="ENCODE_3D_VOLUMETRIC",
            format_version=FORMAT_VERSION,
            payload_bytes=len(payload),
            artifact_bytes=len(artifact_bytes),
            chunk_count=len(chunk_meta),
            compression_ratio=ratio,
            payload_sha256=payload_hash,
            artifact_sha256=artifact_hash,
            virtual_capacity_gib=self.spec.capacity_gib,
            cell_bits=self.spec.cell_bits,
            logical_cube_side_cells=self.spec.cubic_side_cells,
            first_chunk_xyz=first_xyz,
            last_chunk_xyz=last_xyz,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            status="ENCODED_VERIFIABLE",
        )
        return artifact_bytes, receipt

    def decode(self, artifact: bytes) -> tuple[bytes, DecodeReceipt]:
        if not isinstance(artifact, (bytes, bytearray, memoryview)):
            raise TypeError("artifact must be bytes-like")
        artifact = bytes(artifact)
        started = time.perf_counter()

        if len(artifact) < len(MAGIC) + _HEADER_LEN.size or not artifact.startswith(MAGIC):
            raise ArtifactError("invalid volumetric artifact magic")

        cursor = len(MAGIC)
        (header_len,) = _HEADER_LEN.unpack_from(artifact, cursor)
        cursor += _HEADER_LEN.size
        header_end = cursor + header_len
        if header_end > len(artifact):
            raise ArtifactError("truncated volumetric artifact header")

        try:
            header = json.loads(artifact[cursor:header_end].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ArtifactError("invalid volumetric artifact header") from exc
        cursor = header_end

        if header.get("version") != FORMAT_VERSION or header.get("codec") != "zlib":
            raise ArtifactError("unsupported volumetric artifact format")

        reconstructed = bytearray()
        chunk_meta = header.get("chunks")
        if not isinstance(chunk_meta, list):
            raise ArtifactError("artifact chunk table is missing")

        for expected_index, meta in enumerate(chunk_meta):
            if cursor + _CHUNK_LEN.size > len(artifact):
                raise ArtifactError("truncated chunk length")
            (encoded_len,) = _CHUNK_LEN.unpack_from(artifact, cursor)
            cursor += _CHUNK_LEN.size
            chunk_end = cursor + encoded_len
            if chunk_end > len(artifact):
                raise ArtifactError("truncated encoded chunk")
            encoded = artifact[cursor:chunk_end]
            cursor = chunk_end
            try:
                raw = zlib.decompress(encoded)
            except zlib.error as exc:
                raise ArtifactError(f"chunk {expected_index} decompression failed") from exc
            if meta.get("index") != expected_index:
                raise ArtifactError("chunk index sequence is invalid")
            if len(raw) != meta.get("raw_bytes"):
                raise ArtifactError(f"chunk {expected_index} size mismatch")
            if hashlib.sha256(raw).hexdigest() != meta.get("sha256"):
                raise ArtifactError(f"chunk {expected_index} hash mismatch")
            reconstructed.extend(raw)

        if cursor != len(artifact):
            raise ArtifactError("artifact contains trailing bytes")

        payload = bytes(reconstructed)
        expected_size = header.get("payload_bytes")
        expected_hash = header.get("payload_sha256")
        payload_hash = hashlib.sha256(payload).hexdigest()
        if len(payload) != expected_size or payload_hash != expected_hash:
            raise ArtifactError("reconstructed payload failed end-to-end verification")

        receipt = DecodeReceipt(
            operation="DECODE_3D_VOLUMETRIC",
            payload_bytes=len(payload),
            chunk_count=len(chunk_meta),
            payload_sha256=payload_hash,
            verified=True,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            status="RECONSTRUCTED_BIT_EXACT",
        )
        return payload, receipt

    def self_test(self) -> Dict[str, Any]:
        sample = b"Codex_vOmegaXi_3D_Matrix_Stream_Simulation_Data" * 128
        artifact, encoded = self.encode(sample)
        restored, decoded = self.decode(artifact)
        return {
            "ok": restored == sample,
            "grid": self.spec.metrics(),
            "encode": encoded.to_dict(),
            "decode": decoded.to_dict(),
        }
