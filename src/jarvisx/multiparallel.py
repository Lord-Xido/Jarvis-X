"""Bounded deterministic reference for the Jarvis-X 3D multiparallel pipeline.

The module implements a closed set of serializable pipeline stages, deterministic
package ordering, framed compression artifacts, immutable branch snapshots and a
seeded topology search.  It is a Layer 5 research subsystem: none of its results are
authoritative ``CodexVM`` or ``SystemRuntime`` state.

The reference deliberately maps source code into a read-only 3D observation.  Axis
transforms never rewrite source text because line rotation or reordering is not a
general semantics-preserving Python transform.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import keyword
import math
import random
import struct
import time
import tokenize
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from enum import Enum
from threading import RLock
from typing import Iterable, Sequence, Union


_FRAME_MAGIC = b"JXMP"
_FRAME_VERSION = 1
_MAX_TOPOLOGY_NODES = 32
_DEFAULT_DECODE_LIMIT = 64 * 1024 * 1024


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


@dataclass(frozen=True, order=True)
class Vector3:
    """One finite three-dimensional vector."""

    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", _require_finite("x", self.x))
        object.__setattr__(self, "y", _require_finite("y", self.y))
        object.__setattr__(self, "z", _require_finite("z", self.z))

    def transform(self, axis_order: str = "xyz", scale: float = 1.0) -> "Vector3":
        """Return an axis permutation followed by a finite uniform scale."""

        if len(axis_order) != 3 or set(axis_order) != {"x", "y", "z"}:
            raise ValueError("axis_order must be a permutation of 'xyz'")
        factor = _require_finite("scale", scale)
        if factor == 0.0:
            raise ValueError("scale cannot be zero")
        values = {"x": self.x, "y": self.y, "z": self.z}
        return Vector3(
            values[axis_order[0]] * factor,
            values[axis_order[1]] * factor,
            values[axis_order[2]] * factor,
        )


@dataclass(frozen=True)
class VertexBatch:
    """Immutable vector batch used by the dependency-free reference path."""

    vertices: tuple[Vector3, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.vertices, tuple):
            object.__setattr__(self, "vertices", tuple(self.vertices))
        if not all(isinstance(vertex, Vector3) for vertex in self.vertices):
            raise TypeError("vertices must contain only Vector3 values")

    def __len__(self) -> int:
        return len(self.vertices)

    def transform(self, axis_order: str = "xyz", scale: float = 1.0) -> "VertexBatch":
        return VertexBatch(tuple(vertex.transform(axis_order, scale) for vertex in self.vertices))


@dataclass(frozen=True)
class Mesh:
    """Immutable triangular mesh contract.

    Meshes are processed as one package in v1 so that face indices cannot be broken by
    vertex chunking.
    """

    vertices: VertexBatch
    faces: tuple[tuple[int, int, int], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.vertices, VertexBatch):
            raise TypeError("vertices must be a VertexBatch")
        if not isinstance(self.faces, tuple):
            object.__setattr__(self, "faces", tuple(self.faces))
        vertex_count = len(self.vertices)
        for face in self.faces:
            if not isinstance(face, tuple) or len(face) != 3:
                raise TypeError("faces must be integer index triples")
            for index in face:
                if isinstance(index, bool) or not isinstance(index, int):
                    raise TypeError("face indices must be integers")
                if index < 0 or index >= vertex_count:
                    raise ValueError("face index lies outside the vertex batch")

    def transform(self, axis_order: str = "xyz", scale: float = 1.0) -> "Mesh":
        return Mesh(self.vertices.transform(axis_order, scale), self.faces)


class DataKind(str, Enum):
    TEXT = "text"
    BYTES = "bytes"
    VERTICES = "vertices"
    MESH = "mesh"


class Compression(str, Enum):
    NONE = "none"
    ZLIB = "zlib"


PipelineValue = Union[str, bytes, VertexBatch, Mesh, "EncodedChunk", "FramedArtifact"]


def _kind_code(kind: DataKind) -> int:
    return {
        DataKind.TEXT: 1,
        DataKind.BYTES: 2,
        DataKind.VERTICES: 3,
        DataKind.MESH: 4,
    }[kind]


def _kind_from_code(code: int) -> DataKind:
    for kind in DataKind:
        if _kind_code(kind) == code:
            return kind
    raise ValueError(f"unsupported framed data kind {code}")


def _raw_encode(value: Union[str, bytes, VertexBatch, Mesh]) -> tuple[DataKind, bytes]:
    if isinstance(value, str):
        return DataKind.TEXT, value.encode("utf-8")
    if isinstance(value, bytes):
        return DataKind.BYTES, value
    if isinstance(value, VertexBatch):
        payload = bytearray(struct.pack(">Q", len(value.vertices)))
        for vertex in value.vertices:
            payload.extend(struct.pack(">ddd", vertex.x, vertex.y, vertex.z))
        return DataKind.VERTICES, bytes(payload)
    if isinstance(value, Mesh):
        payload = bytearray(
            struct.pack(">QQ", len(value.vertices.vertices), len(value.faces))
        )
        for vertex in value.vertices.vertices:
            payload.extend(struct.pack(">ddd", vertex.x, vertex.y, vertex.z))
        for face in value.faces:
            payload.extend(struct.pack(">III", *face))
        return DataKind.MESH, bytes(payload)
    raise TypeError("pipeline values must be text, bytes, VertexBatch or Mesh")


def _raw_decode(kind: DataKind, payload: bytes) -> Union[str, bytes, VertexBatch, Mesh]:
    if kind is DataKind.TEXT:
        try:
            return payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("text payload is not valid UTF-8") from exc
    if kind is DataKind.BYTES:
        return payload
    if kind is DataKind.VERTICES:
        if len(payload) < 8:
            raise ValueError("vertex payload is truncated")
        (count,) = struct.unpack_from(">Q", payload, 0)
        expected = 8 + count * 24
        if expected != len(payload):
            raise ValueError("vertex payload length does not match its count")
        vertices_list: list[Vector3] = []
        for index in range(count):
            x, y, z = struct.unpack_from(">ddd", payload, 8 + index * 24)
            vertices_list.append(Vector3(x, y, z))
        vertices = tuple(vertices_list)
        return VertexBatch(vertices)
    if kind is DataKind.MESH:
        if len(payload) < 16:
            raise ValueError("mesh payload is truncated")
        vertex_count, face_count = struct.unpack_from(">QQ", payload, 0)
        expected = 16 + vertex_count * 24 + face_count * 12
        if expected != len(payload):
            raise ValueError("mesh payload length does not match its counts")
        mesh_vertices: list[Vector3] = []
        for index in range(vertex_count):
            x, y, z = struct.unpack_from(">ddd", payload, 16 + index * 24)
            mesh_vertices.append(Vector3(x, y, z))
        vertices = tuple(mesh_vertices)
        face_offset = 16 + vertex_count * 24
        face_values: list[tuple[int, int, int]] = []
        for index in range(face_count):
            a, b, c = struct.unpack_from(">III", payload, face_offset + index * 12)
            face_values.append((a, b, c))
        faces = tuple(face_values)
        return Mesh(VertexBatch(vertices), faces)
    raise ValueError(f"unsupported data kind {kind!r}")


def _bounded_zlib_decode(payload: bytes, expected_size: int, limit: int) -> bytes:
    if expected_size < 0 or expected_size > limit:
        raise ValueError("declared raw size exceeds the decode limit")
    decoder = zlib.decompressobj()
    try:
        raw = decoder.decompress(payload, expected_size + 1)
    except zlib.error as exc:
        raise ValueError("compressed payload is invalid") from exc
    if len(raw) != expected_size:
        raise ValueError("compressed payload does not match its declared raw size")
    if not decoder.eof or decoder.unconsumed_tail or decoder.unused_data:
        raise ValueError("compressed payload has trailing or unconsumed data")
    return raw


@dataclass(frozen=True)
class EncodedChunk:
    """One independently verifiable encoded package."""

    kind: DataKind
    payload: bytes
    raw_size: int
    raw_sha256: str
    compression: Compression = Compression.NONE

    def __post_init__(self) -> None:
        if not isinstance(self.kind, DataKind):
            raise TypeError("kind must be a DataKind")
        if not isinstance(self.payload, bytes):
            raise TypeError("payload must be bytes")
        if isinstance(self.raw_size, bool) or not isinstance(self.raw_size, int):
            raise TypeError("raw_size must be an integer")
        if self.raw_size < 0:
            raise ValueError("raw_size must be non-negative")
        if (
            not isinstance(self.raw_sha256, str)
            or len(self.raw_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.raw_sha256)
        ):
            raise ValueError("raw_sha256 must be a lowercase SHA-256 digest")
        if not isinstance(self.compression, Compression):
            raise TypeError("compression must be a Compression value")
        if self.compression is Compression.NONE:
            if len(self.payload) != self.raw_size or _sha256(self.payload) != self.raw_sha256:
                raise ValueError("uncompressed payload does not match its integrity fields")

    @classmethod
    def encode(cls, value: Union[str, bytes, VertexBatch, Mesh]) -> "EncodedChunk":
        kind, raw = _raw_encode(value)
        return cls(kind, raw, len(raw), _sha256(raw))

    def compress(self, level: int = 6) -> "EncodedChunk":
        if isinstance(level, bool) or not isinstance(level, int) or not 0 <= level <= 9:
            raise ValueError("zlib compression level must be an integer in [0, 9]")
        raw = self.raw_bytes()
        return EncodedChunk(
            kind=self.kind,
            payload=zlib.compress(raw, level),
            raw_size=len(raw),
            raw_sha256=self.raw_sha256,
            compression=Compression.ZLIB,
        )

    def raw_bytes(self, max_output_bytes: int = _DEFAULT_DECODE_LIMIT) -> bytes:
        if self.raw_size > max_output_bytes:
            raise ValueError("chunk exceeds the decode limit")
        if self.compression is Compression.NONE:
            raw = self.payload
        elif self.compression is Compression.ZLIB:
            raw = _bounded_zlib_decode(self.payload, self.raw_size, max_output_bytes)
        else:  # pragma: no cover - Enum construction prevents this branch.
            raise ValueError(f"unsupported compression {self.compression!r}")
        if _sha256(raw) != self.raw_sha256:
            raise ValueError("chunk integrity verification failed")
        return raw

    def decode(
        self, max_output_bytes: int = _DEFAULT_DECODE_LIMIT
    ) -> Union[str, bytes, VertexBatch, Mesh]:
        return _raw_decode(self.kind, self.raw_bytes(max_output_bytes))

    @property
    def payload_sha256(self) -> str:
        return _sha256(self.payload)


@dataclass(frozen=True)
class FramedArtifact:
    """Ordered multi-package binary artifact with strict length framing."""

    kind: DataKind
    chunks: tuple[EncodedChunk, ...]

    def __post_init__(self) -> None:
        if not self.chunks:
            raise ValueError("framed artifact requires at least one chunk")
        if any(chunk.kind is not self.kind for chunk in self.chunks):
            raise ValueError("all framed chunks must use the declared data kind")

    def to_bytes(self) -> bytes:
        frame = bytearray(_FRAME_MAGIC)
        frame.extend(struct.pack(">BBI", _FRAME_VERSION, _kind_code(self.kind), len(self.chunks)))
        for chunk in self.chunks:
            compression_code = 0 if chunk.compression is Compression.NONE else 1
            frame.extend(struct.pack(">BQ", compression_code, chunk.raw_size))
            frame.extend(bytes.fromhex(chunk.raw_sha256))
            frame.extend(struct.pack(">Q", len(chunk.payload)))
            frame.extend(chunk.payload)
        return bytes(frame)

    @classmethod
    def from_bytes(
        cls,
        frame: bytes,
        *,
        max_chunks: int = 256,
        max_frame_bytes: int = _DEFAULT_DECODE_LIMIT,
        max_output_bytes: int = _DEFAULT_DECODE_LIMIT,
    ) -> "FramedArtifact":
        if not isinstance(frame, bytes):
            raise TypeError("frame must be bytes")
        if len(frame) > max_frame_bytes:
            raise ValueError("frame exceeds the configured byte limit")
        header_size = len(_FRAME_MAGIC) + 6
        if len(frame) < header_size or not frame.startswith(_FRAME_MAGIC):
            raise ValueError("invalid multiparallel frame magic")
        version, kind_code, count = struct.unpack_from(">BBI", frame, len(_FRAME_MAGIC))
        if version != _FRAME_VERSION:
            raise ValueError(f"unsupported multiparallel frame version {version}")
        if count < 1 or count > max_chunks:
            raise ValueError("framed chunk count is outside the configured bound")
        kind = _kind_from_code(kind_code)
        offset = header_size
        chunks: list[EncodedChunk] = []
        total_raw = 0
        for _ in range(count):
            metadata_size = 1 + 8 + 32 + 8
            if offset + metadata_size > len(frame):
                raise ValueError("multiparallel frame metadata is truncated")
            compression_code, raw_size = struct.unpack_from(">BQ", frame, offset)
            offset += 9
            raw_sha256 = frame[offset : offset + 32].hex()
            offset += 32
            (payload_size,) = struct.unpack_from(">Q", frame, offset)
            offset += 8
            if payload_size > max_frame_bytes or offset + payload_size > len(frame):
                raise ValueError("multiparallel frame payload is truncated or oversized")
            payload = frame[offset : offset + payload_size]
            offset += payload_size
            if compression_code == 0:
                compression = Compression.NONE
            elif compression_code == 1:
                compression = Compression.ZLIB
            else:
                raise ValueError("unsupported framed compression code")
            total_raw += raw_size
            if total_raw > max_output_bytes:
                raise ValueError("framed artifact exceeds the decoded-output limit")
            chunks.append(EncodedChunk(kind, payload, raw_size, raw_sha256, compression))
        if offset != len(frame):
            raise ValueError("multiparallel frame contains trailing bytes")
        artifact = cls(kind, tuple(chunks))
        artifact.decode(max_output_bytes=max_output_bytes)
        return artifact

    def decode(
        self, max_output_bytes: int = _DEFAULT_DECODE_LIMIT
    ) -> Union[str, bytes, VertexBatch, Mesh]:
        total = sum(chunk.raw_size for chunk in self.chunks)
        if total > max_output_bytes:
            raise ValueError("framed artifact exceeds the decoded-output limit")
        decoded = tuple(chunk.decode(max_output_bytes) for chunk in self.chunks)
        if self.kind is DataKind.TEXT:
            return "".join(value for value in decoded if isinstance(value, str))
        if self.kind is DataKind.BYTES:
            return b"".join(value for value in decoded if isinstance(value, bytes))
        if self.kind is DataKind.VERTICES:
            vertices: list[Vector3] = []
            for value in decoded:
                if not isinstance(value, VertexBatch):
                    raise TypeError("decoded vertex artifact contains an incompatible chunk")
                vertices.extend(value.vertices)
            return VertexBatch(tuple(vertices))
        if self.kind is DataKind.MESH:
            if len(decoded) != 1 or not isinstance(decoded[0], Mesh):
                raise ValueError("mesh artifacts must contain exactly one complete mesh chunk")
            return decoded[0]
        raise ValueError(f"unsupported artifact kind {self.kind!r}")

    @property
    def digest_sha256(self) -> str:
        return _sha256(self.to_bytes())

    @property
    def compression_ratio(self) -> float:
        raw = sum(chunk.raw_size for chunk in self.chunks)
        payload = sum(len(chunk.payload) for chunk in self.chunks)
        return payload / raw if raw else 1.0


class StageKind(str, Enum):
    LOAD = "load"
    TRANSFORM = "transform"
    ENCODE = "encode"
    COMPRESS = "compress"
    DECOMPRESS = "decompress"
    DECODE = "decode"
    VERIFY = "verify"


ParameterValue = Union[str, int, float, bool]


@dataclass(frozen=True)
class PipelineNode:
    """One closed, serializable processing stage."""

    node_id: str
    kind: StageKind
    parameters: tuple[tuple[str, ParameterValue], ...] = ()

    def __post_init__(self) -> None:
        if not self.node_id or len(self.node_id) > 80:
            raise ValueError("node_id must contain 1 to 80 characters")
        if not isinstance(self.kind, StageKind):
            raise TypeError("kind must be a StageKind")
        names: set[str] = set()
        for name, value in self.parameters:
            if not name or name in names:
                raise ValueError("node parameter names must be non-empty and unique")
            names.add(name)
            if not isinstance(value, (str, int, float, bool)):
                raise TypeError("node parameters must be JSON scalar values")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("floating node parameters must be finite")

    def parameter(self, name: str, default: ParameterValue) -> ParameterValue:
        return dict(self.parameters).get(name, default)


@dataclass(frozen=True)
class PipelineTopology:
    """Validated v1 linear DAG plus bounded execution hints."""

    nodes: tuple[PipelineNode, ...]
    edges: tuple[tuple[str, str], ...]
    parallelism: int = 4
    batch_size: int = 64
    compression_level: int = 6
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("only topology schema version 1 is supported")
        if not self.nodes or len(self.nodes) > _MAX_TOPOLOGY_NODES:
            raise ValueError(f"topology must contain 1 to {_MAX_TOPOLOGY_NODES} nodes")
        node_ids = tuple(node.node_id for node in self.nodes)
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("topology node identifiers must be unique")
        if isinstance(self.parallelism, bool) or not isinstance(self.parallelism, int):
            raise TypeError("parallelism must be an integer")
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int):
            raise TypeError("batch_size must be an integer")
        if self.parallelism < 1 or self.batch_size < 1:
            raise ValueError("parallelism and batch_size must be positive")
        if (
            isinstance(self.compression_level, bool)
            or not isinstance(self.compression_level, int)
            or not 0 <= self.compression_level <= 9
        ):
            raise ValueError("compression_level must be an integer in [0, 9]")

        valid_ids = set(node_ids)
        if len(set(self.edges)) != len(self.edges):
            raise ValueError("topology edges must be unique")
        incoming = {node_id: 0 for node_id in node_ids}
        outgoing: dict[str, str] = {}
        for source, target in self.edges:
            if source not in valid_ids or target not in valid_ids:
                raise ValueError("topology edge references an unknown node")
            if source == target:
                raise ValueError("topology cannot contain self-edges")
            incoming[target] += 1
            if incoming[target] > 1 or source in outgoing:
                raise ValueError("v1 topology must be a single linear DAG")
            outgoing[source] = target

        if len(self.nodes) == 1:
            if self.edges:
                raise ValueError("single-node topology cannot contain edges")
            return
        sources = [node_id for node_id, count in incoming.items() if count == 0]
        sinks = [node_id for node_id in node_ids if node_id not in outgoing]
        if len(sources) != 1 or len(sinks) != 1 or len(self.edges) != len(self.nodes) - 1:
            raise ValueError("v1 topology must have one source, one sink and one path")
        visited: list[str] = []
        current = sources[0]
        while current not in visited:
            visited.append(current)
            if current not in outgoing:
                break
            current = outgoing[current]
        if len(visited) != len(self.nodes) or visited[-1] != sinks[0]:
            raise ValueError("topology contains a cycle or disconnected node")

    @property
    def ordered_nodes(self) -> tuple[PipelineNode, ...]:
        by_id = {node.node_id: node for node in self.nodes}
        if len(self.nodes) == 1:
            return self.nodes
        incoming = {node.node_id: 0 for node in self.nodes}
        outgoing: dict[str, str] = {}
        for source, target in self.edges:
            incoming[target] += 1
            outgoing[source] = target
        current = next(node_id for node_id, count in incoming.items() if count == 0)
        ordered: list[PipelineNode] = []
        while True:
            ordered.append(by_id[current])
            if current not in outgoing:
                return tuple(ordered)
            current = outgoing[current]

    @property
    def digest_sha256(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "nodes": [
                {
                    "id": node.node_id,
                    "kind": node.kind.value,
                    "parameters": sorted(node.parameters),
                }
                for node in self.ordered_nodes
            ],
            "edges": list(self.edges),
            "parallelism": self.parallelism,
            "batch_size": self.batch_size,
            "compression_level": self.compression_level,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return _sha256(encoded)


def topology_from_stages(
    stages: Sequence[StageKind],
    *,
    parallelism: int = 4,
    batch_size: int = 64,
    compression_level: int = 6,
) -> PipelineTopology:
    if not stages:
        raise ValueError("at least one pipeline stage is required")
    nodes = tuple(
        PipelineNode(f"stage-{index:02d}-{kind.value}", kind)
        for index, kind in enumerate(stages)
    )
    edges = tuple(
        (nodes[index].node_id, nodes[index + 1].node_id)
        for index in range(len(nodes) - 1)
    )
    return PipelineTopology(
        nodes,
        edges,
        parallelism=parallelism,
        batch_size=batch_size,
        compression_level=compression_level,
    )


def default_topology(
    *, parallelism: int = 4, batch_size: int = 64, compression_level: int = 6
) -> PipelineTopology:
    return topology_from_stages(
        (StageKind.LOAD, StageKind.TRANSFORM, StageKind.ENCODE, StageKind.COMPRESS),
        parallelism=parallelism,
        batch_size=batch_size,
        compression_level=compression_level,
    )


@dataclass(frozen=True)
class RuntimeLimits:
    """Resident and execution ceilings for the reference implementation."""

    max_workers: int = 8
    max_packages: int = 256
    max_nodes: int = 16
    max_input_bytes: int = 16 * 1024 * 1024
    max_output_bytes: int = 64 * 1024 * 1024
    max_vertices: int = 1_000_000
    max_branches: int = 128
    max_population: int = 64
    max_generations: int = 64
    max_batch_size: int = 1_048_576

    def __post_init__(self) -> None:
        for name in (
            "max_workers",
            "max_packages",
            "max_nodes",
            "max_input_bytes",
            "max_output_bytes",
            "max_vertices",
            "max_branches",
            "max_population",
            "max_generations",
            "max_batch_size",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class WorkPackage:
    package_id: str
    sequence: int
    priority: int
    start_node: str
    payload: Union[str, bytes, VertexBatch, Mesh]


@dataclass(frozen=True)
class PackageReceipt:
    package_id: str
    sequence: int
    success: bool
    output: PipelineValue | None
    output_digest: str
    stages: tuple[str, ...]
    elapsed_ns: int
    error_type: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class PipelineStats:
    package_count: int
    successful_packages: int
    worker_count: int
    backend: str
    codec_runtime_version: str
    elapsed_ns: int
    throughput_packages_per_second: float
    input_bytes: int
    output_bytes: int
    compression_ratio: float


@dataclass(frozen=True)
class PipelineRun:
    run_id: str
    topology_digest: str
    input_digest: str
    success: bool
    receipts: tuple[PackageReceipt, ...]
    output: PipelineValue | None
    stats: PipelineStats


def _value_bytes(value: PipelineValue) -> bytes:
    if isinstance(value, EncodedChunk):
        return value.payload
    if isinstance(value, FramedArtifact):
        return value.to_bytes()
    _, raw = _raw_encode(value)
    return raw


def _value_digest(value: PipelineValue) -> str:
    if isinstance(value, EncodedChunk):
        payload = (
            value.kind.value.encode("ascii")
            + b"\0"
            + value.compression.value.encode("ascii")
            + b"\0"
            + value.payload
        )
        return _sha256(payload)
    if isinstance(value, FramedArtifact):
        return value.digest_sha256
    kind, raw = _raw_encode(value)
    return _sha256(kind.value.encode("ascii") + b"\0" + raw)


def _apply_stage(
    value: PipelineValue,
    node: PipelineNode,
    topology: PipelineTopology,
    max_output_bytes: int,
) -> PipelineValue:
    if node.kind is StageKind.LOAD:
        return value
    if node.kind is StageKind.TRANSFORM:
        axis_order = str(node.parameter("axis_order", "xyz"))
        scale = float(node.parameter("scale", 1.0))
        if isinstance(value, VertexBatch):
            return value.transform(axis_order, scale)
        if isinstance(value, Mesh):
            return value.transform(axis_order, scale)
        if isinstance(value, (str, bytes)):
            # Code and opaque bytes are intentionally unchanged by spatial transforms.
            return value
        raise TypeError("transform stage requires raw text, bytes, vertices or mesh")
    if node.kind is StageKind.ENCODE:
        if not isinstance(value, (str, bytes, VertexBatch, Mesh)):
            raise TypeError("encode stage requires an unencoded value")
        return EncodedChunk.encode(value)
    if node.kind is StageKind.COMPRESS:
        if not isinstance(value, EncodedChunk):
            raise TypeError("compress stage requires an encoded chunk")
        level = int(node.parameter("level", topology.compression_level))
        return value.compress(level)
    if node.kind is StageKind.DECOMPRESS:
        if not isinstance(value, EncodedChunk):
            raise TypeError("decompress stage requires an encoded chunk")
        raw = value.raw_bytes(max_output_bytes)
        return EncodedChunk(value.kind, raw, len(raw), value.raw_sha256)
    if node.kind is StageKind.DECODE:
        if not isinstance(value, EncodedChunk):
            raise TypeError("decode stage requires an encoded chunk")
        return value.decode(max_output_bytes)
    if node.kind is StageKind.VERIFY:
        if isinstance(value, EncodedChunk):
            value.raw_bytes(max_output_bytes)
        elif isinstance(value, FramedArtifact):
            value.decode(max_output_bytes)
        else:
            _raw_encode(value)
        return value
    raise RuntimeError(f"unhandled stage {node.kind!r}")


def _process_single(
    package: WorkPackage,
    nodes: tuple[PipelineNode, ...],
    topology: PipelineTopology,
    max_output_bytes: int,
) -> PackageReceipt:
    started = time.perf_counter_ns()
    completed: list[str] = []
    try:
        value: PipelineValue = package.payload
        for node in nodes:
            value = _apply_stage(value, node, topology, max_output_bytes)
            completed.append(node.node_id)
        digest = _value_digest(value)
        return PackageReceipt(
            package.package_id,
            package.sequence,
            True,
            value,
            digest,
            tuple(completed),
            time.perf_counter_ns() - started,
        )
    except Exception as exc:  # The receipt is the package transaction boundary.
        return PackageReceipt(
            package.package_id,
            package.sequence,
            False,
            None,
            "",
            tuple(completed),
            time.perf_counter_ns() - started,
            type(exc).__name__,
            str(exc)[:512],
        )


class ParallelPipeline:
    """Execute one validated topology with deterministic reconciliation."""

    def __init__(
        self, topology: PipelineTopology | None = None, limits: RuntimeLimits | None = None
    ) -> None:
        self.limits = limits or RuntimeLimits()
        self.topology = topology or default_topology(
            parallelism=min(4, self.limits.max_workers),
            batch_size=min(64, self.limits.max_batch_size),
        )
        self._validate_topology(self.topology)

    def _validate_topology(self, topology: PipelineTopology) -> None:
        if len(topology.nodes) > self.limits.max_nodes:
            raise ValueError("topology exceeds the configured node limit")
        if topology.parallelism > self.limits.max_workers:
            raise ValueError("topology parallelism exceeds the worker limit")
        if topology.batch_size > self.limits.max_batch_size:
            raise ValueError("topology batch_size exceeds the configured limit")

    def _validate_input(self, value: Union[str, bytes, VertexBatch, Mesh]) -> int:
        _, raw = _raw_encode(value)
        if len(raw) > self.limits.max_input_bytes:
            raise ValueError("input exceeds the configured byte limit")
        vertex_count = 0
        if isinstance(value, VertexBatch):
            vertex_count = len(value)
        elif isinstance(value, Mesh):
            vertex_count = len(value.vertices)
        if vertex_count > self.limits.max_vertices:
            raise ValueError("input exceeds the configured vertex limit")
        return len(raw)

    def _split(
        self, value: Union[str, bytes, VertexBatch, Mesh], workers: int
    ) -> tuple[Union[str, bytes, VertexBatch, Mesh], ...]:
        if isinstance(value, Mesh):
            return (value,)
        units = len(value) if not isinstance(value, VertexBatch) else len(value.vertices)
        if units == 0:
            return (value,)
        by_batch = max(1, math.ceil(units / self.topology.batch_size))
        package_count = min(workers, by_batch, units, self.limits.max_packages)
        package_count = max(1, package_count)
        base, remainder = divmod(units, package_count)
        chunks: list[Union[str, bytes, VertexBatch, Mesh]] = []
        offset = 0
        for sequence in range(package_count):
            width = base + (1 if sequence < remainder else 0)
            end = offset + width
            if isinstance(value, VertexBatch):
                chunks.append(VertexBatch(value.vertices[offset:end]))
            else:
                chunks.append(value[offset:end])
            offset = end
        return tuple(chunks)

    def _packages(
        self, value: Union[str, bytes, VertexBatch, Mesh], workers: int
    ) -> tuple[WorkPackage, ...]:
        input_digest = _value_digest(value)
        source_node = self.topology.ordered_nodes[0].node_id
        packages: list[WorkPackage] = []
        for sequence, chunk in enumerate(self._split(value, workers)):
            material = (
                b"jarvisx-multiparallel-package-v1\0"
                + self.topology.digest_sha256.encode("ascii")
                + input_digest.encode("ascii")
                + struct.pack(">Q", sequence)
                + _value_digest(chunk).encode("ascii")
            )
            packages.append(
                WorkPackage(
                    package_id=_sha256(material),
                    sequence=sequence,
                    priority=0,
                    start_node=source_node,
                    payload=chunk,
                )
            )
        return tuple(packages)

    @staticmethod
    def merge_receipts(receipts: Iterable[PackageReceipt]) -> PipelineValue:
        ordered = tuple(sorted(receipts, key=lambda receipt: receipt.sequence))
        if not ordered or any(not receipt.success or receipt.output is None for receipt in ordered):
            raise ValueError("only a non-empty set of successful receipts can be merged")
        outputs = tuple(receipt.output for receipt in ordered)
        first = outputs[0]
        if isinstance(first, EncodedChunk):
            if not all(isinstance(output, EncodedChunk) for output in outputs):
                raise TypeError("package outputs have incompatible types")
            chunks = tuple(output for output in outputs if isinstance(output, EncodedChunk))
            if any(chunk.kind is not first.kind for chunk in chunks):
                raise TypeError("encoded package outputs have incompatible data kinds")
            return FramedArtifact(first.kind, chunks)
        if isinstance(first, str):
            if not all(isinstance(output, str) for output in outputs):
                raise TypeError("package outputs have incompatible types")
            return "".join(output for output in outputs if isinstance(output, str))
        if isinstance(first, bytes):
            if not all(isinstance(output, bytes) for output in outputs):
                raise TypeError("package outputs have incompatible types")
            return b"".join(output for output in outputs if isinstance(output, bytes))
        if isinstance(first, VertexBatch):
            if not all(isinstance(output, VertexBatch) for output in outputs):
                raise TypeError("package outputs have incompatible types")
            return VertexBatch(
                tuple(
                    vertex
                    for output in outputs
                    if isinstance(output, VertexBatch)
                    for vertex in output.vertices
                )
            )
        if isinstance(first, Mesh):
            if len(outputs) != 1 or not isinstance(first, Mesh):
                raise ValueError("mesh output cannot be merged from multiple packages")
            return first
        if isinstance(first, FramedArtifact):
            if not all(isinstance(output, FramedArtifact) for output in outputs):
                raise TypeError("package outputs have incompatible types")
            artifacts = tuple(
                output for output in outputs if isinstance(output, FramedArtifact)
            )
            if any(artifact.kind is not first.kind for artifact in artifacts):
                raise TypeError("framed outputs have incompatible data kinds")
            return FramedArtifact(
                first.kind,
                tuple(chunk for artifact in artifacts for chunk in artifact.chunks),
            )
        raise TypeError("unsupported package output type")

    def run(
        self,
        value: Union[str, bytes, VertexBatch, Mesh],
        *,
        workers: int | None = None,
        backend: str = "sequential",
    ) -> PipelineRun:
        input_bytes = self._validate_input(value)
        worker_count = self.topology.parallelism if workers is None else workers
        if isinstance(worker_count, bool) or not isinstance(worker_count, int):
            raise TypeError("workers must be an integer")
        if worker_count < 1 or worker_count > self.limits.max_workers:
            raise ValueError("workers must lie inside the configured worker limit")
        if backend not in {"sequential", "process"}:
            raise ValueError("backend must be 'sequential' or 'process'")

        packages = self._packages(value, worker_count)
        nodes = self.topology.ordered_nodes
        started = time.perf_counter_ns()
        if backend == "sequential" or len(packages) == 1:
            receipts = tuple(
                _process_single(package, nodes, self.topology, self.limits.max_output_bytes)
                for package in packages
            )
        else:
            collected: list[PackageReceipt] = []
            with ProcessPoolExecutor(max_workers=min(worker_count, len(packages))) as executor:
                futures = {
                    executor.submit(
                        _process_single,
                        package,
                        nodes,
                        self.topology,
                        self.limits.max_output_bytes,
                    ): package
                    for package in packages
                }
                for future in as_completed(futures):
                    package = futures[future]
                    try:
                        collected.append(future.result())
                    except Exception as exc:
                        collected.append(
                            PackageReceipt(
                                package.package_id,
                                package.sequence,
                                False,
                                None,
                                "",
                                (),
                                0,
                                type(exc).__name__,
                                str(exc)[:512],
                            )
                        )
            receipts = tuple(collected)
        receipts = tuple(sorted(receipts, key=lambda receipt: receipt.sequence))
        elapsed_ns = time.perf_counter_ns() - started
        success = all(receipt.success for receipt in receipts)
        output: PipelineValue | None = None
        if success:
            try:
                output = self.merge_receipts(receipts)
                if len(_value_bytes(output)) > self.limits.max_output_bytes:
                    raise ValueError("merged output exceeds the configured byte limit")
                if isinstance(output, FramedArtifact):
                    output.decode(self.limits.max_output_bytes)
            except Exception as exc:
                success = False
                failed = PackageReceipt(
                    package_id="merge",
                    sequence=len(receipts),
                    success=False,
                    output=None,
                    output_digest="",
                    stages=(),
                    elapsed_ns=0,
                    error_type=type(exc).__name__,
                    error_message=str(exc)[:512],
                )
                receipts = receipts + (failed,)
                output = None

        output_bytes = len(_value_bytes(output)) if output is not None else 0
        if isinstance(output, FramedArtifact):
            compression_ratio = output.compression_ratio
        elif input_bytes:
            compression_ratio = output_bytes / input_bytes
        else:
            compression_ratio = 1.0
        throughput = len(packages) / (elapsed_ns / 1_000_000_000) if elapsed_ns else 0.0
        input_digest = _value_digest(value)
        receipt_material = "".join(
            f"{receipt.sequence}:{receipt.package_id}:{receipt.success}:{receipt.output_digest};"
            for receipt in receipts
        ).encode("utf-8")
        run_id = _sha256(
            b"jarvisx-multiparallel-run-v1\0"
            + self.topology.digest_sha256.encode("ascii")
            + input_digest.encode("ascii")
            + receipt_material
        )
        stats = PipelineStats(
            package_count=len(packages),
            successful_packages=sum(receipt.success for receipt in receipts),
            worker_count=min(worker_count, len(packages)),
            backend=backend,
            codec_runtime_version=zlib.ZLIB_RUNTIME_VERSION,
            elapsed_ns=elapsed_ns,
            throughput_packages_per_second=throughput,
            input_bytes=input_bytes,
            output_bytes=output_bytes,
            compression_ratio=compression_ratio,
        )
        return PipelineRun(
            run_id,
            self.topology.digest_sha256,
            input_digest,
            success,
            receipts,
            output,
            stats,
        )


@dataclass(frozen=True)
class CodePoint:
    line_number: int
    coordinate: Vector3
    line_sha256: str


@dataclass(frozen=True)
class CodeGeometry:
    """Read-only spatial observation derived from validated Python source."""

    source_sha256: str
    source_line_count: int
    points: tuple[CodePoint, ...]
    axis_order: str = "xyz"
    scale: float = 1.0

    def transform(self, axis_order: str = "xyz", scale: float = 1.0) -> "CodeGeometry":
        factor = _require_finite("scale", scale)
        if factor == 0.0:
            raise ValueError("scale cannot be zero")
        return CodeGeometry(
            self.source_sha256,
            self.source_line_count,
            tuple(
                replace(point, coordinate=point.coordinate.transform(axis_order, factor))
                for point in self.points
            ),
            axis_order,
            factor,
        )


class SpatialProcessor:
    """Map Python source into observational coordinates and transform typed assets."""

    @staticmethod
    def map_code(source: str, *, max_source_bytes: int = 4 * 1024 * 1024) -> CodeGeometry:
        if not isinstance(source, str):
            raise TypeError("source must be text")
        encoded = source.encode("utf-8")
        if len(encoded) > max_source_bytes:
            raise ValueError("source exceeds the configured byte limit")
        try:
            ast.parse(source)
        except SyntaxError as exc:
            raise ValueError("source must be syntactically valid Python") from exc

        identifier_counts: dict[int, int] = {}
        try:
            tokens = tokenize.generate_tokens(io.StringIO(source).readline)
            for token in tokens:
                if token.type == tokenize.NAME and not keyword.iskeyword(token.string):
                    identifier_counts[token.start[0]] = identifier_counts.get(token.start[0], 0) + 1
        except tokenize.TokenError as exc:
            raise ValueError("source tokenization failed") from exc

        lines = source.splitlines()
        points: list[CodePoint] = []
        for line_number, line in enumerate(lines, start=1):
            prefix = line[: len(line) - len(line.lstrip(" \t"))]
            indent = len(prefix.expandtabs(4))
            complexity = identifier_counts.get(line_number, 0)
            points.append(
                CodePoint(
                    line_number,
                    Vector3(float(line_number), float(indent), float(complexity)),
                    _sha256(line.encode("utf-8")),
                )
            )
        return CodeGeometry(_sha256(encoded), len(lines), tuple(points))

    @staticmethod
    def transform_asset(
        asset: Union[VertexBatch, Mesh], *, axis_order: str = "xyz", scale: float = 1.0
    ) -> Union[VertexBatch, Mesh]:
        if isinstance(asset, VertexBatch):
            return asset.transform(axis_order, scale)
        if isinstance(asset, Mesh):
            return asset.transform(axis_order, scale)
        raise TypeError("asset must be a VertexBatch or Mesh")


@dataclass(frozen=True)
class BranchSnapshot:
    branch_id: str
    name: str
    sequence: int
    payload: Union[str, bytes, VertexBatch, Mesh]
    payload_digest: str
    topology_digest: str
    run: PipelineRun | None = None


@dataclass(frozen=True)
class CandidateScore:
    topology: PipelineTopology
    fitness: float
    compression_ratio: float
    estimated_work_units: float
    observed_elapsed_ns: int
    success: bool
    error: str | None = None


@dataclass(frozen=True)
class GenerationReport:
    generation: int
    population_size: int
    best_fitness: float
    best_topology_digest: str


@dataclass(frozen=True)
class EvolutionConfig:
    generations: int = 5
    population_size: int = 10
    survivor_fraction: float = 0.5
    seed: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.generations, bool) or not isinstance(self.generations, int):
            raise TypeError("generations must be an integer")
        if isinstance(self.population_size, bool) or not isinstance(self.population_size, int):
            raise TypeError("population_size must be an integer")
        if self.generations < 1 or self.population_size < 2:
            raise ValueError("generations must be positive and population_size at least 2")
        fraction = _require_finite("survivor_fraction", self.survivor_fraction)
        if not 0.1 <= fraction <= 0.9:
            raise ValueError("survivor_fraction must lie in [0.1, 0.9]")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")


@dataclass(frozen=True)
class EvolutionResult:
    initial_topology_digest: str
    selected_topology: PipelineTopology
    best_score: CandidateScore
    history: tuple[GenerationReport, ...]
    promoted: bool = False


_SAFE_STAGE_TEMPLATES: tuple[tuple[StageKind, ...], ...] = (
    (StageKind.LOAD, StageKind.ENCODE),
    (StageKind.LOAD, StageKind.TRANSFORM, StageKind.ENCODE),
    (StageKind.LOAD, StageKind.ENCODE, StageKind.COMPRESS),
    (StageKind.LOAD, StageKind.TRANSFORM, StageKind.ENCODE, StageKind.COMPRESS),
    (
        StageKind.LOAD,
        StageKind.TRANSFORM,
        StageKind.ENCODE,
        StageKind.COMPRESS,
        StageKind.VERIFY,
    ),
)


class TopologyEvolution:
    """Seeded bounded search over a finite, type-safe topology family."""

    def __init__(self, limits: RuntimeLimits | None = None) -> None:
        self.limits = limits or RuntimeLimits()

    @staticmethod
    def _stages(topology: PipelineTopology) -> tuple[StageKind, ...]:
        return tuple(node.kind for node in topology.ordered_nodes)

    def _mutate(self, topology: PipelineTopology, rng: random.Random) -> PipelineTopology:
        stages = self._stages(topology)
        parallelism = topology.parallelism
        batch_size = topology.batch_size
        compression_level = topology.compression_level
        action = rng.randrange(4)
        if action == 0:
            parallelism = min(
                self.limits.max_workers,
                max(1, parallelism + rng.choice((-1, 1))),
            )
        elif action == 1:
            if rng.random() < 0.5:
                batch_size = max(1, batch_size // 2)
            else:
                batch_size = min(self.limits.max_batch_size, batch_size * 2)
        elif action == 2:
            compression_level = min(9, max(0, compression_level + rng.choice((-1, 1))))
        else:
            stages = rng.choice(_SAFE_STAGE_TEMPLATES)
        return topology_from_stages(
            stages,
            parallelism=parallelism,
            batch_size=batch_size,
            compression_level=compression_level,
        )

    def _crossover(
        self, left: PipelineTopology, right: PipelineTopology, rng: random.Random
    ) -> PipelineTopology:
        stages = self._stages(left if rng.random() < 0.5 else right)
        return topology_from_stages(
            stages,
            parallelism=rng.choice((left.parallelism, right.parallelism)),
            batch_size=rng.choice((left.batch_size, right.batch_size)),
            compression_level=rng.choice((left.compression_level, right.compression_level)),
        )

    def evaluate(
        self,
        topology: PipelineTopology,
        test_data: Union[str, bytes, VertexBatch, Mesh],
    ) -> CandidateScore:
        try:
            pipeline = ParallelPipeline(topology, self.limits)
            run = pipeline.run(test_data, workers=topology.parallelism, backend="sequential")
        except Exception as exc:
            return CandidateScore(
                topology,
                float("-inf"),
                float("inf"),
                float("inf"),
                0,
                False,
                f"{type(exc).__name__}: {exc}",
            )
        if not run.success:
            error = next(
                (
                    receipt.error_message
                    for receipt in run.receipts
                    if not receipt.success and receipt.error_message
                ),
                "candidate pipeline failed",
            )
            return CandidateScore(
                topology,
                float("-inf"),
                run.stats.compression_ratio,
                float("inf"),
                run.stats.elapsed_ns,
                False,
                error,
            )

        stage_weight = {
            StageKind.LOAD: 0.05,
            StageKind.TRANSFORM: 0.20,
            StageKind.ENCODE: 0.25,
            StageKind.COMPRESS: 1.00 + topology.compression_level * 0.08,
            StageKind.DECOMPRESS: 0.80,
            StageKind.DECODE: 0.30,
            StageKind.VERIFY: 0.20,
        }
        size = max(1, run.stats.input_bytes)
        active_parallelism = max(1, min(topology.parallelism, run.stats.package_count))
        estimated_work = (
            size * sum(stage_weight[node.kind] for node in topology.ordered_nodes)
        ) / active_parallelism
        estimated_work += run.stats.package_count * 32.0 + len(topology.nodes) * 16.0
        speed_score = 1.0 / (1.0 + estimated_work / size)
        compression_gain = max(-1.0, min(1.0, 1.0 - run.stats.compression_ratio))
        parallel_score = active_parallelism / max(1, run.stats.package_count)
        node_penalty = max(0, len(topology.nodes) - 4) * 0.02
        fitness = (
            0.50 * compression_gain
            + 0.30 * speed_score
            + 0.20 * parallel_score
            - node_penalty
        )
        return CandidateScore(
            topology,
            fitness,
            run.stats.compression_ratio,
            estimated_work,
            run.stats.elapsed_ns,
            True,
        )

    def evolve(
        self,
        base: PipelineTopology,
        test_data: Union[str, bytes, VertexBatch, Mesh],
        config: EvolutionConfig | None = None,
    ) -> EvolutionResult:
        selected_config = config or EvolutionConfig()
        if selected_config.population_size > self.limits.max_population:
            raise ValueError("population exceeds the configured limit")
        if selected_config.generations > self.limits.max_generations:
            raise ValueError("generations exceed the configured limit")
        ParallelPipeline(base, self.limits)

        rng = random.Random(selected_config.seed)
        population = [base]
        while len(population) < selected_config.population_size:
            population.append(self._mutate(base, rng))

        history: list[GenerationReport] = []
        best: CandidateScore | None = None
        for generation in range(selected_config.generations):
            scores = [self.evaluate(topology, test_data) for topology in population]
            scores.sort(key=lambda score: (-score.fitness, score.topology.digest_sha256))
            if not scores[0].success:
                raise RuntimeError("no admissible topology candidate completed successfully")
            generation_best = scores[0]
            if (
                best is None
                or generation_best.fitness > best.fitness
                or (
                    generation_best.fitness == best.fitness
                    and generation_best.topology.digest_sha256
                    < best.topology.digest_sha256
                )
            ):
                best = generation_best
            history.append(
                GenerationReport(
                    generation,
                    len(scores),
                    scores[0].fitness,
                    scores[0].topology.digest_sha256,
                )
            )
            survivor_count = max(
                2,
                min(
                    len(scores),
                    math.ceil(len(scores) * selected_config.survivor_fraction),
                ),
            )
            survivors = scores[:survivor_count]
            next_population = [score.topology for score in survivors]
            while len(next_population) < selected_config.population_size:
                left, right = rng.sample(survivors, 2)
                child = self._crossover(left.topology, right.topology, rng)
                if rng.random() < 0.8:
                    child = self._mutate(child, rng)
                next_population.append(child)
            population = next_population
        assert best is not None
        return EvolutionResult(base.digest_sha256, best.topology, best, tuple(history))


@dataclass(frozen=True)
class EngineStats:
    total_runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    packages_processed: int = 0
    branches_created: int = 0
    evolution_generations: int = 0
    best_fitness: float | None = None


class JarvisX3DEngine:
    """Central controller for the isolated multiparallel research subsystem."""

    def __init__(
        self,
        *,
        code: str | None = None,
        assets: Sequence[Union[VertexBatch, Mesh]] = (),
        topology: PipelineTopology | None = None,
        limits: RuntimeLimits | None = None,
    ) -> None:
        if code is not None and not isinstance(code, str):
            raise TypeError("code must be text when supplied")
        if not all(isinstance(asset, (VertexBatch, Mesh)) for asset in assets):
            raise TypeError("assets must contain VertexBatch or Mesh values")
        self.code = code
        self.assets = tuple(assets)
        self.limits = limits or RuntimeLimits()
        self._topology = topology or default_topology(
            parallelism=min(4, self.limits.max_workers),
            batch_size=min(64, self.limits.max_batch_size),
        )
        ParallelPipeline(self._topology, self.limits)
        self._last_committed_run: PipelineRun | None = None
        self._branches: dict[str, BranchSnapshot] = {}
        self._branch_sequence = 0
        self._stats = EngineStats()
        self._lock = RLock()

    @property
    def topology(self) -> PipelineTopology:
        with self._lock:
            return self._topology

    @property
    def last_committed_run(self) -> PipelineRun | None:
        with self._lock:
            return self._last_committed_run

    @property
    def stats(self) -> EngineStats:
        with self._lock:
            return self._stats

    def _default_input(self) -> Union[str, VertexBatch, Mesh]:
        if self.code is not None:
            return self.code
        if len(self.assets) == 1:
            return self.assets[0]
        raise ValueError("data is required when the engine has no unique default input")

    def process_parallel(
        self,
        data: Union[str, bytes, VertexBatch, Mesh, None] = None,
        *,
        num_workers: int | None = None,
        backend: str = "sequential",
    ) -> PipelineRun:
        with self._lock:
            supplied = self._default_input() if data is None else data
            pipeline = ParallelPipeline(self._topology, self.limits)
            run = pipeline.run(supplied, workers=num_workers, backend=backend)
            self._stats = replace(
                self._stats,
                total_runs=self._stats.total_runs + 1,
                successful_runs=self._stats.successful_runs + int(run.success),
                failed_runs=self._stats.failed_runs + int(not run.success),
                packages_processed=self._stats.packages_processed + run.stats.package_count,
            )
            if run.success:
                self._last_committed_run = run
            return run

    def spatial_process(
        self,
        source: str | None = None,
        *,
        axis_order: str = "xyz",
        scale: float = 1.0,
    ) -> CodeGeometry:
        selected = self.code if source is None else source
        if selected is None:
            raise ValueError("source is required when no code is loaded")
        geometry = SpatialProcessor.map_code(selected, max_source_bytes=self.limits.max_input_bytes)
        return geometry.transform(axis_order, scale)

    def create_branch(
        self, name: str, data: Union[str, bytes, VertexBatch, Mesh, None] = None
    ) -> str:
        if not isinstance(name, str) or not name.strip() or len(name) > 80:
            raise ValueError("branch name must contain 1 to 80 non-blank characters")
        with self._lock:
            if len(self._branches) >= self.limits.max_branches:
                raise ValueError("branch limit reached")
            supplied = self._default_input() if data is None else data
            ParallelPipeline(self._topology, self.limits)._validate_input(supplied)
            sequence = self._branch_sequence
            material = (
                b"jarvisx-multiparallel-branch-v1\0"
                + name.encode("utf-8")
                + struct.pack(">Q", sequence)
                + _value_digest(supplied).encode("ascii")
                + self._topology.digest_sha256.encode("ascii")
            )
            branch_id = _sha256(material)
            self._branch_sequence += 1
            self._branches[branch_id] = BranchSnapshot(
                branch_id,
                name,
                sequence,
                supplied,
                _value_digest(supplied),
                self._topology.digest_sha256,
            )
            self._stats = replace(
                self._stats, branches_created=self._stats.branches_created + 1
            )
            return branch_id

    def branch(self, branch_id: str) -> BranchSnapshot:
        with self._lock:
            try:
                return self._branches[branch_id]
            except KeyError as exc:
                raise KeyError("unknown branch identifier") from exc

    def process_branch(
        self,
        branch_id: str,
        *,
        num_workers: int | None = None,
        backend: str = "sequential",
    ) -> PipelineRun:
        with self._lock:
            snapshot = self.branch(branch_id)
            pipeline = ParallelPipeline(self._topology, self.limits)
            run = pipeline.run(snapshot.payload, workers=num_workers, backend=backend)
            self._stats = replace(
                self._stats,
                total_runs=self._stats.total_runs + 1,
                successful_runs=self._stats.successful_runs + int(run.success),
                failed_runs=self._stats.failed_runs + int(not run.success),
                packages_processed=self._stats.packages_processed + run.stats.package_count,
            )
            if run.success:
                self._branches[branch_id] = replace(snapshot, run=run)
            return run

    def merge_branches(self, branch_ids: Sequence[str]) -> PipelineValue:
        if not branch_ids:
            raise ValueError("at least one branch identifier is required")
        if len(set(branch_ids)) != len(branch_ids):
            raise ValueError("branch identifiers must be unique")
        with self._lock:
            snapshots = tuple(self.branch(branch_id) for branch_id in branch_ids)
            receipts: list[PackageReceipt] = []
            for index, snapshot in enumerate(snapshots):
                run = snapshot.run
                if run is None or not run.success or run.output is None:
                    raise ValueError("every branch must have a successful processed output")
                receipts.append(
                    PackageReceipt(
                        package_id=snapshot.branch_id,
                        sequence=index,
                        success=True,
                        output=run.output,
                        output_digest=_value_digest(run.output),
                        stages=("branch-merge",),
                        elapsed_ns=0,
                    )
                )
            return ParallelPipeline.merge_receipts(receipts)

    def auto_evolve(
        self,
        test_data: Union[str, bytes, VertexBatch, Mesh],
        config: EvolutionConfig | None = None,
    ) -> EvolutionResult:
        with self._lock:
            active_before = self._topology
            evolution = TopologyEvolution(self.limits)
            result = evolution.evolve(active_before, test_data, config)
            verification = ParallelPipeline(result.selected_topology, self.limits).run(
                test_data,
                workers=result.selected_topology.parallelism,
                backend="sequential",
            )
            if not verification.success:
                raise RuntimeError("selected topology failed promotion verification")
            self._topology = result.selected_topology
            self._stats = replace(
                self._stats,
                evolution_generations=self._stats.evolution_generations + len(result.history),
                best_fitness=result.best_score.fitness,
            )
            return replace(result, promoted=True)
