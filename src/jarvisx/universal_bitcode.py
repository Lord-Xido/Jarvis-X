"""Deterministic hierarchical bitcode container for arbitrary digital artifacts.

The runtime treats media as typed byte strings.  It preserves the representation
contract, chunks the byte string, applies a reversible per-chunk codec, and binds
the result with SHA-256 and a Merkle root.  It deliberately does not claim that
lossless byte transport is semantic understanding or cross-modal generation.
"""

from __future__ import annotations

import json
import math
import re
import zlib
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
from pathlib import PurePath
from struct import Struct
from typing import Any

MAGIC = b"JXUBIR1\x00"
FORMAT_VERSION = 1
SCHEMA = "jarvisx.universal-bitcode"
HEADER = Struct(">8sHHIQQ32s32s32s")
HEADER_SIZE = HEADER.size
DEFAULT_CHUNK_SIZE = 64 * 1024

_ALLOWED_FLAGS = 0
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_FORMAT_RE = re.compile(r"^[a-z0-9][a-z0-9._+-]{0,63}$")
_MEDIA_TYPE_RE = re.compile(r"^[a-z0-9][a-z0-9!#$&^_.+-]{0,126}/[a-z0-9][a-z0-9!#$&^_.+-]{0,126}$")


class BitcodeError(ValueError):
    """Base class for universal-bitcode validation failures."""


class BitcodeFormatError(BitcodeError):
    """Raised when a container or representation contract is malformed."""


class IntegrityError(BitcodeError):
    """Raised when a digest, size, or reconstruction invariant fails."""


class ResourceLimitError(BitcodeError):
    """Raised before an input can exceed an explicit runtime budget."""


class MediaKind(str, Enum):
    TEXT = "text"
    DOCUMENT = "document"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    SCENE_3D = "scene-3d"
    CODE = "code"
    MODEL = "model"
    ARCHIVE = "archive"
    BINARY = "binary"


@dataclass(frozen=True)
class BitcodeBudget:
    """Hard limits applied before allocation, parsing, or decompression."""

    max_input_bytes: int = 256 * 1024 * 1024
    max_manifest_bytes: int = 8 * 1024 * 1024
    max_metadata_bytes: int = 64 * 1024
    max_source_name_bytes: int = 4096
    max_chunks: int = 16_384
    max_chunk_size: int = 4 * 1024 * 1024

    def __post_init__(self) -> None:
        for name, value in (
            ("max_input_bytes", self.max_input_bytes),
            ("max_manifest_bytes", self.max_manifest_bytes),
            ("max_metadata_bytes", self.max_metadata_bytes),
            ("max_source_name_bytes", self.max_source_name_bytes),
            ("max_chunks", self.max_chunks),
            ("max_chunk_size", self.max_chunk_size),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


DEFAULT_BUDGET = BitcodeBudget()


def _canonical_json(value: object) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise BitcodeFormatError("value is not canonical JSON data") from exc
    return text.encode("utf-8")


def _normalized_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    raw = _canonical_json(dict(value))
    normalized = json.loads(raw.decode("utf-8"))
    if not isinstance(normalized, dict):
        raise BitcodeFormatError("metadata must be a JSON object")
    return normalized


def _require_int(value: object, name: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise BitcodeFormatError(f"{name} must be an integer >= {minimum}")
    return value


def _require_str(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise BitcodeFormatError(f"{name} must be a string")
    return value


def _require_digest(value: object, name: str) -> str:
    digest = _require_str(value, name)
    if not _DIGEST_RE.fullmatch(digest):
        raise BitcodeFormatError(f"{name} must be a lowercase SHA-256 digest")
    return digest


@dataclass(frozen=True)
class RepresentationContract:
    """The explicit interpretation contract carried beside an opaque byte string."""

    media_kind: MediaKind
    media_type: str
    format_name: str
    source_name: str = ""
    schema: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            kind = (
                self.media_kind
                if isinstance(self.media_kind, MediaKind)
                else MediaKind(str(self.media_kind))
            )
        except ValueError as exc:
            raise BitcodeFormatError("unsupported media_kind") from exc
        media_type = str(self.media_type).strip().lower()
        format_name = str(self.format_name).strip().lower()
        source_name = str(self.source_name)
        schema = None if self.schema is None else str(self.schema).strip()
        if not _MEDIA_TYPE_RE.fullmatch(media_type):
            raise BitcodeFormatError("media_type must be a type/subtype token")
        if not _FORMAT_RE.fullmatch(format_name):
            raise BitcodeFormatError("format_name contains unsupported characters")
        if "\x00" in source_name or any(ord(char) < 32 for char in source_name):
            raise BitcodeFormatError("source_name cannot contain control characters")
        if schema == "":
            raise BitcodeFormatError("schema cannot be blank")
        if schema is not None and ("\x00" in schema or len(schema.encode("utf-8")) > 4096):
            raise BitcodeFormatError("schema is invalid or too long")
        normalized = _normalized_mapping(self.metadata)
        object.__setattr__(self, "media_kind", kind)
        object.__setattr__(self, "media_type", media_type)
        object.__setattr__(self, "format_name", format_name)
        object.__setattr__(self, "source_name", source_name)
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "metadata", normalized)

    def as_dict(self) -> dict[str, object]:
        return {
            "format_name": self.format_name,
            "media_kind": self.media_kind.value,
            "media_type": self.media_type,
            "metadata": dict(self.metadata),
            "schema": self.schema,
            "source_name": self.source_name,
        }

    @classmethod
    def from_dict(cls, value: object) -> "RepresentationContract":
        if not isinstance(value, dict):
            raise BitcodeFormatError("contract must be an object")
        expected = {
            "format_name",
            "media_kind",
            "media_type",
            "metadata",
            "schema",
            "source_name",
        }
        if set(value) != expected:
            raise BitcodeFormatError("contract fields do not match format version 1")
        metadata = value["metadata"]
        if not isinstance(metadata, dict):
            raise BitcodeFormatError("contract metadata must be an object")
        schema = value["schema"]
        if schema is not None and not isinstance(schema, str):
            raise BitcodeFormatError("contract schema must be a string or null")
        try:
            media_kind = MediaKind(_require_str(value["media_kind"], "contract.media_kind"))
        except ValueError as exc:
            raise BitcodeFormatError("unsupported contract media_kind") from exc
        return cls(
            media_kind=media_kind,
            media_type=_require_str(value["media_type"], "contract.media_type"),
            format_name=_require_str(value["format_name"], "contract.format_name"),
            source_name=_require_str(value["source_name"], "contract.source_name"),
            schema=schema,
            metadata=metadata,
        )


@dataclass(frozen=True)
class ChunkDescriptor:
    index: int
    raw_offset: int
    encoded_offset: int
    raw_size: int
    encoded_size: int
    codec: str
    raw_sha256: str
    encoded_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "codec": self.codec,
            "encoded_offset": self.encoded_offset,
            "encoded_sha256": self.encoded_sha256,
            "encoded_size": self.encoded_size,
            "index": self.index,
            "raw_offset": self.raw_offset,
            "raw_sha256": self.raw_sha256,
            "raw_size": self.raw_size,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ChunkDescriptor":
        if not isinstance(value, dict):
            raise BitcodeFormatError("chunk descriptor must be an object")
        expected = {
            "codec",
            "encoded_offset",
            "encoded_sha256",
            "encoded_size",
            "index",
            "raw_offset",
            "raw_sha256",
            "raw_size",
        }
        if set(value) != expected:
            raise BitcodeFormatError("chunk fields do not match format version 1")
        codec = _require_str(value["codec"], "chunk.codec")
        if codec not in {"identity", "zlib"}:
            raise BitcodeFormatError(f"unsupported chunk codec: {codec}")
        return cls(
            index=_require_int(value["index"], "chunk.index"),
            raw_offset=_require_int(value["raw_offset"], "chunk.raw_offset"),
            encoded_offset=_require_int(value["encoded_offset"], "chunk.encoded_offset"),
            raw_size=_require_int(value["raw_size"], "chunk.raw_size", minimum=1),
            encoded_size=_require_int(value["encoded_size"], "chunk.encoded_size", minimum=1),
            codec=codec,
            raw_sha256=_require_digest(value["raw_sha256"], "chunk.raw_sha256"),
            encoded_sha256=_require_digest(value["encoded_sha256"], "chunk.encoded_sha256"),
        )


@dataclass(frozen=True)
class BitcodeManifest:
    contract: RepresentationContract
    chunk_size: int
    raw_size: int
    encoded_size: int
    raw_sha256: str
    payload_sha256: str
    merkle_root_sha256: str
    chunks: tuple[ChunkDescriptor, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "chunk_size": self.chunk_size,
            "chunks": [chunk.as_dict() for chunk in self.chunks],
            "contract": self.contract.as_dict(),
            "encoded_size": self.encoded_size,
            "merkle_root_sha256": self.merkle_root_sha256,
            "payload_sha256": self.payload_sha256,
            "raw_sha256": self.raw_sha256,
            "raw_size": self.raw_size,
            "schema": SCHEMA,
            "version": FORMAT_VERSION,
        }

    @classmethod
    def from_dict(cls, value: object) -> "BitcodeManifest":
        if not isinstance(value, dict):
            raise BitcodeFormatError("manifest must be an object")
        expected = {
            "chunk_size",
            "chunks",
            "contract",
            "encoded_size",
            "merkle_root_sha256",
            "payload_sha256",
            "raw_sha256",
            "raw_size",
            "schema",
            "version",
        }
        if set(value) != expected:
            raise BitcodeFormatError("manifest fields do not match format version 1")
        if value["schema"] != SCHEMA or value["version"] != FORMAT_VERSION:
            raise BitcodeFormatError("unsupported manifest schema or version")
        chunks_value = value["chunks"]
        if not isinstance(chunks_value, list):
            raise BitcodeFormatError("manifest chunks must be a list")
        return cls(
            contract=RepresentationContract.from_dict(value["contract"]),
            chunk_size=_require_int(value["chunk_size"], "manifest.chunk_size", minimum=1),
            raw_size=_require_int(value["raw_size"], "manifest.raw_size"),
            encoded_size=_require_int(value["encoded_size"], "manifest.encoded_size"),
            raw_sha256=_require_digest(value["raw_sha256"], "manifest.raw_sha256"),
            payload_sha256=_require_digest(value["payload_sha256"], "manifest.payload_sha256"),
            merkle_root_sha256=_require_digest(
                value["merkle_root_sha256"], "manifest.merkle_root_sha256"
            ),
            chunks=tuple(ChunkDescriptor.from_dict(item) for item in chunks_value),
        )


@dataclass(frozen=True)
class DecodedArtifact:
    data: bytes
    contract: RepresentationContract
    manifest: BitcodeManifest
    container_sha256: str


@dataclass(frozen=True)
class VerificationReport:
    valid: bool
    container_sha256: str
    raw_sha256: str
    payload_sha256: str
    merkle_root_sha256: str
    raw_size: int
    encoded_size: int
    container_size: int
    chunk_count: int
    codecs: Mapping[str, int]
    media_kind: MediaKind
    media_type: str
    format_name: str

    @property
    def payload_ratio(self) -> float:
        return self.encoded_size / self.raw_size if self.raw_size else 1.0

    def as_dict(self) -> dict[str, object]:
        return {
            "chunk_count": self.chunk_count,
            "codecs": dict(sorted(self.codecs.items())),
            "container_sha256": self.container_sha256,
            "container_size": self.container_size,
            "encoded_size": self.encoded_size,
            "format_name": self.format_name,
            "media_kind": self.media_kind.value,
            "media_type": self.media_type,
            "merkle_root_sha256": self.merkle_root_sha256,
            "payload_ratio": self.payload_ratio,
            "payload_sha256": self.payload_sha256,
            "raw_sha256": self.raw_sha256,
            "raw_size": self.raw_size,
            "valid": self.valid,
        }


@dataclass(frozen=True)
class CycleReceipt:
    container: bytes
    contract: RepresentationContract
    input_sha256: str
    reconstructed_sha256: str
    container_sha256: str
    reencoded_sha256: str
    reality_gap_bytes: int
    fixed_point: bool
    chunk_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "chunk_count": self.chunk_count,
            "container_sha256": self.container_sha256,
            "fixed_point": self.fixed_point,
            "format_name": self.contract.format_name,
            "input_sha256": self.input_sha256,
            "media_kind": self.contract.media_kind.value,
            "reality_gap_bytes": self.reality_gap_bytes,
            "reconstructed_sha256": self.reconstructed_sha256,
            "reencoded_sha256": self.reencoded_sha256,
        }


@dataclass(frozen=True)
class _Detection:
    media_kind: MediaKind
    media_type: str
    format_name: str
    evidence: str


@dataclass(frozen=True)
class _ParsedContainer:
    manifest: BitcodeManifest
    payload: bytes
    container_sha256: str


_EXTENSIONS: dict[str, tuple[MediaKind, str, str]] = {
    ".txt": (MediaKind.TEXT, "text/plain", "text"),
    ".md": (MediaKind.TEXT, "text/markdown", "markdown"),
    ".csv": (MediaKind.TEXT, "text/csv", "csv"),
    ".tsv": (MediaKind.TEXT, "text/tab-separated-values", "tsv"),
    ".json": (MediaKind.DOCUMENT, "application/json", "json"),
    ".jsonl": (MediaKind.DOCUMENT, "application/x-ndjson", "jsonl"),
    ".yaml": (MediaKind.DOCUMENT, "application/yaml", "yaml"),
    ".yml": (MediaKind.DOCUMENT, "application/yaml", "yaml"),
    ".xml": (MediaKind.DOCUMENT, "application/xml", "xml"),
    ".html": (MediaKind.DOCUMENT, "text/html", "html"),
    ".pdf": (MediaKind.DOCUMENT, "application/pdf", "pdf"),
    ".docx": (
        MediaKind.DOCUMENT,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "docx",
    ),
    ".png": (MediaKind.IMAGE, "image/png", "png"),
    ".jpg": (MediaKind.IMAGE, "image/jpeg", "jpeg"),
    ".jpeg": (MediaKind.IMAGE, "image/jpeg", "jpeg"),
    ".gif": (MediaKind.IMAGE, "image/gif", "gif"),
    ".webp": (MediaKind.IMAGE, "image/webp", "webp"),
    ".wav": (MediaKind.AUDIO, "audio/wav", "wav"),
    ".flac": (MediaKind.AUDIO, "audio/flac", "flac"),
    ".mp3": (MediaKind.AUDIO, "audio/mpeg", "mp3"),
    ".mp4": (MediaKind.VIDEO, "video/mp4", "mp4"),
    ".webm": (MediaKind.VIDEO, "video/webm", "webm"),
    ".mov": (MediaKind.VIDEO, "video/quicktime", "quicktime"),
    ".glb": (MediaKind.SCENE_3D, "model/gltf-binary", "glb"),
    ".gltf": (MediaKind.SCENE_3D, "model/gltf+json", "gltf"),
    ".obj": (MediaKind.SCENE_3D, "model/obj", "obj"),
    ".stl": (MediaKind.SCENE_3D, "model/stl", "stl"),
    ".ply": (MediaKind.SCENE_3D, "model/ply", "ply"),
    ".onnx": (MediaKind.MODEL, "application/x-onnx", "onnx"),
    ".safetensors": (MediaKind.MODEL, "application/x-safetensors", "safetensors"),
    ".gguf": (MediaKind.MODEL, "application/x-gguf", "gguf"),
    ".pt": (MediaKind.MODEL, "application/x-pytorch", "pytorch"),
    ".pth": (MediaKind.MODEL, "application/x-pytorch", "pytorch"),
    ".zip": (MediaKind.ARCHIVE, "application/zip", "zip"),
    ".gz": (MediaKind.ARCHIVE, "application/gzip", "gzip"),
    ".tar": (MediaKind.ARCHIVE, "application/x-tar", "tar"),
}

for _suffix, _media_type, _format_name in (
    (".py", "text/x-python", "python"),
    (".js", "text/javascript", "javascript"),
    (".mjs", "text/javascript", "javascript"),
    (".ts", "text/typescript", "typescript"),
    (".c", "text/x-c", "c"),
    (".cpp", "text/x-c++", "cpp"),
    (".h", "text/x-c", "c-header"),
    (".hpp", "text/x-c++", "cpp-header"),
    (".java", "text/x-java-source", "java"),
    (".cs", "text/x-csharp", "csharp"),
    (".rs", "text/x-rust", "rust"),
    (".go", "text/x-go", "go"),
    (".sh", "text/x-shellscript", "shell"),
    (".ps1", "text/x-powershell", "powershell"),
    (".glsl", "text/x-glsl", "glsl"),
    (".wgsl", "text/x-wgsl", "wgsl"),
    (".cx", "text/x-codexlang", "codexlang"),
    (".ebnf", "text/x-ebnf", "ebnf"),
    (".toml", "application/toml", "toml"),
):
    _EXTENSIONS[_suffix] = (MediaKind.CODE, _media_type, _format_name)


def _signature_detection(data: bytes) -> _Detection | None:
    if data.startswith(b"%PDF-"):
        return _Detection(MediaKind.DOCUMENT, "application/pdf", "pdf", "signature")
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return _Detection(MediaKind.IMAGE, "image/png", "png", "signature")
    if data.startswith((b"\xff\xd8\xff",)):
        return _Detection(MediaKind.IMAGE, "image/jpeg", "jpeg", "signature")
    if data.startswith((b"GIF87a", b"GIF89a")):
        return _Detection(MediaKind.IMAGE, "image/gif", "gif", "signature")
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return _Detection(MediaKind.IMAGE, "image/webp", "webp", "signature")
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return _Detection(MediaKind.AUDIO, "audio/wav", "wav", "signature")
    if data.startswith(b"fLaC"):
        return _Detection(MediaKind.AUDIO, "audio/flac", "flac", "signature")
    if data.startswith(b"ID3") or (len(data) >= 2 and data[0] == 0xFF and data[1] & 0xE0 == 0xE0):
        return _Detection(MediaKind.AUDIO, "audio/mpeg", "mp3", "signature")
    if len(data) >= 12 and data[4:8] == b"ftyp":
        return _Detection(MediaKind.VIDEO, "video/mp4", "mp4", "signature")
    if data.startswith(b"\x1aE\xdf\xa3"):
        return _Detection(MediaKind.VIDEO, "video/webm", "webm", "signature")
    if data.startswith(b"glTF"):
        return _Detection(MediaKind.SCENE_3D, "model/gltf-binary", "glb", "signature")
    if data.startswith(b"GGUF"):
        return _Detection(MediaKind.MODEL, "application/x-gguf", "gguf", "signature")
    if data.startswith(b"PK\x03\x04"):
        return _Detection(MediaKind.ARCHIVE, "application/zip", "zip", "signature")
    if data.startswith(b"\x1f\x8b"):
        return _Detection(MediaKind.ARCHIVE, "application/gzip", "gzip", "signature")
    if data.startswith(b"\x7fELF"):
        return _Detection(MediaKind.CODE, "application/x-elf", "elf", "signature")
    if data.startswith(b"MZ"):
        return _Detection(
            MediaKind.CODE, "application/vnd.microsoft.portable-executable", "pe", "signature"
        )
    return None


def _looks_like_text(data: bytes) -> bool:
    if b"\x00" in data:
        return False
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    if not text:
        return True
    printable = sum(char.isprintable() or char in "\n\r\t" for char in text)
    return printable / len(text) >= 0.95


def detect_contract(data: bytes, *, source_name: str = "") -> RepresentationContract:
    """Infer a bounded representation hint from signatures, names, and UTF-8 shape."""

    raw = bytes(data)
    suffix = PurePath(source_name).suffix.lower()
    extension = _EXTENSIONS.get(suffix)
    detected = _signature_detection(raw)
    metadata: dict[str, object] = {}
    if detected is not None:
        if detected.format_name == "zip" and extension is not None:
            kind, media_type, format_name = extension
            detected = _Detection(kind, media_type, format_name, "signature+extension")
        elif extension is not None and extension[2] != detected.format_name:
            metadata["extension_hint"] = extension[2]
    elif extension is not None:
        kind, media_type, format_name = extension
        detected = _Detection(kind, media_type, format_name, "extension")
    elif _looks_like_text(raw):
        stripped = raw.lstrip()
        if stripped.startswith((b"{", b"[")):
            try:
                json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                pass
            else:
                detected = _Detection(MediaKind.DOCUMENT, "application/json", "json", "utf8+json")
        if detected is None:
            detected = _Detection(MediaKind.TEXT, "text/plain", "text", "utf8")
    else:
        detected = _Detection(MediaKind.BINARY, "application/octet-stream", "raw", "fallback")
    metadata["detected_by"] = detected.evidence
    return RepresentationContract(
        media_kind=detected.media_kind,
        media_type=detected.media_type,
        format_name=detected.format_name,
        source_name=source_name,
        metadata=metadata,
    )


def _merkle_root(digests: tuple[str, ...]) -> str:
    if not digests:
        return sha256(b"").hexdigest()
    nodes = [sha256(b"\x00" + bytes.fromhex(digest)).digest() for digest in digests]
    while len(nodes) > 1:
        if len(nodes) % 2:
            nodes.append(nodes[-1])
        nodes = [
            sha256(b"\x01" + nodes[index] + nodes[index + 1]).digest()
            for index in range(0, len(nodes), 2)
        ]
    return nodes[0].hex()


def _reality_gap(left: bytes, right: bytes) -> int:
    mismatch = sum(a != b for a, b in zip(left, right))
    return mismatch + abs(len(left) - len(right))


class UniversalBitcodeRuntime:
    """Compile, verify, decode, and close the loop over typed digital artifacts."""

    def __init__(self, *, budget: BitcodeBudget | None = None) -> None:
        self.budget = budget or DEFAULT_BUDGET

    def encode(
        self,
        data: bytes | bytearray | memoryview,
        *,
        contract: RepresentationContract | None = None,
        source_name: str = "",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> bytes:
        raw = bytes(data)
        self._check_raw_budget(raw)
        self._check_chunk_size(chunk_size)
        active_contract = contract or detect_contract(raw, source_name=source_name)
        self._check_contract_budget(active_contract)
        chunk_count = math.ceil(len(raw) / chunk_size) if raw else 0
        if chunk_count > self.budget.max_chunks:
            raise ResourceLimitError(
                f"chunk count {chunk_count} exceeds budget {self.budget.max_chunks}"
            )

        payload_parts: list[bytes] = []
        descriptors: list[ChunkDescriptor] = []
        encoded_offset = 0
        for index, raw_offset in enumerate(range(0, len(raw), chunk_size)):
            raw_chunk = raw[raw_offset : raw_offset + chunk_size]
            compressed = zlib.compress(raw_chunk, level=9)
            if len(compressed) < len(raw_chunk):
                codec = "zlib"
                encoded = compressed
            else:
                codec = "identity"
                encoded = raw_chunk
            descriptor = ChunkDescriptor(
                index=index,
                raw_offset=raw_offset,
                encoded_offset=encoded_offset,
                raw_size=len(raw_chunk),
                encoded_size=len(encoded),
                codec=codec,
                raw_sha256=sha256(raw_chunk).hexdigest(),
                encoded_sha256=sha256(encoded).hexdigest(),
            )
            descriptors.append(descriptor)
            payload_parts.append(encoded)
            encoded_offset += len(encoded)

        payload = b"".join(payload_parts)
        raw_digest = sha256(raw).hexdigest()
        payload_digest = sha256(payload).hexdigest()
        chunks = tuple(descriptors)
        manifest = BitcodeManifest(
            contract=active_contract,
            chunk_size=chunk_size,
            raw_size=len(raw),
            encoded_size=len(payload),
            raw_sha256=raw_digest,
            payload_sha256=payload_digest,
            merkle_root_sha256=_merkle_root(tuple(chunk.raw_sha256 for chunk in chunks)),
            chunks=chunks,
        )
        manifest_bytes = _canonical_json(manifest.as_dict())
        if len(manifest_bytes) > self.budget.max_manifest_bytes:
            raise ResourceLimitError(
                f"manifest size {len(manifest_bytes)} exceeds budget "
                f"{self.budget.max_manifest_bytes}"
            )
        header = HEADER.pack(
            MAGIC,
            FORMAT_VERSION,
            _ALLOWED_FLAGS,
            len(manifest_bytes),
            len(raw),
            len(payload),
            bytes.fromhex(raw_digest),
            sha256(manifest_bytes).digest(),
            bytes.fromhex(payload_digest),
        )
        return header + manifest_bytes + payload

    def inspect(self, container: bytes | bytearray | memoryview) -> BitcodeManifest:
        """Validate the envelope and encoded payload without decompressing chunks."""

        return self._parse(container).manifest

    def decode(self, container: bytes | bytearray | memoryview) -> DecodedArtifact:
        parsed = self._parse(container)
        raw_parts: list[bytes] = []
        for descriptor in parsed.manifest.chunks:
            start = descriptor.encoded_offset
            end = start + descriptor.encoded_size
            encoded = parsed.payload[start:end]
            if sha256(encoded).hexdigest() != descriptor.encoded_sha256:
                raise IntegrityError(f"encoded chunk {descriptor.index} digest mismatch")
            raw_chunk = self._decode_chunk(encoded, descriptor)
            if sha256(raw_chunk).hexdigest() != descriptor.raw_sha256:
                raise IntegrityError(f"raw chunk {descriptor.index} digest mismatch")
            raw_parts.append(raw_chunk)
        raw = b"".join(raw_parts)
        if len(raw) != parsed.manifest.raw_size:
            raise IntegrityError("decoded size does not match manifest")
        if sha256(raw).hexdigest() != parsed.manifest.raw_sha256:
            raise IntegrityError("decoded artifact digest mismatch")
        root = _merkle_root(tuple(chunk.raw_sha256 for chunk in parsed.manifest.chunks))
        if root != parsed.manifest.merkle_root_sha256:
            raise IntegrityError("chunk Merkle root mismatch")
        return DecodedArtifact(
            data=raw,
            contract=parsed.manifest.contract,
            manifest=parsed.manifest,
            container_sha256=parsed.container_sha256,
        )

    def verify(self, container: bytes | bytearray | memoryview) -> VerificationReport:
        raw_container = bytes(container)
        decoded = self.decode(raw_container)
        manifest = decoded.manifest
        codecs = Counter(chunk.codec for chunk in manifest.chunks)
        return VerificationReport(
            valid=True,
            container_sha256=decoded.container_sha256,
            raw_sha256=manifest.raw_sha256,
            payload_sha256=manifest.payload_sha256,
            merkle_root_sha256=manifest.merkle_root_sha256,
            raw_size=manifest.raw_size,
            encoded_size=manifest.encoded_size,
            container_size=len(raw_container),
            chunk_count=len(manifest.chunks),
            codecs=dict(codecs),
            media_kind=manifest.contract.media_kind,
            media_type=manifest.contract.media_type,
            format_name=manifest.contract.format_name,
        )

    def close_loop(
        self,
        data: bytes | bytearray | memoryview,
        *,
        contract: RepresentationContract | None = None,
        source_name: str = "",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> CycleReceipt:
        """Execute bits -> IR -> bits and verify the canonical fixed point."""

        raw = bytes(data)
        container = self.encode(
            raw,
            contract=contract,
            source_name=source_name,
            chunk_size=chunk_size,
        )
        decoded = self.decode(container)
        reencoded = self.encode(
            decoded.data,
            contract=decoded.contract,
            chunk_size=decoded.manifest.chunk_size,
        )
        return CycleReceipt(
            container=container,
            contract=decoded.contract,
            input_sha256=sha256(raw).hexdigest(),
            reconstructed_sha256=sha256(decoded.data).hexdigest(),
            container_sha256=sha256(container).hexdigest(),
            reencoded_sha256=sha256(reencoded).hexdigest(),
            reality_gap_bytes=_reality_gap(raw, decoded.data),
            fixed_point=container == reencoded,
            chunk_count=len(decoded.manifest.chunks),
        )

    def _parse(self, container: bytes | bytearray | memoryview) -> _ParsedContainer:
        raw_container = bytes(container)
        if len(raw_container) < HEADER_SIZE:
            raise BitcodeFormatError("container is shorter than the fixed header")
        (
            magic,
            version,
            flags,
            manifest_size,
            raw_size,
            payload_size,
            raw_digest,
            manifest_digest,
            payload_digest,
        ) = HEADER.unpack_from(raw_container)
        if magic != MAGIC:
            raise BitcodeFormatError("container magic mismatch")
        if version != FORMAT_VERSION:
            raise BitcodeFormatError(f"unsupported container version: {version}")
        if flags != _ALLOWED_FLAGS:
            raise BitcodeFormatError("container uses unsupported flags")
        if manifest_size < 2 or manifest_size > self.budget.max_manifest_bytes:
            raise ResourceLimitError("manifest size is outside the runtime budget")
        if raw_size > self.budget.max_input_bytes or payload_size > self.budget.max_input_bytes:
            raise ResourceLimitError("container payload exceeds the runtime byte budget")
        expected_size = HEADER_SIZE + manifest_size + payload_size
        if len(raw_container) != expected_size:
            raise BitcodeFormatError("container length does not match header")
        manifest_start = HEADER_SIZE
        payload_start = manifest_start + manifest_size
        manifest_bytes = raw_container[manifest_start:payload_start]
        payload = raw_container[payload_start:]
        if sha256(manifest_bytes).digest() != manifest_digest:
            raise IntegrityError("manifest digest mismatch")
        if sha256(payload).digest() != payload_digest:
            raise IntegrityError("payload digest mismatch")
        try:
            manifest_value = json.loads(manifest_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BitcodeFormatError("manifest is not valid UTF-8 JSON") from exc
        if _canonical_json(manifest_value) != manifest_bytes:
            raise BitcodeFormatError("manifest is not canonical JSON")
        manifest = BitcodeManifest.from_dict(manifest_value)
        if manifest.raw_size != raw_size or bytes.fromhex(manifest.raw_sha256) != raw_digest:
            raise IntegrityError("manifest raw identity does not match header")
        if (
            manifest.encoded_size != payload_size
            or bytes.fromhex(manifest.payload_sha256) != payload_digest
        ):
            raise IntegrityError("manifest payload identity does not match header")
        self._check_chunk_size(manifest.chunk_size)
        self._check_contract_budget(manifest.contract)
        self._validate_chunk_layout(manifest)
        return _ParsedContainer(
            manifest=manifest,
            payload=payload,
            container_sha256=sha256(raw_container).hexdigest(),
        )

    def _validate_chunk_layout(self, manifest: BitcodeManifest) -> None:
        if len(manifest.chunks) > self.budget.max_chunks:
            raise ResourceLimitError("manifest chunk count exceeds the runtime budget")
        expected_count = (
            math.ceil(manifest.raw_size / manifest.chunk_size) if manifest.raw_size else 0
        )
        if len(manifest.chunks) != expected_count:
            raise BitcodeFormatError("chunk count does not cover the declared raw size")
        raw_cursor = 0
        encoded_cursor = 0
        for expected_index, chunk in enumerate(manifest.chunks):
            if chunk.index != expected_index:
                raise BitcodeFormatError("chunk indexes must be contiguous")
            if chunk.raw_offset != raw_cursor or chunk.encoded_offset != encoded_cursor:
                raise BitcodeFormatError("chunk offsets must be contiguous")
            if chunk.raw_size > manifest.chunk_size:
                raise BitcodeFormatError("chunk raw size exceeds manifest chunk_size")
            if chunk.encoded_size > self.budget.max_chunk_size:
                raise ResourceLimitError("encoded chunk exceeds the runtime chunk budget")
            raw_cursor += chunk.raw_size
            encoded_cursor += chunk.encoded_size
        if raw_cursor != manifest.raw_size or encoded_cursor != manifest.encoded_size:
            raise BitcodeFormatError("chunk layout does not close at declared sizes")

    @staticmethod
    def _decode_chunk(encoded: bytes, descriptor: ChunkDescriptor) -> bytes:
        if descriptor.codec == "identity":
            if len(encoded) != descriptor.raw_size:
                raise IntegrityError("identity chunk size mismatch")
            return encoded
        decompressor = zlib.decompressobj()
        try:
            raw = decompressor.decompress(encoded, descriptor.raw_size + 1)
        except zlib.error as exc:
            raise IntegrityError("zlib chunk could not be decoded") from exc
        if (
            len(raw) != descriptor.raw_size
            or decompressor.unconsumed_tail
            or decompressor.unused_data
            or not decompressor.eof
        ):
            raise IntegrityError("zlib chunk violates its bounded size contract")
        return raw

    def _check_raw_budget(self, raw: bytes) -> None:
        if len(raw) > self.budget.max_input_bytes:
            raise ResourceLimitError(
                f"input size {len(raw)} exceeds budget {self.budget.max_input_bytes}"
            )

    def _check_chunk_size(self, chunk_size: int) -> None:
        if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
            raise BitcodeFormatError("chunk_size must be a positive integer")
        if chunk_size > self.budget.max_chunk_size:
            raise ResourceLimitError(
                f"chunk_size {chunk_size} exceeds budget {self.budget.max_chunk_size}"
            )

    def _check_contract_budget(self, contract: RepresentationContract) -> None:
        source_size = len(contract.source_name.encode("utf-8"))
        if source_size > self.budget.max_source_name_bytes:
            raise ResourceLimitError("source_name exceeds the runtime metadata budget")
        metadata_size = len(_canonical_json(dict(contract.metadata)))
        if metadata_size > self.budget.max_metadata_bytes:
            raise ResourceLimitError("contract metadata exceeds the runtime metadata budget")


__all__ = [
    "BitcodeBudget",
    "BitcodeError",
    "BitcodeFormatError",
    "BitcodeManifest",
    "ChunkDescriptor",
    "CycleReceipt",
    "DecodedArtifact",
    "DEFAULT_CHUNK_SIZE",
    "FORMAT_VERSION",
    "HEADER_SIZE",
    "IntegrityError",
    "MAGIC",
    "MediaKind",
    "RepresentationContract",
    "ResourceLimitError",
    "UniversalBitcodeRuntime",
    "VerificationReport",
    "detect_contract",
]
