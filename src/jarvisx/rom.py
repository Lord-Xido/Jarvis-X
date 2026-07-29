"""Deterministic Jarvis-X bytecode ROM container.

The ROM format is deliberately small and dependency-free. It wraps the existing
64-bit Jarvis-X words with source and payload SHA-256 digests plus canonical JSON
metadata, making compilation artifacts reproducible and tamper-evident.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Tuple

MAGIC = b"JXROM\x00\x01\x00"
FORMAT_VERSION = 1
_HEADER = struct.Struct(">8sHHII32s32s")
_WORD = struct.Struct(">Q")


class RomFormatError(ValueError):
    """Raised when a ROM image is malformed or fails integrity checks."""


def _canonical_metadata(metadata: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(metadata), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


@dataclass(frozen=True)
class RomImage:
    language: str
    words: Tuple[int, ...]
    source_digest: bytes
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: int = FORMAT_VERSION

    @classmethod
    def build(
        cls,
        language: str,
        source: str,
        words: Iterable[int],
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "RomImage":
        normalized_words = tuple(int(word) for word in words)
        for word in normalized_words:
            if not 0 <= word <= 0xFFFFFFFFFFFFFFFF:
                raise RomFormatError("bytecode words must fit in unsigned 64 bits")
        return cls(
            language=language,
            words=normalized_words,
            source_digest=hashlib.sha256(source.encode("utf-8")).digest(),
            metadata=dict(metadata or {}),
        )

    @property
    def source_sha256(self) -> str:
        return self.source_digest.hex()

    @property
    def payload_sha256(self) -> str:
        return hashlib.sha256(self.payload_bytes()).hexdigest()

    def payload_bytes(self) -> bytes:
        return b"".join(_WORD.pack(word) for word in self.words)

    def to_bytes(self) -> bytes:
        language = self.language.encode("utf-8")
        metadata = _canonical_metadata(self.metadata)
        payload = self.payload_bytes()
        if len(language) > 0xFFFF:
            raise RomFormatError("language identifier is too long")
        header = _HEADER.pack(
            MAGIC,
            self.version,
            len(language),
            len(metadata),
            len(self.words),
            self.source_digest,
            hashlib.sha256(payload).digest(),
        )
        return header + language + metadata + payload

    @classmethod
    def from_bytes(cls, blob: bytes) -> "RomImage":
        if len(blob) < _HEADER.size:
            raise RomFormatError("ROM image is shorter than its header")
        (
            magic,
            version,
            language_size,
            metadata_size,
            word_count,
            source_digest,
            expected_payload_digest,
        ) = _HEADER.unpack_from(blob)
        if magic != MAGIC:
            raise RomFormatError("invalid ROM magic")
        if version != FORMAT_VERSION:
            raise RomFormatError("unsupported ROM format version: %s" % version)

        language_start = _HEADER.size
        metadata_start = language_start + language_size
        payload_start = metadata_start + metadata_size
        expected_size = payload_start + (word_count * _WORD.size)
        if len(blob) != expected_size:
            raise RomFormatError("ROM size does not match the encoded header")

        try:
            language = blob[language_start:metadata_start].decode("utf-8")
            metadata = json.loads(blob[metadata_start:payload_start].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RomFormatError("ROM text section is invalid") from exc
        if not isinstance(metadata, dict):
            raise RomFormatError("ROM metadata must decode to an object")

        payload = blob[payload_start:]
        if hashlib.sha256(payload).digest() != expected_payload_digest:
            raise RomFormatError("ROM payload digest mismatch")
        words = tuple(
            _WORD.unpack_from(payload, offset)[0]
            for offset in range(0, len(payload), _WORD.size)
        )
        return cls(
            language=language,
            words=words,
            source_digest=source_digest,
            metadata=metadata,
            version=version,
        )


class BytecodeAutoencoder:
    """Encodes canonical bytecode words into a deterministic ROM image."""

    def encode(
        self,
        language: str,
        source: str,
        words: Iterable[int],
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> bytes:
        return RomImage.build(language, source, words, metadata).to_bytes()


class BytecodeAutodecoder:
    """Decodes and verifies a ROM image before exposing bytecode words."""

    def decode(self, blob: bytes) -> RomImage:
        return RomImage.from_bytes(blob)
