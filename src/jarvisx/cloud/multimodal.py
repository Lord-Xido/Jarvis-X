"""Typed multimodal envelopes used by the cloud runtime."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256


class MediaKind(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    BINARY = "binary"


@dataclass(frozen=True)
class MediaEnvelope:
    """Immutable media unit normalized before encoding or routing."""

    kind: MediaKind
    payload: bytes
    content_type: str = "application/octet-stream"

    def __post_init__(self) -> None:
        if not isinstance(self.payload, bytes):
            raise TypeError("payload must be bytes")
        if not self.payload:
            raise ValueError("payload must not be empty")
        if not self.content_type:
            raise ValueError("content_type must not be empty")

    @property
    def digest(self) -> str:
        return sha256(self.payload).hexdigest()

    @property
    def size_bytes(self) -> int:
        return len(self.payload)

    def descriptor(self) -> dict[str, str | int]:
        return {
            "kind": self.kind.value,
            "content_type": self.content_type,
            "sha256": self.digest,
            "size_bytes": self.size_bytes,
        }
