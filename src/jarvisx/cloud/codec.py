"""Deterministic lossless multimodal encode/decode substrate.

This is intentionally a codec, not a claim of a learned neural autoencoder. It
provides an executable encode/decode path for every supported media kind while
model-specific learned codecs can be registered behind the same boundary.
"""

from __future__ import annotations

import base64
import zlib
from dataclasses import dataclass
from hashlib import sha256

from .multimodal import MediaEnvelope, MediaKind


@dataclass(frozen=True)
class LatentPacket:
    kind: MediaKind
    content_type: str
    algorithm: str
    source_sha256: str
    source_size: int
    payload: bytes

    @property
    def encoded_size(self) -> int:
        return len(self.payload)

    @property
    def compression_ratio(self) -> float:
        return self.encoded_size / self.source_size

    def descriptor(self) -> dict[str, str | int | float]:
        return {
            "kind": self.kind.value,
            "content_type": self.content_type,
            "algorithm": self.algorithm,
            "source_sha256": self.source_sha256,
            "source_size": self.source_size,
            "encoded_size": self.encoded_size,
            "compression_ratio": self.compression_ratio,
            "latent_sha256": sha256(self.payload).hexdigest(),
        }

    def to_base64(self) -> str:
        return base64.b64encode(self.payload).decode("ascii")


class LosslessMultimodalCodec:
    """Reference codec supporting text/image/audio/video/binary bytes."""

    algorithm = "zlib-deflate-v1"

    def __init__(self, compression_level: int = 6) -> None:
        if not 0 <= compression_level <= 9:
            raise ValueError("compression_level must be between 0 and 9")
        self.compression_level = compression_level

    def encode(self, media: MediaEnvelope) -> LatentPacket:
        return LatentPacket(
            kind=media.kind,
            content_type=media.content_type,
            algorithm=self.algorithm,
            source_sha256=media.digest,
            source_size=media.size_bytes,
            payload=zlib.compress(media.payload, self.compression_level),
        )

    def decode(self, packet: LatentPacket) -> MediaEnvelope:
        if packet.algorithm != self.algorithm:
            raise ValueError(f"unsupported codec algorithm: {packet.algorithm}")
        payload = zlib.decompress(packet.payload)
        if len(payload) != packet.source_size:
            raise RuntimeError("decoded size differs from source size")
        if sha256(payload).hexdigest() != packet.source_sha256:
            raise RuntimeError("decoded digest differs from source digest")
        return MediaEnvelope(
            kind=packet.kind,
            payload=payload,
            content_type=packet.content_type,
        )

    def roundtrip(self, media: MediaEnvelope) -> dict[str, str | int | float | bool]:
        packet = self.encode(media)
        reconstructed = self.decode(packet)
        return {
            **packet.descriptor(),
            "reconstructed_sha256": reconstructed.digest,
            "lossless": reconstructed.payload == media.payload,
        }
