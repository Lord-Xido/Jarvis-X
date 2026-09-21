"""Canonical Dr Moagi bytecoding preservation and verification contract.

This module deliberately separates three different claims:

1. byte transport can be exact;
2. quantized latent reconstruction is approximate unless the representation is exact;
3. semantic preservation is task-specific and must be measured by a supplied metric.

The module never executes payload bytes. It only packs, verifies, unpacks, and
measures the declared bytecoding invariants.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from hashlib import sha256
from typing import Callable, Mapping, Sequence

FRAME_MAGIC = b"DMBYTE1\0"
FRAME_VERSION = 1
FRAME_HEADER = struct.Struct("<8sHHII")
FRAME_DIGEST_BYTES = 32


@dataclass(frozen=True)
class DecodedFrame:
    flags: int
    metadata: dict[str, object]
    payload: bytes
    digest: str


@dataclass(frozen=True)
class QuantizedInt8:
    values: tuple[int, ...]
    scale: float
    clipped_count: int


@dataclass(frozen=True)
class VerificationReceipt:
    valid: bool
    decision: str
    reason: str
    payload_bytes: int = 0
    metadata_bytes: int = 0
    digest: str = ""


def _canonical_metadata(metadata: Mapping[str, object]) -> bytes:
    return json.dumps(
        dict(metadata),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def pack_frame(payload: bytes, metadata: Mapping[str, object], *, flags: int = 0) -> bytes:
    """Pack payload bytes into an integrity-protected deterministic envelope."""

    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    if not 0 <= flags <= 0xFFFF:
        raise ValueError("flags must fit uint16")
    metadata_bytes = _canonical_metadata(metadata)
    header = FRAME_HEADER.pack(
        FRAME_MAGIC,
        FRAME_VERSION,
        flags,
        len(metadata_bytes),
        len(payload),
    )
    body = header + metadata_bytes + payload
    return body + sha256(body).digest()


def unpack_frame(frame: bytes) -> DecodedFrame:
    """Verify and decode one frame without executing its payload."""

    minimum = FRAME_HEADER.size + FRAME_DIGEST_BYTES
    if len(frame) < minimum:
        raise ValueError("bytecoding frame is truncated")

    body = frame[:-FRAME_DIGEST_BYTES]
    digest = frame[-FRAME_DIGEST_BYTES:]
    if sha256(body).digest() != digest:
        raise ValueError("bytecoding frame SHA-256 mismatch")

    magic, version, flags, metadata_len, payload_len = FRAME_HEADER.unpack_from(body, 0)
    if magic != FRAME_MAGIC:
        raise ValueError("invalid bytecoding frame magic")
    if version != FRAME_VERSION:
        raise ValueError("unsupported bytecoding frame version")

    expected = FRAME_HEADER.size + metadata_len + payload_len
    if len(body) != expected:
        raise ValueError("bytecoding frame length metadata mismatch")

    metadata_start = FRAME_HEADER.size
    payload_start = metadata_start + metadata_len
    raw_metadata = body[metadata_start:payload_start]
    try:
        decoded = json.loads(raw_metadata.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid bytecoding metadata") from exc
    if not isinstance(decoded, dict):
        raise ValueError("bytecoding metadata must be a JSON object")

    payload = bytes(body[payload_start:])
    return DecodedFrame(
        flags=flags,
        metadata=decoded,
        payload=payload,
        digest=sha256(body).hexdigest(),
    )


def verify_frame(
    frame: bytes,
    *,
    max_payload_bytes: int,
    max_metadata_bytes: int = 1 << 20,
) -> VerificationReceipt:
    """Return the VERIFY -> ACCEPT/ROLLBACK decision for one inert byte frame."""

    if max_payload_bytes <= 0 or max_metadata_bytes <= 0:
        raise ValueError("verification limits must be positive")

    try:
        decoded = unpack_frame(frame)
    except (TypeError, ValueError) as exc:
        return VerificationReceipt(False, "ROLLBACK", str(exc))

    metadata_bytes = len(_canonical_metadata(decoded.metadata))
    if len(decoded.payload) > max_payload_bytes:
        return VerificationReceipt(
            False,
            "ROLLBACK",
            "payload exceeds configured byte budget",
            len(decoded.payload),
            metadata_bytes,
            decoded.digest,
        )
    if metadata_bytes > max_metadata_bytes:
        return VerificationReceipt(
            False,
            "ROLLBACK",
            "metadata exceeds configured byte budget",
            len(decoded.payload),
            metadata_bytes,
            decoded.digest,
        )

    return VerificationReceipt(
        True,
        "ACCEPT",
        "digest, structure, and configured byte bounds verified",
        len(decoded.payload),
        metadata_bytes,
        decoded.digest,
    )


def quantize_int8(values: Sequence[float], scale: float) -> QuantizedInt8:
    """Quantize finite values into signed INT8 and report saturation explicitly."""

    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be finite and positive")

    quantized: list[int] = []
    clipped_count = 0
    for value in values:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("quantization input must be finite")
        raw = int(round(value / scale))
        clipped = min(127, max(-128, raw))
        clipped_count += int(clipped != raw)
        quantized.append(clipped)

    return QuantizedInt8(tuple(quantized), scale, clipped_count)


def dequantize_int8(quantized: QuantizedInt8) -> tuple[float, ...]:
    return tuple(value * quantized.scale for value in quantized.values)


def max_abs_error(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("error operands must have equal length")
    if not left:
        return 0.0
    return max(abs(float(a) - float(b)) for a, b in zip(left, right))


def quantization_preserved(
    reference: Sequence[float],
    quantized: QuantizedInt8,
    *,
    epsilon: float,
) -> bool:
    """Check numerical preservation against an explicit tolerance.

    No unclipped INT8 guarantee is inferred if saturation occurred; the actual
    reconstruction error is measured directly.
    """

    if not math.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError("epsilon must be finite and non-negative")
    reconstructed = dequantize_int8(quantized)
    return max_abs_error(reference, reconstructed) <= epsilon


def semantic_preserved(
    reference: object,
    reconstructed: object,
    *,
    metric: Callable[[object, object], float],
    epsilon: float,
) -> bool:
    """Evaluate a task-supplied semantic distance instead of assuming semantics."""

    if not math.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError("epsilon must be finite and non-negative")
    distance = float(metric(reference, reconstructed))
    if not math.isfinite(distance) or distance < 0.0:
        raise ValueError("semantic metric must return a finite non-negative distance")
    return distance <= epsilon


__all__ = [
    "DecodedFrame",
    "FRAME_DIGEST_BYTES",
    "FRAME_HEADER",
    "FRAME_MAGIC",
    "FRAME_VERSION",
    "QuantizedInt8",
    "VerificationReceipt",
    "dequantize_int8",
    "max_abs_error",
    "pack_frame",
    "quantization_preserved",
    "quantize_int8",
    "semantic_preserved",
    "unpack_frame",
    "verify_frame",
]
