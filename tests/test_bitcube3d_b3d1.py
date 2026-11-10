from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import pytest

from jarvisx.bitcube3d_b3d1 import (
    BRICK_BYTES,
    HEADER,
    SIDE,
    B3D1Snapshot,
    deserialize,
    from_points,
)

VECTORS = Path(__file__).resolve().parents[1] / (
    "native/bitcube3d/tests/fixtures/b3d1-v1.json"
)


@pytest.mark.parametrize("vector", json.loads(VECTORS.read_text())["vectors"])
def test_golden_vectors_exact_binary_roundtrip(vector: dict[str, object]) -> None:
    points = [tuple(v) for v in vector["bits"]]
    rounds = vector["rounds"]
    snapshot = from_points(points, rounds=rounds)
    encoded = snapshot.serialize()

    assert encoded[:4] == b"B3D1"
    assert len(encoded) == vector["expected_bytes"]
    assert len(snapshot.bricks) == vector["expected_bricks"]
    assert hashlib.sha256(encoded).hexdigest() == vector["sha256"]
    assert deserialize(encoded).serialize() == encoded
    assert deserialize(encoded).active_points(inverse_rounds=True) == sorted(set(points))


def test_rejects_corrupt_lengths_and_version_metadata() -> None:
    raw = from_points([(0, 0, 0)]).serialize()

    for corrupt in (raw[:-1], raw + b"X"):
        with pytest.raises(ValueError, match="length"):
            deserialize(corrupt)

    altered = bytearray(raw)
    altered[4:8] = struct.pack("<I", SIDE - 1)
    with pytest.raises(ValueError, match="unsupported"):
        deserialize(bytes(altered))

    altered = bytearray(raw)
    altered[12:20] = struct.pack("<Q", 1_000_000)
    with pytest.raises(ValueError, match="limit"):
        deserialize(bytes(altered))


def test_rejects_out_of_range_duplicate_unsorted_and_zero_bricks() -> None:
    raw = from_points([(0, 0, 0), (16, 0, 0)]).serialize()
    header = raw[:HEADER.size]
    first = raw[HEADER.size:HEADER.size + 12 + BRICK_BYTES]
    second = raw[HEADER.size + 12 + BRICK_BYTES:]

    for invalid in (header + second + first, header + first + first):
        with pytest.raises(ValueError, match="sorted"):
            deserialize(invalid)

    altered = bytearray(raw)
    altered[HEADER.size:HEADER.size + 4] = struct.pack("<I", SIDE // 16)
    with pytest.raises(ValueError, match="coordinate"):
        deserialize(bytes(altered))

    with pytest.raises(ValueError, match="empty brick"):
        deserialize(header + first[:12] + bytes(BRICK_BYTES) + second)


def test_limits_and_empty_snapshot_roundtrip() -> None:
    assert from_points([]).serialize() == B3D1Snapshot(0, {}).serialize()
    with pytest.raises(ValueError, match="limit"):
        from_points([(0, 0, 0)], max_bricks=0)
    with pytest.raises(ValueError, match="round"):
        from_points([(0, 0, 0)], rounds=10_001)
    with pytest.raises(ValueError, match="coordinate"):
        from_points([(SIDE, 0, 0)])
    with pytest.raises(TypeError, match="bytes"):
        deserialize(bytearray(from_points([]).serialize()))
