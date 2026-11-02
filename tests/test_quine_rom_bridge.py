"""Deterministic bounds, Morton geometry, round-trip and corruption checks."""
import json
import pytest
from jarvisx.quine_rom_bridge import (
    AXIS_BITS, HEADER, Q_DEC, Q_ENC, Q_FOLD, Q_JMP_SELF, Q_NOP,
    QuineROMBridge, morton_decode_90, morton_encode_90, restore, sha256,
)
from jarvisx.dm3d_rom import TILE_BYTES


@pytest.mark.parametrize("index", [0, 1, 7, 8, 1023, 2**63 + 17, 2**90 - 1])
def test_morton_90_roundtrip(index):
    assert morton_encode_90(*morton_decode_90(index)) == index


def test_empty_payload_stays_empty(tmp_path):
    bridge = QuineROMBridge(b"")
    report = bridge.run(cycles=2)
    assert report["bit_exact"] and report["materialized_chunks"] == 0
    archive = tmp_path / "empty.qrom"
    bridge.export(archive)
    assert restore(archive) == b""


@pytest.mark.parametrize("size", [1, 512, 4097, TILE_BYTES - 1, TILE_BYTES + 91])
def test_exact_file_roundtrip_and_3d_mapping(size, tmp_path):
    data = bytes((i * 47 + i // 257 + 23) & 255 for i in range(size))
    bridge = QuineROMBridge(data)
    report = bridge.run(cycles=2)
    assert report["source_sha256"] == sha256(data)
    assert report["roundtrip_sha256"] == sha256(data)
    assert report["instructions_executed"] == 12
    assert report["materialized_chunks"] == (size + TILE_BYTES - 1) // TILE_BYTES
    assert all(0 <= c < AXIS_BITS for item in bridge.folded for c in item["morton_xyz"])
    archive = tmp_path / "artifact.qrom"
    bridge.export(archive)
    assert restore(archive) == data


def test_tamper_detection(tmp_path):
    bridge = QuineROMBridge(b"abc" * 999)
    bridge.run(cycles=1)
    archive = tmp_path / "a.qrom"
    bridge.export(archive)
    wire = archive.read_bytes()
    doc = json.loads(wire[HEADER.size:])
    doc["source_sha256"] = "0" * 64
    raw = json.dumps(doc).encode()
    archive.write_bytes(HEADER.pack(b"JXQRB1\x00\x00", len(raw)) + raw)
    with pytest.raises(ValueError, match="SHA-256"):
        restore(archive)


def test_unbounded_cycles_unknown_opcode_and_corrupt_header(tmp_path):
    bridge = QuineROMBridge(b"hello")
    with pytest.raises(ValueError, match="cycles"):
        bridge.run(cycles=1000000)
    with pytest.raises(ValueError, match="opcode"):
        bridge.run(program=[0x12, Q_JMP_SELF, Q_NOP])
    with pytest.raises(ValueError, match="begin"):
        bridge.run(program=[Q_DEC, Q_FOLD, Q_ENC, Q_JMP_SELF, Q_NOP])
    f = tmp_path / "wrong.qrom"
    f.write_bytes(b"not a valid QROM")
    with pytest.raises(ValueError):
        restore(f)
