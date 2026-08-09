import hashlib

import pytest

from jarvisx.volumetric_ae import ArtifactError, Universal3DAutoEncoder, VirtualVolumeSpec
from jarvisx.volumetric_api import dashboard, execute_cycle


def test_6400_gib_q16_16_metrics_are_virtual() -> None:
    spec = VirtualVolumeSpec(capacity_gib=6400, cell_bits=32, chunk_bytes=1024)
    metrics = spec.metrics()
    assert metrics["capacity_bytes"] == 6400 * 1024**3
    assert metrics["total_cells"] == (6400 * 1024**3) // 4
    assert metrics["logical_cube_side_cells"] ** 3 >= metrics["total_cells"]
    assert metrics["allocation_mode"] == "sparse_virtual"
    assert metrics["resident_bytes_at_idle"] == 0


def test_round_trip_is_bit_exact_across_multiple_chunks() -> None:
    engine = Universal3DAutoEncoder(VirtualVolumeSpec(chunk_bytes=257))
    payload = (bytes(range(256)) * 20) + b"Jarvis-X"
    artifact, encode_receipt = engine.encode(payload)
    restored, decode_receipt = engine.decode(artifact)

    assert restored == payload
    assert encode_receipt.chunk_count > 1
    assert encode_receipt.payload_sha256 == hashlib.sha256(payload).hexdigest()
    assert decode_receipt.verified is True
    assert decode_receipt.status == "RECONSTRUCTED_BIT_EXACT"


def test_empty_payload_round_trip() -> None:
    engine = Universal3DAutoEncoder(VirtualVolumeSpec(chunk_bytes=64))
    artifact, receipt = engine.encode(b"")
    restored, decoded = engine.decode(artifact)
    assert restored == b""
    assert receipt.chunk_count == 0
    assert decoded.chunk_count == 0


def test_corruption_is_detected() -> None:
    engine = Universal3DAutoEncoder(VirtualVolumeSpec(chunk_bytes=64))
    artifact, _ = engine.encode(b"A" * 512)
    damaged = bytearray(artifact)
    damaged[-1] ^= 0xFF
    with pytest.raises(ArtifactError):
        engine.decode(bytes(damaged))


def test_self_test_reports_success() -> None:
    report = Universal3DAutoEncoder(VirtualVolumeSpec(chunk_bytes=256)).self_test()
    assert report["ok"] is True
    assert report["decode"]["verified"] is True


def test_api_cycle_executes_real_encode_decode_verification() -> None:
    payload = b"DM-vOmegaXi-live-cycle" * 2048
    report = execute_cycle(payload)
    assert report["verified"] is True
    assert report["encode"]["payload_bytes"] == len(payload)
    assert report["decode"]["payload_sha256"] == hashlib.sha256(payload).hexdigest()


def test_dashboard_is_packaged_and_targets_live_runtime() -> None:
    response = dashboard()
    body = response.body.decode("utf-8")
    assert "Live Runtime Telemetry" in body
    assert "/v1/volumetric/metrics" in body
    assert "/v1/volumetric/cycle" in body
    assert "Measured Ratio" in body
