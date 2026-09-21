from __future__ import annotations

import math

import pytest

from jarvisx.dr_moagi_bytecoding_contract import (
    dequantize_int8,
    max_abs_error,
    pack_frame,
    quantization_preserved,
    quantize_int8,
    semantic_preserved,
    unpack_frame,
    verify_frame,
)


def test_pack_unpack_preserves_payload_and_metadata_exactly() -> None:
    payload = bytes(range(64))
    metadata = {"isa": "DM3D", "cycle": 7, "shape": [16, 16, 16]}

    frame = pack_frame(payload, metadata, flags=3)
    decoded = unpack_frame(frame)

    assert decoded.payload == payload
    assert decoded.metadata == metadata
    assert decoded.flags == 3


def test_digest_tampering_rolls_back() -> None:
    frame = bytearray(pack_frame(b"verified", {"version": 1}))
    frame[-33] ^= 0x01

    receipt = verify_frame(bytes(frame), max_payload_bytes=1024)

    assert receipt.valid is False
    assert receipt.decision == "ROLLBACK"
    assert "SHA-256" in receipt.reason


def test_payload_budget_rolls_back_without_execution() -> None:
    frame = pack_frame(b"x" * 32, {"kind": "inert-bytecode"})

    receipt = verify_frame(frame, max_payload_bytes=16)

    assert receipt.valid is False
    assert receipt.decision == "ROLLBACK"
    assert receipt.payload_bytes == 32


def test_unclipped_int8_quantization_respects_half_step_bound() -> None:
    values = (-1.0, -0.37, 0.0, 0.21, 0.99)
    scale = 0.01
    quantized = quantize_int8(values, scale)
    reconstructed = dequantize_int8(quantized)

    assert quantized.clipped_count == 0
    assert max_abs_error(values, reconstructed) <= scale / 2.0 + 1e-12


def test_clipping_is_reported_and_tolerance_is_measured_not_assumed() -> None:
    values = (0.0, 10.0)
    quantized = quantize_int8(values, 0.01)

    assert quantized.clipped_count == 1
    assert quantization_preserved(values, quantized, epsilon=0.01) is False


def test_semantic_preservation_uses_supplied_metric() -> None:
    reference = {"class": "mesh", "score": 1.0}
    reconstructed = {"class": "mesh", "score": 0.97}

    def metric(left: object, right: object) -> float:
        lhs = left
        rhs = right
        assert isinstance(lhs, dict)
        assert isinstance(rhs, dict)
        if lhs["class"] != rhs["class"]:
            return 1.0
        return abs(float(lhs["score"]) - float(rhs["score"]))

    assert semantic_preserved(reference, reconstructed, metric=metric, epsilon=0.05)
    assert not semantic_preserved(reference, reconstructed, metric=metric, epsilon=0.01)


@pytest.mark.parametrize("scale", [0.0, -1.0, math.inf, math.nan])
def test_quantization_rejects_invalid_scales(scale: float) -> None:
    with pytest.raises(ValueError):
        quantize_int8((0.0,), scale)
