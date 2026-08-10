import hashlib

import pytest

from jarvisx.dr_moagi_codec_3d import (
    CodecConfig,
    CodecFormatError,
    DrMoagiCodec3D,
    Volume3D,
)


def _fixture(offset: float = 0.0) -> Volume3D:
    values = tuple(
        offset + float((x * 3 + y * 2 + z) % 11) / 3.0
        for x in range(4)
        for y in range(4)
        for z in range(4)
    )
    return Volume3D((4, 4, 4), values)


def test_codec_is_deterministic_and_respects_quantization_error_bound() -> None:
    volume = _fixture()
    config = CodecConfig(quant_step=0.2)
    codec = DrMoagiCodec3D(config)

    first = codec.encode(volume)
    second = codec.encode(volume)
    reconstructed = codec.decode(first)

    assert first == second
    assert reconstructed.shape == volume.shape
    assert volume.mse(reconstructed) <= (config.quant_step / 2.0) ** 2 + 1.0e-12
    metadata = codec.inspect(first)
    payload = first[-metadata.payload_bytes :]
    assert metadata.payload_digest_sha256 == hashlib.sha256(payload).hexdigest()


def test_codec_rejects_corrupted_payload() -> None:
    codec = DrMoagiCodec3D()
    bitstream = bytearray(codec.encode(_fixture()))
    bitstream[-1] ^= 0x01

    with pytest.raises(CodecFormatError, match="integrity"):
        codec.decode(bytes(bitstream))


def test_transaction_commits_then_rolls_back_anchor_drift() -> None:
    codec = DrMoagiCodec3D(CodecConfig(quant_step=0.1, max_anchor_mse=1.0))
    first = codec.process(_fixture())
    assert first.committed
    memory_after_first = codec.memory

    second = codec.process(_fixture(offset=10.0))

    assert not second.committed
    assert second.rejection_reason == "anchor distortion exceeds configured ceiling"
    assert codec.memory == memory_after_first
    assert second.memory_before == second.memory_after


def test_virtual_depth_is_telemetry_not_invented_throughput() -> None:
    codec = DrMoagiCodec3D(CodecConfig(virtual_depth=1_000_000))
    result = codec.process(_fixture())

    assert result.committed
    assert result.virtual_depth == 1_000_000
    assert result.measured_microsteps_executed == 1
    assert result.measured_throughput_voxels_per_second >= 0.0


def test_invalid_shape_and_resource_limits_fail_closed() -> None:
    with pytest.raises(ValueError, match="expected"):
        Volume3D((2, 2, 2), (0.0,))

    codec = DrMoagiCodec3D(CodecConfig(max_voxels=4))
    with pytest.raises(ValueError, match="max_voxels"):
        codec.encode(_fixture())
