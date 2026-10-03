import pytest

from jarvisx.jxp_c3d import (
    IntegrityError,
    JXPC3DFrame,
    JXPC3DRuntime,
    Modality,
    Opcode,
    ReferenceInwardCodec,
    SpatialAddress,
    route_coordinate,
)


@pytest.mark.parametrize(
    "address",
    [
        SpatialAddress(0, 0, 0),
        SpatialAddress(1, 2, 3),
        SpatialAddress(511, 512, 513),
        SpatialAddress(999, 999, 999),
    ],
)
def test_morton30_round_trip(address):
    assert SpatialAddress.from_morton30(address.morton30) == address
    assert address.morton30 < (1 << 30)


def test_route_coordinate_is_deterministic_and_bounded():
    a = route_coordinate("stream-a", 42)
    b = route_coordinate("stream-a", 42)
    c = route_coordinate("stream-a", 43)

    assert a == b
    assert a != c
    assert 0 <= a.linear < 1000**3


def test_reference_codec_contracts_latent_spread_and_reconstructs_exactly():
    codec = ReferenceInwardCodec()
    source = bytes(range(256))

    latent, residual, before, after = codec.encode(source, iterations=5, rho=0.5)
    reconstructed = codec.decode(latent, residual)

    assert reconstructed == source
    assert after <= before
    assert len(latent) == len(source)
    assert len(residual) == 2 * len(source)


def test_transfer_materializes_only_verified_active_voxels():
    runtime = JXPC3DRuntime()
    source = b"Jarvis-X JXP-C3D" * 32

    receipt = runtime.transfer(
        source,
        stream_id="media-1",
        sequence=7,
        modality=Modality.BINARY,
        iterations=4,
    )

    assert receipt.verified is True
    assert receipt.active_cells == 1
    assert runtime.store.logical_cells == 1000**3
    assert runtime.reconstruct(receipt.address) == source

    runtime.transfer(
        b"second",
        stream_id="media-1",
        sequence=8,
        modality=Modality.TEXT,
        iterations=2,
    )
    assert runtime.store.active_cells == 2


def test_frame_detects_payload_corruption():
    source = b"abc123"
    codec = ReferenceInwardCodec()
    latent, residual, _, _ = codec.encode(source, iterations=2)
    frame = JXPC3DFrame(
        stream_id="s",
        sequence=0,
        address=SpatialAddress(1, 2, 3),
        modality=Modality.BINARY,
        opcode=Opcode.COMMIT,
        iteration=2,
        latent=latent,
        residual=residual,
        source_sha256=__import__("hashlib").sha256(source).hexdigest(),
    )
    wire = bytearray(frame.pack())
    wire[-1] ^= 0x01

    with pytest.raises(IntegrityError, match="payload hash mismatch"):
        JXPC3DFrame.unpack(bytes(wire))


def test_failed_candidate_is_not_committed():
    class BadCodec(ReferenceInwardCodec):
        def decode(self, latent: bytes, residual: bytes) -> bytes:
            decoded = super().decode(latent, residual)
            return decoded[:-1] + bytes([decoded[-1] ^ 1]) if decoded else b"x"

    runtime = JXPC3DRuntime(codec=BadCodec())

    with pytest.raises(IntegrityError, match="source reconstruction"):
        runtime.transfer(b"must-not-commit", stream_id="bad")

    assert runtime.store.active_cells == 0


def test_empty_payload_round_trip():
    runtime = JXPC3DRuntime()
    receipt = runtime.transfer(b"", stream_id="empty", iterations=0)

    assert receipt.verified is True
    assert receipt.spread_before == 0
    assert receipt.spread_after == 0
    assert runtime.reconstruct(receipt.address) == b""
