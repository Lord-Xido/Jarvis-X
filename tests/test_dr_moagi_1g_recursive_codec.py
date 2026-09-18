import random
import zlib

import pytest

from jarvisx.dr_moagi_1g_recursive_codec import (
    BRICK_BYTES,
    BRICK_COUNT,
    BRICK_SIDE,
    GIB_BYTES,
    VOLUME_SIDE,
    CodecPolicy,
    Recursive3DCodec,
    VolumeGeometry1GiB,
)


def structured_brick() -> bytes:
    pattern = bytes(range(64)) + bytes(reversed(range(64)))
    repeats, tail = divmod(BRICK_BYTES, len(pattern))
    return pattern * repeats + pattern[:tail]


def test_exact_1gib_geometry_and_brick_partition():
    geometry = VolumeGeometry1GiB()
    assert VOLUME_SIDE**3 == GIB_BYTES == 1 << 30
    assert BRICK_SIDE**3 == BRICK_BYTES == 262_144
    assert BRICK_COUNT == 4096
    assert geometry.brick_count * geometry.brick_bytes == geometry.volume_bytes


@pytest.mark.parametrize(
    "xyz",
    [(0, 0, 0), (1, 2, 3), (63, 127, 511), (1023, 1023, 1023)],
)
def test_coordinate_roundtrip(xyz):
    geometry = VolumeGeometry1GiB()
    offset = geometry.xyz_to_offset(*xyz)
    assert geometry.offset_to_xyz(offset) == xyz


def test_brick_origin_is_inside_volume():
    geometry = VolumeGeometry1GiB()
    assert geometry.brick_origin(0) == (0, 0, 0)
    assert geometry.brick_origin(BRICK_COUNT - 1) == (960, 960, 960)


def test_lossless_latent_xor_roundtrip_and_bounded_refinement():
    source = structured_brick()
    codec = Recursive3DCodec(CodecPolicy(latent_side=8, compression_level=6))
    frame = codec.encode_brick(source, brick_index=17)
    assert codec.decode_brick(frame) == source
    assert frame.refine_steps <= codec.policy.max_refine_steps
    assert frame.converged
    assert frame.encoded_size <= BRICK_BYTES + 24


def test_high_entropy_input_still_roundtrips_under_no_expansion_fallback():
    source = random.Random(7).randbytes(BRICK_BYTES)
    codec = Recursive3DCodec()
    frame = codec.encode_brick(source)
    assert codec.decode_brick(frame) == source
    assert frame.encoded_size <= BRICK_BYTES + 24


def test_optimizer_is_candidate_first_and_promotion_is_explicit():
    source = structured_brick()
    baseline = CodecPolicy(latent_side=32, compression_level=1)
    codec = Recursive3DCodec(baseline)
    decision = codec.optimize_policy(
        source,
        candidates=[
            CodecPolicy(latent_side=4, compression_level=9),
            CodecPolicy(latent_side=8, compression_level=9),
        ],
    )

    assert codec.policy == baseline
    assert decision.exact_roundtrip
    if decision.accepted:
        assert decision.candidate_size < decision.baseline_size
        assert codec.promote(decision)
        assert codec.policy == decision.candidate_policy
    else:
        assert not codec.promote(decision)
        assert codec.policy == baseline


def test_corrupt_payload_fails_closed():
    source = structured_brick()
    codec = Recursive3DCodec()
    frame = codec.encode_brick(source)
    if frame.mode == "latent+xor":
        corrupt = type(frame)(
            **{
                **frame.__dict__,
                "residual_payload": frame.residual_payload[:-1] + b"x",
            }
        )
        with pytest.raises((ValueError, zlib.error)):
            codec.decode_brick(corrupt)
