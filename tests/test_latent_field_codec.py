from __future__ import annotations

import math

import pytest

from jarvisx.latent_field_codec import (
    ADDRESS_BITS,
    LATENT_BYTES,
    WORLD_EXTENT,
    FieldObservation,
    LatentFieldCodec,
    LatentFieldConfig,
    LatentFieldState,
)


def smooth_field(x: int, y: int, z: int) -> float:
    extent = WORLD_EXTENT - 1
    nx, ny, nz = x / extent, y / extent, z / extent
    return 0.1 + 0.2 * nx + 0.25 * ny + 0.3 * nz


def test_default_codec_is_exactly_one_kib_and_addresses_the_full_cube():
    codec = LatentFieldCodec()

    assert codec.latent_bytes == 1024
    assert codec.virtual_voxels == 1024**3
    assert codec.raw_u8_storage_bytes == 1024**3
    assert codec.logical_expansion_ratio == 1_048_576
    assert ADDRESS_BITS == 30
    assert codec.linear_address(1023, 1023, 1023) == 2**30 - 1


def test_config_requires_exactly_1024_latent_cells():
    with pytest.raises(ValueError, match="product"):
        LatentFieldConfig(latent_shape=(8, 8, 8))

    with pytest.raises(ValueError, match="1,024"):
        LatentFieldConfig(latent_shape=(8, 8, 16), latent_bytes=2048)


def test_constant_field_round_trips_with_only_quantization_error():
    codec = LatentFieldCodec()
    state = codec.encode_field(lambda _x, _y, _z: 0.5)

    assert isinstance(state, LatentFieldState)
    assert len(state.payload) == LATENT_BYTES
    assert codec.decode_voxel(state, 0, 0, 0) == pytest.approx(128 / 255)
    assert codec.decode_voxel(state, 512, 512, 512) == pytest.approx(128 / 255)
    assert codec.self_consistency_error(state) < 1.0e-15


def test_smooth_field_is_reconstructed_by_trilinear_latent_decoder():
    codec = LatentFieldCodec()
    state = codec.encode_field(smooth_field)

    for coordinate in ((0, 0, 0), (123, 456, 789), (512, 512, 512), (1023, 1023, 1023)):
        expected = smooth_field(*coordinate)
        assert codec.decode_voxel(state, *coordinate) == pytest.approx(expected, abs=0.004)


def test_materialization_is_lazy_and_bounded_to_requested_slice():
    codec = LatentFieldCodec()
    state = codec.encode_field(smooth_field)

    slice_2d = codec.materialize_slice(state, axis="z", index=512, resolution=12)

    assert len(slice_2d) == 12
    assert all(len(row) == 12 for row in slice_2d)
    assert all(0.0 <= value <= 1.0 for row in slice_2d for value in row)


def test_inward_refinement_reduces_observation_loss_transactionally():
    codec = LatentFieldCodec()
    state = LatentFieldState(bytes([64] * LATENT_BYTES))
    observations = [
        FieldObservation(128, 128, 128, 0.8),
        FieldObservation(512, 512, 512, 0.7),
        FieldObservation(900, 700, 400, 0.2),
    ]

    report = codec.refine(state, observations, learning_rate=0.5)

    assert report.committed
    assert report.state.revision == 1
    assert report.loss_after <= report.loss_before
    assert report.learning_rate_used > 0.0
    assert report.state.payload != state.payload


def test_decode_points_matches_individual_queries():
    codec = LatentFieldCodec()
    state = codec.encode_field(smooth_field)
    points = ((1, 2, 3), (100, 200, 300), (1023, 1023, 1023))

    batch = codec.decode_points(state, points)

    assert batch == pytest.approx(tuple(codec.decode_voxel(state, *point) for point in points))


@pytest.mark.parametrize(
    "call",
    [
        lambda codec, state: codec.decode_voxel(state, -1, 0, 0),
        lambda codec, state: codec.decode_voxel(state, 0, 1024, 0),
        lambda codec, state: codec.materialize_slice(state, axis="q", index=0),
        lambda codec, state: codec.materialize_slice(state, axis="z", index=1024),
    ],
)
def test_invalid_queries_are_rejected(call):
    codec = LatentFieldCodec()
    state = codec.encode_field(smooth_field)

    with pytest.raises((TypeError, ValueError)):
        call(codec, state)


def test_encoder_rejects_non_finite_or_out_of_range_fields():
    codec = LatentFieldCodec()

    with pytest.raises(ValueError):
        codec.encode_field(lambda _x, _y, _z: math.inf)

    with pytest.raises(ValueError):
        codec.encode_field(lambda _x, _y, _z: 1.1)
