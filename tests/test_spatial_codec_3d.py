from __future__ import annotations

import pytest

from jarvisx.auto_codec_loop import AutoCodecLoop, AutoCodecLoopConfig
from jarvisx.dr_moagi_field_runtime import DrMoagiFieldConfig, DrMoagiFieldRuntime
from jarvisx.spatial_codec_3d import (
    MORTON_MAX_COORDINATE,
    MortonQuantizedFieldCodec3D,
    SpatialAutoCodec3DSystem,
    measure_spatial_field,
    morton_decode_3d,
    morton_encode_3d,
    spatial_frame,
)


def test_morton_round_trip_uses_all_three_axes():
    coordinates = [
        (0, 0, 0),
        (1, 2, 3),
        (17, 31, 63),
        (511, 255, 127),
        (MORTON_MAX_COORDINATE, MORTON_MAX_COORDINATE, MORTON_MAX_COORDINATE),
    ]
    for coordinate in coordinates:
        assert morton_decode_3d(morton_encode_3d(*coordinate)) == coordinate


def test_morton_codec_quantizes_values_and_preserves_spatial_addresses():
    codec = MortonQuantizedFieldCodec3D(step=0.1, side=16)
    source = {(1, 2, 3): 0.26, (4, 5, 6): -0.37}

    latent = codec.encode(source)
    reconstructed = codec.decode(latent, tuple(source))

    assert set(latent.entries) == {
        morton_encode_3d(1, 2, 3),
        morton_encode_3d(4, 5, 6),
    }
    assert reconstructed[(1, 2, 3)] == pytest.approx(0.3)
    assert reconstructed[(4, 5, 6)] == pytest.approx(-0.4)


def test_spatial_metrics_measure_bounds_centroid_links_and_energy():
    field = {
        (1, 1, 1): 1.0,
        (2, 1, 1): -2.0,
        (2, 2, 1): 3.0,
    }

    metrics = measure_spatial_field(field, side=8)

    assert metrics.active_cells == 3
    assert metrics.bounds_min == (1, 1, 1)
    assert metrics.bounds_max == (2, 2, 1)
    assert metrics.centroid == pytest.approx((5 / 3, 4 / 3, 1.0))
    assert metrics.l1_energy == pytest.approx(6.0)
    assert metrics.l2_energy == pytest.approx(14.0)
    assert metrics.six_face_links == 2
    assert metrics.occupancy_ratio == pytest.approx(3 / 512)


def test_spatial_frame_is_deterministically_bounded_in_morton_order():
    field = {(index, 0, 0): float(index) for index in range(10)}

    first = spatial_frame(field, side=16, cycle=3, max_render_points=4)
    second = spatial_frame(field, side=16, cycle=3, max_render_points=4)

    assert first == second
    assert len(first.points) <= 4
    assert first.cycle == 3


def test_3d_system_executes_closed_loop_and_emits_authoritative_frames():
    codec = MortonQuantizedFieldCodec3D(step=0.1, side=16)
    runtime = DrMoagiFieldRuntime(
        codec,
        DrMoagiFieldConfig(
            side=16,
            alpha=1.0,
            lambda_residual=0.0,
            eta=0.0,
            dt=0.1,
            expand_halo=False,
        ),
    )
    loop = AutoCodecLoop(
        runtime,
        AutoCodecLoopConfig(
            max_cycles=8,
            min_cycles=1,
            reconstruction_mse_target=0.0,
            stop_on_fixed_point=True,
        ),
    )
    system = SpatialAutoCodec3DSystem(
        loop,
        codec,
        side=16,
        frame_stride=1,
        max_render_points=64,
        max_frames=16,
    )

    summary = system.run({(2, 2, 2): 0.26, (3, 2, 2): -0.37})
    payload = summary.to_dict()

    assert payload["spatial_mode"] == "3d-morton-quantized"
    assert payload["cycles_executed"] >= 1
    assert payload["journal_verified"] is True
    assert payload["latent_entries"] >= 1
    assert payload["latent_digest"]
    assert payload["final_frame"]["state_digest"] == payload["final_state_digest"]
    assert payload["frames"][0]["cycle"] == 0
    assert payload["frames"][-1]["state_digest"] == payload["final_state_digest"]


def test_3d_system_requires_runtime_and_codec_identity():
    runtime_codec = MortonQuantizedFieldCodec3D(step=0.1, side=8)
    other_codec = MortonQuantizedFieldCodec3D(step=0.1, side=8)
    runtime = DrMoagiFieldRuntime(
        runtime_codec,
        DrMoagiFieldConfig(
            side=8,
            alpha=0.0,
            lambda_residual=0.0,
            eta=0.0,
            dt=0.1,
            expand_halo=False,
        ),
    )
    loop = AutoCodecLoop(runtime)

    with pytest.raises(ValueError, match="same spatial codec"):
        SpatialAutoCodec3DSystem(loop, other_codec, side=8)
