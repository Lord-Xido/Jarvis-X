import pytest

from jarvisx.fractal_octree import build_fractal_octree
from jarvisx.psi_engine import OctreeSpatialMask, PsiEngine, identity_operator


def test_equation_composition_matches_operator_order():
    engine = PsiEngine(
        encoder=lambda values: (2.0 * values[0], 2.0 * values[1]),
        fusion=lambda values: (values[0] + 1.0, values[1] - 1.0),
        decoder=lambda values: (values[0] + values[1], values[0] - values[1]),
        octree_mask=lambda _x, _y, _z: 0.5,
    )

    trace = engine.evaluate_blocks(
        x=0.1,
        y=0.2,
        z=0.3,
        t=4.0,
        blocks=((1.0, 2.0), (3.0, 4.0)),
    )

    assert trace.encoded_blocks == ((2.0, 4.0), (6.0, 8.0))
    assert trace.latent_sum == (4.0, 6.0)
    assert trace.fused_latent == (5.0, 5.0)
    assert trace.output == (10.0, 0.0)
    assert trace.octree_mask == 0.5
    assert trace.block_count == 2


def test_octree_spatial_mask_projects_culled_octants_to_zero():
    root = build_fractal_octree(size=1.0, max_depth=1)
    mask = OctreeSpatialMask(root)

    assert mask(0.25, 0.25, 0.25) == 1.0
    assert mask(0.75, 0.75, 0.75) == 0.0
    assert mask(1.25, 0.25, 0.25) == 0.0


def test_evaluate_field_samples_every_block_at_same_spacetime_coordinate():
    calls = []

    def field(x, y, z, t, block_index):
        calls.append((x, y, z, t, block_index))
        return (x + y + z + t + block_index,)

    engine = PsiEngine(
        encoder=identity_operator,
        fusion=identity_operator,
        decoder=identity_operator,
        octree_mask=lambda _x, _y, _z: 1.0,
    )

    trace = engine.evaluate_field(1.0, 2.0, 3.0, 4.0, 3, field)

    assert calls == [
        (1.0, 2.0, 3.0, 4.0, 0),
        (1.0, 2.0, 3.0, 4.0, 1),
        (1.0, 2.0, 3.0, 4.0, 2),
    ]
    assert trace.latent_sum == (33.0,)
    assert trace.output == (33.0,)
    assert trace.time == 4.0


def test_culled_region_zeroes_aggregate_before_fusion_and_decode():
    root = build_fractal_octree(size=1.0, max_depth=1)
    engine = PsiEngine(
        encoder=identity_operator,
        fusion=identity_operator,
        decoder=identity_operator,
        octree_mask=OctreeSpatialMask(root),
    )

    trace = engine.evaluate_blocks(
        x=0.75,
        y=0.75,
        z=0.75,
        t=0.0,
        blocks=((2.0, -3.0), (5.0, 7.0)),
    )

    assert trace.octree_mask == 0.0
    assert trace.latent_sum == (0.0, 0.0)
    assert trace.output == (0.0, 0.0)


def test_encoder_outputs_must_share_one_latent_dimension():
    engine = PsiEngine(
        encoder=identity_operator,
        fusion=identity_operator,
        decoder=identity_operator,
        octree_mask=lambda _x, _y, _z: 1.0,
    )

    with pytest.raises(ValueError, match="same dimension"):
        engine.evaluate_blocks(0.0, 0.0, 0.0, 0.0, ((1.0,), (2.0, 3.0)))


def test_evaluate_field_rejects_nonpositive_block_count():
    engine = PsiEngine(
        encoder=identity_operator,
        fusion=identity_operator,
        decoder=identity_operator,
        octree_mask=lambda _x, _y, _z: 1.0,
    )

    with pytest.raises(ValueError, match="n_blocks must be positive"):
        engine.evaluate_field(0.0, 0.0, 0.0, 0.0, 0, lambda *_args: (1.0,))
