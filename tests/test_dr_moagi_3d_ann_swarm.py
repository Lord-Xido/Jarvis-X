"""Numerical regression tests for the bounded 3D trainable swarm."""
import numpy as np
import pytest

from jarvisx.dr_moagi_3d_ann_swarm import (
    VIRTUAL_SIDE, SelfOptimisingSwarm3DANN, _digest_position_and_features,
)


def test_sha_embedding_is_finite_and_in_range():
    for i in range(2000):
        pos, feats = _digest_position_and_features(f"seed-{i}")
        assert np.all(np.isfinite(pos)) and np.all((pos >= 0) & (pos < 1))
        assert np.all(np.isfinite(feats)) and np.all(np.abs(feats) <= 1)


def test_contraction_preserves_centroid_and_pairwise_differences():
    swarm = SelfOptimisingSwarm3DANN(n_particles=12, min_particles=8, max_iterations=1)
    swarm.embed("test")
    before = np.stack([p.pos for p in swarm.particles])
    centroid = before.mean(axis=0)
    swarm.contractive_step()
    after = np.stack([p.pos for p in swarm.particles])
    gain = swarm.contract_strength * 0.15
    np.testing.assert_allclose(after[0] - after[1], (1 - gain) * (before[0] - before[1]), atol=2e-7)
    np.testing.assert_allclose(after.mean(axis=0), centroid, atol=2e-7)


def test_training_reduces_loss_without_pruning():
    s = SelfOptimisingSwarm3DANN(n_particles=8, min_particles=8, max_iterations=130)
    r = s.evolve("Neural particles occupy a virtual volumetric state")
    assert r["final_recon_error"] < s.history_error[0] * 0.08
    assert r["final_particles"] == 8


def test_pruning_and_repeatability_and_no_shape_crash():
    s = SelfOptimisingSwarm3DANN(
        n_particles=96, min_particles=28, max_iterations=200, meta_every=8, seed=42
    )
    first = s.evolve("def contractive_map(x): return x*0.5")
    first_pos = s.get_fixed_point_positions().copy()
    second = s.evolve("def contractive_map(x): return x*0.5")
    np.testing.assert_array_equal(first_pos, s.get_fixed_point_positions())
    assert first["final_recon_error"] == second["final_recon_error"]
    assert 28 <= first["final_particles"] < 96
    assert np.isfinite(first["final_recon_error"])
    assert s.virtual_coordinates().min() >= 0
    assert s.virtual_coordinates().max() < VIRTUAL_SIDE
    assert s.get_swarm_latents().shape == (first["final_particles"], 16)


def test_input_validation_and_new_state_on_repeated_run():
    with pytest.raises(ValueError):
        SelfOptimisingSwarm3DANN(n_particles=3, min_particles=4)
    with pytest.raises(ValueError):
        SelfOptimisingSwarm3DANN(contract_strength=1)
    s = SelfOptimisingSwarm3DANN(n_particles=5, min_particles=3, max_iterations=4)
    a = s.evolve("first input")
    b = s.evolve("new input")
    assert a["iterations"] == b["iterations"] == 4
    assert len(s.history_error) == 4
