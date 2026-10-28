"""Non-disruptive integration tests for the optional Hyper-Matrix research engine."""

import numpy as np
import pytest

pytest.importorskip("scipy")

from jarvisx.dr_moagi_hypermatrix_engine import (
    Config,
    morton_decode,
    morton_encode,
    octree_decode,
    octree_encode,
    run_engine,
)


def test_morton_unique_and_reversible():
    indices = [
        morton_encode(x, y, z, 3)
        for x in range(8)
        for y in range(8)
        for z in range(8)
    ]
    assert len(set(indices)) == 512
    for index in indices:
        assert morton_encode(*morton_decode(index, 3), 3) == index


def test_lossless_integer_octree_and_reject_unrepresentable():
    rng = np.random.default_rng(20261028)
    volume = rng.integers(-5000, 5000, (8, 8, 8), dtype=np.int64)
    anchor, residuals = octree_encode(volume)
    np.testing.assert_array_equal(octree_decode(anchor, residuals), volume)
    with pytest.raises(ValueError):
        octree_encode(np.ones((3, 3, 3), dtype=np.int64))


def test_end_to_end_small_bounded_run(tmp_path):
    config = Config(nu=8, nv=8, side=8, steps=4)
    result = run_engine(config, outdir=tmp_path)
    assert result["surface_nodes"] == 64
    assert result["morton_bijection_test"]
    assert result["octree"]["exact_integer_roundtrip"]
    assert result["octree"]["independent_disk_roundtrip"]
    assert result["octree"]["sha256_original"] == result["octree"]["sha256_decoded"]
    assert result["graph_fixed_point"]["stopped_at_tolerance"]
    assert (tmp_path / "verification.json").is_file()
    assert (tmp_path / "spatial_results.npz").is_file()
    with np.load(tmp_path / "spatial_results.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(
            saved["input_voxels"], saved["reconstructed_voxels"]
        )
