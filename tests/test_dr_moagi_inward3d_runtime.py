import numpy as np
import pytest

pytest.importorskip("torch")

from jarvisx.dr_moagi_inward3d_runtime import (
    Inward3DConfig,
    InwardRecursive3DRuntime,
    VirtualBitVolume,
    bit_identity,
    xnor_popcount_dot,
)


def test_virtual_bit_volume_round_trip_and_mapping() -> None:
    payload = b"def"
    volume = VirtualBitVolume(16)
    tiles = volume.encode(payload)
    assert len(tiles) == 1
    assert tiles[0].shape == (8, 16, 16, 16)
    assert volume.decode(tiles, len(payload)) == payload
    assert tiles[0][:, 0, 0, 0].astype(int).tolist() == [0, 0, 1, 0, 0, 1, 1, 0]


def test_xnor_popcount_matches_bipolar_dot_product() -> None:
    a = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
    b = np.array([1, 1, 1, 0, 0, 0, 1, 1], dtype=np.uint8)
    expected = int(np.dot(2 * a.astype(int) - 1, 2 * b.astype(int) - 1))
    assert xnor_popcount_dot(a, b) == expected == 2


def test_exact_identity_gate() -> None:
    payload = b"Jarvis-X"
    exact = bit_identity(payload, payload)
    assert exact.exact
    assert exact.differing_bits == 0
    changed = bit_identity(payload, b"Jarvis-Y")
    assert not changed.exact
    assert changed.differing_bits > 0


def test_tile_edge_boundary_is_explicit() -> None:
    with pytest.raises(ValueError, match="tile_edge"):
        Inward3DConfig(tile_edge=8).validate()


def test_true_latent_recursion_changes_state() -> None:
    config = Inward3DConfig(base_channels=2, latent_channels=4, recursion_depth=3, seed=11)
    runtime = InwardRecursive3DRuntime(config)
    tile = runtime.volume.encode(b"print('echo')\n")[0]
    _, folds = runtime.reconstruct_tile(tile)
    assert len(folds) == 3
    assert all(fold.latent_delta_rms > 0 for fold in folds)


def test_guarded_training_never_returns_worse_accepted_score() -> None:
    config = Inward3DConfig(base_channels=2, latent_channels=4, recursion_depth=2, seed=3)
    runtime = InwardRecursive3DRuntime(config)
    tile = runtime.volume.encode(b"def f(x):\n    return x + 1\n")[0]
    accepted, before, after = runtime.guarded_train_tile(tile, steps=1)
    if accepted:
        assert after < before
    else:
        assert after == pytest.approx(before)
