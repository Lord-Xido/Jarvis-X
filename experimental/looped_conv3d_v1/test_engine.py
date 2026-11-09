import math
from pathlib import Path

import pytest
import torch

from engine import (
    ARCHIVE_SIZE, LATENT_DIM, LoopedConv3D, demo, load_checkpoint, optimise,
    pack_latent, reproducible, run_from_latent, save_checkpoint, score,
    source_volume, training_objective, unpack_latent,
)


@pytest.fixture(autouse=True)
def deterministic_cpu():
    reproducible(42)


@pytest.mark.parametrize("side", [64, 128])
def test_five_stride2_shapes(side):
    model = LoopedConv3D(side).eval()
    original = source_volume(side)
    with torch.no_grad():
        encoded = model.encoder(original)
        assert encoded.shape == (1, 16, side // 32, side // 32, side // 32)
        z = model.encode(original)
        assert z.shape == (1, 512)
        reconstructed = model.decode(z)
        assert reconstructed.shape == original.shape
        assert (reconstructed >= 0).all() and (reconstructed <= 1).all()
        assert math.isfinite(score(original, reconstructed)["mse_norm"])


def test_loopback_from_latent_and_guard_boundary():
    model = LoopedConv3D(64).eval()
    original = source_volume(64)
    with torch.no_grad():
        z = model.encode(original)
        y, records = run_from_latent(model, z, cycles=3, reference=original)
        assert y.shape == original.shape
        assert len(records) == 3
        assert all(record["mse_norm"] is not None for record in records)
        guarded, history = run_from_latent(model, z, cycles=3, reference=original, guard=True)
        assert guarded.shape == original.shape
        assert score(original, guarded)["mse_norm"] <= min(r["mse_norm"] for r in history) + 1e-9
        with pytest.raises(ValueError, match="guard requires ORIGINAL"):
            run_from_latent(model, z, guard=True)
        decoded, observations = run_from_latent(model, z, cycles=2)
        assert decoded.shape == original.shape
        assert all(r["mse_norm"] is None and r["status"] == "unverified" for r in observations)


def test_monotone_training_and_failure_rolls_back():
    model = LoopedConv3D(64)
    original = source_volume(64)
    result = optimise(model, original, cycles=2, updates=2, lr=0.1)
    assert result.after <= result.before
    assert result.accepted >= 1
    with torch.no_grad():
        observed = float(training_objective(model, original, cycles=2))
    assert abs(observed - result.after) < 1e-8
    state_before = {k: v.clone() for k, v in model.state_dict().items()}
    # Guaranteed failure: a huge full-gradient step, without backtracking.
    failure = optimise(model, original, cycles=2, updates=1, lr=1e12, attempts=1)
    assert failure.accepted == 0
    assert failure.rejected == 1
    assert all(torch.equal(state_before[k], v) for k, v in model.state_dict().items())


def test_archive_checkpoint_and_corruption(tmp_path: Path):
    model = LoopedConv3D(64).eval()
    original = source_volume(64)
    checkpoint = tmp_path / "model.safetensors"
    digest = save_checkpoint(model, checkpoint)
    stored = load_checkpoint(checkpoint, 64, expected_sha=digest)
    archive = pack_latent(model, original, digest)
    assert len(archive) == ARCHIVE_SIZE == 2128
    latent = unpack_latent(archive, stored, digest)
    with torch.no_grad():
        before = model.decode(model.encode(original))
        after = stored.decode(latent)
    assert torch.equal(before, after)
    with pytest.raises(ValueError, match="model weights"):
        unpack_latent(archive, stored, "0" * 64)
    with pytest.raises(ValueError, match="truncated"):
        unpack_latent(archive[:-1], stored, digest)
    corrupted = bytearray(archive)
    corrupted[100] ^= 0x80
    with pytest.raises(ValueError, match="checksum"):
        unpack_latent(bytes(corrupted), stored, digest)
    with pytest.raises(ValueError, match="configuration mismatch"):
        load_checkpoint(checkpoint, 128)
    with pytest.raises(ValueError, match="digest mismatch"):
        load_checkpoint(checkpoint, 64, expected_sha="0" * 64)


def test_full_demo_replay(tmp_path: Path):
    evidence = demo(64, updates=1, cycles=2, output=tmp_path)
    assert evidence["replay_exact_same_machine"]
    assert evidence["archive_bytes"] == 2128
    assert evidence["training"]["after"] <= evidence["training"]["before"]
    assert evidence["latent_values"] == LATENT_DIM
    assert (tmp_path / "trace.json").exists()
