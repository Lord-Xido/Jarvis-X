"""Focused geometry, certificate, replay, input validation and CLI tests."""
from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from jarvisx.inward_loop_certified import (
    Certificate,
    Field,
    FourierDecoder,
    FourierEncoder,
    InwardLoop,
    canonical_bytes,
    projection_jacobian,
    run,
    torus,
    verify_manifest,
)


def test_exact_latent_shape_and_angles() -> None:
    P = torus(np.array([0.2, 0.2]), np.array([0.3, 2.1]))
    enc = FourierEncoder()
    Z = enc(P)
    assert Z.shape == (2, 16)
    assert Z[0, 0] == pytest.approx(np.cos(0.2))
    assert Z[0, 4] == pytest.approx(np.sin(0.2))
    assert Z[0, 8] == pytest.approx(np.cos(0.3))
    assert Z[0, 12] == pytest.approx(np.sin(0.3))
    assert not np.array_equal(Z[0], Z[1]), "v-harmonics must not be silently discarded"


def test_roundtrip_and_idempotence_full_torus() -> None:
    u, v = np.meshgrid(np.linspace(0, 2*np.pi, 41), np.linspace(0, 2*np.pi, 29))
    P = torus(u.ravel(), v.ravel())
    enc, dec = FourierEncoder(), FourierDecoder()
    X = dec(enc(P))
    assert np.max(np.linalg.norm(X - P, axis=1)) < 1e-12
    assert np.max(np.abs(dec(enc(X)) - X)) < 1e-12


def test_pointwise_projector_jacobian_has_rank_two() -> None:
    enc, dec = FourierEncoder(), FourierDecoder()
    p = torus(np.array([1.1]), np.array([0.7]))[0]
    J = projection_jacobian(p, enc, dec)
    assert np.max(np.abs(J @ J - J)) < 1e-8
    assert np.max(np.abs(np.sort(np.linalg.eigvals(J).real) - [0, 1, 1])) < 1e-8


def test_deterministic_chain_and_different_seed() -> None:
    first = run(frames=4, n=64, seed=13)
    second = run(frames=4, n=64, seed=13)
    other = run(frames=4, n=64, seed=14)
    assert first == second
    assert first["final_root"] != other["final_root"]
    assert len(first["final_root"]) == 64
    assert all(m["residual_max"] < 1e-12 for m in first["metrics"])
    assert all(m["eig_error"] < 1e-8 for m in first["metrics"])


def test_manifest_replay_rejects_tampering() -> None:
    payload = run(frames=3, n=32, seed=4)
    assert verify_manifest(payload)
    corrupted = json.loads(json.dumps(payload))
    corrupted["metrics"][1]["residual_mean"] = 0.3
    assert not verify_manifest(corrupted)
    corrupted = json.loads(json.dumps(payload))
    corrupted["metrics"][0]["cert_root"] = "00" * 32
    assert not verify_manifest(corrupted)


def test_invalid_shapes_and_inputs() -> None:
    with pytest.raises(ValueError):
        FourierEncoder()(np.zeros((4, 4)))
    with pytest.raises(ValueError):
        FourierEncoder()(np.zeros((4, 3)))  # revolution axis has no unique u
    with pytest.raises(ValueError):
        FourierDecoder()(np.zeros((2, 16)))
    with pytest.raises(ValueError):
        Field(n=0)
    with pytest.raises(ValueError):
        InwardLoop(dt=0)
    with pytest.raises(ValueError):
        run(frames=0)


def test_byte_encoding_distinguishes_shapes() -> None:
    assert canonical_bytes(np.arange(6).reshape(2, 3)) != canonical_bytes(np.arange(6).reshape(3, 2))
    assert Certificate(seed=1, n=16, dt=1/60).chain != Certificate(seed=2, n=16, dt=1/60).chain


def test_cli_generate_and_verify_manifest(tmp_path) -> None:
    target = tmp_path / "trace.json"
    command = [sys.executable, "-m", "jarvisx.inward_loop_certified"]
    subprocess.run(command + ["--frames", "3", "--particles", "32", "--json", str(target)], check=True)
    subprocess.run(command + ["--verify", str(target)], check=True)
    payload = json.loads(target.read_text())
    assert payload["configuration"]["frames"] == 3
