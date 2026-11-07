"""Executable invariants for the independent MagnusStream numerical verifier.

Run: python -m pytest -q -o addopts='' apps/magnusstream-3d/test_runtime.py
"""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import verify_magnusstream_3d as m  # noqa: E402


def test_dimensions_and_encoded_geometry():
    x = m.make_field()
    z = m.encode(x)
    assert x.shape == (16, 16, 16)
    assert z.shape == (8, 8, 8)
    assert m.decode(z).shape == x.shape
    assert z.size == 512 and x.size == 4096


def test_constant_fixed_point():
    x = np.full((16, 16, 16), 0.625)
    np.testing.assert_allclose(m.F(x), x, rtol=0, atol=1e-12)


def test_mean_and_variance_invariants():
    x = m.make_field()
    before = float(x.mean())
    initial_variance = float(x.var())
    for _ in range(120):
        x = m.F(x)
        assert abs(float(x.mean()) - before) < 1e-11
        assert float(x.var()) <= initial_variance + 1e-12
        initial_variance = float(x.var())


def test_binary_transport_roundtrip_and_corruption_rejection():
    latent = m.encode(m.make_field())
    payload = m.pack_latent(latent)
    assert len(payload) == 2080
    assert 16384 / len(payload) == pytest.approx(7.876923076923077)
    restored = m.unpack_latent(payload)
    assert np.max(np.abs(restored - latent)) < 4e-8
    assert m.pack_latent(restored) == payload  # exact container-byte identity
    with pytest.raises(ValueError):
        m.unpack_latent(b'FAIL' + payload[4:])
    with pytest.raises(ValueError):
        m.unpack_latent(payload[:-1])
    bad_shape = bytearray(payload)
    bad_shape[8] ^= 0x01
    with pytest.raises(ValueError):
        m.unpack_latent(bytes(bad_shape))
    nonfinite = bytearray(payload)
    nonfinite[32:36] = b'\x00\x00\xc0\x7f'  # little-endian quiet NaN
    with pytest.raises(ValueError):
        m.unpack_latent(bytes(nonfinite))
    with pytest.raises(ValueError):
        m.pack_latent(np.full((8, 8, 8), np.nan))


def test_fixed_point_is_not_faithful_reconstruction():
    source = m.make_field()
    state = source.copy()
    for _ in range(1000):
        state = m.F(state)
    assert m.metrics(state, m.F(state))['residual'] < 1e-8
    assert m.metrics(source, state)['mse'] > 0.018
    assert float(state.var()) < 1e-18


def test_receipt_reproducible_and_bounded(tmp_path):
    target = tmp_path / 'verification.json'
    result = subprocess.run(
        [sys.executable, str(ROOT / 'verify_magnusstream_3d.py'),
         '--loops', '1000', '--json', str(target)],
        cwd=str(ROOT), capture_output=True, text=True, check=True,
    )
    assert result.returncode == 0
    receipt = json.loads(target.read_text())
    golden = json.loads((ROOT / 'reference_results.json').read_text())
    assert receipt['source_sha256_float32'] == golden['source_sha256_float32']
    assert receipt['encoded_sha256'] == golden['encoded_sha256']
    assert receipt['iterations'] == 1000
    assert receipt['original_artifact_verified_as_real_codec'] is False
    assert receipt['raw_bytes'] == 16384 and receipt['encoded_bytes_with_header'] == 2080
    assert all(v is True for v in receipt['checks'].values())
    assert receipt['final_relative_residual'] < 1e-8
    assert abs(receipt['final_reconstruction_metrics']['mse'] -
               golden['final_reconstruction_metrics']['mse']) < 1e-11
