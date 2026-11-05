from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("torch")


MODULE_PATH = Path(__file__).resolve().parents[1] / "examples" / "jarvisx_3d_ide_sdk.py"
SPEC = importlib.util.spec_from_file_location("jarvisx_3d_ide_sdk", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_reference_density_matches_8k_rgb8_120fps():
    density = MODULE.DensitySpec()
    assert density.bytes_per_frame == 99_532_800
    assert density.bytes_per_second == 11_943_936_000
    assert density.bits_per_second / 1e9 == pytest.approx(95.551488)


def test_shared_latent_is_explicitly_three_dimensional():
    cfg = MODULE.Config()
    sdk = MODULE.Permeation3DSDK(cfg, device="cpu")
    z = sdk.encode(MODULE.Modality.CODE, "def f(x):\n    return x\n")
    assert tuple(z.shape) == (1, cfg.latent_channels, cfg.latent_side, cfg.latent_side, cfg.latent_side)


def test_generated_python_is_syntax_valid():
    sdk = MODULE.Permeation3DSDK(device="cpu")
    generated = sdk.generate_python("build a clamp function")
    assert generated["syntax_valid"] is True
    ast.parse(generated["source"])
    compile(generated["source"], "<generated>", "exec")


def test_refinement_has_commit_or_rollback_boundary():
    sdk = MODULE.Permeation3DSDK(device="cpu")
    source = "def evolve(state, dt):\n    return state + dt * (-0.1 * state)\n"
    before = sdk.state()["state_hash"]
    report = sdk.refine(MODULE.Modality.CODE, source, cycles=1)[0]
    after = sdk.state()["state_hash"]

    assert report.update_ratio >= 0.0
    if report.accepted:
        assert after != before
    else:
        assert after == before
