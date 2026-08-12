from __future__ import annotations

import pytest

from jarvisx.dr_moagi_field_runtime import (
    DrMoagiFieldConfig,
    DrMoagiFieldRuntime,
    IdentityFieldCodec,
)


class ZeroCodec:
    def encode(self, field):
        return None

    def decode(self, latent, support):
        return {coordinate: 0.0 for coordinate in support}


def test_conservative_step_guard_rejects_large_dt():
    with pytest.raises(ValueError, match="conservative"):
        DrMoagiFieldConfig(dt=1.0)


def test_identity_codec_has_zero_reconstruction_residual():
    runtime = DrMoagiFieldRuntime(
        IdentityFieldCodec(),
        DrMoagiFieldConfig(
            side=5,
            alpha=1.0,
            lambda_residual=0.0,
            eta=0.0,
            dt=0.1,
            expand_halo=False,
        ),
    )
    runtime.load({(2, 2, 2): 0.5})

    metrics = runtime.step()

    assert metrics.committed
    assert metrics.reconstruction_mse == pytest.approx(0.0)
    assert metrics.max_abs_residual == pytest.approx(0.0)
    assert runtime.snapshot()[(2, 2, 2)] == pytest.approx(0.5)


def test_autoencoder_closure_pulls_state_toward_reconstruction():
    runtime = DrMoagiFieldRuntime(
        ZeroCodec(),
        DrMoagiFieldConfig(
            side=5,
            alpha=1.0,
            lambda_residual=0.0,
            eta=0.0,
            dt=0.1,
            expand_halo=False,
        ),
    )
    runtime.load({(2, 2, 2): 1.0})

    metrics = runtime.step()

    assert metrics.committed
    assert runtime.snapshot()[(2, 2, 2)] == pytest.approx(0.9)


def test_moagi_glyph_uses_six_face_neighbours():
    runtime = DrMoagiFieldRuntime(
        IdentityFieldCodec(),
        DrMoagiFieldConfig(
            side=5,
            alpha=0.0,
            lambda_residual=0.0,
            eta=0.1,
            dt=0.1,
            expand_halo=True,
        ),
    )
    runtime.load({(2, 2, 2): 1.0})

    metrics = runtime.step()
    state = runtime.snapshot()

    assert metrics.committed
    assert state[(2, 2, 2)] == pytest.approx(1.0)
    expected_neighbour = -1.0 / 600.0
    assert state[(1, 2, 2)] == pytest.approx(expected_neighbour)
    assert state[(3, 2, 2)] == pytest.approx(expected_neighbour)
    assert state[(2, 1, 2)] == pytest.approx(expected_neighbour)
    assert state[(2, 3, 2)] == pytest.approx(expected_neighbour)
    assert state[(2, 2, 1)] == pytest.approx(expected_neighbour)
    assert state[(2, 2, 3)] == pytest.approx(expected_neighbour)


def test_validator_rolls_back_candidate_and_preserves_anchor():
    runtime = DrMoagiFieldRuntime(
        ZeroCodec(),
        DrMoagiFieldConfig(
            side=5,
            alpha=1.0,
            lambda_residual=0.0,
            eta=0.0,
            dt=0.1,
            expand_halo=False,
        ),
    )
    initial = {(2, 2, 2): 0.75}
    runtime.load(initial)

    metrics = runtime.step(validator=lambda candidate, telemetry: False)

    assert not metrics.committed
    assert metrics.rejection_reason == "validator rejected candidate"
    assert runtime.snapshot() == initial
    assert runtime.anchor_snapshot() == initial
    assert runtime.cycle == 1


def test_decoder_cannot_escape_requested_sparse_support():
    class EscapingCodec:
        def encode(self, field):
            return None

        def decode(self, latent, support):
            return {(0, 0, 0): 1.0}

    runtime = DrMoagiFieldRuntime(
        EscapingCodec(),
        DrMoagiFieldConfig(
            side=5,
            alpha=0.0,
            lambda_residual=0.0,
            eta=0.0,
            dt=0.1,
            expand_halo=False,
        ),
    )
    runtime.load({(2, 2, 2): 0.5})

    with pytest.raises(ValueError, match="outside requested support"):
        runtime.step()

    assert runtime.snapshot() == {(2, 2, 2): 0.5}


def test_support_closure_fails_before_dense_materialization():
    runtime = DrMoagiFieldRuntime(
        IdentityFieldCodec(),
        DrMoagiFieldConfig(
            side=5,
            alpha=0.0,
            lambda_residual=0.0,
            eta=0.0,
            dt=0.1,
            expand_halo=True,
            max_active_cells=1,
        ),
    )
    runtime.load({(2, 2, 2): 0.5})

    with pytest.raises(RuntimeError, match="support-closure budget"):
        runtime.step()
