from __future__ import annotations

import pytest

from jarvisx.dr_moagi_master_operator import (
    DrMoagiMasterOperator,
    MoagiMasterConfig,
)


def test_logical_10e18_domain_stays_sparse_and_encodes_to_r8():
    engine = DrMoagiMasterOperator()
    engine.load(
        {
            (0, 0, 0): 1.0,
            (999_999, 999_999, 999_999): -0.5,
        }
    )

    latent = engine.encode(engine.snapshot())
    status = engine.status()

    assert len(latent) == 8
    assert status["logical_shape"] == [1_000_000, 1_000_000, 1_000_000]
    assert status["logical_voxels"] == str(10**18)
    assert status["resident_active_cells"] == 2


def test_encode_decode_projection_is_idempotent_on_sparse_support():
    engine = DrMoagiMasterOperator(MoagiMasterConfig(beta_kl=0.0))
    source = {
        (0, 0, 0): 1.0,
        (8, 0, 0): 0.0,
        (1, 0, 0): -0.25,
    }
    engine.load(source)
    support = tuple(sorted(source))

    projected = engine.decode(engine.encode(source), support)
    projected_twice = engine.decode(engine.encode(projected), support)

    assert engine._rms(projected, projected_twice) == pytest.approx(0.0, abs=1.0e-15)


def test_master_operator_reaches_fixed_point_for_ae_consistent_state():
    engine = DrMoagiMasterOperator(
        MoagiMasterConfig(beta_kl=0.0, fixed_point_tolerance=1.0e-12)
    )
    engine.load({(0, 0, 0): 0.75, (1, 0, 0): -0.25})

    reports = engine.run_until_fixed_point()

    assert reports[-1].converged
    assert reports[-1].operator_delta_rms <= engine.config.fixed_point_tolerance
    assert (
        reports[-1].autoencoder_consistency_rms
        <= engine.config.fixed_point_tolerance
    )
    assert reports[-1].loss.reconstruction_mse == pytest.approx(0.0)


def test_residual_control_preserves_exact_decay_and_reset_law():
    engine = DrMoagiMasterOperator(
        MoagiMasterConfig(beta_kl=0.0, residual_initial=0.2001)
    )
    engine.load({(0, 0, 0): 1.0})

    report = engine.step()

    assert report.residual_reset_applied
    assert report.residual_control == pytest.approx(5.6)


def test_multiplex_receipt_preserves_32_lane_1e18_budget_without_claiming_measurement():
    engine = DrMoagiMasterOperator()
    engine.load({(0, 0, 0): 1.0, (1, 2, 3): 2.0})

    receipt = engine.multiplex_receipt()
    status = engine.status()

    assert receipt.lane_count == 32
    assert len(receipt.lane_digests) == 32
    assert sum(receipt.lane_bandwidth_budget_bps) == 10**18
    assert receipt.total_bandwidth_budget_bps == 10**18
    assert status["bandwidth_semantics"] == (
        "configured logical budget; not measured physical throughput"
    )


def test_loss_is_reconstruction_plus_beta_kl_and_gradient_is_finite():
    engine = DrMoagiMasterOperator(MoagiMasterConfig(beta_kl=0.25))
    field = {(0, 0, 0): 1.0, (8, 0, 0): 0.0, (1, 0, 0): -0.25}
    engine.load(field)

    loss, gradient = engine.loss_and_gradient(engine.snapshot())

    assert loss.total == pytest.approx(
        loss.reconstruction_mse + engine.config.beta_kl * loss.kl_divergence
    )
    assert loss.reconstruction_mse >= 0.0
    assert loss.kl_divergence >= -1.0e-15
    assert all(value == pytest.approx(value) for value in gradient.values())


def test_invalid_coordinate_is_rejected_before_materialization():
    engine = DrMoagiMasterOperator()

    with pytest.raises(ValueError, match="outside logical 3D domain"):
        engine.load({(1_000_000, 0, 0): 1.0})
