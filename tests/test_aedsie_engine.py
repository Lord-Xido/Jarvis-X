import math

from jarvisx.aedsie_engine import (
    PROGRAM,
    AEDSIEConfig,
    AEDSIEVirtualEngine,
    Mechanics,
    ResidualAutoencoder,
    run_aedsie,
)


def test_end_to_end_engine_is_deterministic():
    first = run_aedsie(cycles=2, inward=True)
    second = run_aedsie(cycles=2, inward=True)

    assert first["ledger_head"] == second["ledger_head"]
    assert first["final_cycle"]["state_hash"] == second["final_cycle"]["state_hash"]
    assert first["final_cycle"]["predicted_class"] == second["final_cycle"]["predicted_class"]


def test_program_executes_every_declared_stage():
    report = AEDSIEVirtualEngine().step()

    assert report.program_trace == PROGRAM
    assert report.cycle == 1
    assert len(report.ledger_head) == 64
    assert len(report.state_hash) == 64


def test_field_latent_metric_and_routing_invariants():
    engine = AEDSIEVirtualEngine()
    report = engine.step()

    assert len(engine.field) == 4 * 8 * 4
    assert len(report.routing_weights) == 9
    assert math.isclose(sum(report.routing_weights), 1.0, rel_tol=0.0, abs_tol=1e-12)
    assert report.metric_min > 0.0
    assert report.metric_max >= report.metric_min
    assert report.reconstruction_mse >= 0.0
    assert math.isfinite(report.aoa_degrees)


def test_ledger_is_parent_chained_and_advances_once_per_commit():
    engine = AEDSIEVirtualEngine()
    first = engine.step()
    second = engine.step()

    assert first.ledger_head != second.ledger_head
    assert second.ledger_head == engine.ledger.head
    assert len(engine.ledger.records) == 2


def test_inward_turn_never_commits_a_worse_candidate():
    engine = AEDSIEVirtualEngine()
    report = engine.step(inward=True)

    if report.inward.accepted:
        assert report.inward.shadow_cost < report.inward.baseline_cost
        assert report.mechanics.version == 1
    else:
        assert report.mechanics.version == 0
    assert report.inward.analysis_share <= engine.config.analysis_budget
    assert report.mechanics.admissible()


def test_disabling_inward_turn_preserves_mechanics():
    engine = AEDSIEVirtualEngine()
    report = engine.step(inward=False)

    assert not report.inward.attempted
    assert not report.inward.accepted
    assert report.mechanics == Mechanics()
    assert len(engine.mechanics_history) == 1


def test_residual_autoencoder_is_finite_and_shape_preserving():
    autoencoder = ResidualAutoencoder(field_cells=8, latent_dim=4)
    field = tuple((float(i), float(i % 3), -float(i) / 2.0) for i in range(8))

    latent, base = autoencoder.encode(field)
    reconstructed = autoencoder.decode(latent, base)

    assert len(latent) == 4
    assert len(reconstructed) == len(field)
    assert all(math.isfinite(value) for vector in reconstructed for value in vector)


def test_configuration_and_stability_guards():
    AEDSIEConfig().validate()
    assert Mechanics().admissible()
    assert not Mechanics(alpha=2.0, coherence=2.0, dt=1.0).admissible()

    invalid = AEDSIEConfig(fft_bins=64, samples_per_frame=32)
    try:
        invalid.validate()
    except ValueError:
        pass
    else:
        raise AssertionError("invalid FFT configuration was accepted")
