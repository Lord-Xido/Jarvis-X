from jarvisx.dr_moagi_inward_3d_bits import Inward3DBitConfig, Inward3DBitLoop


def _config(**overrides):
    values = dict(
        tile=3,
        bits=24,
        latent=6,
        iterations=8,
        alpha=0.65,
        beta=0.50,
        omega_feedback=0.0,
        seed=1337,
        epsilon=0.0,
    )
    values.update(overrides)
    return Inward3DBitConfig(**values)


def test_inward_loop_reenters_committed_state_on_next_iteration():
    engine = Inward3DBitLoop(_config())
    initial = engine.snapshot()

    first = engine.step()

    assert first.committed
    assert engine.snapshot() != initial
    committed_hash = engine.authority_hash()

    second = engine.step()

    assert second.input_hash == committed_hash
    assert second.active_cells == 27


def test_inward_loop_is_deterministic_across_fresh_engines():
    config = _config(iterations=6)

    first_engine = Inward3DBitLoop(config)
    second_engine = Inward3DBitLoop(config)
    first = list(first_engine.run())
    second = list(second_engine.run())

    assert first == second
    assert first_engine.snapshot() == second_engine.snapshot()
    assert first_engine.omega_snapshot() == second_engine.omega_snapshot()
    assert first_engine.latent_snapshot() == second_engine.latent_snapshot()
    assert first_engine.authority_hash() == second_engine.authority_hash()


def test_external_gate_rejection_is_atomic():
    engine = Inward3DBitLoop(_config(), gate=lambda candidate, omega: False)
    before_state = engine.snapshot()
    before_omega = engine.omega_snapshot()
    before_latent = engine.latent_snapshot()
    before_hash = engine.authority_hash()

    report = engine.step()

    assert not report.committed
    assert report.rejection_reason == "external gate rejected candidate"
    assert engine.snapshot() == before_state
    assert engine.omega_snapshot() == before_omega
    assert engine.latent_snapshot() == before_latent
    assert engine.authority_hash() == before_hash
    assert engine.iteration == 0


def test_full_fixed_point_includes_residual_memory_not_only_x_gap():
    engine = Inward3DBitLoop(_config(beta=0.0, iterations=4))

    first = engine.step()
    second = engine.step()

    assert first.committed
    assert first.reality_gap == 0.0
    assert not first.fixed_point
    assert second.committed
    assert second.reality_gap == 0.0
    assert second.fixed_point


def test_all_materialized_states_remain_within_declared_bit_widths():
    engine = Inward3DBitLoop(_config(omega_feedback=0.25))

    history = list(engine.run())

    assert history
    assert engine.active_cells == engine.c.tile**3
    assert all(0 <= value < (1 << engine.c.bits) for value in engine.state.values())
    assert all(0 <= value < (1 << engine.c.bits) for value in engine.omega.values())
    assert all(0 <= value < (1 << engine.c.latent) for value in engine.latent.values())


def test_summary_separates_source_and_latent_extent():
    engine = Inward3DBitLoop(_config(tile=2, bits=16, latent=4, iterations=2))
    history = list(engine.run())
    summary = engine.summary(history)

    assert summary["active_cells"] == 8
    assert summary["source_bits"] == 128
    assert summary["latent_bits"] == 32
    assert summary["nominal_representation_ratio"] == 4.0
    assert summary["mode"] == "inward-recursive-3d-bit-ae-ad"
