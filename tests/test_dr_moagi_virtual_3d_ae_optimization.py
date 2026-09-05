from jarvisx.dr_moagi_virtual_3d_ae import (
    Config,
    DrMoagiVirtual3DAE,
    Tile,
    couple,
    latent_balance_loss,
)


def test_default_spatial_coupling_is_in_active_regime():
    assert Config().alpha > 0.5


def test_six_neighbour_coupling_can_flip_surrounded_latent_bit():
    config = Config(tile=3, bits=12, latent=1, alpha=0.65, beta=0.5)
    tile = Tile(config)
    latent = {point: 0 for point in tile.coords}
    center = (1, 1, 1)
    latent[center] = 1

    coupled = couple(tile, latent, d=1, alpha=config.alpha)

    assert coupled[center] == 0


def test_latent_balance_penalty_rejects_collapse():
    balanced = {(0, 0, 0): 0b01, (1, 0, 0): 0b10}
    collapsed = {(0, 0, 0): 0, (1, 0, 0): 0}

    assert latent_balance_loss(balanced, 2) == 0.0
    assert latent_balance_loss(collapsed, 2) == 1.0


def test_bounded_optimizer_is_deterministic_and_non_regressive():
    config = Config(
        tile=3,
        bits=24,
        latent=6,
        passes=4,
        alpha=0.65,
        beta=0.65,
        alpha_candidates=(0.55, 0.65, 0.80),
        beta_candidates=(0.35, 0.50, 0.65),
    )

    first = DrMoagiVirtual3DAE(config)
    first_result = first.optimize()
    second_result = DrMoagiVirtual3DAE(config).optimize()

    assert first_result == second_result
    assert first_result.score <= first_result.baseline_score
    assert first.c.alpha == first_result.alpha
    assert first.c.beta == first_result.beta
    assert first_result.candidates_evaluated == 9


def test_optimized_engine_still_reaches_fixed_point():
    config = Config(
        tile=3,
        bits=24,
        latent=6,
        passes=4,
        alpha=0.65,
        beta=0.65,
        alpha_candidates=(0.55, 0.65),
        beta_candidates=(0.35, 0.65),
        epsilon=0.0,
    )
    engine = DrMoagiVirtual3DAE(config)
    tuning = engine.optimize()
    history = engine.run()

    assert tuning.score <= tuning.baseline_score
    assert history[-1].reality_gap == 0.0
    assert history[-1].changed_bits == 0


def test_summary_exposes_optimization_provenance():
    engine = DrMoagiVirtual3DAE(
        Config(
            tile=2,
            bits=16,
            latent=4,
            passes=3,
            alpha_candidates=(0.55, 0.65),
            beta_candidates=(0.35, 0.65),
        )
    )
    engine.optimize()
    summary = engine.summary(engine.run())

    assert summary["tuning"] is not None
    assert summary["tuning"]["candidates_evaluated"] == 4
    assert summary["alpha"] == engine.c.alpha
    assert summary["beta"] == engine.c.beta
