import math

import pytest

from jarvisx.dr_moagi_inward_bit_engine import (
    BitAutoEncoder,
    Config,
    Inward3DBitEngine,
)


def test_even_group_tie_resolution_is_unbiased_over_complete_four_bit_domain():
    codec = BitAutoEncoder(source_bits=4, latent_bits=1)
    encoded_ones = sum(codec.encode(source) for source in range(16))
    assert encoded_ones == 8


def test_latent_cycle_is_exact_identity_but_is_not_used_as_coupling_metric():
    codec = BitAutoEncoder(source_bits=16, latent_bits=4)
    assert all(codec.latent_cycle(latent) == latent for latent in range(16))

    engine = Inward3DBitEngine(Config(side=3, bits=16, latent_bits=4, iterations=1))
    metrics = engine.step()
    assert metrics.latent_cycle_loss == 0.0
    assert metrics.latent_coupling_loss >= 0.0


def test_precomputed_masks_are_deterministic_and_have_expected_cardinality():
    config = Config(side=2, bits=16, latent_bits=4, beta=0.5, omega_gain=0.25)
    left = Inward3DBitEngine(config)
    right = Inward3DBitEngine(config)

    assert left.reconstruction_masks == right.reconstruction_masks
    assert left.omega_injection_masks == right.omega_injection_masks
    assert all(mask.bit_count() == 8 for mask in left.reconstruction_masks.values())
    assert all(mask.bit_count() == 4 for mask in left.omega_injection_masks.values())


def test_omega_memory_is_contractive_when_reconstruction_error_is_zero():
    config = Config(
        side=2,
        bits=16,
        latent_bits=4,
        omega_retention=0.5,
        omega_capture=1.0,
    )
    engine = Inward3DBitEngine(config)
    engine.omega = {point: engine.full_mask for point in engine.grid.coords}

    next_omega = engine.update_omega(engine.state, engine.state)
    before = sum(value.bit_count() for value in engine.omega.values())
    after = sum(value.bit_count() for value in next_omega.values())

    assert after <= before
    assert all(value.bit_count() <= 8 for value in next_omega.values())


def test_fixed_point_requires_source_latent_and_omega_state_to_stop_changing():
    engine = Inward3DBitEngine(
        Config(
            side=2,
            bits=8,
            latent_bits=4,
            iterations=4,
            alpha=0.0,
            beta=0.0,
            omega_gain=0.0,
            omega_retention=0.0,
            omega_capture=0.0,
            tolerance=0.0,
        )
    )

    first = engine.step()
    second = engine.step()

    assert not first.fixed_point
    assert first.changed_bits == 0
    assert first.latent_changed_bits > 0
    assert second.fixed_point
    assert second.changed_bits == 0
    assert second.latent_changed_bits == 0
    assert second.omega_changed_bits == 0


def test_metrics_separate_local_anchor_and_drift_errors():
    engine = Inward3DBitEngine(Config(side=3, bits=16, latent_bits=4, iterations=2))
    first = engine.step()
    second = engine.step()

    for metrics in (first, second):
        assert 0.0 <= metrics.local_reconstruction_loss <= 1.0
        assert 0.0 <= metrics.anchor_reconstruction_loss <= 1.0
        assert 0.0 <= metrics.anchor_drift <= 1.0
        assert 0.0 <= metrics.full_state_gap <= 1.0
        assert math.isfinite(metrics.full_state_gap)

    assert first.local_reconstruction_loss == first.anchor_reconstruction_loss


def test_bitplane_adapter_exposes_state_and_latent_for_spectral_analysis():
    engine = Inward3DBitEngine(Config(side=2, bits=8, latent_bits=4, iterations=1))
    state_plane = engine.bitplane_field(0)
    assert set(state_plane) == set(engine.grid.coords)
    assert set(state_plane.values()) <= {-1.0, 1.0}

    with pytest.raises(RuntimeError, match="before the first step"):
        engine.bitplane_field(0, latent=True)

    engine.step()
    latent_plane = engine.bitplane_field(0, latent=True, spins=False)
    assert set(latent_plane.values()) <= {0.0, 1.0}


def test_default_recurrence_is_deterministic_and_bounded():
    config = Config(side=3, bits=16, latent_bits=4, iterations=8)
    left = Inward3DBitEngine(config)
    right = Inward3DBitEngine(config)

    left_metrics = list(left.run())
    right_metrics = list(right.run())

    assert left_metrics == right_metrics
    assert left.state == right.state
    assert left.omega == right.omega
    assert left.latent == right.latent
    assert all(0 <= value <= left.full_mask for value in left.state.values())


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError):
        Config(bits=8, latent_bits=9)
    with pytest.raises(ValueError):
        Config(omega_retention=1.1)
    with pytest.raises(ValueError):
        Config(iterations=0)
