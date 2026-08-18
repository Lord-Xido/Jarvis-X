from __future__ import annotations

import math

import pytest

from jarvisx.dr_moagi_permeation import (
    PermeationConfig,
    PermeationIntegrityError,
    ReceivedFrame,
    absorb,
    modulate,
    propagate,
    simulate_round_trip,
)


def _state() -> dict[str, object]:
    return {
        "latent": [index / 10.0 for index in range(16)],
        "geometry": {"vpx_density": 0.72, "vpx_velocity": [0.1, -0.2, 0.3]},
        "intent": "fixed-point-description",
    }


def test_declared_default_field_numbers_are_reproducible():
    config = PermeationConfig()

    assert config.wavelength_m == pytest.approx(0.8993863678636786)
    assert config.wave_number_rad_m == pytest.approx(6.986080211671541)
    assert config.propagation_delay_ns == pytest.approx(3.3356409519815204)
    assert abs(config.channel_coefficient) == pytest.approx(0.941 / (4.0 * math.pi))


def test_round_trip_reconstructs_16d_state_exactly():
    result = simulate_round_trip(_state(), PermeationConfig())

    assert result["physical_rf"] is False
    assert result["verified"] is True
    assert result["reconstructed"] == _state()
    assert result["symbol_count"] == result["payload_bytes"] * 8


def test_focus_rotates_the_quadrupole_axis_without_claiming_one_sided_beamforming():
    aligned = PermeationConfig(axis=(0.0, 1.0, 0.0), receiver_direction=(0.0, 1.0, 0.0))
    perpendicular = aligned.focused((1.0, 0.0, 0.0))
    opposite = PermeationConfig(
        axis=(0.0, -1.0, 0.0), receiver_direction=(0.0, 1.0, 0.0)
    )

    assert aligned.angular_gain == pytest.approx(1.0)
    assert perpendicular.angular_gain == pytest.approx(0.4)
    assert opposite.angular_gain == pytest.approx(1.0)


def test_absorb_rejects_symbol_corruption():
    config = PermeationConfig()
    received = propagate(modulate(_state(), config), config)
    samples = list(received.samples)
    samples[0] = -samples[0]
    corrupted = ReceivedFrame(
        payload_digest=received.payload_digest,
        payload_length=received.payload_length,
        samples=tuple(samples),
        channel_coefficient=received.channel_coefficient,
    )

    with pytest.raises(PermeationIntegrityError, match="digest mismatch"):
        absorb(corrupted)


def test_payload_budget_is_enforced_before_symbol_expansion():
    with pytest.raises(ValueError, match="max_payload_bytes"):
        modulate({"payload": "x" * 100}, PermeationConfig(max_payload_bytes=16))


def test_channel_null_is_rejected():
    with pytest.raises(ValueError, match="channel null"):
        PermeationConfig(
            omni_weight=0.2,
            quadrupole_weight=0.4,
            axis=(0.0, 1.0, 0.0),
            receiver_direction=(1.0, 0.0, 0.0),
        )
