from __future__ import annotations

import json
from pathlib import Path

import pytest

from jarvisx.dm_vomegaxi_mixed_signal import (
    DeltaSigmaBank,
    DMvOmegaXiMixedSignalEngine,
    HardwareInterlock,
    HardwareInterlockLimits,
    HardwareTelemetry,
    HBridgeGateFrame,
    MixedSignalConfig,
    MixedSignalError,
    OmegaRegisterBank,
    PulseDensityBank,
    pack_bits_16,
    rotate_left_16,
    unpack_bits_16,
    xnor_popcount,
)


def _config(**overrides: object) -> MixedSignalConfig:
    values: dict[str, object] = {
        "sensor_channels": 1,
        "oversample": 4,
        "sensor_min": -1.0,
        "sensor_max": 1.0,
        "score_bound": 4,
        "memory_words": 1,
        "omega_rotate_bits": 0,
        "omega_persistence_mask": 0,
        "pdm_period": 8,
        "max_duty_cycle": 0.8,
        "max_outputs": 16,
        "fixed_point_tolerance": 0.0,
    }
    values.update(overrides)
    return MixedSignalConfig(**values)  # type: ignore[arg-type]


def _limits(**overrides: object) -> HardwareInterlockLimits:
    values: dict[str, object] = {
        "max_abs_current_a": 5.0,
        "max_abs_voltage_v": 24.0,
        "max_temperature_c": 80.0,
        "watchdog_timeout_ticks": 3,
        "min_dead_time_ticks": 2,
    }
    values.update(overrides)
    return HardwareInterlockLimits(**values)  # type: ignore[arg-type]


def _telemetry(**overrides: object) -> HardwareTelemetry:
    values: dict[str, object] = {
        "current_a": 1.0,
        "voltage_v": 12.0,
        "temperature_c": 30.0,
        "watchdog_age_ticks": 1,
        "observed_dead_time_ticks": 2,
        "emergency_stop": False,
        "bridge_overlap_detected": False,
    }
    values.update(overrides)
    return HardwareTelemetry(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "overrides",
    [
        {"sensor_channels": 0},
        {"oversample": True},
        {"sensor_min": 1.0, "sensor_max": 1.0},
        {"omega_rotate_bits": 16},
        {"omega_persistence_mask": 1 << 16},
        {"max_duty_cycle": 1.1},
        {"fixed_point_tolerance": -1.0},
    ],
)
def test_config_rejects_invalid_bounds(overrides: dict[str, object]) -> None:
    with pytest.raises(MixedSignalError):
        _config(**overrides)


def test_delta_sigma_extremes_and_zero_are_deterministic() -> None:
    bank = DeltaSigmaBank(_config())
    assert bank.encode([-1.0]) == (0, 0, 0, 0)
    bank.reset()
    assert bank.encode([1.0]) == (1, 1, 1, 1)
    bank.reset()
    assert bank.encode([0.0]) == (1, 0, 1, 0)
    assert bank.state == (0.0,)


def test_delta_sigma_is_time_major_and_rejects_bad_samples() -> None:
    bank = DeltaSigmaBank(_config(sensor_channels=2, oversample=2))
    assert bank.encode([-1.0, 1.0]) == (0, 1, 0, 1)
    before = bank.state
    with pytest.raises(MixedSignalError, match="sample count"):
        bank.encode([0.0])
    with pytest.raises(MixedSignalError, match="outside"):
        bank.encode([0.0, 2.0])
    assert bank.state == before


def test_xnor_popcount_matches_bipolar_dot_identity() -> None:
    bits = (1, 0, 1, 0)
    assert xnor_popcount(bits, bits) == (4, 4)
    assert xnor_popcount(bits, (0, 1, 0, 1)) == (0, -4)
    assert xnor_popcount(bits, (1, 1, 0, 0)) == (2, 0)
    with pytest.raises(MixedSignalError, match="equally sized"):
        xnor_popcount(bits, (1,))


def test_pack_unpack_and_rotate_use_explicit_16_bit_order() -> None:
    words = pack_bits_16((1, 0, 1, 1), word_count=2)
    assert words == (0b1101, 0)
    assert unpack_bits_16(words, bit_count=4) == (1, 0, 1, 1)
    assert rotate_left_16(0x8001, 1) == 0x0003
    assert rotate_left_16(0x1234, 16) == 0x1234
    with pytest.raises(MixedSignalError, match="capacity"):
        pack_bits_16((1,) * 17, word_count=1)


def test_omega_update_is_xor_then_rotate_with_optional_retention() -> None:
    bank = OmegaRegisterBank(_config(omega_rotate_bits=1))
    assert bank.update((0x0001,)) == (0x0002,)
    assert bank.update((0x0001,)) == (0x0006,)

    retained = OmegaRegisterBank(_config(omega_rotate_bits=1, omega_persistence_mask=0xFFFF))
    assert retained.update((0xFFFF,)) == (0x0000,)


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"emergency_stop": True}, "emergency-stop"),
        ({"current_a": -5.1}, "overcurrent"),
        ({"voltage_v": 24.1}, "overvoltage"),
        ({"temperature_c": 80.1}, "overtemperature"),
        ({"watchdog_age_ticks": 4}, "watchdog-timeout"),
        ({"bridge_overlap_detected": True}, "bridge-overlap"),
        ({"observed_dead_time_ticks": 1}, "insufficient-dead-time"),
    ],
)
def test_independent_interlock_detects_each_trip(overrides: dict[str, object], reason: str) -> None:
    trips = HardwareInterlock(_limits()).evaluate(_telemetry(**overrides))
    assert reason in trips


def test_independent_interlock_accepts_telemetry_at_the_limits() -> None:
    telemetry = _telemetry(
        current_a=-5.0,
        voltage_v=24.0,
        temperature_c=80.0,
        watchdog_age_ticks=3,
        observed_dead_time_ticks=2,
    )
    assert HardwareInterlock(_limits()).evaluate(telemetry) == ()


def test_hardware_contract_types_are_strict() -> None:
    with pytest.raises(MixedSignalError, match="current and voltage"):
        _limits(max_abs_current_a=0.0)
    with pytest.raises(MixedSignalError, match="finite"):
        _limits(max_abs_voltage_v=float("inf"))
    with pytest.raises(MixedSignalError, match="emergency_stop"):
        _telemetry(emergency_stop=1)
    with pytest.raises(MixedSignalError, match="limits must"):
        HardwareInterlock(object())  # type: ignore[arg-type]


def test_pulse_density_is_bounded_and_stateful() -> None:
    bank = PulseDensityBank(channels=2, period=8)
    first = bank.encode((0.5, 1.0))
    assert sum(first[0]) == 4
    assert first[1] == (1,) * 8
    assert all(bit in (0, 1) for pattern in first for bit in pattern)
    assert bank.state == pytest.approx((0.0, 0.0))
    with pytest.raises(MixedSignalError, match=r"\[0, 1\]"):
        bank.encode((1.1, 0.0))


def test_engine_executes_the_complete_bounded_stack() -> None:
    engine = DMvOmegaXiMixedSignalEngine(
        weights=[(1, 1, 1, 1)],
        interlock_limits=_limits(),
        config=_config(),
    )
    report = engine.step([1.0], _telemetry(), target_bits=(1,))

    assert report.bitstream == (1, 1, 1, 1)
    assert report.xnor_matches == (4,)
    assert report.raw_scores == (4,)
    assert report.bounded_scores == (4,)
    assert report.latent_bits == (1,)
    assert report.omega_words == (1,)
    assert report.theta_output_bits == (1,)
    assert report.emitted_bits == (1,)
    assert report.hamming_error_bits == 0
    assert report.hamming_error_rate == 0.0
    assert report.actuation_permitted
    assert report.emission_active
    assert report.duty_cycles == pytest.approx((0.8,))
    assert report.gate_frames[0].direction == 1
    assert not any(
        positive & negative
        for positive, negative in zip(
            report.gate_frames[0].positive_gate,
            report.gate_frames[0].negative_gate,
        )
    )
    assert len(report.state_sha256) == 64
    assert engine.status()["hardware_io"] is False


def test_theta_mask_blocks_output_and_exposes_hamming_error() -> None:
    engine = DMvOmegaXiMixedSignalEngine(
        weights=[(1, 1, 1, 1)],
        interlock_limits=_limits(),
        config=_config(),
        theta_mask=(0,),
    )
    report = engine.step([1.0], _telemetry(), target_bits=(1,))
    assert report.theta_candidate_bits == (1,)
    assert report.theta_output_bits == (0,)
    assert report.hamming_error_bits == 1
    assert report.hamming_error_rate == 1.0
    assert report.emitted_bits == (0,)
    assert not report.emission_active


def test_hardware_interlock_overrides_permitted_theta_output() -> None:
    engine = DMvOmegaXiMixedSignalEngine(
        weights=[(1, 1, 1, 1)],
        interlock_limits=_limits(),
        config=_config(),
    )
    report = engine.step([1.0], _telemetry(current_a=8.0), target_bits=(1,))
    assert report.theta_output_bits == (1,)
    assert report.emitted_bits == (0,)
    assert report.interlock_trips == ("overcurrent",)
    assert not report.actuation_permitted
    assert not report.emission_active
    assert report.duty_cycles == (0.0,)
    assert not any(report.gate_frames[0].positive_gate)
    assert not any(report.gate_frames[0].negative_gate)


def test_negative_polarity_uses_only_the_negative_logic_frame() -> None:
    engine = DMvOmegaXiMixedSignalEngine(
        weights=[(1, 1, 1, 1)],
        interlock_limits=_limits(),
        config=_config(),
        actuator_polarities=(-1,),
    )
    frame = engine.step([1.0], _telemetry()).gate_frames[0]
    assert frame.direction == -1
    assert not any(frame.positive_gate)
    assert any(frame.negative_gate)


def test_replay_is_deterministic_and_reset_restores_initial_state() -> None:
    kwargs = {
        "weights": [(1, 1, 1, 1), (0, 0, 0, 0)],
        "interlock_limits": _limits(),
        "config": _config(),
    }
    first = DMvOmegaXiMixedSignalEngine(**kwargs)
    second = DMvOmegaXiMixedSignalEngine(**kwargs)
    first_report = first.step([0.5], _telemetry(), target_bits=(1, 0))
    second_report = second.step([0.5], _telemetry(), target_bits=(1, 0))
    assert first_report.as_dict() == second_report.as_dict()

    first.reset()
    replay = first.step([0.5], _telemetry(), target_bits=(1, 0))
    assert replay.as_dict() == first_report.as_dict()


def test_exact_discrete_state_can_report_an_internal_fixed_point() -> None:
    config = _config(oversample=2, score_bound=2, pdm_period=2)
    engine = DMvOmegaXiMixedSignalEngine(
        weights=[(0, 1)],
        interlock_limits=_limits(),
        config=config,
    )
    report = engine.step([0.0], _telemetry(), target_bits=(0,))
    assert report.bitstream == (1, 0)
    assert report.emitted_bits == (0,)
    assert report.state_gap_bits == 0
    assert report.state_gap_numeric == pytest.approx(0.0)
    assert report.internal_fixed_point


def test_invalid_target_is_rejected_before_state_mutation() -> None:
    engine = DMvOmegaXiMixedSignalEngine(
        weights=[(1, 1, 1, 1)],
        interlock_limits=_limits(),
        config=_config(),
    )
    before = (engine.delta_sigma.state, engine.omega.state, engine.pdm.state)
    with pytest.raises(MixedSignalError, match="target_bits length"):
        engine.step([1.0], _telemetry(), target_bits=(1, 0))
    after = (engine.delta_sigma.state, engine.omega.state, engine.pdm.state)
    assert after == before


def test_constructor_rejects_shape_capacity_and_empty_masks() -> None:
    with pytest.raises(MixedSignalError, match="at least one"):
        DMvOmegaXiMixedSignalEngine([], _limits(), config=_config())
    with pytest.raises(MixedSignalError, match="input width"):
        DMvOmegaXiMixedSignalEngine([(1, 1)], _limits(), config=_config())
    with pytest.raises(MixedSignalError, match="max_outputs"):
        DMvOmegaXiMixedSignalEngine(
            [(1, 1, 1, 1)] * 2,
            _limits(),
            config=_config(max_outputs=1),
        )
    with pytest.raises(MixedSignalError, match="theta_mask length"):
        DMvOmegaXiMixedSignalEngine(
            [(1, 1, 1, 1)],
            _limits(),
            config=_config(),
            theta_mask=(),
        )
    with pytest.raises(MixedSignalError, match="polarities"):
        DMvOmegaXiMixedSignalEngine(
            [(1, 1, 1, 1)],
            _limits(),
            config=_config(),
            actuator_polarities=(True,),
        )


def test_gate_frame_rejects_shoot_through_by_construction() -> None:
    with pytest.raises(MixedSignalError, match="cannot overlap"):
        HBridgeGateFrame(
            direction=1,
            duty_cycle=0.5,
            positive_gate=(1, 0),
            negative_gate=(1, 0),
        )
    with pytest.raises(MixedSignalError, match="zero duty"):
        HBridgeGateFrame(
            direction=0,
            duty_cycle=0.5,
            positive_gate=(0, 0),
            negative_gate=(0, 0),
        )


def test_reference_config_requires_explicit_hardware_limits() -> None:
    path = Path(__file__).parents[1] / "configs" / "dm_vomegaxi_mixed_signal_reference.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    assert config["schema"] == "jarvisx.dm-vomegaxi-mixed-signal-reference/v1"
    assert config["hardware_limits"]["required"] is True
    assert config["hardware_limits"]["defaults_provided"] is False
    assert config["hardware_io"] is False
    assert config["zero_latency_claim"] is False
