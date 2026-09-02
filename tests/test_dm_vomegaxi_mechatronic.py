from __future__ import annotations

import pytest

from jarvisx.dm_vomegaxi_mechatronic import (
    DMVomegaxiMechatronicLoop,
    ConstraintGovernor,
    DeltaSigmaBank,
    SafetyLimits,
    bipolar_dot,
    rotate_u16,
    xnor_popcount,
)


def test_xnor_popcount_is_exact_bipolar_dot_product():
    lhs = (1, 0, 1, 1, 0, 0)
    rhs = (1, 1, 0, 1, 0, 0)
    assert xnor_popcount(lhs, rhs) == 4
    assert bipolar_dot(lhs, rhs) == 2
    explicit = sum((1 if a else -1) * (1 if b else -1) for a, b in zip(lhs, rhs))
    assert bipolar_dot(lhs, rhs) == explicit


def test_delta_sigma_density_tracks_normalized_signal():
    encoder = DeltaSigmaBank(1)
    bits = [encoder.encode((0.25,))[0] for _ in range(4096)]
    reconstructed = 2.0 * (sum(bits) / len(bits)) - 1.0
    assert reconstructed == pytest.approx(0.25, abs=1 / 4096)


def test_memory_rotation_stays_in_u16_domain():
    assert rotate_u16(0x8001, 1) == 0x0003
    assert rotate_u16(0x1234, 16) == 0x1234


def test_theta_clamps_slew_and_never_commands_shoot_through():
    governor = ConstraintGovernor(SafetyLimits(max_duty_ppm=500_000, max_slew_ppm=100_000))
    command = governor.apply(900_000)
    assert command.signed_duty_ppm == 100_000
    assert command.reason == "forward"
    assert command.shoot_through_safe


def test_theta_emergency_stop_and_direction_reversal_are_inhibited():
    governor = ConstraintGovernor(
        SafetyLimits(max_duty_ppm=800_000, max_slew_ppm=800_000, reversal_dead_ticks=1)
    )
    assert governor.apply(400_000).reason == "forward"
    reversal = governor.apply(-400_000)
    assert reversal.inhibited
    assert reversal.reason == "reversal-dead-time"
    assert reversal.signed_duty_ppm == 0
    stopped = governor.apply(400_000, emergency_stop=True)
    assert stopped.inhibited
    assert stopped.reason == "emergency-stop"


def test_complete_loop_is_deterministic_and_exposes_each_stage():
    limits = SafetyLimits(max_duty_ppm=750_000, max_slew_ppm=250_000)
    a = DMVomegaxiMechatronicLoop((1, 0, 1, 0), limits=limits)
    b = DMVomegaxiMechatronicLoop((1, 0, 1, 0), limits=limits)
    inputs = (0.8, -0.4, 0.2, -0.9)
    trace_a = a.step(inputs, target_bit=1)
    trace_b = b.step(inputs, target_bit=1)
    assert trace_a == trace_b
    assert trace_a.tick == 1
    assert len(trace_a.pulse_bits) == 4
    assert trace_a.signed_dot == 2 * trace_a.popcount - 4
    assert 0 <= trace_a.memory_u16 <= 0xFFFF
    assert abs(trace_a.command.signed_duty_ppm) <= limits.max_slew_ppm
    assert trace_a.command.shoot_through_safe


def test_invalid_bit_and_sensor_domains_fail_closed():
    with pytest.raises(ValueError):
        bipolar_dot((1, 2), (1, 0))
    loop = DMVomegaxiMechatronicLoop((1, 0))
    with pytest.raises(ValueError):
        loop.step((1.1, 0.0), target_bit=1)
