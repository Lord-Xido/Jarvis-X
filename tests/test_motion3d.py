import math

import pytest

from jarvisx.motion3d import (
    DrMoagiMotionEngine,
    MotionConstraints,
    MotionObservation,
    MotionState,
    quaternion_multiply,
    quaternion_normalize,
)


def test_constant_force_advances_state_with_semi_implicit_integration():
    engine = DrMoagiMotionEngine()
    result = engine.step(
        MotionState(mass=2.0),
        0.5,
        external_force=(4.0, 0.0, 0.0),
    )
    assert result.state.acceleration == pytest.approx((2.0, 0.0, 0.0))
    assert result.state.velocity == pytest.approx((1.0, 0.0, 0.0))
    assert result.state.position == pytest.approx((0.5, 0.0, 0.0))
    assert result.kinetic_energy == pytest.approx(1.0)


def test_rotation_remains_on_unit_quaternion_manifold():
    engine = DrMoagiMotionEngine()
    result = engine.step(
        MotionState(),
        0.25,
        external_torque=(0.0, 0.0, 2.0),
    )
    magnitude = math.sqrt(sum(value * value for value in result.state.orientation))
    assert magnitude == pytest.approx(1.0)
    assert result.state.angular_velocity[2] == pytest.approx(0.5)
    assert result.state.orientation != (1.0, 0.0, 0.0, 0.0)


def test_observation_residual_corrects_and_updates_memory():
    engine = DrMoagiMotionEngine(position_gain=0.5, velocity_gain=0.5)
    observation = MotionObservation(
        position=(2.0, 0.0, 0.0),
        velocity=(1.0, 0.0, 0.0),
        confidence=1.0,
    )
    result = engine.step(MotionState(), 0.1, observation=observation)
    assert result.residual_position == pytest.approx((2.0, 0.0, 0.0))
    assert result.state.position == pytest.approx((1.0, 0.0, 0.0))
    assert result.state.velocity == pytest.approx((0.5, 0.0, 0.0))
    assert result.state.memory_position[0] > 0.0
    assert result.state.memory_velocity[0] > 0.0


def test_constraint_projection_limits_speed_and_resolves_floor_contact():
    engine = DrMoagiMotionEngine(
        MotionConstraints(max_speed=2.0, floor_z=0.0, restitution=0.5)
    )
    result = engine.step(
        MotionState(position=(0.0, 0.0, 0.1), velocity=(10.0, 0.0, -4.0)),
        0.2,
    )
    speed = math.sqrt(sum(value * value for value in result.state.velocity))
    assert speed <= 2.0 + 1e-12
    assert result.state.position[2] == 0.0
    assert result.state.velocity[2] >= 0.0


def test_state_hash_is_deterministic_for_identical_motion():
    engine = DrMoagiMotionEngine()
    first = engine.step(MotionState(), 0.1, external_force=(1.0, 2.0, 3.0))
    second = engine.step(MotionState(), 0.1, external_force=(1.0, 2.0, 3.0))
    assert first.state == second.state
    assert first.state_hash == second.state_hash


def test_invalid_state_and_step_inputs_fail_closed():
    engine = DrMoagiMotionEngine()
    with pytest.raises(ValueError):
        engine.step(MotionState(mass=0.0), 0.1)
    with pytest.raises(ValueError):
        engine.step(MotionState(), 0.0)
    with pytest.raises(ValueError):
        engine.step(MotionState(), float("nan"))
    with pytest.raises(ValueError):
        MotionObservation(confidence=1.5).validate()


def test_quaternion_operations_normalize_product():
    a = quaternion_normalize((1.0, 1.0, 0.0, 0.0))
    b = quaternion_normalize((1.0, 0.0, 1.0, 0.0))
    product = quaternion_normalize(quaternion_multiply(a, b))
    magnitude = math.sqrt(sum(value * value for value in product))
    assert magnitude == pytest.approx(1.0)


def test_motion_cli_emits_final_state(capsys):
    from jarvisx.motion3d_cli import main

    main(
        [
            '{"mass":2,"position":[0,0,0]}',
            "--dt",
            "0.5",
            "--steps",
            "2",
            "--force",
            "[4,0,0]",
            "--summary-only",
        ]
    )
    output = capsys.readouterr().out
    assert '"steps": 2' in output
    assert '"position"' in output
