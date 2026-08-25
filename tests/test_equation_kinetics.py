import math

import pytest

from jarvisx.equation_kinetics import (
    EquationKineticConfig,
    EquationKineticState,
    dual_synchronous_step,
    population_moments,
    propose_step,
    residual_energy,
    residual_force,
)


def identity_residual(x):
    return x


def identity_jacobian(x):
    del x
    return ((1.0,),)


def zero_residual(x):
    return tuple(0.0 for _ in x)


def scalar_identity_jacobian(x):
    del x
    return ((1.0,),)


def test_residual_energy_and_force_match_linear_identity_case():
    residual = (2.0, -3.0)
    jacobian = ((1.0, 0.0), (0.0, 1.0))

    assert residual_energy(residual) == pytest.approx(6.5)
    assert residual_force(residual, jacobian) == pytest.approx((-2.0, 3.0))


def test_nonlinear_residual_force_uses_jacobian_chain_rule():
    # F(x) = x^2 - 1 at x=2 -> F=3 and J=4, hence -J^T F = -12.
    force = residual_force((3.0,), ((4.0,),))
    assert force == pytest.approx((-12.0,))


def test_stable_damped_step_moves_toward_root_and_reduces_total_local_energy():
    state = EquationKineticState(position=(1.0,), velocity=(0.0,))
    config = EquationKineticConfig(
        dt=0.1,
        mass=1.0,
        damping=0.4,
        coupling=0.0,
        memory_gain=0.0,
        max_speed=10.0,
    )

    step = propose_step(state, identity_residual, identity_jacobian, config)

    assert step.state.position[0] < state.position[0]
    assert step.state.velocity[0] < 0.0
    before_total = step.metrics_before.residual_energy + step.metrics_before.kinetic_energy
    after_total = step.metrics_after.residual_energy + step.metrics_after.kinetic_energy
    assert after_total < before_total


def test_dual_synchronous_coupling_reduces_disagreement_from_same_snapshot():
    state_a = EquationKineticState(position=(-1.0,), velocity=(0.0,))
    state_b = EquationKineticState(position=(1.0,), velocity=(0.0,))
    config = EquationKineticConfig(
        dt=1.0,
        mass=1.0,
        damping=0.0,
        coupling=0.25,
        memory_gain=0.0,
        max_speed=10.0,
    )

    result = dual_synchronous_step(
        state_a,
        state_b,
        zero_residual,
        scalar_identity_jacobian,
        zero_residual,
        scalar_identity_jacobian,
        config,
    )

    assert result.committed
    assert result.disagreement_before == pytest.approx(2.0)
    assert result.disagreement_after == pytest.approx(1.0)
    assert result.state_a.step == result.state_b.step == 1
    assert result.total_energy_after < result.total_energy_before


def test_dual_step_rolls_back_both_states_when_joint_validator_fails():
    state_a = EquationKineticState(position=(-1.0,), velocity=(0.0,))
    state_b = EquationKineticState(position=(1.0,), velocity=(0.0,))
    config = EquationKineticConfig(
        dt=1.0,
        damping=0.0,
        coupling=0.25,
        memory_gain=0.0,
    )

    result = dual_synchronous_step(
        state_a,
        state_b,
        zero_residual,
        scalar_identity_jacobian,
        zero_residual,
        scalar_identity_jacobian,
        config,
        validator=lambda candidate: abs(candidate.position[0]) < 0.25,
    )

    assert not result.committed
    assert result.state_a == state_a
    assert result.state_b == state_b
    assert result.disagreement_after == result.disagreement_before
    assert result.total_energy_after == result.total_energy_before


def test_max_disagreement_gate_is_atomic():
    state_a = EquationKineticState(position=(-2.0,), velocity=(0.0,))
    state_b = EquationKineticState(position=(2.0,), velocity=(0.0,))
    config = EquationKineticConfig(
        dt=0.1,
        damping=0.0,
        coupling=0.0,
        memory_gain=0.0,
        max_disagreement=1.0,
    )

    result = dual_synchronous_step(
        state_a,
        state_b,
        zero_residual,
        scalar_identity_jacobian,
        zero_residual,
        scalar_identity_jacobian,
        config,
    )

    assert not result.committed
    assert result.state_a == state_a
    assert result.state_b == state_b


def test_population_moments_define_equation_kinetic_temperature_as_velocity_variance():
    states = (
        EquationKineticState(position=(0.0,), velocity=(-1.0,)),
        EquationKineticState(position=(2.0,), velocity=(1.0,)),
    )

    moments = population_moments(states)

    assert moments.count == 2
    assert moments.mean_position == pytest.approx((1.0,))
    assert moments.mean_velocity == pytest.approx((0.0,))
    assert moments.velocity_covariance_trace == pytest.approx(1.0)
    assert moments.kinetic_temperature == pytest.approx(1.0)


def test_memory_is_bounded_finite_state_and_updates_from_local_force():
    state = EquationKineticState(position=(1.0,), velocity=(0.0,), memory=(0.0,))
    config = EquationKineticConfig(
        dt=0.05,
        damping=0.2,
        memory_retention=0.5,
        memory_gain=0.1,
    )

    step = propose_step(state, identity_residual, identity_jacobian, config)

    assert step.state.memory == pytest.approx((-0.5,))
    assert all(math.isfinite(value) for value in step.state.memory)


def test_coupled_processors_must_share_logical_step():
    state_a = EquationKineticState(position=(0.0,), velocity=(0.0,), step=0)
    state_b = EquationKineticState(position=(0.0,), velocity=(0.0,), step=1)

    with pytest.raises(ValueError, match="same logical step"):
        dual_synchronous_step(
            state_a,
            state_b,
            zero_residual,
            scalar_identity_jacobian,
            zero_residual,
            scalar_identity_jacobian,
            EquationKineticConfig(),
        )


def test_non_finite_equation_state_is_rejected():
    with pytest.raises(ValueError, match="finite"):
        EquationKineticState(position=(float("nan"),), velocity=(0.0,))
