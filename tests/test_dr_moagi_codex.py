from __future__ import annotations

import math

import pytest

from jarvisx.dr_moagi_codex import (
    DrMoagiCodex,
    DrMoagiCodexConfig,
    fixed_point_recurse,
    helmholtz_permeate,
    l2_norm,
    project_l2_ball,
    update_parameters,
)


def halve(latent, time_index, condition):
    return tuple(0.5 * value for value in latent)


def test_fixed_point_recursion_converges_with_actual_iteration_count():
    config = DrMoagiCodexConfig(
        fixed_point_tolerance=1e-6,
        max_fixed_point_iterations=64,
        claimed_contraction=0.5,
    )
    result = fixed_point_recurse((1.0,), halve, time_index=0, condition=None, config=config)
    assert result.converged
    assert result.iterations == 20
    assert result.final_delta <= 1e-6
    assert result.max_observed_contraction == pytest.approx(0.5)


def test_claimed_contraction_fails_closed_when_trajectory_violates_it():
    config = DrMoagiCodexConfig(
        fixed_point_tolerance=1e-12,
        max_fixed_point_iterations=8,
        claimed_contraction=0.4,
    )
    with pytest.raises(RuntimeError, match="claimed_contraction"):
        fixed_point_recurse((1.0,), halve, time_index=0, condition=None, config=config)


def test_projection_is_l2_ball_not_componentwise_clip():
    projected = project_l2_ball((3.0, 4.0), 2.0)
    assert projected == pytest.approx((1.2, 1.6))
    assert l2_norm(projected) == pytest.approx(2.0)


def test_parameter_update_is_separate_from_latent_state():
    theta_next = update_parameters((1.0, -1.0), (0.5, 0.25), eta_theta=0.2)
    assert theta_next == pytest.approx((0.9, -1.05))


def test_helmholtz_k_zero_matches_static_green_kernel_at_unit_distance():
    phi = helmholtz_permeate(
        {(0.0, 0.0, 0.0): 1.0},
        [(1.0, 0.0, 0.0)],
        wave_number=0.0,
        cell_volume=1.0,
        softening=1e-6,
    )
    assert phi[(1.0, 0.0, 0.0)].real == pytest.approx(1.0 / (4.0 * math.pi))
    assert phi[(1.0, 0.0, 0.0)].imag == pytest.approx(0.0)


def test_green_kernel_self_source_is_regularized():
    phi = helmholtz_permeate(
        {(0.0, 0.0, 0.0): 1.0},
        [(0.0, 0.0, 0.0)],
        wave_number=0.0,
        cell_volume=1.0,
        softening=0.5,
    )
    assert math.isfinite(phi[(0.0, 0.0, 0.0)].real)
    assert phi[(0.0, 0.0, 0.0)].real == pytest.approx(1.0 / (2.0 * math.pi))


def test_codex_executes_end_to_end_and_projects_before_decode():
    scene = {(0.0, 0.0, 0.0): 1.0, (1.0, 0.0, 0.0): 2.0}

    def encoder(field):
        return (field[(0.0, 0.0, 0.0)], field[(1.0, 0.0, 0.0)])

    decoded_inputs = []

    def decoder(latent):
        decoded_inputs.append(tuple(latent))
        return {(0.0, 0.0, 0.0): latent[0], (1.0, 0.0, 0.0): latent[1]}

    def source_mapper(latent):
        return {(0.0, 0.0, 0.0): latent[0], (1.0, 0.0, 0.0): latent[1]}

    codex = DrMoagiCodex(
        encoder=encoder,
        decoder=decoder,
        inward_operator=halve,
        source_mapper=source_mapper,
        config=DrMoagiCodexConfig(
            lambda_max=0.25,
            fixed_point_tolerance=1e-6,
            max_fixed_point_iterations=64,
            claimed_contraction=0.5,
            gamma=1.0,
            beta=0.0,
            green_softening=0.25,
            eta_theta=0.1,
        ),
    )
    result = codex.execute(
        scene,
        theta=(1.0, 2.0),
        theta_gradient=(0.5, -0.5),
    )

    assert result.fixed_point.converged
    assert l2_norm(result.projected_latent) <= 0.25 + 1e-12
    assert decoded_inputs[-1] == pytest.approx(result.projected_latent)
    assert result.theta_after == pytest.approx((0.95, 2.05))
    assert result.virtual_depth_label == "1000000^1000000"
    assert result.fixed_point.iterations < 64
    assert set(result.permeation_field) == set(result.source_charge)


def test_nonconvergent_operator_rejected_when_convergence_required():
    def translate(latent, time_index, condition):
        return tuple(value + 1.0 for value in latent)

    config = DrMoagiCodexConfig(
        max_fixed_point_iterations=3,
        fixed_point_tolerance=1e-12,
        require_convergence=True,
    )
    with pytest.raises(RuntimeError, match="did not converge"):
        fixed_point_recurse((0.0,), translate, time_index=0, condition=None, config=config)
