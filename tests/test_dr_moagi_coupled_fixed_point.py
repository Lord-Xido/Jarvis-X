from __future__ import annotations

import math

import pytest

from jarvisx.dr_moagi_coupled_fixed_point import (
    WireSpec,
    centerline_point,
    coupled_iteration,
    damped_newton_step,
    eikonal_residual,
    fixed_point_residual,
    hard_density,
    image_feature_vector,
    numerical_gradient_hessian,
    optimize_coupled,
    patch_tokens,
    render_pixel,
    sdf,
    self_attention,
)


def test_default_hairpin_geometry_matches_requested_dimensions():
    spec = WireSpec()
    assert spec.radius_mm == pytest.approx(0.5)
    assert spec.centerline_separation_mm == pytest.approx(3.0)
    assert spec.bend_radius_mm == pytest.approx(1.5)
    assert spec.bend_curvature_per_mm == pytest.approx(2.0 / 3.0)
    assert 2.0 * spec.leg_length_mm + math.pi * spec.bend_radius_mm == pytest.approx(1000.0)

    left_start = centerline_point(0.0, spec)
    right_end = centerline_point(spec.total_length_mm, spec)
    assert left_start == pytest.approx((-1.5, 0.0, 0.0))
    assert right_end == pytest.approx((1.5, 0.0, 0.0))


def test_sdf_is_zero_on_straight_leg_surface_and_density_is_solid():
    spec = WireSpec()
    y = 0.5 * spec.leg_length_mm
    surface = (-spec.bend_radius_mm - spec.radius_mm, y, 0.0)
    inside = (-spec.bend_radius_mm, y, 0.0)
    outside = (-spec.bend_radius_mm - spec.radius_mm - 0.2, y, 0.0)

    assert sdf(surface, spec) == pytest.approx(0.0, abs=1.0e-12)
    assert sdf(inside, spec) < 0.0
    assert sdf(outside, spec) > 0.0
    assert hard_density(inside, spec) == spec.copper_density
    assert hard_density(outside, spec) == 0.0


def test_eikonal_property_holds_away_from_nonsmooth_medial_loci():
    spec = WireSpec()
    y = 0.5 * spec.leg_length_mm
    point = (-spec.bend_radius_mm - 1.2, y, 0.3)
    assert eikonal_residual(point, spec) < 1.0e-5


def test_renderer_distinguishes_wire_from_empty_space():
    spec = WireSpec()
    y = 0.5 * spec.leg_length_mm
    on_wire = render_pixel(-spec.bend_radius_mm, y, spec, samples=48)
    empty = render_pixel(6.0, y, spec, samples=48)

    assert 0.0 <= empty <= 1.0
    assert 0.0 <= on_wire <= 1.0
    assert on_wire > empty + 0.5


def test_patch_tokens_and_attention_preserve_expected_shape():
    image = (
        (0.0, 0.0, 1.0, 1.0),
        (0.0, 0.0, 1.0, 1.0),
        (1.0, 1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0, 0.0),
    )
    tokens = patch_tokens(image, patch=2)
    attended = self_attention(tokens)

    assert len(tokens) == 4
    assert len(attended) == 4
    assert all(len(token) == 3 for token in attended)
    assert all(all(math.isfinite(value) for value in token) for token in attended)


def test_damped_newton_step_reduces_quadratic_loss():
    def loss(z):
        return (z[0] - 1.0) ** 2 + 2.0 * (z[1] - 2.0) ** 2

    z = (1.5, 1.0)
    updated = damped_newton_step(loss, z, trust_radius=2.0)
    assert loss(updated) < loss(z)


def test_fixed_point_residual_is_zero_for_idempotent_projection():
    def phi(z):
        return (max(0.0, z[0]), max(0.0, z[1]))

    assert fixed_point_residual(phi, (-2.0, 3.0)) == pytest.approx(0.0)


def test_coupled_iteration_moves_bad_geometry_toward_target():
    target = WireSpec()
    target_features = image_feature_vector(target, width=16, height=16, samples=8)
    base = WireSpec()
    initial = (1.35, 2.45)

    result = coupled_iteration(
        base,
        initial,
        target_features,
        trust_radius=0.3,
        render_samples=8,
    )

    assert result.loss_after <= result.loss_before
    assert result.z_after != result.z_before
    assert math.isfinite(result.fixed_point_residual)



def test_shared_gradient_hessian_stencil_uses_nine_unique_evaluations():
    calls = 0

    def loss(z):
        nonlocal calls
        calls += 1
        return (z[0] - 1.0) ** 2 + (z[1] - 2.0) ** 2

    gradient, hessian, f0 = numerical_gradient_hessian(loss, (1.4, 1.6))
    assert calls == 9
    assert f0 > 0.0
    assert gradient[0] > 0.0
    assert gradient[1] < 0.0
    assert hessian[0][0] == pytest.approx(2.0, rel=1.0e-5)
    assert hessian[1][1] == pytest.approx(2.0, rel=1.0e-5)


def test_coupled_iteration_reuses_loss_evaluations():
    target = WireSpec()
    target_features = image_feature_vector(target, width=16, height=16, samples=6)
    result = coupled_iteration(
        WireSpec(),
        (1.25, 2.35),
        target_features,
        trust_radius=0.25,
        render_samples=6,
    )

    assert result.loss_after <= result.loss_before
    assert result.loss_evaluations <= 24
    assert result.step_norm >= 0.0


def test_auto_optimizer_reduces_external_objective():
    target = WireSpec()
    target_features = image_feature_vector(target, width=16, height=16, samples=6)
    result = optimize_coupled(
        WireSpec(),
        (1.4, 2.5),
        target_features,
        max_iterations=4,
        render_samples=6,
    )

    assert result.iterations <= 4
    assert result.loss_final < result.loss_initial
    assert result.total_loss_evaluations > 0
    assert math.dist(result.z_final, (1.0, 2.0)) < math.dist(result.z_initial, (1.0, 2.0))
