"""Independent numerical and integration checks for Phi_M's optional 3D operator."""

import numpy as np
import pytest

from jarvisx.dr_moagi_phi_m_3d import (
    PhiMConfig,
    gradient,
    graph_candidate,
    integrated_gradient_hessian,
    numerical_self_test,
    phi_m,
    potential,
    spatial_jacobian,
    static_source_integral,
    temporal_derivative,
)
from jarvisx.geometric_diffusion_runtime import GeometricGraph, GeometricNode, GeometricDiffusionRuntime


def sample_graph():
    return GeometricGraph(
        nodes=(
            GeometricNode((0, 0, 0), (1.0, 2.0)),
            GeometricNode((0.2, 0.4, -0.3), (3.0, 4.0)),
        ),
        edges=((0, 1),),
    )


def test_analytic_derivatives_and_stationary_integral():
    report = numerical_self_test()
    assert report["passed"], report
    assert report["spatial_jacobian_max_abs_error"] < 2e-7
    assert report["temporal_derivative_max_abs_error"] < 2e-7
    assert report["stationary_integral_max_abs_error"] < 2e-8


def test_origin_zero_time_and_batch_shapes():
    p = PhiMConfig()
    x = np.zeros((2, 3, 3))
    y = phi_m(x, 0.0, p)
    assert y.shape == x.shape
    np.testing.assert_array_equal(y, 0.0)
    assert spatial_jacobian(x, 0.0, p).shape == (2, 3, 3, 3)
    assert potential(x, 1.0, p).shape == (2, 3)
    assert gradient(x, 1.0, p).shape == (2, 3, 3)
    assert temporal_derivative(x, 1.0, p).shape == x.shape
    assert np.isfinite(y).all()
    assert np.max(np.abs(phi_m(x + 0.5, 1.0, p))) < 1.0


def test_integral_zero_and_closed_form_frequency_zero():
    p = PhiMConfig(velocity=(0.0, 0.0, 0.0), frequency=0.0)
    x = np.array([[0.2, 0.5, 0.7], [1.0, -0.2, 0.1]])
    g0, h0 = integrated_gradient_hessian(x, 0.0, p)
    np.testing.assert_array_equal(g0, 0.0)
    np.testing.assert_array_equal(h0, 0.0)
    g, _ = integrated_gradient_hessian(x, 0.8, p)
    np.testing.assert_allclose(g, static_source_integral(x, 0.8, p), atol=1e-12)


def test_graph_adapter_preserves_authority_topology_and_features():
    graph = sample_graph()
    runtime = GeometricDiffusionRuntime(graph)
    old = runtime.state
    proposed = graph_candidate(graph, 1.5, max_position_step=0.1, max_rms=1.0)
    assert proposed.edges == graph.edges
    assert [x.feature for x in proposed.nodes] == [x.feature for x in graph.nodes]
    assert proposed != graph
    for old_node, new_node in zip(graph.nodes, proposed.nodes):
        assert max(abs(a - b) for a, b in zip(old_node.position, new_node.position)) <= 0.1000000001
    assert runtime.state is old
    assert graph.nodes[0].position == (0.0, 0.0, 0.0)


def test_graph_candidate_rms_rejection_does_not_commit():
    graph = sample_graph()
    runtime = GeometricDiffusionRuntime(graph)
    prior = runtime.state
    with pytest.raises(RuntimeError, match="RMS"):
        graph_candidate(graph, 1.5, max_position_step=0.25, max_rms=0.0)
    assert runtime.state is prior


@pytest.mark.parametrize("invalid", [
    {"spatial_sigma": 0.0}, {"modulation": 1.01},
    {"alpha": float("nan")}, {"quadrature_order": 0},
    {"quadrature_order": 65}, {"max_points": 10000},
    {"velocity": (0.0, 0.0)}, {"center": (float("inf"), 0.0, 0.0)},
])
def test_invalid_config_fails_closed(invalid):
    with pytest.raises((ValueError, TypeError)):
        PhiMConfig(**invalid)


@pytest.mark.parametrize("coords", [
    np.zeros((0, 3)), [1.0, 2.0], [[float("inf"), 0.0, 0.0]],
    [[0, 0, 1001]], np.zeros((4097, 3)),
])
def test_invalid_geometry_fails_closed(coords):
    with pytest.raises(ValueError):
        phi_m(coords, 0.5)


@pytest.mark.parametrize("t", [-1.0, float("nan"), float("inf"), 11.0])
def test_invalid_time_fails_closed(t):
    with pytest.raises(ValueError):
        phi_m([0.0, 0.0, 0.0], t)


def test_graph_rejects_too_many_nodes():
    node = GeometricNode((0.0, 0.0, 0.0), (0.0,))
    graph = GeometricGraph((node,) * 3, ())
    with pytest.raises(ValueError, match="budget"):
        graph_candidate(graph, 0.5, PhiMConfig(max_points=2))
