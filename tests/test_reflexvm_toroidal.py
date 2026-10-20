"""Numerical and topological regressions for the independent T³ operator."""
import numpy as np
import pytest

from jarvisx.reflexvm_toroidal import (
    benchmark,
    cycle_explicit,
    decode6,
    encode6,
    fold,
    periodic_max_error,
    refine,
    toroidal_contraction,
)


def test_large_coordinates_fold():
    np.testing.assert_allclose(
        fold([[1234567.89, 987654.321, 314159.265]]),
        [[567.89, 654.321, 159.265]], atol=2e-10,
    )


def test_codec_roundtrip():
    rng = np.random.default_rng(3)
    points = rng.uniform(0, 1000, (10000, 3))
    assert np.max(periodic_max_error(points, decode6(encode6(points)))) < 3e-12


def test_seam_continuity():
    x = [[0.0001, 400, 700]]
    y = [[999.9999, 400, 700]]
    anchor = [[500, 500, 500]]
    error = periodic_max_error(toroidal_contraction(x, anchor), toroidal_contraction(y, anchor))
    assert error[0] <= 0.5 * 0.0002 + 1e-10


def test_global_lipschitz_randomized():
    rng = np.random.default_rng(7)
    x, y = rng.uniform(0, 1000, (2, 20000, 3))
    anchor = [[500, 500, 500]]
    lhs = periodic_max_error(toroidal_contraction(x, anchor), toroidal_contraction(y, anchor))
    rhs = 0.5 * periodic_max_error(x, y)
    assert np.all(lhs <= rhs + 1e-10)


def test_anchor_is_fixed_point():
    anchor = [[0, 500, 999.9], [10, 15, 20]]
    assert np.max(periodic_max_error(toroidal_contraction(anchor, anchor), anchor)) < 1e-10


def test_geometric_convergence():
    x, history, steps = refine([[567.89, 654.321, 159.265]], [[500, 500, 500]],
                               tol=1e-11, max_iters=60)
    assert 1 <= steps <= 60
    assert history[-1] <= 1e-11
    assert np.all(history[1:] <= 0.5 * history[:-1] + 1e-10)
    assert np.max(periodic_max_error(x, [[500, 500, 500]])) <= 1e-11


def test_distinct_anchors_preserve_distinct_points():
    anchors = [[1, 2, 3], [701, 601, 501]]
    points = fold(np.asarray(anchors) + [[100, -50, 25], [-90, 30, 90]])
    out, hist, _ = refine(points, anchors, tol=1e-10)
    assert hist[-1] < 1e-10
    assert periodic_max_error(out[:1], out[1:])[0] > 200


def test_fused_matches_explicit_cycle():
    rng = np.random.default_rng(5)
    x, anchor = rng.uniform(0, 1000, (2, 2000, 3))
    assert np.max(periodic_max_error(
        toroidal_contraction(x, anchor), cycle_explicit(x, anchor)
    )) < 1e-11


@pytest.mark.parametrize("period,q", [(-1, 0.5), (1000, 0), (1000, 1), (1000, float('nan'))])
def test_invalid_parameters(period, q):
    with pytest.raises(ValueError):
        toroidal_contraction([[1, 2, 3]], [[4, 5, 6]], period=period, q=q)


def test_invalid_shapes_and_nonfinite():
    with pytest.raises(ValueError):
        fold([[1, 2]])
    with pytest.raises(ValueError):
        fold([[1, float('inf'), 3]])
    with pytest.raises(ValueError):
        toroidal_contraction([[1, 2, 3], [4, 5, 6]], [[1, 2, 3]] * 3)
    with pytest.raises(ValueError):
        decode6([[0, 0, 0, 0, 0, 0]])


def test_zero_iterations_and_invalid_q():
    x, history, steps = refine([[1, 2, 3]], [[4, 5, 6]], max_iters=0)
    assert steps == 0 and len(history) == 1
    np.testing.assert_allclose(x, [[1, 2, 3]])
    with pytest.raises(ValueError):
        refine([[1, 2, 3]], [[1, 2, 3]], q=1.5)


def test_benchmark_returns_units():
    data = benchmark(n=50, steps=4)
    assert data['points'] == 50 and data['steps'] == 4
    assert data['point_steps_per_second'] == 4 * data['trajectories_per_second']
