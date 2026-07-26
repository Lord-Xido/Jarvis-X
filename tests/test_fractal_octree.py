import math

import pytest

from jarvisx.fractal_octree import FractalOctreeNode, build_fractal_octree


def test_depth_three_matches_closed_form_metrics():
    root = build_fractal_octree(size=1.0, max_depth=3)
    measured = root.metrics()
    expected = root.expected_metrics()

    assert measured.active_nodes == 85
    assert measured.active_leaves == 64
    assert measured.retained_volume == pytest.approx(0.125)
    assert measured == expected
    assert root.calculate_metrics() == pytest.approx((85, 0.125))


def test_each_active_parent_has_four_active_octants():
    root = build_fractal_octree(size=1.0, max_depth=1)
    active_children = [child for child in root.children if child.is_active]
    inactive_children = [child for child in root.children if not child.is_active]

    assert len(root.children) == 8
    assert len(active_children) == 4
    assert len(inactive_children) == 4
    assert {(child.x, child.y, child.z) for child in active_children} == {
        (0.0, 0.0, 0.0),
        (0.5, 0.0, 0.0),
        (0.0, 0.5, 0.0),
        (0.0, 0.0, 0.5),
    }


def test_depth_zero_retains_the_root_cube():
    root = build_fractal_octree(size=2.0, max_depth=0)
    metrics = root.metrics()

    assert metrics.active_nodes == 1
    assert metrics.active_leaves == 1
    assert metrics.retained_volume == pytest.approx(8.0)
    assert list(root.iter_active()) == [root]


def test_repeated_subdivision_is_idempotent():
    root = build_fractal_octree(size=1.0, max_depth=3)
    first = root.metrics()

    root.subdivide_and_optimize()
    second = root.metrics()

    assert first == second
    assert len(root.children) == 8


def test_general_closed_form_invariants():
    for depth in range(6):
        root = build_fractal_octree(size=1.0, max_depth=depth)
        metrics = root.metrics()

        assert metrics.active_leaves == 4 ** depth
        assert metrics.active_nodes == (4 ** (depth + 1) - 1) // 3
        assert metrics.retained_volume == pytest.approx(2.0 ** (-depth))
        assert math.isclose(metrics.fractal_dimension, 2.0)


def test_invalid_geometry_is_rejected():
    with pytest.raises(ValueError, match="size must be positive"):
        FractalOctreeNode(0.0, 0.0, 0.0, 0.0)

    with pytest.raises(ValueError, match="max_depth"):
        FractalOctreeNode(0.0, 0.0, 0.0, 1.0, depth=2, max_depth=1)

    with pytest.raises(ValueError, match="octant coordinates"):
        FractalOctreeNode.survives(2, 0, 0)
