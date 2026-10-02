from __future__ import annotations

import pytest

from jarvisx.cognitive_constraint_field3d import (
    CognitiveConstraintError,
    CognitiveConstraintField3D,
    CognitiveVariable3D,
    LinearConstraint,
    angle_demo_130_50,
)


def test_angle_demo_reaches_exact_100_degree_fixed_point() -> None:
    receipt = angle_demo_130_50()

    assert receipt.accepted is True
    assert receipt.fixed_point_reached is True
    assert receipt.unresolved == ()
    assert receipt.max_residual == pytest.approx(0.0)
    assert receipt.value("top_interior") == pytest.approx(50.0)
    assert receipt.value("bottom_interior") == pytest.approx(80.0)
    assert receipt.value("d") == pytest.approx(100.0)


def test_angle_demo_exposes_auditable_3d_reasoning_trajectory() -> None:
    receipt = angle_demo_130_50()

    assert [step.variable for step in receipt.trajectory] == [
        "top_interior",
        "bottom_interior",
        "d",
    ]
    assert [step.constraint for step in receipt.trajectory] == [
        "top_straight_line",
        "triangle_sum",
        "bottom_straight_line",
    ]
    assert receipt.trajectory[-1].coordinate == pytest.approx((0.80, 0.85, 0.95))


def test_inconsistent_fully_known_state_fails_closed() -> None:
    variables = (
        CognitiveVariable3D("a", (0.0, 0.0, 0.0), 100.0),
        CognitiveVariable3D("b", (1.0, 1.0, 1.0), 90.0),
    )
    constraints = (
        LinearConstraint(
            "supplement",
            (("a", 1.0), ("b", 1.0)),
            180.0,
            "straight-line invariant",
        ),
    )

    receipt = CognitiveConstraintField3D(variables, constraints).solve()

    assert receipt.fixed_point_reached is True
    assert receipt.accepted is False
    assert receipt.max_residual == pytest.approx(10.0)


def test_underdetermined_state_is_stable_but_not_accepted() -> None:
    variables = (
        CognitiveVariable3D("a", (0.0, 0.0, 0.0)),
        CognitiveVariable3D("b", (1.0, 0.0, 0.0)),
    )
    constraints = (
        LinearConstraint(
            "sum",
            (("a", 1.0), ("b", 1.0)),
            180.0,
            "two-variable sum",
        ),
    )

    receipt = CognitiveConstraintField3D(variables, constraints).solve()

    assert receipt.fixed_point_reached is True
    assert receipt.accepted is False
    assert receipt.unresolved == ("a", "b")


def test_unknown_constraint_variable_is_rejected() -> None:
    variable = CognitiveVariable3D("a", (0.0, 0.0, 0.0), 1.0)
    constraint = LinearConstraint(
        "bad",
        (("missing", 1.0),),
        1.0,
        "invalid reference",
    )

    with pytest.raises(CognitiveConstraintError, match="unknown variable"):
        CognitiveConstraintField3D((variable,), (constraint,))


def test_coordinate_contract_requires_exactly_three_finite_axes() -> None:
    with pytest.raises(CognitiveConstraintError, match="exactly three"):
        CognitiveVariable3D("bad", (0.0, 1.0), 1.0)

    with pytest.raises(CognitiveConstraintError, match="finite"):
        CognitiveVariable3D("bad", (0.0, 1.0, float("inf")), 1.0)
