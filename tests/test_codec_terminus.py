from __future__ import annotations

import pytest

from jarvisx.codec_terminus import (
    AxisAssignment,
    CodecMarkovOperator,
    LinearGeometricCost,
    TerminusCandidate,
    ThroughputBounds,
    audit_local_sections,
    kleene_iterate,
    select_terminus,
)


def test_codec_markov_operator_composes_and_mixes():
    operator = CodecMarkovOperator.from_components(
        encoder=((1.0, 0.0), (0.0, 1.0)),
        transport=((0.75, 0.25), (0.25, 0.75)),
        decoder=((1.0, 0.0), (0.0, 1.0)),
    )

    receipt = operator.mix((1.0, 0.0), tolerance=1.0e-9)

    assert receipt.converged
    assert receipt.probabilities == pytest.approx((0.5, 0.5), abs=1.0e-8)
    assert receipt.l1_residual <= 1.0e-9
    assert operator.x_size == 2
    assert operator.y_size == 2


def test_codec_rejects_non_stochastic_components():
    with pytest.raises(ValueError, match="sum to 1"):
        CodecMarkovOperator.from_components(
            encoder=((0.8, 0.0), (0.0, 1.0)),
            transport=((1.0, 0.0), (0.0, 1.0)),
            decoder=((1.0, 0.0), (0.0, 1.0)),
        )


def test_five_bound_throughput_reports_active_bottleneck():
    bounds = ThroughputBounds(
        synthesis=120.0,
        bandwidth=90.0,
        quantization=110.0,
        thermal=80.0,
        channel=95.0,
    )

    assert bounds.effective == pytest.approx(80.0)
    assert bounds.active_bounds == ("thermal",)


def test_geometric_axis_assignment_and_linear_cost_are_explicit():
    axes = AxisAssignment()
    cost = LinearGeometricCost(n=256, coefficient=4.0)

    assert (axes.x, axes.y, axes.z) == ("position", "bitplane", "stage")
    assert cost.total == pytest.approx(1024.0)
    assert cost.normalized == pytest.approx(4.0)
    assert cost.asymptotic_class == "Theta(n)"


def test_cech_graph_audit_accepts_consistent_acyclic_cover():
    audit = audit_local_sections(
        {
            "U0": {"a": 1, "b": 2},
            "U1": {"b": 2, "c": 3},
            "U2": {"c": 3, "d": 4},
        }
    )

    assert audit.passed
    assert audit.obstruction_zero
    assert audit.h1_dimension == 0
    assert audit.edges == 2
    assert audit.components == 1
    assert audit.conflicts == ()


def test_cech_graph_audit_rejects_overlap_conflict():
    audit = audit_local_sections(
        {
            "U0": {"shared": 1},
            "U1": {"shared": 2},
        }
    )

    assert not audit.passed
    assert audit.h1_dimension == 0
    assert len(audit.conflicts) == 1


def test_cech_graph_audit_exposes_nonzero_h1_cycle():
    audit = audit_local_sections(
        {
            "U0": {"a": 1, "c": 3},
            "U1": {"a": 1, "b": 2},
            "U2": {"b": 2, "c": 3},
        }
    )

    assert not audit.passed
    assert audit.conflicts == ()
    assert audit.h1_dimension == 1
    assert not audit.obstruction_zero


def test_kleene_iteration_converges_on_ascending_chain():
    target = frozenset({0, 1, 2})

    def step(value: frozenset[int]) -> frozenset[int]:
        for item in sorted(target):
            if item not in value:
                return value | {item}
        return value

    receipt = kleene_iterate(
        frozenset(),
        step,
        leq=lambda left, right: left.issubset(right),
        max_iterations=8,
    )

    assert receipt.converged
    assert receipt.value == target
    assert receipt.iterations == 4


def test_kleene_iteration_fails_closed_on_descending_step():
    with pytest.raises(ValueError, match="ascending order"):
        kleene_iterate(
            frozenset({1}),
            lambda _: frozenset(),
            leq=lambda left, right: left.issubset(right),
        )


def test_terminus_selects_minimum_cost_audited_fixed_point_only():
    good_audit = audit_local_sections(
        {
            "U0": {"a": 1},
            "U1": {"b": 2},
        }
    )
    bad_audit = audit_local_sections(
        {
            "U0": {"shared": 1},
            "U1": {"shared": 2},
        }
    )

    candidates = (
        TerminusCandidate(value=8, cost=8.0, audit=good_audit),
        TerminusCandidate(value=3, cost=3.0, audit=good_audit),
        TerminusCandidate(value=1, cost=1.0, audit=bad_audit),
    )

    receipt = select_terminus(candidates, operator=lambda value: value)

    assert receipt.value == 3
    assert receipt.cost == pytest.approx(3.0)
    assert receipt.eligible_candidates == 2
