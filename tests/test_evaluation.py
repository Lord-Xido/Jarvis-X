import math

import pytest

from jarvisx.evaluation import (
    Metric,
    canonical_gate_names,
    qualifies_for_ten,
    score_category,
    score_system,
    summarize,
)


def test_category_uses_weighted_geometric_mean():
    result = score_category(
        "example",
        [Metric("a", 1.0), Metric("b", 0.25)],
    )
    assert result.score == pytest.approx(5.0)


def test_zero_metric_forces_zero_category_score():
    result = score_category(
        "safety",
        [Metric("policy", 1.0), Metric("critical_findings_zero", 0.0)],
    )
    assert result.score == 0.0


def test_system_score_is_weakest_link_not_average():
    strong = score_category("strong", [Metric("complete", 1.0)])
    weak = score_category("weak", [Metric("partial", 0.4)])
    assert score_system([strong, weak]) == pytest.approx(4.0)


def test_ten_requires_every_metric_to_be_complete():
    complete = score_category(
        "complete",
        [Metric("one", 1.0), Metric("two", 1.0)],
    )
    partial = score_category("partial", [Metric("three", 0.999)])

    assert qualifies_for_ten([complete]) is True
    assert qualifies_for_ten([complete, partial]) is False


def test_summary_is_machine_readable():
    category = score_category("audit", [Metric("replay", 1.0)])
    report = summarize([category])

    assert report["system_score"] == 10.0
    assert report["qualifies_for_ten"] is True
    assert report["categories"]["audit"]["metrics"]["replay"]["value"] == 1.0


def test_metric_contract_rejects_invalid_values():
    with pytest.raises(ValueError):
        Metric("bad", -0.1)
    with pytest.raises(ValueError):
        Metric("bad", 1.1)
    with pytest.raises(ValueError):
        Metric("bad", 0.5, weight=0.0)


def test_canonical_gates_cover_runtime_risk_families():
    gates = canonical_gate_names()
    assert "mathematical_completeness" in gates
    assert "scientific_validation" in gates
    assert "security" in gates
    assert "deterministic_replay_match" in gates["testability_auditability"]
