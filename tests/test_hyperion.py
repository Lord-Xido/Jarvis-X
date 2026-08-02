from dataclasses import replace

import pytest

from jarvisx.hyperion import (
    FILTER_NAMES,
    HyperionConfig,
    HyperionEngine,
    Observation,
    ScoreModel,
    TrainingExample,
    binary_cross_entropy,
)


def obs(
    source,
    timestamp,
    value,
    event,
    *,
    quantity="amount",
    unit="ZAR",
    label="known",
    confidence=1.0,
):
    return Observation(
        source=source,
        timestamp_ms=timestamp,
        value=value,
        quantity=quantity,
        unit=unit,
        correlation_id=event,
        label=label,
        confidence=confidence,
    )


def test_semantically_incompatible_video_identity_is_not_averaged() -> None:
    engine = HyperionEngine()
    report = engine.audit(
        [
            obs("csv", 1000, 100.0, "e1"),
            obs("cpu", 1001, 100.0, "e1"),
            obs(
                "video",
                1002,
                1.0,
                "e1",
                quantity="identity_match",
                unit="boolean",
            ),
        ]
    )
    assert report.points[0].fused_value == 100.0
    assert len(report.witnesses[0].terms) == 2


def test_time_correct_derivatives_use_seconds() -> None:
    engine = HyperionEngine()
    report = engine.audit(
        [
            obs("csv", 0, 0.0, "e0"),
            obs("csv", 1000, 10.0, "e1"),
            obs("csv", 3000, 30.0, "e2"),
        ]
    )
    assert report.points[1].velocity == pytest.approx(10.0)
    assert report.points[2].velocity == pytest.approx(10.0)
    assert report.points[2].acceleration == pytest.approx(0.0)


def test_continuity_residual_is_not_tautologically_zero() -> None:
    engine = HyperionEngine()
    report = engine.audit(
        [
            obs("csv", 0, 10.0, "e0"),
            obs("csv", 1000, 11.0, "e1"),
            obs("csv", 2000, 12.0, "e2"),
            obs("csv", 3000, 100.0, "e3"),
        ]
    )
    final = report.points[-1]
    assert final.continuity_residual != 0.0
    assert final.flags["continuity"]


def test_bytecode_divergence_requires_same_event_and_detects_mismatch() -> None:
    engine = HyperionEngine()
    report = engine.audit(
        [
            obs("csv", 1000, 100.0, "e1"),
            obs("cpu", 1001, 80.0, "e1"),
            obs("cpu", 1002, 100.0, "other"),
        ]
    )
    point = next(point for point in report.points if point.event_id == "e1")
    assert point.flags["bytecode_divergence"]
    assert point.severities["bytecode_divergence"] > 0.0


def test_fixed_point_witness_and_report_verify_and_tampering_fails() -> None:
    engine = HyperionEngine()
    report = engine.audit(
        [
            obs("csv", 1000, 100.01, "e1", confidence=1.0),
            obs("audio", 1001, 99.99, "e1", confidence=0.8),
            obs("cpu", 1002, 100.00, "e1", confidence=1.0),
        ]
    )
    assert report.verify()
    witness = report.witnesses[0]
    assert witness.verify()
    tampered = replace(witness, fused_int=witness.fused_int + 1)
    assert not tampered.verify()


def test_cas_and_ghs_are_bounded() -> None:
    engine = HyperionEngine()
    report = engine.audit(
        [obs("csv", index * 1000, float(index), f"e{index}") for index in range(20)]
    )
    assert all(0.0 <= point.cas <= 1.0 for point in report.points)
    assert 0.0 <= report.geometric_health_score <= 100.0


def test_supervised_training_requires_labels_and_reduces_loss() -> None:
    zeros = {name: 0.0 for name in FILTER_NAMES}
    positives = {name: 1.0 for name in FILTER_NAMES}
    examples = [
        TrainingExample(zeros, 0),
        TrainingExample(zeros, 0),
        TrainingExample(positives, 1),
        TrainingExample(positives, 1),
    ]
    model = ScoreModel()
    before = binary_cross_entropy(model, examples)
    trained = model.fit_supervised(examples)
    after = binary_cross_entropy(trained, examples)
    assert trained.version == model.version + 1
    assert trained.training_digest
    assert after < before
    with pytest.raises(ValueError):
        model.fit_supervised([])


def test_deterministic_replay_yields_identical_report_digest() -> None:
    observations = [
        obs("csv", index * 1000, 100.0 + index, f"e{index}")
        for index in range(8)
    ] + [
        obs("cpu", index * 1000 + 1, 100.0 + index, f"e{index}")
        for index in range(8)
    ]
    engine = HyperionEngine()
    first = engine.audit(observations)
    second = engine.audit(list(reversed(observations)))
    assert first.report_digest == second.report_digest
    assert first.input_root == second.input_root


def test_precision_strike_uses_projected_balance_and_lower_bound() -> None:
    config = HyperionConfig(lower_bound=-100.0, robust_window=10)
    engine = HyperionEngine(config=config)
    values = [0.0, -5.0, -10.0, -15.0, -20.0, -25.0, -30.0, -35.0, -40.0, -99.0]
    report = engine.audit(
        [
            obs("csv", index * 1000, value, f"e{index}")
            for index, value in enumerate(values)
        ]
    )
    assert report.points[-1].flags["precision_strike"]


def test_configuration_rejects_unbounded_ghs_weights() -> None:
    with pytest.raises(ValueError):
        HyperionConfig(ghs_exposure_weight=0.8, ghs_frequency_weight=0.3)


def test_duplicate_source_does_not_gain_extra_fusion_weight() -> None:
    engine = HyperionEngine()
    report = engine.audit(
        [
            obs("csv", 1000, 100.0, "e1", confidence=0.8),
            obs("csv", 1001, 1000.0, "e1", confidence=0.1),
            obs("cpu", 1000, 100.0, "e1", confidence=1.0),
        ]
    )
    assert report.points[0].fused_value == 100.0
    assert len(report.witnesses[0].terms) == 2


def test_simultaneous_events_use_declared_minimum_time_resolution() -> None:
    engine = HyperionEngine(HyperionConfig(minimum_dt_ms=1))
    report = engine.audit(
        [
            obs("csv", 1000, 10.0, "e1"),
            obs("csv", 1000, 11.0, "e2"),
        ]
    )
    assert report.points[1].velocity == pytest.approx(1000.0)


def test_witness_exports_integer_circuit_relation() -> None:
    report = HyperionEngine().audit(
        [
            obs("csv", 1000, 100.0, "e1"),
            obs("cpu", 1000, 100.0, "e1"),
        ]
    )
    circuit = report.witnesses[0].circuit_inputs()
    assert circuit["numerator"] == (
        circuit["denominator"] * circuit["fused_int"] + circuit["remainder"]
    )
