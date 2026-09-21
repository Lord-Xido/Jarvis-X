import pytest

from jarvisx.adaptive_orchestrator import SecurityState
from jarvisx.continual_optimizer import (
    BenchmarkHarness,
    CanaryObservation,
    ContinualOptimizer,
    EvidencePromotionGate,
    FrontierSnapshot,
    MetricDirection,
    MetricSpec,
    OptimizationCandidate,
    PromotionStage,
    ReleaseController,
)


SPECS = (
    MetricSpec("throughput", MetricDirection.MAXIMIZE, weight=2.0, max_regression=0.0),
    MetricSpec("latency", MetricDirection.MINIMIZE, weight=1.0, maximum=100.0, max_regression=0.0),
    MetricSpec("safety", MetricDirection.MAXIMIZE, weight=3.0, minimum=0.99, max_regression=0.0),
)


def evaluator(subject, repetition):
    base = {"throughput": 100.0, "latency": 50.0, "safety": 1.0}
    if subject == "challenger":
        base = {"throughput": 120.0, "latency": 40.0, "safety": 1.0}
    elif subject == "bad":
        base = {"throughput": 140.0, "latency": 35.0, "safety": 0.95}
    return base


def make_optimizer():
    harness = BenchmarkHarness(SPECS, repetitions=3)
    gate = EvidencePromotionGate(SPECS, min_utility_gain=0.01, max_risk_score=0.4)
    release = ReleaseController(canary_min_observations=3, canary_min_success_rate=1.0)
    return ContinualOptimizer(harness=harness, gate=gate, release=release)


def test_evidence_gate_admits_measured_improvement_to_shadow():
    opt = make_optimizer()
    opt.bootstrap("champion", evaluator)
    candidate = OptimizationCandidate("challenger", "champion", "faster", 0.2)
    decision = opt.challenge(
        candidate,
        evaluator=evaluator,
        security=SecurityState(),
        frontier=FrontierSnapshot("public-bench", "2026-08", {"throughput": 125.0}),
    )
    assert decision.allowed
    assert decision.stage is PromotionStage.SHADOW
    assert decision.metric_deltas["throughput"] == 20.0
    assert decision.frontier_deltas["throughput"] == -5.0


def test_hard_metric_floor_blocks_candidate_even_when_other_metrics_improve():
    opt = make_optimizer()
    opt.bootstrap("champion", evaluator)
    decision = opt.challenge(
        OptimizationCandidate("bad", "champion", "unsafe speedup", 0.2),
        evaluator=evaluator,
        security=SecurityState(),
    )
    assert not decision.allowed
    assert "safety" in decision.reason


def test_security_state_can_contract_optimization_authority():
    opt = make_optimizer()
    opt.bootstrap("champion", evaluator)
    decision = opt.challenge(
        OptimizationCandidate("challenger", "champion", "faster", 0.2),
        evaluator=evaluator,
        security=SecurityState(confidence=0.3, reason="intrusion watch"),
    )
    assert not decision.allowed
    assert "intrusion watch" in decision.reason


def test_candidate_cannot_self_promote_past_canary():
    opt = make_optimizer()
    opt.bootstrap("champion", evaluator)
    candidate = OptimizationCandidate("challenger", "champion", "faster", 0.2)
    shadow = opt.challenge(candidate, evaluator=evaluator, security=SecurityState())
    canary = opt.release.advance(shadow)
    assert canary.stage is PromotionStage.CANARY
    production = opt.release.advance(
        canary,
        [
            CanaryObservation(True, 0.2),
            CanaryObservation(True, 0.1),
            CanaryObservation(True, 0.15),
        ],
    )
    assert production.allowed
    assert production.stage is PromotionStage.PRODUCTION
    opt.commit_production(candidate, production, evaluator=evaluator)
    assert opt.champion_id == "challenger"
    assert opt.generation == 1


def test_canary_regression_rolls_back():
    opt = make_optimizer()
    opt.bootstrap("champion", evaluator)
    candidate = OptimizationCandidate("challenger", "champion", "faster", 0.2)
    canary = opt.release.advance(
        opt.challenge(candidate, evaluator=evaluator, security=SecurityState())
    )
    rejected = opt.release.advance(
        canary,
        [
            CanaryObservation(True, 0.1),
            CanaryObservation(False, -0.5),
            CanaryObservation(True, 0.1),
        ],
    )
    assert not rejected.allowed
    assert rejected.stage is PromotionStage.REJECTED
    assert "rollback" in rejected.reason


def test_parent_must_be_active_champion():
    opt = make_optimizer()
    opt.bootstrap("champion", evaluator)
    decision = opt.challenge(
        OptimizationCandidate("challenger", "stale-parent", "faster", 0.2),
        evaluator=evaluator,
        security=SecurityState(),
    )
    assert not decision.allowed
    assert "active champion" in decision.reason


def test_benchmark_rejects_non_finite_metrics():
    harness = BenchmarkHarness(SPECS, repetitions=2)

    def broken(subject, repetition):
        return {"throughput": float("nan"), "latency": 1.0, "safety": 1.0}

    with pytest.raises(ValueError, match="finite"):
        harness.evaluate("x", broken)
