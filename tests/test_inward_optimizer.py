import math

from jarvisx.inward_optimizer import (
    InwardOptimizer,
    MechanicsState,
    ObjectiveWeights,
    SafetyPolicy,
    Telemetry,
)


def synthetic_evaluator(state: MechanicsState) -> Telemetry:
    target_learning_rate = 0.0015
    target_layers = 3
    target_momentum = 0.85
    task_loss = (
        (state.hyper.learning_rate - target_learning_rate) ** 2 * 1_000_000.0
        + (state.architecture.layers - target_layers) ** 2 * 0.05
        + (state.rule.momentum - target_momentum) ** 2
        + 0.1
    )
    latency_ms = 5.0 + state.architecture.layers * 1.5 + state.architecture.experts * 0.4
    memory_mb = 128.0 + state.architecture.latent_dim * state.architecture.experts * 0.5
    return Telemetry(
        task_loss=task_loss,
        latency_ms=latency_ms,
        memory_mb=memory_mb,
        semantic_distance=0.0,
        gradient_norm=1.0,
    )


def test_optimizer_commits_best_safe_candidate():
    optimizer = InwardOptimizer()

    result = optimizer.optimize_once(synthetic_evaluator)

    assert result.committed is True
    assert result.transformation == "learning_rate_x1.5"
    assert result.active_state.version == 1
    assert result.active_state.hyper.learning_rate == 0.0015
    assert any(entry.committed for entry in optimizer.journal)


def test_rollback_restores_previous_immutable_state():
    optimizer = InwardOptimizer()
    original = optimizer.active_state
    optimizer.optimize_once(synthetic_evaluator)

    assert optimizer.active_state != original
    assert optimizer.rollback() is True
    assert optimizer.active_state == original
    assert optimizer.rollback() is False


def test_non_finite_shadow_candidate_is_rejected():
    optimizer = InwardOptimizer()
    active = optimizer.active_state

    def evaluator(state: MechanicsState) -> Telemetry:
        if state == active:
            return Telemetry(task_loss=1.0, latency_ms=10.0, memory_mb=100.0)
        return Telemetry(task_loss=math.nan, latency_ms=1.0, memory_mb=1.0)

    result = optimizer.optimize_once(evaluator)

    assert result.committed is False
    assert optimizer.active_state == active
    assert all(not evaluation.gate.accepted for evaluation in result.evaluations)


def test_semantic_distance_blocks_apparent_performance_gain():
    optimizer = InwardOptimizer(policy=SafetyPolicy(max_semantic_distance=1e-6))
    active = optimizer.active_state

    def evaluator(state: MechanicsState) -> Telemetry:
        if state == active:
            return Telemetry(task_loss=1.0, latency_ms=10.0, memory_mb=100.0)
        return Telemetry(
            task_loss=0.01,
            latency_ms=1.0,
            memory_mb=10.0,
            semantic_distance=0.1,
        )

    result = optimizer.optimize_once(evaluator)

    assert result.committed is False
    assert any(
        "semantic distance exceeds policy" in evaluation.gate.reasons
        for evaluation in result.evaluations
    )


def test_task_integrity_gate_prevents_metric_hacking():
    weights = ObjectiveWeights(task_loss=0.01, latency_ms=1.0, memory_mb=0.0, rollback_rate=0.0)
    optimizer = InwardOptimizer(weights=weights)
    active = optimizer.active_state

    def evaluator(state: MechanicsState) -> Telemetry:
        if state == active:
            return Telemetry(task_loss=1.0, latency_ms=100.0, memory_mb=100.0)
        return Telemetry(task_loss=1.2, latency_ms=1.0, memory_mb=100.0)

    result = optimizer.optimize_once(evaluator)

    assert result.committed is False
    assert any(
        "task-loss regression exceeds policy" in evaluation.gate.reasons
        for evaluation in result.evaluations
    )
