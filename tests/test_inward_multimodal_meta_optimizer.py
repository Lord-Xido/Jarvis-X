import pytest

from jarvisx.inward_multimodal_meta_optimizer import (
    InwardMultimodalMetaOptimizer,
    MetaFitness,
    MetaSearchConfig,
)
from jarvisx.inward_multimodal_swarm3d import Swarm3DConfig


def fitness(
    task_score: float,
    *,
    fixed_point_error: float = 0.10,
    resource_cost: float = 0.20,
    stable: bool = True,
) -> MetaFitness:
    return MetaFitness(
        task_score=task_score,
        semantic_coherence=0.80,
        feature_coherence=0.80,
        fixed_point_error=fixed_point_error,
        resource_cost=resource_cost,
        stable=stable,
    )


def test_candidate_generation_is_deterministic_for_generation_and_seed():
    optimizer = InwardMultimodalMetaOptimizer(
        MetaSearchConfig(generations=2, branch_width=6, seed=17)
    )
    config = Swarm3DConfig()

    left = optimizer.candidate_configs(config, generation=3)
    right = optimizer.candidate_configs(config, generation=3)
    other = optimizer.candidate_configs(config, generation=4)

    assert left == right
    assert left != other
    assert len(left) == 6


def test_optimizer_promotes_only_verified_fitness_improvement():
    optimizer = InwardMultimodalMetaOptimizer(
        MetaSearchConfig(
            generations=4,
            branch_width=20,
            seed=9,
            mutation_fraction=0.50,
            improvement_threshold=0.0,
        )
    )
    baseline = Swarm3DConfig(task_gain=1.0)

    def evaluator(config: Swarm3DConfig) -> MetaFitness:
        task_score = min(1.0, 0.35 + 0.25 * config.task_gain)
        return fitness(task_score)

    report = optimizer.optimize(baseline, evaluator)

    assert report.evaluated_candidates == 80
    assert report.final_fitness.score >= report.baseline_fitness.score
    assert report.promoted
    assert report.final_config.task_gain > baseline.task_gain


def test_fixed_point_integrity_gate_blocks_better_scalar_score():
    optimizer = InwardMultimodalMetaOptimizer(
        MetaSearchConfig(
            generations=3,
            branch_width=24,
            seed=11,
            mutation_fraction=0.60,
            improvement_threshold=0.0,
            max_fixed_point_regression=0.0,
        )
    )
    baseline = Swarm3DConfig(task_gain=1.0)

    def evaluator(config: Swarm3DConfig) -> MetaFitness:
        if config.task_gain > baseline.task_gain:
            return fitness(
                min(1.0, 0.85 + 0.05 * config.task_gain),
                fixed_point_error=0.20,
            )
        return fitness(
            max(0.0, 0.80 - 0.10 * (baseline.task_gain - config.task_gain)),
            fixed_point_error=0.10,
        )

    report = optimizer.optimize(baseline, evaluator)

    assert not report.promoted
    assert report.final_config == baseline
    assert report.final_fitness == report.baseline_fitness
    assert any(
        "fixed-point" in evaluation.reason
        for evaluation in report.evaluations
        if evaluation.config.task_gain > baseline.task_gain
    )


def test_unstable_incumbent_cannot_enter_self_optimization():
    optimizer = InwardMultimodalMetaOptimizer(
        MetaSearchConfig(generations=1, branch_width=2)
    )

    with pytest.raises(ValueError, match="incumbent runtime must be stable"):
        optimizer.optimize(
            Swarm3DConfig(),
            lambda _: fitness(0.8, stable=False),
        )


def test_meta_fitness_rejects_invalid_normalized_quality():
    with pytest.raises(ValueError, match="task_score"):
        MetaFitness(
            task_score=1.1,
            semantic_coherence=0.8,
            feature_coherence=0.8,
            fixed_point_error=0.1,
            resource_cost=0.2,
            stable=True,
        )
