"""Evidence-based evaluation for Jarvis-X architecture claims.

A category score is a weighted geometric mean in [0, 10].  The system score is
weakest-link constrained: the minimum category score.  This prevents strong
performance in one area from masking a critical gap in safety, correctness,
reproducibility, or scientific validation.
"""

from dataclasses import dataclass
from math import exp, log
from typing import Dict, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class Metric:
    """Normalized evidence metric.

    ``value`` is constrained to [0, 1], where 1 means the declared acceptance
    criterion has been fully satisfied. ``weight`` must be positive.
    """

    name: str
    value: float
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("metric name must not be empty")
        if not 0.0 <= self.value <= 1.0:
            raise ValueError("metric value must lie in [0, 1]")
        if self.weight <= 0.0:
            raise ValueError("metric weight must be positive")


@dataclass(frozen=True)
class CategoryResult:
    name: str
    score: float
    metrics: Sequence[Metric]

    @property
    def qualifies_for_ten(self) -> bool:
        return self.score == 10.0 and all(metric.value == 1.0 for metric in self.metrics)


def score_category(name: str, metrics: Iterable[Metric]) -> CategoryResult:
    """Return 10 times the weighted geometric mean of normalized metrics."""

    collected = tuple(metrics)
    if not name:
        raise ValueError("category name must not be empty")
    if not collected:
        raise ValueError("a category requires at least one metric")
    if any(metric.value == 0.0 for metric in collected):
        return CategoryResult(name=name, score=0.0, metrics=collected)
    total_weight = sum(metric.weight for metric in collected)
    weighted_log = sum(metric.weight * log(metric.value) for metric in collected)
    score = 10.0 * exp(weighted_log / total_weight)
    return CategoryResult(name=name, score=score, metrics=collected)


def score_system(categories: Iterable[CategoryResult]) -> float:
    """Return the weakest category score.

    Jarvis-X cannot claim a global 10/10 while any critical category remains
    below 10.  This intentionally rejects arithmetic averaging.
    """

    collected = tuple(categories)
    if not collected:
        raise ValueError("system evaluation requires at least one category")
    return min(category.score for category in collected)


def qualifies_for_ten(categories: Iterable[CategoryResult]) -> bool:
    collected = tuple(categories)
    return bool(collected) and score_system(collected) == 10.0 and all(
        category.qualifies_for_ten for category in collected
    )


def summarize(categories: Iterable[CategoryResult]) -> Mapping[str, object]:
    collected = tuple(categories)
    return {
        "system_score": score_system(collected),
        "qualifies_for_ten": qualifies_for_ten(collected),
        "categories": {
            category.name: {
                "score": category.score,
                "qualifies_for_ten": category.qualifies_for_ten,
                "metrics": {
                    metric.name: {"value": metric.value, "weight": metric.weight}
                    for metric in category.metrics
                },
            }
            for category in collected
        },
    }


def canonical_gate_names() -> Dict[str, Sequence[str]]:
    """Return the minimum evidence families required by Runtime v1."""

    return {
        "conceptual_coherence": (
            "canonical_state_defined",
            "single_transition_algebra",
            "symbol_conflicts_zero",
        ),
        "mathematical_completeness": (
            "operators_typed",
            "stability_bound_defined",
            "resource_bound_defined",
            "convergence_metrics_distinguished",
        ),
        "implementation_maturity": (
            "versioned_api",
            "continuous_integration",
            "recovery_path_tested",
            "supported_backend_implemented",
        ),
        "testability_auditability": (
            "critical_tests_pass",
            "deterministic_replay_match",
            "journal_chain_verified",
            "fault_injection_pass",
        ),
        "scientific_validation": (
            "external_benchmark_run",
            "baseline_comparison",
            "ablation_completed",
            "confidence_intervals_reported",
        ),
        "security": (
            "threat_model_complete",
            "policy_projection_enforced",
            "critical_findings_zero",
            "release_provenance_signed",
        ),
        "observability": (
            "traces_complete",
            "metrics_complete",
            "failure_logs_replayable",
        ),
    }
