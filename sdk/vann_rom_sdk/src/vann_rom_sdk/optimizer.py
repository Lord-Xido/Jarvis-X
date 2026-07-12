from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class RuntimePolicy:
    learning_rate: float = 0.01
    sparsity_threshold: float = 0.0
    prefetch_depth: int = 2
    precision_bits: int = 32
    fusion_level: int = 1

    def validate(self) -> None:
        if not 0.0001 <= self.learning_rate <= 0.05:
            raise ValueError("learning_rate outside safe range")
        if not 0.0 <= self.sparsity_threshold <= 0.25:
            raise ValueError("sparsity_threshold outside safe range")
        if not 0 <= self.prefetch_depth <= 16:
            raise ValueError("prefetch_depth outside safe range")
        if self.precision_bits not in {8, 16, 32, 64}:
            raise ValueError("unsupported precision_bits")
        if not 1 <= self.fusion_level <= 8:
            raise ValueError("fusion_level outside safe range")


@dataclass(slots=True)
class RuntimeMetrics:
    cycles: int = 0
    instructions: int = 0
    rom_fetches: int = 0
    cache_hits: int = 0
    demand_accesses: int = 0
    demand_hits: int = 0
    demand_misses: int = 0
    prefetch_requests: int = 0
    useful_prefetches: int = 0
    commits: int = 0
    rollbacks: int = 0
    reconstruction_error: float = 0.0
    loss: float = 0.0
    latent_sparsity: float = 0.0
    elapsed_seconds: float = 0.0

    @property
    def cache_hit_rate(self) -> float:
        return self.demand_hits / self.demand_accesses if self.demand_accesses else 0.0

    @property
    def prefetch_accuracy(self) -> float:
        if not self.prefetch_requests:
            return 0.0
        return self.useful_prefetches / self.prefetch_requests

    @property
    def instructions_per_second(self) -> float:
        return self.instructions / self.elapsed_seconds if self.elapsed_seconds > 0 else 0.0

    def as_dict(self) -> dict[str, float | int]:
        result = asdict(self)
        result["cache_hit_rate"] = self.cache_hit_rate
        result["prefetch_accuracy"] = self.prefetch_accuracy
        result["instructions_per_second"] = self.instructions_per_second
        return result


class AutoOptimizer:
    """Bounded policy controller with reversible, conservative updates."""

    def __init__(self, policy: RuntimePolicy | None = None) -> None:
        self.policy = policy or RuntimePolicy()
        self.policy.validate()
        self.history: list[RuntimePolicy] = []
        self.last_decision: dict[str, object] = {
            "accepted": False,
            "reason": "not evaluated",
        }

    def propose(self, metrics: RuntimeMetrics) -> RuntimePolicy:
        current = self.policy
        candidate = RuntimePolicy(**asdict(current))

        if metrics.reconstruction_error > 0.18:
            candidate.learning_rate = min(0.05, current.learning_rate * 1.10)
            candidate.precision_bits = max(16, current.precision_bits)
            candidate.sparsity_threshold = max(0.0, current.sparsity_threshold - 0.01)
        elif metrics.reconstruction_error < 0.06:
            candidate.learning_rate = max(0.0005, current.learning_rate * 0.98)
            candidate.sparsity_threshold = min(0.20, current.sparsity_threshold + 0.005)

        if metrics.cache_hit_rate < 0.50 and metrics.demand_accesses >= 2:
            candidate.prefetch_depth = min(8, current.prefetch_depth + 1)
        elif metrics.cache_hit_rate > 0.90 and metrics.prefetch_accuracy < 0.50:
            candidate.prefetch_depth = max(0, current.prefetch_depth - 1)

        if metrics.instructions > 20 and metrics.rollbacks == 0:
            candidate.fusion_level = min(4, current.fusion_level + 1)

        candidate.validate()
        return candidate

    @staticmethod
    def objective(metrics: RuntimeMetrics, policy: RuntimePolicy) -> float:
        """Conservative predicted cost used for policy shadow comparison."""

        error_cost = 100.0 * metrics.reconstruction_error
        rollback_cost = 25.0 * metrics.rollbacks
        miss_cost = 4.0 * (1.0 - metrics.cache_hit_rate)
        prefetch_waste = 2.0 * (1.0 - metrics.prefetch_accuracy)
        precision_cost = policy.precision_bits / 64.0
        sparsity_risk = 20.0 * policy.sparsity_threshold * metrics.reconstruction_error
        fusion_credit = min(policy.fusion_level, 4) * 0.05
        return (
            error_cost
            + rollback_cost
            + miss_cost
            + prefetch_waste
            + precision_cost
            + sparsity_risk
            - fusion_credit
        )

    def consider(self, candidate: RuntimePolicy, metrics: RuntimeMetrics) -> bool:
        candidate.validate()
        if candidate == self.policy:
            self.last_decision = {
                "accepted": False,
                "reason": "candidate equals active policy",
                "baseline_score": self.objective(metrics, self.policy),
                "candidate_score": self.objective(metrics, candidate),
            }
            return False

        baseline_score = self.objective(metrics, self.policy)
        candidate_score = self.objective(metrics, candidate)
        accepted = candidate_score <= baseline_score + 1e-12
        self.last_decision = {
            "accepted": accepted,
            "reason": "predicted constrained cost improved" if accepted else "no predicted improvement",
            "baseline_score": baseline_score,
            "candidate_score": candidate_score,
            "candidate": asdict(candidate),
        }
        if accepted:
            self.commit(candidate)
        return accepted

    def commit(self, candidate: RuntimePolicy) -> None:
        candidate.validate()
        self.history.append(RuntimePolicy(**asdict(self.policy)))
        self.policy = RuntimePolicy(**asdict(candidate))

    def rollback(self) -> None:
        if self.history:
            self.policy = self.history.pop()
