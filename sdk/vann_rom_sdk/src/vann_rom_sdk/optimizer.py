from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class RuntimePolicy:
    learning_rate: float = 0.01
    sparsity_threshold: float = 0.0
    prefetch_depth: int = 2
    precision_bits: int = 32
    fusion_level: int = 1


@dataclass(slots=True)
class RuntimeMetrics:
    cycles: int = 0
    instructions: int = 0
    rom_fetches: int = 0
    cache_hits: int = 0
    commits: int = 0
    rollbacks: int = 0
    reconstruction_error: float = 0.0
    loss: float = 0.0
    latent_sparsity: float = 0.0
    elapsed_seconds: float = 0.0

    @property
    def cache_hit_rate(self) -> float:
        total = self.rom_fetches + self.cache_hits
        return self.cache_hits / total if total else 0.0

    @property
    def instructions_per_second(self) -> float:
        return self.instructions / self.elapsed_seconds if self.elapsed_seconds > 0 else 0.0

    def as_dict(self) -> dict[str, float | int]:
        result = asdict(self)
        result["cache_hit_rate"] = self.cache_hit_rate
        result["instructions_per_second"] = self.instructions_per_second
        return result


class AutoOptimizer:
    """Bounded policy controller with reversible, conservative updates."""

    def __init__(self, policy: RuntimePolicy | None = None) -> None:
        self.policy = policy or RuntimePolicy()
        self.history: list[RuntimePolicy] = []

    def propose(self, metrics: RuntimeMetrics) -> RuntimePolicy:
        current = self.policy
        candidate = RuntimePolicy(**asdict(current))

        if metrics.reconstruction_error > 0.18:
            candidate.learning_rate = min(0.05, current.learning_rate * 1.10)
            candidate.precision_bits = min(32, max(16, current.precision_bits))
            candidate.sparsity_threshold = max(0.0, current.sparsity_threshold - 0.01)
        elif metrics.reconstruction_error < 0.06:
            candidate.learning_rate = max(0.0005, current.learning_rate * 0.98)
            candidate.sparsity_threshold = min(0.20, current.sparsity_threshold + 0.005)

        if metrics.cache_hit_rate < 0.75:
            candidate.prefetch_depth = min(8, current.prefetch_depth + 1)
        elif metrics.cache_hit_rate > 0.95:
            candidate.prefetch_depth = max(1, current.prefetch_depth - 1)

        if metrics.instructions > 20:
            candidate.fusion_level = min(4, current.fusion_level + 1)

        return candidate

    def commit(self, candidate: RuntimePolicy) -> None:
        self.history.append(self.policy)
        self.policy = candidate

    def rollback(self) -> None:
        if self.history:
            self.policy = self.history.pop()
