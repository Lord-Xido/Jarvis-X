"""Deterministic 3D inward-folded document-swarm reference model.

The module models a logical 4,843^3 execution lattice and a bounded local
3D self-optimisation loop. ``virtual_ns`` is model time, not measured hardware
latency, and ``logical_capacity`` is not a physical transformer throughput claim.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import random
from typing import Iterator

DEFAULT_SIDE = 4_843
DEFAULT_CAPACITY = DEFAULT_SIDE**3


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class SwarmState:
    section_words: int = 1_000
    tokens_per_word: float = 1.30
    parallel_fraction: float = 0.92
    coherence_weight: float = 0.35
    verification_weight: float = 0.25
    novelty_weight: float = 0.10
    compression_weight: float = 0.15
    stability_weight: float = 0.15
    mutation_scale: float = 0.12
    memory_decay: float = 0.90
    refinement_passes: int = 3


@dataclass(frozen=True)
class SwarmMetrics:
    virtual_ns: float
    logical_utilisation: float
    coherence: float
    verification: float
    compression: float
    stability: float
    novelty: float
    memory_cost: float
    fitness: float


@dataclass(frozen=True)
class OptimisationStep:
    generation: int
    direction: tuple[int, int, int]
    state: SwarmState
    metrics: SwarmMetrics


@dataclass(frozen=True)
class OptimisationResult:
    initial_state: SwarmState
    initial_metrics: SwarmMetrics
    final_state: SwarmState
    final_metrics: SwarmMetrics
    history: tuple[OptimisationStep, ...]


class Inward3DDocumentSwarm:
    """Bounded 3D virtual scheduler plus recursive state optimiser."""

    def __init__(
        self,
        *,
        side: int = DEFAULT_SIDE,
        document_words: int = 1_000_000,
        seed: int = 7,
    ) -> None:
        if side <= 0:
            raise ValueError("side must be positive")
        if document_words <= 0:
            raise ValueError("document_words must be positive")
        self.side = side
        self.logical_capacity = side**3
        self.document_words = document_words
        self._rng = random.Random(seed)

    def document_tokens(self, state: SwarmState) -> int:
        return math.ceil(self.document_words * state.tokens_per_word)

    def section_count(self, state: SwarmState) -> int:
        return math.ceil(self.document_words / state.section_words)

    def linear_to_xyz(self, index: int) -> tuple[int, int, int]:
        if not 0 <= index < self.logical_capacity:
            raise ValueError("index outside logical cube")
        x = index % self.side
        y = (index // self.side) % self.side
        z = index // (self.side * self.side)
        return x, y, z

    def xyz_to_linear(self, x: int, y: int, z: int) -> int:
        if not all(0 <= axis < self.side for axis in (x, y, z)):
            raise ValueError("coordinate outside logical cube")
        return x + self.side * (y + self.side * z)

    @staticmethod
    def encode_state(state: SwarmState) -> tuple[float, ...]:
        return (
            math.log10(state.section_words) / 5.0,
            state.tokens_per_word / 2.0,
            state.parallel_fraction,
            state.coherence_weight,
            state.verification_weight,
            state.novelty_weight,
            state.compression_weight,
            state.stability_weight,
            state.mutation_scale,
            state.memory_decay,
            state.refinement_passes / 10.0,
        )

    @staticmethod
    def decode_state(latent: tuple[float, ...]) -> SwarmState:
        if len(latent) != 11:
            raise ValueError("latent state must have 11 components")
        return SwarmState(
            section_words=int(
                _clamp(round(10 ** (latent[0] * 5.0)), 128, 20_000)
            ),
            tokens_per_word=_clamp(latent[1] * 2.0, 1.05, 1.80),
            parallel_fraction=_clamp(latent[2], 0.10, 0.9999),
            coherence_weight=_clamp(latent[3], 0.05, 0.60),
            verification_weight=_clamp(latent[4], 0.05, 0.60),
            novelty_weight=_clamp(latent[5], 0.00, 0.30),
            compression_weight=_clamp(latent[6], 0.02, 0.40),
            stability_weight=_clamp(latent[7], 0.05, 0.60),
            mutation_scale=_clamp(latent[8], 0.005, 0.30),
            memory_decay=_clamp(latent[9], 0.50, 0.999),
            refinement_passes=int(_clamp(round(latent[10] * 10.0), 1, 8)),
        )

    @staticmethod
    def candidate_directions(radius: int = 2) -> Iterator[tuple[int, int, int]]:
        if radius <= 0:
            raise ValueError("radius must be positive")
        for x in range(-radius, radius + 1):
            for y in range(-radius, radius + 1):
                for z in range(-radius, radius + 1):
                    yield x, y, z

    def fold_candidate(
        self,
        state: SwarmState,
        direction: tuple[int, int, int],
        *,
        radius: int = 2,
    ) -> SwarmState:
        if radius <= 0:
            raise ValueError("radius must be positive")
        x, y, z = direction
        if any(abs(axis) > radius for axis in direction):
            raise ValueError("direction lies outside search radius")

        latent = list(self.encode_state(state))
        scale = state.mutation_scale
        dx = x * scale / radius
        dy = y * scale / radius
        dz = z * scale / radius

        # X: partitioning / throughput
        latent[0] += 0.35 * dx
        latent[1] -= 0.10 * dx
        latent[2] += 0.55 * dx

        # Y: coherence / verification
        latent[3] += 0.35 * dy
        latent[4] += 0.35 * dy
        latent[5] -= 0.10 * abs(dy)

        # Z: compression / stability / recursion
        latent[6] += 0.25 * dz
        latent[7] += 0.40 * dz
        latent[8] -= 0.20 * dz
        latent[9] += 0.20 * dz
        latent[10] += 0.15 * dz

        # Seeded isotropic exploration keeps runs deterministic.
        jitter = 0.015 * scale
        latent = [v + self._rng.uniform(-jitter, jitter) for v in latent]
        return self.decode_state(tuple(latent))

    def evaluate(self, state: SwarmState) -> SwarmMetrics:
        tokens = self.document_tokens(state)
        sections = self.section_count(state)

        planning_ops = sections + 1
        generation_ops = tokens
        verification_ops = tokens * state.refinement_passes
        effective_ops = (
            planning_ops + generation_ops + verification_ops
        ) / max(state.parallel_fraction, 1e-12)

        ticks = max(1, math.ceil(effective_ops / self.logical_capacity))
        virtual_ns = ticks + 0.10 * state.refinement_passes
        utilisation = min(
            1.0, effective_ops / (ticks * self.logical_capacity)
        )

        coherence = _clamp(
            0.55
            + 0.23
            * math.tanh(state.coherence_weight * state.refinement_passes)
            + 0.10 * state.memory_decay
            - 1.60 * max(0.0, state.parallel_fraction - 0.97),
            0.0,
            1.0,
        )
        verification = _clamp(
            0.50
            + 0.30
            * math.tanh(
                state.verification_weight * (1 + state.refinement_passes)
            )
            + 0.08 * state.stability_weight,
            0.0,
            1.0,
        )
        compression = _clamp(
            0.45
            + 0.28 * state.compression_weight
            + 0.12 * math.log10(max(128, state.section_words)) / 5.0,
            0.0,
            1.0,
        )
        stability = _clamp(
            0.52
            + 0.28 * state.stability_weight
            + 0.15 * state.memory_decay
            - 0.25 * state.mutation_scale,
            0.0,
            1.0,
        )
        novelty = _clamp(
            0.35 + 0.90 * state.novelty_weight + 0.25 * state.mutation_scale,
            0.0,
            1.0,
        )
        memory_cost = _clamp(
            (state.section_words / 20_000) * 0.45
            + (state.refinement_passes / 8) * 0.35
            + state.memory_decay * 0.20,
            0.0,
            1.0,
        )

        speed_score = 1.0 / (1.0 + virtual_ns)
        fitness = (
            0.24 * coherence
            + 0.23 * verification
            + 0.17 * stability
            + 0.11 * compression
            + 0.07 * novelty
            + 0.10 * speed_score
            + 0.04 * utilisation
            - 0.04 * memory_cost
        )
        return SwarmMetrics(
            virtual_ns=virtual_ns,
            logical_utilisation=utilisation,
            coherence=coherence,
            verification=verification,
            compression=compression,
            stability=stability,
            novelty=novelty,
            memory_cost=memory_cost,
            fitness=fitness,
        )

    def optimise(
        self,
        state: SwarmState | None = None,
        *,
        generations: int = 24,
        radius: int = 2,
    ) -> OptimisationResult:
        if generations <= 0:
            raise ValueError("generations must be positive")
        current = state or SwarmState()
        initial = current
        initial_metrics = self.evaluate(current)
        history: list[OptimisationStep] = []

        for generation in range(1, generations + 1):
            incumbent_metrics = self.evaluate(current)
            best_direction = (0, 0, 0)
            best_state = current
            best_metrics = incumbent_metrics

            for direction in self.candidate_directions(radius):
                candidate = self.fold_candidate(
                    current, direction, radius=radius
                )
                metrics = self.evaluate(candidate)
                if metrics.fitness > best_metrics.fitness:
                    best_direction = direction
                    best_state = candidate
                    best_metrics = metrics

            # Anneal mutation when no local candidate improves the incumbent.
            if best_state == current:
                current = replace(
                    current,
                    mutation_scale=max(0.005, current.mutation_scale * 0.92),
                )
                best_metrics = self.evaluate(current)
            else:
                current = best_state

            history.append(
                OptimisationStep(
                    generation=generation,
                    direction=best_direction,
                    state=current,
                    metrics=best_metrics,
                )
            )

        return OptimisationResult(
            initial_state=initial,
            initial_metrics=initial_metrics,
            final_state=current,
            final_metrics=self.evaluate(current),
            history=tuple(history),
        )

    def schedule_summary(self, state: SwarmState) -> dict[str, float | int]:
        tokens = self.document_tokens(state)
        sections = self.section_count(state)
        raw_ops = sections + 1 + tokens * (1 + state.refinement_passes)
        effective_ops = raw_ops / state.parallel_fraction
        ticks = max(1, math.ceil(effective_ops / self.logical_capacity))
        return {
            "side": self.side,
            "logical_capacity": self.logical_capacity,
            "document_words": self.document_words,
            "document_tokens": tokens,
            "sections": sections,
            "raw_ops": raw_ops,
            "effective_ops": effective_ops,
            "logical_ticks": ticks,
            "virtual_ns": ticks + 0.10 * state.refinement_passes,
        }
