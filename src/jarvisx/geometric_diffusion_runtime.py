"""Bounded 3D geometric diffusion and kinetic adaptation reference runtime.

This module is deliberately small and dependency-free.  It provides a deterministic
research fixture for geometry-conditioned diffusion, candidate-first publication,
working-memory residuals, and bounded runtime-mutation promotion.  It does not claim
that a language model literally stores state in physical 3D space, nor that diffusion
can reproduce an arbitrary target image exactly from an underspecified prompt.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

Vec3 = tuple[float, float, float]
Feature = tuple[float, ...]
Edge = tuple[int, int]


def _finite_sequence(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    result = tuple(float(v) for v in values)
    if not result:
        raise ValueError(f"{name} must not be empty")
    if not all(math.isfinite(v) for v in result):
        raise ValueError(f"{name} must contain only finite values")
    return result


@dataclass(frozen=True)
class GeometricNode:
    """One virtual 3D node with an associated semantic/latent feature vector."""

    position: Vec3
    feature: Feature

    def __post_init__(self) -> None:
        position = _finite_sequence(self.position, name="position")
        if len(position) != 3:
            raise ValueError("position must contain exactly three coordinates")
        feature = _finite_sequence(self.feature, name="feature")
        object.__setattr__(self, "position", (position[0], position[1], position[2]))
        object.__setattr__(self, "feature", feature)


@dataclass(frozen=True)
class GeometricGraph:
    """Finite undirected graph used as the relational geometry boundary."""

    nodes: tuple[GeometricNode, ...]
    edges: tuple[Edge, ...]

    def __post_init__(self) -> None:
        if not self.nodes:
            raise ValueError("graph must contain at least one node")
        feature_width = len(self.nodes[0].feature)
        if any(len(node.feature) != feature_width for node in self.nodes):
            raise ValueError("all graph nodes must use the same feature width")

        count = len(self.nodes)
        normalized: set[Edge] = set()
        for edge in self.edges:
            if len(edge) != 2:
                raise ValueError("edges must contain exactly two node indices")
            a, b = edge
            if isinstance(a, bool) or isinstance(b, bool):
                raise TypeError("edge indices must be integers")
            if not isinstance(a, int) or not isinstance(b, int):
                raise TypeError("edge indices must be integers")
            if a == b:
                raise ValueError("self edges are not admissible")
            if a < 0 or b < 0 or a >= count or b >= count:
                raise ValueError("edge index is outside the node buffer")
            normalized.add((a, b) if a < b else (b, a))
        object.__setattr__(self, "edges", tuple(sorted(normalized)))

    @property
    def feature_width(self) -> int:
        return len(self.nodes[0].feature)

    def adjacency(self) -> tuple[tuple[int, ...], ...]:
        neighbors: list[list[int]] = [[] for _ in self.nodes]
        for a, b in self.edges:
            neighbors[a].append(b)
            neighbors[b].append(a)
        return tuple(tuple(sorted(items)) for items in neighbors)


@dataclass(frozen=True)
class GeometricDiffusionConfig:
    """Resource and numerical bounds for one geometric diffusion cycle."""

    beta: float = 0.15
    denoise_steps: int = 4
    geometry_gain: float = 0.55
    graph_gain: float = 0.15
    memory_retention: float = 0.75
    memory_gain: float = 0.10
    max_position_step: float = 0.75
    max_feature_step: float = 2.0
    max_cycle_rms: float = 0.35
    max_nodes: int = 100_000
    max_edges: int = 400_000
    branch_width: int = 4
    verification_threshold: float = 0.90

    def __post_init__(self) -> None:
        unit_fields = (
            "beta",
            "geometry_gain",
            "graph_gain",
            "memory_retention",
            "verification_threshold",
        )
        for name in unit_fields:
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and within [0, 1]")
        for name in ("memory_gain", "max_position_step", "max_feature_step", "max_cycle_rms"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in ("denoise_steps", "max_nodes", "max_edges", "branch_width"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class DiffusionMetrics:
    cycle: int
    node_count: int
    edge_count: int
    position_rms: float
    feature_rms: float
    combined_rms: float
    edge_energy: float
    verification_score: float


@dataclass(frozen=True)
class DiffusionState:
    cycle: int
    graph: GeometricGraph
    memory: tuple[Feature, ...]
    metrics: DiffusionMetrics


StateValidator = Callable[[DiffusionState], bool]


def _rms(values: Iterable[float]) -> float:
    data = tuple(float(v) for v in values)
    if not data:
        return 0.0
    return math.sqrt(sum(v * v for v in data) / len(data))


def graph_rms(reference: GeometricGraph, candidate: GeometricGraph) -> tuple[float, float, float]:
    """Return position, feature and combined RMS for topology-compatible graphs."""

    _require_same_topology(reference, candidate)
    position_delta: list[float] = []
    feature_delta: list[float] = []
    for left, right in zip(reference.nodes, candidate.nodes):
        position_delta.extend(a - b for a, b in zip(left.position, right.position))
        feature_delta.extend(a - b for a, b in zip(left.feature, right.feature))
    pos = _rms(position_delta)
    feat = _rms(feature_delta)
    combined = _rms((*position_delta, *feature_delta))
    return pos, feat, combined


def edge_energy(graph: GeometricGraph) -> float:
    """Mean squared feature difference across graph edges.

    This is a deterministic smoothness/dispersion metric, not Shannon entropy.
    """

    if not graph.edges:
        return 0.0
    total = 0.0
    samples = 0
    for a, b in graph.edges:
        left = graph.nodes[a].feature
        right = graph.nodes[b].feature
        for x, y in zip(left, right):
            total += (x - y) ** 2
            samples += 1
    return total / samples if samples else 0.0


def forward_diffuse(graph: GeometricGraph, *, beta: float, seed: int) -> GeometricGraph:
    """Apply deterministic seeded Gaussian corruption to position and feature state."""

    beta = float(beta)
    if not math.isfinite(beta) or not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be finite and within [0, 1]")
    keep = math.sqrt(1.0 - beta)
    noise_scale = math.sqrt(beta)
    rng = random.Random(seed)
    nodes: list[GeometricNode] = []
    for node in graph.nodes:
        position = tuple(keep * x + noise_scale * rng.gauss(0.0, 1.0) for x in node.position)
        feature = tuple(keep * x + noise_scale * rng.gauss(0.0, 1.0) for x in node.feature)
        nodes.append(GeometricNode(position=position, feature=feature))
    return GeometricGraph(nodes=tuple(nodes), edges=graph.edges)


def reverse_denoise_step(
    current: GeometricGraph,
    anchor: GeometricGraph,
    *,
    geometry_gain: float,
    graph_gain: float,
    memory: Sequence[Feature] | None = None,
    memory_gain: float = 0.0,
) -> GeometricGraph:
    """One geometry-conditioned reverse step.

    The step contracts state toward the immutable anchor and applies a graph-Laplacian
    feature smoother.  Optional memory is additive but remains bounded by the caller's
    projection gate.
    """

    _require_same_topology(current, anchor)
    for name, value in (("geometry_gain", geometry_gain), ("graph_gain", graph_gain)):
        value = float(value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be finite and within [0, 1]")
    if not math.isfinite(float(memory_gain)) or memory_gain < 0.0:
        raise ValueError("memory_gain must be finite and non-negative")

    if memory is None:
        memory = tuple((0.0,) * current.feature_width for _ in current.nodes)
    if len(memory) != len(current.nodes):
        raise ValueError("memory must provide one feature vector per node")
    if any(len(item) != current.feature_width for item in memory):
        raise ValueError("memory feature width must match the graph")
    if any(not all(math.isfinite(float(x)) for x in item) for item in memory):
        raise ValueError("memory must contain only finite values")

    adjacency = current.adjacency()
    nodes: list[GeometricNode] = []
    for index, (node, target) in enumerate(zip(current.nodes, anchor.nodes)):
        position = tuple(
            x + geometry_gain * (a - x) for x, a in zip(node.position, target.position)
        )
        neighbors = adjacency[index]
        if neighbors:
            neighbor_mean = tuple(
                sum(current.nodes[j].feature[k] for j in neighbors) / len(neighbors)
                for k in range(current.feature_width)
            )
        else:
            neighbor_mean = node.feature
        feature = tuple(
            x
            + geometry_gain * (a - x)
            + graph_gain * (neighbor_mean[k] - x)
            + memory_gain * float(memory[index][k])
            for k, (x, a) in enumerate(zip(node.feature, target.feature))
        )
        nodes.append(GeometricNode(position=position, feature=feature))
    return GeometricGraph(nodes=tuple(nodes), edges=current.edges)


def project_candidate(
    candidate: GeometricGraph,
    anchor: GeometricGraph,
    *,
    max_position_step: float,
    max_feature_step: float,
) -> GeometricGraph:
    """Project candidate node deltas into explicit per-component limits."""

    _require_same_topology(candidate, anchor)
    if max_position_step < 0.0 or max_feature_step < 0.0:
        raise ValueError("projection bounds must be non-negative")
    if not math.isfinite(max_position_step) or not math.isfinite(max_feature_step):
        raise ValueError("projection bounds must be finite")

    nodes: list[GeometricNode] = []
    for node, base in zip(candidate.nodes, anchor.nodes):
        position = tuple(
            a + max(-max_position_step, min(max_position_step, x - a))
            for x, a in zip(node.position, base.position)
        )
        feature = tuple(
            a + max(-max_feature_step, min(max_feature_step, x - a))
            for x, a in zip(node.feature, base.feature)
        )
        nodes.append(GeometricNode(position=position, feature=feature))
    return GeometricGraph(nodes=tuple(nodes), edges=candidate.edges)


def _require_same_topology(left: GeometricGraph, right: GeometricGraph) -> None:
    if len(left.nodes) != len(right.nodes) or left.edges != right.edges:
        raise ValueError("graphs must share node count and edge topology")
    if left.feature_width != right.feature_width:
        raise ValueError("graphs must share feature width")


class GeometricDiffusionRuntime:
    """Candidate-first encode/diffuse/denoise/project/verify reference loop."""

    def __init__(self, anchor: GeometricGraph, config: GeometricDiffusionConfig | None = None):
        self.config = config or GeometricDiffusionConfig()
        self._check_budget(anchor)
        self._anchor = anchor
        zero_memory = tuple((0.0,) * anchor.feature_width for _ in anchor.nodes)
        metrics = DiffusionMetrics(
            cycle=0,
            node_count=len(anchor.nodes),
            edge_count=len(anchor.edges),
            position_rms=0.0,
            feature_rms=0.0,
            combined_rms=0.0,
            edge_energy=edge_energy(anchor),
            verification_score=1.0,
        )
        self._state = DiffusionState(cycle=0, graph=anchor, memory=zero_memory, metrics=metrics)

    @property
    def anchor(self) -> GeometricGraph:
        return self._anchor

    @property
    def state(self) -> DiffusionState:
        return self._state

    def step(
        self,
        observation: GeometricGraph,
        *,
        seed: int = 0,
        validator: StateValidator | None = None,
    ) -> DiffusionState:
        """Execute one full candidate cycle and publish only after all gates pass."""

        self._check_budget(observation)
        _require_same_topology(self._anchor, observation)

        # The observation is the immutable per-cycle reconstruction target.  Forward
        # corruption models exploratory diffusion; reverse steps are geometry-conditioned.
        candidate = forward_diffuse(observation, beta=self.config.beta, seed=seed)
        for _ in range(self.config.denoise_steps):
            candidate = reverse_denoise_step(
                candidate,
                observation,
                geometry_gain=self.config.geometry_gain,
                graph_gain=self.config.graph_gain,
                memory=self._state.memory,
                memory_gain=self.config.memory_gain,
            )
        candidate = project_candidate(
            candidate,
            observation,
            max_position_step=self.config.max_position_step,
            max_feature_step=self.config.max_feature_step,
        )

        pos_rms, feat_rms, combined_rms = graph_rms(observation, candidate)
        verification_score = 1.0 / (1.0 + combined_rms)
        next_cycle = self._state.cycle + 1
        next_memory = self._update_memory(observation, candidate)
        metrics = DiffusionMetrics(
            cycle=next_cycle,
            node_count=len(candidate.nodes),
            edge_count=len(candidate.edges),
            position_rms=pos_rms,
            feature_rms=feat_rms,
            combined_rms=combined_rms,
            edge_energy=edge_energy(candidate),
            verification_score=verification_score,
        )
        pending = DiffusionState(
            cycle=next_cycle,
            graph=candidate,
            memory=next_memory,
            metrics=metrics,
        )

        if combined_rms > self.config.max_cycle_rms:
            raise RuntimeError("candidate exceeds configured cycle RMS")
        if verification_score < self.config.verification_threshold:
            raise RuntimeError("candidate verification score is below threshold")
        if validator is not None and not bool(validator(pending)):
            raise RuntimeError("candidate rejected by geometric diffusion validator")

        # Atomic publication boundary.
        self._state = pending
        return pending

    def branch_candidates(
        self, observation: GeometricGraph, *, seed: int = 0
    ) -> tuple[GeometricGraph, ...]:
        """Generate a bounded deterministic family of exploratory diffusion candidates."""

        self._check_budget(observation)
        _require_same_topology(self._anchor, observation)
        return tuple(
            forward_diffuse(observation, beta=self.config.beta, seed=seed + branch)
            for branch in range(self.config.branch_width)
        )

    def _update_memory(
        self, observation: GeometricGraph, candidate: GeometricGraph
    ) -> tuple[Feature, ...]:
        retention = self.config.memory_retention
        memory: list[Feature] = []
        for previous, source, result in zip(self._state.memory, observation.nodes, candidate.nodes):
            residual = tuple(a - b for a, b in zip(source.feature, result.feature))
            memory.append(
                tuple(
                    retention * old + (1.0 - retention) * err
                    for old, err in zip(previous, residual)
                )
            )
        return tuple(memory)

    def _check_budget(self, graph: GeometricGraph) -> None:
        if len(graph.nodes) > self.config.max_nodes:
            raise RuntimeError("graph exceeds configured node budget")
        if len(graph.edges) > self.config.max_edges:
            raise RuntimeError("graph exceeds configured edge budget")


@dataclass(frozen=True)
class FitnessMetrics:
    """Normalized system-level metrics used by the bounded mutation gate."""

    quality: float
    reliability: float
    efficiency: float
    coherence: float
    fault_probability: float = 0.0

    def __post_init__(self) -> None:
        for name in ("quality", "reliability", "efficiency", "coherence", "fault_probability"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and within [0, 1]")

    @property
    def score(self) -> float:
        return (
            0.30 * self.quality
            + 0.30 * self.reliability
            + 0.20 * self.efficiency
            + 0.20 * self.coherence
            - 0.25 * self.fault_probability
        )


@dataclass(frozen=True)
class RuntimeMutation:
    """Versioned bounded configuration candidate; never executable source-code mutation."""

    identifier: str
    config: GeometricDiffusionConfig

    def __post_init__(self) -> None:
        if not self.identifier or not self.identifier.strip():
            raise ValueError("mutation identifier must not be empty")


@dataclass(frozen=True)
class PromotionReceipt:
    mutation_id: str
    current_fitness: float
    candidate_fitness: float
    verification_score: float
    promoted: bool
    reason: str


def evaluate_mutation(
    mutation: RuntimeMutation,
    *,
    current_metrics: FitnessMetrics,
    candidate_metrics: FitnessMetrics,
    verification_score: float,
) -> PromotionReceipt:
    """Apply the system-wide candidate-first promotion/rollback law."""

    verification_score = float(verification_score)
    if not math.isfinite(verification_score) or not 0.0 <= verification_score <= 1.0:
        raise ValueError("verification_score must be finite and within [0, 1]")

    current = current_metrics.score
    candidate = candidate_metrics.score
    threshold = mutation.config.verification_threshold
    if verification_score < threshold:
        return PromotionReceipt(
            mutation_id=mutation.identifier,
            current_fitness=current,
            candidate_fitness=candidate,
            verification_score=verification_score,
            promoted=False,
            reason="verification threshold not met",
        )
    if candidate <= current:
        return PromotionReceipt(
            mutation_id=mutation.identifier,
            current_fitness=current,
            candidate_fitness=candidate,
            verification_score=verification_score,
            promoted=False,
            reason="candidate fitness did not improve",
        )
    return PromotionReceipt(
        mutation_id=mutation.identifier,
        current_fitness=current,
        candidate_fitness=candidate,
        verification_score=verification_score,
        promoted=True,
        reason="fitness improved and verification gate passed",
    )
