"""Bounded 3D geometric diffusion and kinetic adaptation reference runtime.

The module is dependency-free and deterministic under an explicit seed. It is a
research fixture for virtual 3D relational state, geometry-conditioned diffusion,
candidate-first publication, bounded residual memory, and configuration promotion.
It does not claim that an LLM physically stores hidden state in Euclidean 3D space.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

Vec3 = tuple[float, float, float]
Feature = tuple[float, ...]
Edge = tuple[int, int]


def _finite(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result:
        raise ValueError(f"{name} must not be empty")
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _unit(value: float, *, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and within [0, 1]")
    return value


def _non_negative(value: float, *, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


@dataclass(frozen=True)
class GeometricNode:
    """One virtual 3D point and its semantic/latent feature vector."""

    position: Vec3
    feature: Feature

    def __post_init__(self) -> None:
        position = _finite(self.position, name="position")
        if len(position) != 3:
            raise ValueError("position must contain exactly three coordinates")
        feature = _finite(self.feature, name="feature")
        object.__setattr__(self, "position", (position[0], position[1], position[2]))
        object.__setattr__(self, "feature", feature)


@dataclass(frozen=True)
class GeometricGraph:
    """Finite validated undirected relational graph."""

    nodes: tuple[GeometricNode, ...]
    edges: tuple[Edge, ...]

    def __post_init__(self) -> None:
        if not self.nodes:
            raise ValueError("graph must contain at least one node")
        width = len(self.nodes[0].feature)
        if any(len(node.feature) != width for node in self.nodes):
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
    """Numerical, verification, exploration and resource bounds."""

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
        for name in (
            "beta",
            "geometry_gain",
            "graph_gain",
            "memory_retention",
            "verification_threshold",
        ):
            _unit(getattr(self, name), name=name)
        for name in (
            "memory_gain",
            "max_position_step",
            "max_feature_step",
            "max_cycle_rms",
        ):
            _non_negative(getattr(self, name), name=name)
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
    data = tuple(float(value) for value in values)
    return math.sqrt(sum(value * value for value in data) / len(data)) if data else 0.0


def _same_topology(left: GeometricGraph, right: GeometricGraph) -> None:
    if len(left.nodes) != len(right.nodes) or left.edges != right.edges:
        raise ValueError("graphs must share node count and edge topology")
    if left.feature_width != right.feature_width:
        raise ValueError("graphs must share feature width")


def graph_rms(reference: GeometricGraph, candidate: GeometricGraph) -> tuple[float, float, float]:
    """Return position, feature and combined RMS residuals."""

    _same_topology(reference, candidate)
    position_delta: list[float] = []
    feature_delta: list[float] = []
    for left, right in zip(reference.nodes, candidate.nodes):
        position_delta.extend(a - b for a, b in zip(left.position, right.position))
        feature_delta.extend(a - b for a, b in zip(left.feature, right.feature))
    return (
        _rms(position_delta),
        _rms(feature_delta),
        _rms((*position_delta, *feature_delta)),
    )


def edge_energy(graph: GeometricGraph) -> float:
    """Mean squared feature difference over edges; not Shannon entropy."""

    residuals = [
        x - y
        for a, b in graph.edges
        for x, y in zip(graph.nodes[a].feature, graph.nodes[b].feature)
    ]
    return sum(value * value for value in residuals) / len(residuals) if residuals else 0.0


def forward_diffuse(graph: GeometricGraph, *, beta: float, seed: int) -> GeometricGraph:
    """Apply reproducible Gaussian corruption to positions and feature state."""

    beta = _unit(beta, name="beta")
    keep = math.sqrt(1.0 - beta)
    noise = math.sqrt(beta)
    rng = random.Random(seed)
    nodes: list[GeometricNode] = []
    for node in graph.nodes:
        position: Vec3 = (
            keep * node.position[0] + noise * rng.gauss(0.0, 1.0),
            keep * node.position[1] + noise * rng.gauss(0.0, 1.0),
            keep * node.position[2] + noise * rng.gauss(0.0, 1.0),
        )
        feature = tuple(keep * value + noise * rng.gauss(0.0, 1.0) for value in node.feature)
        nodes.append(GeometricNode(position, feature))
    return GeometricGraph(tuple(nodes), graph.edges)


def reverse_denoise_step(
    current: GeometricGraph,
    anchor: GeometricGraph,
    *,
    geometry_gain: float,
    graph_gain: float,
    memory: Sequence[Feature] | None = None,
    memory_gain: float = 0.0,
) -> GeometricGraph:
    """Contract toward the anchor and smooth features over graph adjacency."""

    _same_topology(current, anchor)
    geometry_gain = _unit(geometry_gain, name="geometry_gain")
    graph_gain = _unit(graph_gain, name="graph_gain")
    memory_gain = _non_negative(memory_gain, name="memory_gain")
    if memory is None:
        memory = tuple((0.0,) * current.feature_width for _ in current.nodes)
    if len(memory) != len(current.nodes):
        raise ValueError("memory must provide one feature vector per node")
    if any(len(item) != current.feature_width for item in memory):
        raise ValueError("memory feature width must match the graph")
    if any(not all(math.isfinite(float(value)) for value in item) for item in memory):
        raise ValueError("memory must contain only finite values")

    adjacency = current.adjacency()
    nodes: list[GeometricNode] = []
    for index, (node, target) in enumerate(zip(current.nodes, anchor.nodes)):
        position: Vec3 = (
            node.position[0] + geometry_gain * (target.position[0] - node.position[0]),
            node.position[1] + geometry_gain * (target.position[1] - node.position[1]),
            node.position[2] + geometry_gain * (target.position[2] - node.position[2]),
        )
        neighbors = adjacency[index]
        neighbor_mean = (
            tuple(
                sum(current.nodes[j].feature[k] for j in neighbors) / len(neighbors)
                for k in range(current.feature_width)
            )
            if neighbors
            else node.feature
        )
        feature = tuple(
            value
            + geometry_gain * (target.feature[k] - value)
            + graph_gain * (neighbor_mean[k] - value)
            + memory_gain * float(memory[index][k])
            for k, value in enumerate(node.feature)
        )
        nodes.append(GeometricNode(position, feature))
    return GeometricGraph(tuple(nodes), current.edges)


def _bounded(value: float, anchor: float, limit: float) -> float:
    return anchor + max(-limit, min(limit, value - anchor))


def project_candidate(
    candidate: GeometricGraph,
    anchor: GeometricGraph,
    *,
    max_position_step: float,
    max_feature_step: float,
) -> GeometricGraph:
    """Project candidate deltas into explicit component-wise bounds."""

    _same_topology(candidate, anchor)
    max_position_step = _non_negative(max_position_step, name="max_position_step")
    max_feature_step = _non_negative(max_feature_step, name="max_feature_step")
    nodes: list[GeometricNode] = []
    for node, base in zip(candidate.nodes, anchor.nodes):
        position: Vec3 = (
            _bounded(node.position[0], base.position[0], max_position_step),
            _bounded(node.position[1], base.position[1], max_position_step),
            _bounded(node.position[2], base.position[2], max_position_step),
        )
        feature = tuple(
            _bounded(value, origin, max_feature_step)
            for value, origin in zip(node.feature, base.feature)
        )
        nodes.append(GeometricNode(position, feature))
    return GeometricGraph(tuple(nodes), candidate.edges)


class GeometricDiffusionRuntime:
    """Candidate-first diffuse, denoise, project, verify and commit loop."""

    def __init__(
        self,
        anchor: GeometricGraph,
        config: GeometricDiffusionConfig | None = None,
    ) -> None:
        self.config = config or GeometricDiffusionConfig()
        self._check_budget(anchor)
        self._anchor = anchor
        memory = tuple((0.0,) * anchor.feature_width for _ in anchor.nodes)
        metrics = DiffusionMetrics(
            0,
            len(anchor.nodes),
            len(anchor.edges),
            0.0,
            0.0,
            0.0,
            edge_energy(anchor),
            1.0,
        )
        self._state = DiffusionState(0, anchor, memory, metrics)

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
        """Publish a new state only after numerical and optional external gates pass."""

        self._check_budget(observation)
        _same_topology(self._anchor, observation)
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
        position_rms, feature_rms, combined_rms = graph_rms(observation, candidate)
        verification = 1.0 / (1.0 + combined_rms)
        cycle = self._state.cycle + 1
        pending = DiffusionState(
            cycle,
            candidate,
            self._update_memory(observation, candidate),
            DiffusionMetrics(
                cycle,
                len(candidate.nodes),
                len(candidate.edges),
                position_rms,
                feature_rms,
                combined_rms,
                edge_energy(candidate),
                verification,
            ),
        )
        if combined_rms > self.config.max_cycle_rms:
            raise RuntimeError("candidate exceeds configured cycle RMS")
        if verification < self.config.verification_threshold:
            raise RuntimeError("candidate verification score is below threshold")
        if validator is not None and not bool(validator(pending)):
            raise RuntimeError("candidate rejected by geometric diffusion validator")
        self._state = pending
        return pending

    def branch_candidates(
        self,
        observation: GeometricGraph,
        *,
        seed: int = 0,
    ) -> tuple[GeometricGraph, ...]:
        """Generate a bounded deterministic family of exploratory candidates."""

        self._check_budget(observation)
        _same_topology(self._anchor, observation)
        return tuple(
            forward_diffuse(observation, beta=self.config.beta, seed=seed + index)
            for index in range(self.config.branch_width)
        )

    def _update_memory(
        self,
        observation: GeometricGraph,
        candidate: GeometricGraph,
    ) -> tuple[Feature, ...]:
        retention = self.config.memory_retention
        return tuple(
            tuple(
                retention * old + (1.0 - retention) * (source - result)
                for old, source, result in zip(previous, left.feature, right.feature)
            )
            for previous, left, right in zip(
                self._state.memory,
                observation.nodes,
                candidate.nodes,
            )
        )

    def _check_budget(self, graph: GeometricGraph) -> None:
        if len(graph.nodes) > self.config.max_nodes:
            raise RuntimeError("graph exceeds configured node budget")
        if len(graph.edges) > self.config.max_edges:
            raise RuntimeError("graph exceeds configured edge budget")


@dataclass(frozen=True)
class FitnessMetrics:
    quality: float
    reliability: float
    efficiency: float
    coherence: float
    fault_probability: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "quality",
            "reliability",
            "efficiency",
            "coherence",
            "fault_probability",
        ):
            _unit(getattr(self, name), name=name)

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
    """Versioned configuration candidate, not arbitrary source-code mutation."""

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
    """Promote only a verified configuration whose fitness strictly improves."""

    verification_score = _unit(verification_score, name="verification_score")
    current = current_metrics.score
    candidate = candidate_metrics.score
    if verification_score < mutation.config.verification_threshold:
        promoted, reason = False, "verification threshold not met"
    elif candidate <= current:
        promoted, reason = False, "candidate fitness did not improve"
    else:
        promoted, reason = True, "fitness improved and verification gate passed"
    return PromotionReceipt(
        mutation.identifier,
        current,
        candidate,
        verification_score,
        promoted,
        reason,
    )
