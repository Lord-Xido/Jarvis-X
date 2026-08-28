"""Bounded 10x10x10 inward-4D graph autoencoder reference.

The fourth coordinate in this module is a geometric embedding coordinate, not
time, consciousness, or an extra physical spatial dimension.  The reference
keeps one latent activation per lattice node and therefore does not claim
compression.  It exists to make the supplied inward-fold arithmetic executable,
deterministic, and falsifiable without adding a numerical dependency.

The authoritative adaptive state is deliberately small: symmetric edge weights
and encoder/decoder biases.  Every proposed update is evaluated against the
complete objective before it is committed.  A rejected update leaves those
parameters and the active topology unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

Coordinate3D = tuple[int, int, int]
Point4D = tuple[float, float, float, float]
NumericSequence = Sequence[float | int]


def coordinate_to_index(coordinate: Coordinate3D, side: int = 10) -> int:
    """Map ``(x, y, z)`` to ``side**2*x + side*y + z``."""

    if isinstance(side, bool) or not isinstance(side, int) or side < 2:
        raise ValueError("side must be an integer of at least 2")
    if (
        not isinstance(coordinate, tuple)
        or len(coordinate) != 3
        or any(isinstance(value, bool) or not isinstance(value, int) for value in coordinate)
    ):
        raise TypeError("coordinate must be an integer (x, y, z) tuple")
    x, y, z = coordinate
    if not all(0 <= value < side for value in coordinate):
        raise ValueError("coordinate is outside the lattice")
    return side * side * x + side * y + z


def index_to_coordinate(index: int, side: int = 10) -> Coordinate3D:
    """Invert :func:`coordinate_to_index`."""

    if isinstance(side, bool) or not isinstance(side, int) or side < 2:
        raise ValueError("side must be an integer of at least 2")
    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError("index must be an integer")
    if not 0 <= index < side**3:
        raise ValueError("index is outside the lattice")
    x, remainder = divmod(index, side * side)
    y, z = divmod(remainder, side)
    return x, y, z


@dataclass(frozen=True)
class Inward4DConfig:
    """Numerical and structural contract for the reference engine."""

    side: int = 10
    spacing: float = 2.2
    fold_factor: float = 1.0
    core_radius: float = 4.5
    inner_radius_min: float = 2.5
    inner_radius_max: float = 4.0
    proximity_radius: float = 5.5
    decay: float = 0.88
    learning_rate: float = 0.005
    prune_threshold: float = 0.15
    prune_interval: int = 25
    base_weight: float = 0.40
    max_weight: float = 1.0
    max_abs_bias: float = 2.0
    edge_energy_weight: float = 0.02
    homeostasis_weight: float = 0.01
    bias_regularization_weight: float = 1.0e-4
    min_degree: int = 2
    max_backtracks: int = 6
    convergence_tolerance: float = 1.0e-6
    seed: int = 41

    def __post_init__(self) -> None:
        if isinstance(self.side, bool) or not isinstance(self.side, int) or self.side < 3:
            raise ValueError("side must be an integer of at least 3")
        positive = {
            "spacing": self.spacing,
            "core_radius": self.core_radius,
            "inner_radius_min": self.inner_radius_min,
            "inner_radius_max": self.inner_radius_max,
            "proximity_radius": self.proximity_radius,
            "learning_rate": self.learning_rate,
            "max_weight": self.max_weight,
            "max_abs_bias": self.max_abs_bias,
            "convergence_tolerance": self.convergence_tolerance,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.inner_radius_max < self.inner_radius_min:
            raise ValueError("inner_radius_max must be at least inner_radius_min")
        if not 0.0 <= self.fold_factor <= 1.0:
            raise ValueError("fold_factor must be in [0, 1]")
        if not 0.0 < self.decay < 1.0:
            raise ValueError("decay must be in (0, 1)")
        if not 0.0 <= self.prune_threshold <= self.max_weight:
            raise ValueError("prune_threshold must be in [0, max_weight]")
        if not 0.0 <= self.base_weight <= self.max_weight:
            raise ValueError("base_weight must be in [0, max_weight]")
        nonnegative = {
            "edge_energy_weight": self.edge_energy_weight,
            "homeostasis_weight": self.homeostasis_weight,
            "bias_regularization_weight": self.bias_regularization_weight,
        }
        for name, value in nonnegative.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if (
            isinstance(self.prune_interval, bool)
            or not isinstance(self.prune_interval, int)
            or self.prune_interval < 1
        ):
            raise ValueError("prune_interval must be a positive integer")
        if (
            isinstance(self.min_degree, bool)
            or not isinstance(self.min_degree, int)
            or not 1 <= self.min_degree <= 6
        ):
            raise ValueError("min_degree must be an integer in [1, 6]")
        if (
            isinstance(self.max_backtracks, bool)
            or not isinstance(self.max_backtracks, int)
            or self.max_backtracks < 0
        ):
            raise ValueError("max_backtracks must be a non-negative integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")


@dataclass(frozen=True)
class EdgeGeometry:
    """Immutable geometry for one undirected synapse."""

    source: int
    target: int
    axis: int
    wraps: bool
    distance4d: float
    geometry_gain: float


@dataclass(frozen=True)
class ForwardPass:
    latent: tuple[float, ...]
    reconstruction: tuple[float, ...]


@dataclass(frozen=True)
class LossTerms:
    reconstruction_mse: float
    edge_energy: float
    homeostasis: float
    bias_penalty: float
    total: float


@dataclass(frozen=True)
class EvaluationMetrics:
    loss: LossTerms
    description_residual_rms: float
    max_abs_residual: float
    converged: bool


@dataclass(frozen=True)
class GradientSnapshot:
    evaluation: EvaluationMetrics
    edge_weights: tuple[float, ...]
    encoder_bias: tuple[float, ...]
    decoder_bias: tuple[float, ...]


@dataclass(frozen=True)
class Inward4DSnapshot:
    epoch: int
    weights: tuple[float, ...]
    encoder_bias: tuple[float, ...]
    decoder_bias: tuple[float, ...]
    active_edges: tuple[bool, ...]


@dataclass(frozen=True)
class StepMetrics:
    epoch: int
    committed: bool
    rejection_reason: str | None
    learning_rate_used: float
    loss_before: float
    loss_after: float
    description_residual_rms: float
    max_abs_residual: float
    active_synapses: int
    pruned_synapses: int
    converged: bool


@dataclass(frozen=True)
class OptimizationReport:
    initial: EvaluationMetrics
    final: EvaluationMetrics
    attempted_epochs: int
    committed_epochs: int
    converged: bool
    history: tuple[StepMetrics, ...]


CandidateValidator = Callable[[EvaluationMetrics], bool]


class Inward4DANN:
    """A deterministic graph autoencoder on a folded 3D lattice.

    The graph uses positive-axis six-neighbour support.  At ``fold_factor=0``
    the three flat boundaries remain open, yielding 2,700 edges for a 10-cube.
    At ``fold_factor=1`` all three seams close, yielding exactly 3,000
    undirected edges.  Folded 4D distance modulates coupling strength and also
    gates seam candidates through ``proximity_radius``.
    """

    def __init__(self, config: Inward4DConfig | None = None) -> None:
        self.config = config or Inward4DConfig()
        self._node_count = self.config.side**3
        self._positions = tuple(
            self._folded_position(index_to_coordinate(index, self.config.side))
            for index in range(self._node_count)
        )
        self._edges = self._build_edges()
        self._incident_edges = self._build_incident_edges()
        self._weights = [
            self._initial_weight(index, edge) for index, edge in enumerate(self._edges)
        ]
        self._encoder_bias = [0.0] * self._node_count
        self._decoder_bias = [0.0] * self._node_count
        self._active = [True] * len(self._edges)
        self._epoch = 0
        self._validate_topology(self._active)

    @property
    def node_count(self) -> int:
        return self._node_count

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def positions(self) -> tuple[Point4D, ...]:
        return self._positions

    @property
    def edge_geometries(self) -> tuple[EdgeGeometry, ...]:
        return self._edges

    @property
    def active_synapse_count(self) -> int:
        return sum(self._active)

    @property
    def active_wrap_synapse_count(self) -> int:
        return sum(is_active and edge.wraps for is_active, edge in zip(self._active, self._edges))

    @property
    def degrees(self) -> tuple[int, ...]:
        return tuple(self._degrees(self._active))

    def arithmetic_summary(self) -> dict[str, int | float]:
        """Return the exact default-table quantities as machine-readable data."""

        side = self.config.side
        return {
            "side": side,
            "nodes": self.node_count,
            "flat_synapses": 3 * (side - 1) * side * side,
            "periodic_synapses": 3 * side**3,
            "topological_wrap_candidates": 3 * side * side,
            "active_synapses": self.active_synapse_count,
            "active_wrap_synapses": self.active_wrap_synapse_count,
            "fold_factor": self.config.fold_factor,
            "spacing": self.config.spacing,
            "core_radius": self.config.core_radius,
            "inner_radius_min": self.config.inner_radius_min,
            "inner_radius_max": self.config.inner_radius_max,
            "decay": self.config.decay,
            "learning_rate": self.config.learning_rate,
            "prune_threshold": self.config.prune_threshold,
        }

    def forward(self, values: NumericSequence) -> ForwardPass:
        vector = self._validated_vector(values, "values")
        _, latent, _, reconstruction, _ = self._forward_with(
            vector,
            self._weights,
            self._encoder_bias,
            self._decoder_bias,
            self._active,
        )
        return ForwardPass(tuple(latent), tuple(reconstruction))

    def encode(self, values: NumericSequence) -> tuple[float, ...]:
        return self.forward(values).latent

    def decode(self, latent: NumericSequence) -> tuple[float, ...]:
        vector = self._validated_vector(latent, "latent")
        _, decoded = self._layer(
            vector,
            self._decoder_bias,
            self._weights,
            self._active,
        )
        return tuple(decoded)

    def describe(self, values: NumericSequence) -> tuple[float, ...]:
        """Compute ``Decode(Encode(values))``."""

        return self.forward(values).reconstruction

    def evaluate(self, target: NumericSequence) -> EvaluationMetrics:
        vector = self._validated_vector(target, "target")
        return self._evaluate_with(
            vector,
            self._weights,
            self._encoder_bias,
            self._decoder_bias,
            self._active,
        )

    def gradients(self, target: NumericSequence) -> GradientSnapshot:
        """Return the exact reverse-mode gradient of the declared smooth loss."""

        vector = self._validated_vector(target, "target")
        encoder_potential, latent, decoder_potential, reconstruction, degrees = self._forward_with(
            vector,
            self._weights,
            self._encoder_bias,
            self._decoder_bias,
            self._active,
        )
        del encoder_potential, decoder_potential
        evaluation = self._evaluation_from_forward(
            vector,
            latent,
            reconstruction,
            self._weights,
            self._encoder_bias,
            self._decoder_bias,
            self._active,
        )

        node_count = self.node_count
        edge_count = self.active_synapse_count
        interaction = 1.0 - self.config.decay
        d_decoder = [
            (2.0 / node_count)
            * (reconstruction[index] - vector[index])
            * (1.0 - reconstruction[index] ** 2)
            for index in range(node_count)
        ]
        gradient_weights = [0.0] * len(self._edges)
        d_latent = [self.config.decay * value for value in d_decoder]

        for edge_index, edge in enumerate(self._edges):
            if not self._active[edge_index]:
                continue
            source = edge.source
            target_index = edge.target
            coefficient = interaction * edge.geometry_gain
            gradient_weights[edge_index] += coefficient * (
                d_decoder[source] * latent[target_index] / degrees[source]
                + d_decoder[target_index] * latent[source] / degrees[target_index]
            )
            coupling = self._weights[edge_index] * coefficient
            d_latent[target_index] += d_decoder[source] * coupling / degrees[source]
            d_latent[source] += d_decoder[target_index] * coupling / degrees[target_index]

        energy_scale = self.config.edge_energy_weight / edge_count
        homeostasis_scale = self.config.homeostasis_weight / edge_count
        for edge_index, edge in enumerate(self._edges):
            if not self._active[edge_index]:
                continue
            source = edge.source
            target_index = edge.target
            difference = latent[source] - latent[target_index]
            weighted_scale = energy_scale * edge.geometry_gain
            gradient_weights[edge_index] += weighted_scale * difference * difference
            latent_gradient = 2.0 * weighted_scale * self._weights[edge_index] * difference
            d_latent[source] += latent_gradient
            d_latent[target_index] -= latent_gradient
            gradient_weights[edge_index] += (
                2.0 * homeostasis_scale * (self._weights[edge_index] - self.config.base_weight)
            )

        d_encoder = [d_latent[index] * (1.0 - latent[index] ** 2) for index in range(node_count)]
        for edge_index, edge in enumerate(self._edges):
            if not self._active[edge_index]:
                continue
            source = edge.source
            target_index = edge.target
            coefficient = interaction * edge.geometry_gain
            gradient_weights[edge_index] += coefficient * (
                d_encoder[source] * vector[target_index] / degrees[source]
                + d_encoder[target_index] * vector[source] / degrees[target_index]
            )

        bias_scale = self.config.bias_regularization_weight / node_count
        gradient_encoder_bias = [
            d_encoder[index] + bias_scale * self._encoder_bias[index] for index in range(node_count)
        ]
        gradient_decoder_bias = [
            d_decoder[index] + bias_scale * self._decoder_bias[index] for index in range(node_count)
        ]
        return GradientSnapshot(
            evaluation=evaluation,
            edge_weights=tuple(gradient_weights),
            encoder_bias=tuple(gradient_encoder_bias),
            decoder_bias=tuple(gradient_decoder_bias),
        )

    def train_step(
        self,
        target: NumericSequence,
        *,
        validator: CandidateValidator | None = None,
    ) -> StepMetrics:
        """Propose, evaluate, and atomically commit one bounded update."""

        vector = self._validated_vector(target, "target")
        gradient = self.gradients(vector)
        before = gradient.evaluation
        next_epoch = self._epoch + 1
        rejection_reason = "candidate did not improve objective"

        for backtrack in range(self.config.max_backtracks + 1):
            learning_rate = self.config.learning_rate * (0.5**backtrack)
            candidate_weights = list(self._weights)
            for index, value in enumerate(candidate_weights):
                if not self._active[index]:
                    continue
                candidate_weights[index] = min(
                    self.config.max_weight,
                    max(0.0, value - learning_rate * gradient.edge_weights[index]),
                )
            candidate_encoder_bias = [
                min(
                    self.config.max_abs_bias,
                    max(
                        -self.config.max_abs_bias,
                        value - learning_rate * gradient.encoder_bias[index],
                    ),
                )
                for index, value in enumerate(self._encoder_bias)
            ]
            candidate_decoder_bias = [
                min(
                    self.config.max_abs_bias,
                    max(
                        -self.config.max_abs_bias,
                        value - learning_rate * gradient.decoder_bias[index],
                    ),
                )
                for index, value in enumerate(self._decoder_bias)
            ]
            candidate_active = list(self._active)
            pruned = 0
            if next_epoch % self.config.prune_interval == 0:
                pruned = self._propose_pruning(candidate_weights, candidate_active)

            candidate = self._evaluate_with(
                vector,
                candidate_weights,
                candidate_encoder_bias,
                candidate_decoder_bias,
                candidate_active,
            )
            if validator is not None and not bool(validator(candidate)):
                rejection_reason = "validator rejected candidate"
                break
            if candidate.loss.total <= before.loss.total + 1.0e-12:
                self._weights = candidate_weights
                self._encoder_bias = candidate_encoder_bias
                self._decoder_bias = candidate_decoder_bias
                self._active = candidate_active
                self._epoch = next_epoch
                return StepMetrics(
                    epoch=self._epoch,
                    committed=True,
                    rejection_reason=None,
                    learning_rate_used=learning_rate,
                    loss_before=before.loss.total,
                    loss_after=candidate.loss.total,
                    description_residual_rms=candidate.description_residual_rms,
                    max_abs_residual=candidate.max_abs_residual,
                    active_synapses=sum(candidate_active),
                    pruned_synapses=pruned,
                    converged=candidate.converged,
                )

        self._epoch = next_epoch
        return StepMetrics(
            epoch=self._epoch,
            committed=False,
            rejection_reason=rejection_reason,
            learning_rate_used=0.0,
            loss_before=before.loss.total,
            loss_after=before.loss.total,
            description_residual_rms=before.description_residual_rms,
            max_abs_residual=before.max_abs_residual,
            active_synapses=self.active_synapse_count,
            pruned_synapses=0,
            converged=before.converged,
        )

    def optimize(
        self,
        target: NumericSequence,
        *,
        max_epochs: int = 100,
        tolerance: float | None = None,
        validator: CandidateValidator | None = None,
    ) -> OptimizationReport:
        """Run a finite optimization budget and return its full audit history."""

        if isinstance(max_epochs, bool) or not isinstance(max_epochs, int) or max_epochs < 0:
            raise ValueError("max_epochs must be a non-negative integer")
        resolved_tolerance = (
            self.config.convergence_tolerance if tolerance is None else float(tolerance)
        )
        if not math.isfinite(resolved_tolerance) or resolved_tolerance <= 0.0:
            raise ValueError("tolerance must be finite and positive")
        vector = self._validated_vector(target, "target")
        initial = self.evaluate(vector)
        history: list[StepMetrics] = []
        current = initial
        for _ in range(max_epochs):
            if current.description_residual_rms < resolved_tolerance:
                break
            step = self.train_step(vector, validator=validator)
            history.append(step)
            current = self.evaluate(vector)
            if not step.committed and step.rejection_reason == "validator rejected candidate":
                break
        final = self.evaluate(vector)
        return OptimizationReport(
            initial=initial,
            final=final,
            attempted_epochs=len(history),
            committed_epochs=sum(step.committed for step in history),
            converged=final.description_residual_rms < resolved_tolerance,
            history=tuple(history),
        )

    def snapshot(self) -> Inward4DSnapshot:
        return Inward4DSnapshot(
            epoch=self._epoch,
            weights=tuple(self._weights),
            encoder_bias=tuple(self._encoder_bias),
            decoder_bias=tuple(self._decoder_bias),
            active_edges=tuple(self._active),
        )

    def restore(self, snapshot: Inward4DSnapshot) -> None:
        """Atomically restore a validated in-memory snapshot."""

        if not isinstance(snapshot, Inward4DSnapshot):
            raise TypeError("snapshot must be an Inward4DSnapshot")
        if isinstance(snapshot.epoch, bool) or snapshot.epoch < 0:
            raise ValueError("snapshot epoch must be non-negative")
        if len(snapshot.weights) != len(self._edges):
            raise ValueError("snapshot weight count does not match topology")
        if len(snapshot.active_edges) != len(self._edges):
            raise ValueError("snapshot active-edge count does not match topology")
        if (
            len(snapshot.encoder_bias) != self.node_count
            or len(snapshot.decoder_bias) != self.node_count
        ):
            raise ValueError("snapshot bias count does not match node count")
        weights = [self._finite(value, "snapshot weight") for value in snapshot.weights]
        if any(not 0.0 <= value <= self.config.max_weight for value in weights):
            raise ValueError("snapshot weight is outside configured bounds")
        encoder_bias = [
            self._finite(value, "snapshot encoder bias") for value in snapshot.encoder_bias
        ]
        decoder_bias = [
            self._finite(value, "snapshot decoder bias") for value in snapshot.decoder_bias
        ]
        if any(abs(value) > self.config.max_abs_bias for value in encoder_bias + decoder_bias):
            raise ValueError("snapshot bias is outside configured bounds")
        active = list(snapshot.active_edges)
        if any(not isinstance(value, bool) for value in active):
            raise TypeError("snapshot active-edge flags must be bool")
        self._validate_topology(active)
        self._weights = weights
        self._encoder_bias = encoder_bias
        self._decoder_bias = decoder_bias
        self._active = active
        self._epoch = snapshot.epoch

    def _folded_position(self, coordinate: Coordinate3D) -> Point4D:
        side = self.config.side
        x, y, z = coordinate
        center = (side - 1) / 2.0
        box = (
            self.config.spacing * (x - center),
            self.config.spacing * (y - center),
            self.config.spacing * (z - center),
            0.0,
        )
        theta = 2.0 * math.pi * x / side
        phi = 2.0 * math.pi * y / side
        psi = 2.0 * math.pi * z / side
        layer = z / (side - 1)
        inner_radius = (
            self.config.inner_radius_min
            + (self.config.inner_radius_max - self.config.inner_radius_min) * layer
        )
        ring = self.config.core_radius + inner_radius * math.cos(phi)
        cross_section = inner_radius * math.sin(phi)
        toroidal = (
            ring * math.cos(theta),
            ring * math.sin(theta),
            cross_section * math.cos(psi),
            cross_section * math.sin(psi),
        )
        gamma = self.config.fold_factor
        return tuple(
            (1.0 - gamma) * box[axis] + gamma * toroidal[axis] for axis in range(4)
        )  # type: ignore[return-value]

    def _build_edges(self) -> tuple[EdgeGeometry, ...]:
        side = self.config.side
        edges: list[EdgeGeometry] = []
        seen: set[tuple[int, int]] = set()
        for source in range(self.node_count):
            coordinate = list(index_to_coordinate(source, side))
            for axis in range(3):
                wraps = coordinate[axis] == side - 1
                if wraps and self.config.fold_factor == 0.0:
                    continue
                neighbour = list(coordinate)
                neighbour[axis] = (neighbour[axis] + 1) % side
                target = coordinate_to_index(tuple(neighbour), side)  # type: ignore[arg-type]
                pair = min(source, target), max(source, target)
                if pair in seen:
                    continue
                distance = math.dist(self._positions[source], self._positions[target])
                if distance > self.config.proximity_radius + 1.0e-12:
                    continue
                fold_scale = self.config.fold_factor if wraps else 1.0
                geometry_gain = fold_scale * math.exp(
                    -0.5 * (distance / self.config.proximity_radius) ** 2
                )
                edges.append(
                    EdgeGeometry(
                        source=pair[0],
                        target=pair[1],
                        axis=axis,
                        wraps=wraps,
                        distance4d=distance,
                        geometry_gain=geometry_gain,
                    )
                )
                seen.add(pair)
        edges.sort(key=lambda edge: (edge.source, edge.target, edge.axis))
        return tuple(edges)

    def _build_incident_edges(self) -> tuple[tuple[int, ...], ...]:
        incident: list[list[int]] = [[] for _ in range(self.node_count)]
        for edge_index, edge in enumerate(self._edges):
            incident[edge.source].append(edge_index)
            incident[edge.target].append(edge_index)
        return tuple(tuple(values) for values in incident)

    def _initial_weight(self, edge_index: int, edge: EdgeGeometry) -> float:
        mixed = (
            edge.source * 73_856_093
            ^ edge.target * 19_349_663
            ^ edge.axis * 83_492_791
            ^ edge_index * 2_654_435_761
            ^ self.config.seed
        )
        jitter = ((mixed % 2001) - 1000) / 100_000.0
        return min(self.config.max_weight, max(0.0, self.config.base_weight + jitter))

    def _layer(
        self,
        values: list[float],
        bias: Sequence[float],
        weights: Sequence[float],
        active: Sequence[bool],
    ) -> tuple[list[float], list[float]]:
        degrees = self._degrees(active)
        neighbour_sum = [0.0] * self.node_count
        for edge_index, edge in enumerate(self._edges):
            if not active[edge_index]:
                continue
            coupling = weights[edge_index] * edge.geometry_gain
            neighbour_sum[edge.source] += coupling * values[edge.target]
            neighbour_sum[edge.target] += coupling * values[edge.source]
        interaction = 1.0 - self.config.decay
        potential = [
            self.config.decay * values[index]
            + interaction * neighbour_sum[index] / degrees[index]
            + bias[index]
            for index in range(self.node_count)
        ]
        return potential, [math.tanh(value) for value in potential]

    def _forward_with(
        self,
        values: list[float],
        weights: Sequence[float],
        encoder_bias: Sequence[float],
        decoder_bias: Sequence[float],
        active: Sequence[bool],
    ) -> tuple[list[float], list[float], list[float], list[float], list[int]]:
        degrees = self._degrees(active)
        encoder_potential, latent = self._layer(values, encoder_bias, weights, active)
        decoder_potential, reconstruction = self._layer(latent, decoder_bias, weights, active)
        return encoder_potential, latent, decoder_potential, reconstruction, degrees

    def _evaluate_with(
        self,
        target: list[float],
        weights: Sequence[float],
        encoder_bias: Sequence[float],
        decoder_bias: Sequence[float],
        active: Sequence[bool],
    ) -> EvaluationMetrics:
        _, latent, _, reconstruction, _ = self._forward_with(
            target, weights, encoder_bias, decoder_bias, active
        )
        return self._evaluation_from_forward(
            target,
            latent,
            reconstruction,
            weights,
            encoder_bias,
            decoder_bias,
            active,
        )

    def _evaluation_from_forward(
        self,
        target: Sequence[float],
        latent: Sequence[float],
        reconstruction: Sequence[float],
        weights: Sequence[float],
        encoder_bias: Sequence[float],
        decoder_bias: Sequence[float],
        active: Sequence[bool],
    ) -> EvaluationMetrics:
        residuals = [reconstruction[index] - target[index] for index in range(self.node_count)]
        reconstruction_mse = sum(value * value for value in residuals) / self.node_count
        edge_count = sum(active)
        edge_energy = 0.0
        homeostasis = 0.0
        for edge_index, edge in enumerate(self._edges):
            if not active[edge_index]:
                continue
            difference = latent[edge.source] - latent[edge.target]
            edge_energy += weights[edge_index] * edge.geometry_gain * difference * difference
            weight_residual = weights[edge_index] - self.config.base_weight
            homeostasis += weight_residual * weight_residual
        edge_energy /= edge_count
        homeostasis /= edge_count
        bias_penalty = (
            sum(value * value for value in encoder_bias)
            + sum(value * value for value in decoder_bias)
        ) / (2.0 * self.node_count)
        total = (
            reconstruction_mse
            + self.config.edge_energy_weight * edge_energy
            + self.config.homeostasis_weight * homeostasis
            + self.config.bias_regularization_weight * bias_penalty
        )
        residual_rms = math.sqrt(reconstruction_mse)
        return EvaluationMetrics(
            loss=LossTerms(
                reconstruction_mse=reconstruction_mse,
                edge_energy=edge_energy,
                homeostasis=homeostasis,
                bias_penalty=bias_penalty,
                total=total,
            ),
            description_residual_rms=residual_rms,
            max_abs_residual=max((abs(value) for value in residuals), default=0.0),
            converged=residual_rms < self.config.convergence_tolerance,
        )

    def _propose_pruning(self, weights: Sequence[float], active: list[bool]) -> int:
        degrees = self._degrees(active)
        pruned = 0
        candidates = sorted(
            (
                (weights[index], index)
                for index, is_active in enumerate(active)
                if is_active and weights[index] < self.config.prune_threshold
            ),
            key=lambda item: (item[0], self._edges[item[1]].source, self._edges[item[1]].target),
        )
        for _, edge_index in candidates:
            edge = self._edges[edge_index]
            if (
                degrees[edge.source] <= self.config.min_degree
                or degrees[edge.target] <= self.config.min_degree
            ):
                continue
            active[edge_index] = False
            if self._is_connected(active):
                degrees[edge.source] -= 1
                degrees[edge.target] -= 1
                pruned += 1
            else:
                active[edge_index] = True
        return pruned

    def _degrees(self, active: Sequence[bool]) -> list[int]:
        degrees = [0] * self.node_count
        for edge_index, edge in enumerate(self._edges):
            if active[edge_index]:
                degrees[edge.source] += 1
                degrees[edge.target] += 1
        if any(value == 0 for value in degrees):
            raise RuntimeError("active topology contains an isolated node")
        return degrees

    def _is_connected(self, active: Sequence[bool]) -> bool:
        seen = {0}
        stack = [0]
        while stack:
            node = stack.pop()
            for edge_index in self._incident_edges[node]:
                if not active[edge_index]:
                    continue
                edge = self._edges[edge_index]
                neighbour = edge.target if edge.source == node else edge.source
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        return len(seen) == self.node_count

    def _validate_topology(self, active: Sequence[bool]) -> None:
        if not self._edges:
            raise ValueError("proximity radius removed every synapse")
        degrees = self._degrees(active)
        if min(degrees) < self.config.min_degree:
            raise ValueError("topology violates the configured minimum degree")
        if not self._is_connected(active):
            raise ValueError("topology is disconnected")

    def _validated_vector(self, values: NumericSequence, name: str) -> list[float]:
        if isinstance(values, (str, bytes)) or len(values) != self.node_count:
            raise ValueError(f"{name} must contain exactly {self.node_count} values")
        return [self._finite(value, f"{name} value") for value in values]

    @staticmethod
    def _finite(value: float | int, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{name} must be finite")
        return result
