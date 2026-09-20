"""Bounded reference runtime for the Dr Moagi trillion-cubed kinetic hyper-swarm.

The logical geometry is 10^12 sites per axis, therefore 10^36 possible lattice
addresses. Each logical site may denote a model with 10^12 parameter-address
positions. None of those virtual extents are densely allocated.

Only active_nodes are materialized in a deterministic contiguous 3D block.
Each active node executes the five-stage kinetic cycle:

    encode local model sketch
    -> cognitive proposal
    -> 3D neighbourhood consensus
    -> topological projection
    -> bounded memory update + decode
    -> recur

For discrete step size dt the reference decomposes the observed state velocity
exactly as

    (x[t+1] - x[t]) / dt
      = v_cog + v_cons + v_proj

up to floating-point round-off.

The implementation is a deterministic software laboratory. Virtual scale is
metadata and addressing semantics, not evidence of 10^36 resident processors,
10^48 resident parameters, or hardware throughput.
"""

from __future__ import annotations

import hashlib
import math
import random
import struct
from dataclasses import dataclass
from typing import Sequence

TRILLION = 10**12
VIRTUAL_SIDE = TRILLION
LOGICAL_SITES = TRILLION**3
LOGICAL_PARAMETERS_PER_SITE = TRILLION
LOGICAL_PARAMETER_POSITIONS = LOGICAL_SITES * LOGICAL_PARAMETERS_PER_SITE

_OFFSETS_6 = (
    (-1, 0, 0),
    (1, 0, 0),
    (0, -1, 0),
    (0, 1, 0),
    (0, 0, -1),
    (0, 0, 1),
)


def _offsets_26() -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (dx, dy, dz)
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if (dx, dy, dz) != (0, 0, 0)
    )


_OFFSETS_26 = _offsets_26()


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _l2(values: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def _mse(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("mse inputs must have the same width")
    if not a:
        return 0.0
    return sum((x - y) ** 2 for x, y in zip(a, b)) / len(a)


def _mean(vectors: Sequence[Sequence[float]], width: int) -> list[float]:
    if not vectors:
        return [0.0] * width
    inv = 1.0 / len(vectors)
    return [sum(vector[j] for vector in vectors) * inv for j in range(width)]


def _integer_cube_side(count: int) -> int:
    side = 1
    while side**3 < count:
        side += 1
    return side


def _anchor_coordinate(axis: int, side: int, seed: int) -> int:
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2")
    if not 1 <= side <= VIRTUAL_SIDE:
        raise ValueError("materialized block side outside virtual domain")
    span = VIRTUAL_SIDE - side + 1
    payload = f"jarvisx:trillion3-kinetic:{seed}:{axis}".encode("ascii")
    digest = hashlib.blake2b(payload, digest_size=16).digest()
    return int.from_bytes(digest, "big") % span


def chart_coordinate(value: int) -> float:
    """Project an integer lattice coordinate into the finite local [-1, 1] chart."""
    if not 0 <= value < VIRTUAL_SIDE:
        raise ValueError("coordinate outside trillion-axis domain")
    return 2.0 * (value / (VIRTUAL_SIDE - 1)) - 1.0


@dataclass(frozen=True)
class Trillion3KineticConfig:
    active_nodes: int = 125
    state_dim: int = 6
    model_sketch_dim: int = 8
    latent_dim: int = 8
    neighbor_mode: int = 6
    dt: float = 1.0
    cognitive_self_gain: float = 0.48
    input_gain: float = 0.24
    memory_gain: float = 0.10
    neighbor_gain: float = 0.18
    self_model_gain: float = 0.72
    consensus_gain: float = 0.55
    memory_beta: float = 0.92
    manifold_radius: float = 1.0
    consensus_epsilon: float = 1.0e-9
    velocity_tolerance: float = 1.0e-7
    seed: int = 7

    def __post_init__(self) -> None:
        if not 1 <= self.active_nodes <= 1_000_000:
            raise ValueError("active_nodes must be in [1, 1_000_000]")
        if self.state_dim <= 0 or self.model_sketch_dim <= 0 or self.latent_dim <= 0:
            raise ValueError("state, sketch and latent dimensions must be positive")
        if self.neighbor_mode not in (6, 26):
            raise ValueError("neighbor_mode must be 6 or 26")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        for name, value in (
            ("cognitive_self_gain", self.cognitive_self_gain),
            ("input_gain", self.input_gain),
            ("memory_gain", self.memory_gain),
            ("neighbor_gain", self.neighbor_gain),
            ("self_model_gain", self.self_model_gain),
            ("consensus_gain", self.consensus_gain),
            ("memory_beta", self.memory_beta),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.manifold_radius <= 0.0:
            raise ValueError("manifold_radius must be positive")
        if self.consensus_epsilon <= 0.0:
            raise ValueError("consensus_epsilon must be positive")
        if self.velocity_tolerance < 0.0:
            raise ValueError("velocity_tolerance must be non-negative")


@dataclass
class KineticNode:
    address: tuple[int, int, int]
    model_sketch: list[float]
    state: list[float]
    memory: list[float]


@dataclass(frozen=True)
class KineticCycleReceipt:
    cycle: int
    logical_side: int
    logical_sites: int
    logical_parameters_per_site: int
    logical_parameter_positions: int
    active_nodes: int
    active_edges: int
    resident_scalars: int
    mean_prediction_error: float
    max_prediction_error: float
    mean_cognitive_speed: float
    mean_consensus_speed: float
    mean_projection_speed: float
    mean_total_speed: float
    kinetic_balance_error: float
    pre_projection_violations: int
    post_projection_violations: int
    converged_nodes: int
    model_sketch_mse: float
    mean_memory_norm: float


class DrMoagiTrillion3KineticSwarm:
    """Sparse executable reference for a 10^12 x 10^12 x 10^12 kinetic lattice."""

    def __init__(self, config: Trillion3KineticConfig | None = None) -> None:
        self.config = config or Trillion3KineticConfig()
        self.rng = random.Random(self.config.seed)
        self.block_side = _integer_cube_side(self.config.active_nodes)
        self.nodes = self._materialize_nodes()
        self.address_to_index = {node.address: index for index, node in enumerate(self.nodes)}
        self.neighbors = self._build_neighbors()
        self.active_edges = sum(len(items) for items in self.neighbors) // 2

        self.w_encode = self._random_matrix(
            self.config.latent_dim, self.config.model_sketch_dim
        )
        self.w_input = self._random_matrix(self.config.state_dim, self.config.latent_dim)
        self.w_decode = self._random_matrix(
            self.config.model_sketch_dim, self.config.state_dim
        )
        self.b_encode = [0.0] * self.config.latent_dim
        self.b_input = [0.0] * self.config.state_dim
        self.b_decode = [0.0] * self.config.model_sketch_dim

        self.cycle = 0
        self.receipts: list[KineticCycleReceipt] = []

    def _random_matrix(self, rows: int, cols: int) -> list[list[float]]:
        scale = 1.0 / math.sqrt(max(1, cols))
        return [
            [self.rng.uniform(-scale, scale) for _ in range(cols)]
            for _ in range(rows)
        ]

    def _model_sketch(self, address: tuple[int, int, int]) -> list[float]:
        charts = [chart_coordinate(value) for value in address]
        sketch: list[float] = []
        for index in range(self.config.model_sketch_dim):
            axis_value = charts[index % 3]
            phase = (index + 1) * math.pi * axis_value
            cross = charts[(index + 1) % 3] * charts[(index + 2) % 3]
            sketch.append(math.tanh(math.sin(phase) + 0.5 * cross))
        return sketch

    def _materialize_nodes(self) -> list[KineticNode]:
        anchors = tuple(
            _anchor_coordinate(axis, self.block_side, self.config.seed) for axis in range(3)
        )
        nodes: list[KineticNode] = []
        for index in range(self.config.active_nodes):
            lx = index % self.block_side
            ly = (index // self.block_side) % self.block_side
            lz = index // (self.block_side * self.block_side)
            address = (
                anchors[0] + lx,
                anchors[1] + ly,
                anchors[2] + lz,
            )
            sketch = self._model_sketch(address)
            state = [
                0.35 * sketch[j % self.config.model_sketch_dim]
                for j in range(self.config.state_dim)
            ]
            memory = [0.0] * self.config.state_dim
            nodes.append(KineticNode(address, sketch, state, memory))
        return nodes

    def _build_neighbors(self) -> list[list[int]]:
        offsets = _OFFSETS_6 if self.config.neighbor_mode == 6 else _OFFSETS_26
        topology: list[list[int]] = []
        for node in self.nodes:
            x, y, z = node.address
            local: list[int] = []
            for dx, dy, dz in offsets:
                candidate = (x + dx, y + dy, z + dz)
                index = self.address_to_index.get(candidate)
                if index is not None:
                    local.append(index)
            topology.append(local)
        return topology

    def _encode(self) -> list[list[float]]:
        return [
            [
                math.tanh(_dot(row, node.model_sketch) + bias)
                for row, bias in zip(self.w_encode, self.b_encode)
            ]
            for node in self.nodes
        ]

    def _project_input(self, latent: Sequence[float]) -> list[float]:
        return [
            math.tanh(_dot(row, latent) + bias)
            for row, bias in zip(self.w_input, self.b_input)
        ]

    def _raw_proposals(
        self, latent: Sequence[Sequence[float]], old_states: Sequence[Sequence[float]]
    ) -> tuple[list[list[float]], list[float], list[list[float]]]:
        raw: list[list[float]] = []
        errors: list[float] = []
        v_cog: list[list[float]] = []

        for index, node in enumerate(self.nodes):
            neighbor_states = [old_states[j] for j in self.neighbors[index]]
            neighbor_mean = _mean(neighbor_states, self.config.state_dim)
            input_state = self._project_input(latent[index])

            proposal = [
                math.tanh(
                    self.config.cognitive_self_gain * old_states[index][j]
                    + self.config.input_gain * input_state[j]
                    + self.config.memory_gain * node.memory[j]
                    + self.config.neighbor_gain * neighbor_mean[j]
                )
                for j in range(self.config.state_dim)
            ]
            self_prediction = [
                math.tanh(
                    self.config.self_model_gain * old_states[index][j]
                    + (1.0 - self.config.self_model_gain) * node.memory[j]
                )
                for j in range(self.config.state_dim)
            ]
            raw.append(proposal)
            errors.append(_mse(proposal, self_prediction))
            v_cog.append(
                [
                    (proposal[j] - old_states[index][j]) / self.config.dt
                    for j in range(self.config.state_dim)
                ]
            )

        return raw, errors, v_cog

    def _consensus(
        self, raw: Sequence[Sequence[float]], errors: Sequence[float]
    ) -> tuple[list[list[float]], list[list[float]]]:
        aggregated: list[list[float]] = []
        v_cons: list[list[float]] = []

        for index in range(len(self.nodes)):
            members = [index, *self.neighbors[index]]
            weights = [
                1.0 / (self.config.consensus_epsilon + errors[member]) for member in members
            ]
            normalizer = sum(weights)
            centroid = [
                sum(weight * raw[member][j] for weight, member in zip(weights, members))
                / normalizer
                for j in range(self.config.state_dim)
            ]
            alpha = self.config.consensus_gain
            value = [
                (1.0 - alpha) * raw[index][j] + alpha * centroid[j]
                for j in range(self.config.state_dim)
            ]
            aggregated.append(value)
            v_cons.append(
                [
                    (value[j] - raw[index][j]) / self.config.dt
                    for j in range(self.config.state_dim)
                ]
            )

        return aggregated, v_cons

    def _project_manifold(
        self, aggregated: Sequence[Sequence[float]]
    ) -> tuple[list[list[float]], list[list[float]], int, int]:
        projected: list[list[float]] = []
        v_proj: list[list[float]] = []
        pre_violations = 0
        post_violations = 0
        radius = self.config.manifold_radius

        for vector in aggregated:
            norm = _l2(vector)
            if norm > radius:
                pre_violations += 1
                scale = radius / norm
                value = [component * scale for component in vector]
            else:
                value = list(vector)
            if _l2(value) > radius + 1.0e-12:
                post_violations += 1
            projected.append(value)
            v_proj.append(
                [
                    (value[j] - vector[j]) / self.config.dt
                    for j in range(self.config.state_dim)
                ]
            )

        return projected, v_proj, pre_violations, post_violations

    def _decode_model_sketches(
        self, states: Sequence[Sequence[float]]
    ) -> list[list[float]]:
        return [
            [
                math.tanh(_dot(row, state) + bias)
                for row, bias in zip(self.w_decode, self.b_decode)
            ]
            for state in states
        ]

    def step(self) -> KineticCycleReceipt:
        old_states = [list(node.state) for node in self.nodes]
        latent = self._encode()
        raw, errors, v_cog = self._raw_proposals(latent, old_states)
        aggregated, v_cons = self._consensus(raw, errors)
        projected, v_proj, pre_violations, post_violations = self._project_manifold(
            aggregated
        )

        total_speeds: list[float] = []
        cognitive_speeds: list[float] = []
        consensus_speeds: list[float] = []
        projection_speeds: list[float] = []
        balance_error = 0.0
        converged_nodes = 0

        for index, node in enumerate(self.nodes):
            total_velocity = [
                v_cog[index][j] + v_cons[index][j] + v_proj[index][j]
                for j in range(self.config.state_dim)
            ]
            observed_velocity = [
                (projected[index][j] - old_states[index][j]) / self.config.dt
                for j in range(self.config.state_dim)
            ]
            balance_error = max(
                balance_error,
                max(
                    abs(observed_velocity[j] - total_velocity[j])
                    for j in range(self.config.state_dim)
                ),
            )

            speed = _l2(total_velocity)
            total_speeds.append(speed)
            cognitive_speeds.append(_l2(v_cog[index]))
            consensus_speeds.append(_l2(v_cons[index]))
            projection_speeds.append(_l2(v_proj[index]))
            if speed <= self.config.velocity_tolerance:
                converged_nodes += 1

            beta = self.config.memory_beta
            node.memory = [
                beta * node.memory[j] + (1.0 - beta) * projected[index][j]
                for j in range(self.config.state_dim)
            ]
            node.state = list(projected[index])

        reconstructed = self._decode_model_sketches(projected)
        model_mse = sum(
            _mse(node.model_sketch, estimate)
            for node, estimate in zip(self.nodes, reconstructed)
        ) / len(self.nodes)

        self.cycle += 1
        resident_scalars = self.config.active_nodes * (
            self.config.model_sketch_dim + 2 * self.config.state_dim
        )
        receipt = KineticCycleReceipt(
            cycle=self.cycle,
            logical_side=VIRTUAL_SIDE,
            logical_sites=LOGICAL_SITES,
            logical_parameters_per_site=LOGICAL_PARAMETERS_PER_SITE,
            logical_parameter_positions=LOGICAL_PARAMETER_POSITIONS,
            active_nodes=self.config.active_nodes,
            active_edges=self.active_edges,
            resident_scalars=resident_scalars,
            mean_prediction_error=sum(errors) / len(errors),
            max_prediction_error=max(errors),
            mean_cognitive_speed=sum(cognitive_speeds) / len(cognitive_speeds),
            mean_consensus_speed=sum(consensus_speeds) / len(consensus_speeds),
            mean_projection_speed=sum(projection_speeds) / len(projection_speeds),
            mean_total_speed=sum(total_speeds) / len(total_speeds),
            kinetic_balance_error=balance_error,
            pre_projection_violations=pre_violations,
            post_projection_violations=post_violations,
            converged_nodes=converged_nodes,
            model_sketch_mse=model_mse,
            mean_memory_norm=sum(_l2(node.memory) for node in self.nodes) / len(self.nodes),
        )
        self.receipts.append(receipt)
        return receipt

    def run(self, cycles: int) -> list[KineticCycleReceipt]:
        if cycles < 0:
            raise ValueError("cycles must be non-negative")
        return [self.step() for _ in range(cycles)]

    def state_digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(str(self.cycle).encode("ascii"))
        for node in self.nodes:
            for coordinate in node.address:
                digest.update(coordinate.to_bytes(5, "big"))
            for value in node.model_sketch + node.state + node.memory:
                digest.update(struct.pack("!d", value))
        return digest.hexdigest()


def demo(cycles: int = 8) -> None:
    engine = DrMoagiTrillion3KineticSwarm()
    print("Dr Moagi 10^12 x 10^12 x 10^12 Kinetic Hyper-Swarm")
    print("logical side               : 10^12")
    print("logical sites              : 10^36")
    print("logical parameters/site    : 10^12")
    print("logical parameter positions: 10^48")
    print(f"materialized nodes         : {engine.config.active_nodes}")
    print(f"active undirected edges    : {engine.active_edges}")
    for receipt in engine.run(cycles):
        print(
            f"cycle={receipt.cycle:03d} "
            f"err={receipt.mean_prediction_error:.6e} "
            f"vcog={receipt.mean_cognitive_speed:.6e} "
            f"vcons={receipt.mean_consensus_speed:.6e} "
            f"vproj={receipt.mean_projection_speed:.6e} "
            f"v={receipt.mean_total_speed:.6e} "
            f"balance={receipt.kinetic_balance_error:.3e} "
            f"viol={receipt.post_projection_violations} "
            f"mse={receipt.model_sketch_mse:.6e}"
        )


if __name__ == "__main__":
    demo()
