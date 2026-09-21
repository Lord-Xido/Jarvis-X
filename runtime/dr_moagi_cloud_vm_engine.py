#!/usr/bin/env python3
"""Executable reference model for the Dr Moagi Cloud VM master equation.

The runtime models a distributed product state with local auto-encoding,
residual-driven memory, bounded gossip, shadow-tested parameter candidates,
constraint projection, transactional commit/rollback, and a SHA-384 journal.

It is intentionally dependency-free and deterministic so the mechanics can be
inspected, tested, and replayed without pretending to be a trained neural model.
"""

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

Vector = List[float]


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def mse(values: Sequence[float]) -> float:
    return mean([value * value for value in values])


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def stable_chunks(values: Sequence[float], count: int) -> List[List[float]]:
    """Partition values into exactly count deterministic, non-empty-ish chunks."""
    if count <= 0:
        raise ValueError("count must be positive")
    if not values:
        return [[] for _ in range(count)]
    chunks: List[List[float]] = [[] for _ in range(count)]
    for index, value in enumerate(values):
        chunks[index % count].append(float(value))
    return chunks


def morton_code(x: int, y: int, z: int) -> int:
    """Interleave ten low bits of x, y and z into one Morton/Z-order code."""
    code = 0
    for bit in range(10):
        code |= ((x >> bit) & 1) << (3 * bit)
        code |= ((y >> bit) & 1) << (3 * bit + 1)
        code |= ((z >> bit) & 1) << (3 * bit + 2)
    return code


def morton_order(values: Sequence[float]) -> Vector:
    """Embed a flat signal into the smallest cube and return Morton ordering."""
    if not values:
        return []
    side = int(math.ceil(len(values) ** (1.0 / 3.0)))
    cells: List[Tuple[int, float]] = []
    for index, value in enumerate(values):
        x = index % side
        y = (index // side) % side
        z = index // (side * side)
        cells.append((morton_code(x, y, z), float(value)))
    cells.sort(key=lambda item: item[0])
    return [value for _code, value in cells]


@dataclass
class Parameters:
    latent_dim: int = 4
    learning_rate: float = 0.05
    memory_decay: float = 0.80
    gossip_gain: float = 0.25
    predictor_gain: float = 0.75
    latency_weight: float = 0.01
    cost_weight: float = 0.01

    def projected(self) -> "Parameters":
        return Parameters(
            latent_dim=max(1, int(self.latent_dim)),
            learning_rate=clamp(self.learning_rate, 0.0001, 0.25),
            memory_decay=clamp(self.memory_decay, 0.0, 0.999),
            gossip_gain=clamp(self.gossip_gain, 0.0, 1.0),
            predictor_gain=clamp(self.predictor_gain, 0.0, 1.5),
            latency_weight=max(0.0, self.latency_weight),
            cost_weight=max(0.0, self.cost_weight),
        )


@dataclass
class NodeState:
    node_id: str
    coordinates: Tuple[int, int, int]
    x: Vector
    neighbors: List[str]
    parameters: Parameters = field(default_factory=Parameters)
    z: Vector = field(default_factory=list)
    predicted_z: Vector = field(default_factory=list)
    x_hat: Vector = field(default_factory=list)
    error: Vector = field(default_factory=list)
    omega: Vector = field(default_factory=list)
    version: int = 0

    def structural_identity(self) -> Tuple[str, Tuple[int, int, int], Tuple[str, ...], int]:
        return (
            self.node_id,
            self.coordinates,
            tuple(sorted(self.neighbors)),
            len(self.x),
        )


@dataclass
class Telemetry:
    reconstruction_loss: float
    latency_proxy: float
    cost_proxy: float
    objective: float


@dataclass
class JournalEntry:
    cycle: int
    accepted: bool
    previous_objective: float
    candidate_objective: float
    state_digest: str
    previous_chain: str
    chain_digest: str
    parameter_digest: str


class LocalOperators:
    @staticmethod
    def encode(x: Sequence[float], latent_dim: int) -> Vector:
        ordered = morton_order(x)
        return [math.tanh(mean(chunk)) for chunk in stable_chunks(ordered, latent_dim)]

    @staticmethod
    def project_memory(omega: Sequence[float], latent_dim: int) -> Vector:
        return [mean(chunk) for chunk in stable_chunks(omega, latent_dim)]

    @staticmethod
    def predict(z: Sequence[float], omega: Sequence[float], gain: float) -> Vector:
        memory = LocalOperators.project_memory(omega, len(z))
        return [
            math.tanh(gain * z_i + (1.0 - min(gain, 1.0)) * m_i)
            for z_i, m_i in zip(z, memory)
        ]

    @staticmethod
    def decode(predicted_z: Sequence[float], output_dim: int) -> Vector:
        if not predicted_z:
            return [0.0] * output_dim
        return [math.tanh(predicted_z[index % len(predicted_z)]) for index in range(output_dim)]

    @staticmethod
    def residual(x: Sequence[float], x_hat: Sequence[float]) -> Vector:
        return [actual - predicted for actual, predicted in zip(x, x_hat)]

    @staticmethod
    def memory_write(error: Sequence[float], latent_dim: int) -> Vector:
        return [math.tanh(mean(chunk)) for chunk in stable_chunks(error, latent_dim)]

    @classmethod
    def advance(cls, node: NodeState, parameters: Parameters) -> NodeState:
        next_node = copy.deepcopy(node)
        projected = parameters.projected()
        next_node.parameters = projected
        next_node.z = cls.encode(node.x, projected.latent_dim)
        next_node.predicted_z = cls.predict(
            next_node.z,
            node.omega or [0.0] * projected.latent_dim,
            projected.predictor_gain,
        )
        next_node.x_hat = cls.decode(next_node.predicted_z, len(node.x))
        next_node.error = cls.residual(node.x, next_node.x_hat)
        write = cls.memory_write(next_node.error, projected.latent_dim)
        previous = cls.project_memory(node.omega, projected.latent_dim)
        next_node.omega = [
            projected.memory_decay * old + projected.learning_rate * delta
            for old, delta in zip(previous, write)
        ]
        next_node.version = node.version + 1
        return next_node


class CloudVMEngine:
    """Transactional distributed execution of the corrected master equation.

    Sigma_cloud is represented as a keyed product of node states. A cycle:
    1. advances every local auto-encoding loop;
    2. performs bounded neighbour gossip on Omega;
    3. proposes bounded global parameter candidates;
    4. scores candidates on a deterministic shadow cohort;
    5. projects against structural and monotonic invariants;
    6. commits or rolls back atomically and extends a SHA-384 journal.
    """

    def __init__(self, nodes: Iterable[NodeState], monotonic_tolerance: float = 1e-12):
        self.nodes: Dict[str, NodeState] = {
            node.node_id: copy.deepcopy(node) for node in nodes
        }
        if not self.nodes:
            raise ValueError("at least one node is required")
        self._validate_graph(self.nodes)
        self.cycle = 0
        self.monotonic_tolerance = monotonic_tolerance
        self.last_objective = float("inf")
        self.chain = "0" * 96
        self.journal: List[JournalEntry] = []

    @staticmethod
    def _validate_graph(nodes: Dict[str, NodeState]) -> None:
        identities = set()
        for node in nodes.values():
            if node.node_id in identities:
                raise ValueError("duplicate node id")
            identities.add(node.node_id)
            for neighbor in node.neighbors:
                if neighbor not in nodes:
                    raise ValueError("unknown neighbor %s for %s" % (neighbor, node.node_id))

    @staticmethod
    def _state_payload(nodes: Dict[str, NodeState]) -> bytes:
        payload = {
            node_id: {
                "coordinates": list(node.coordinates),
                "x": node.x,
                "z": node.z,
                "predicted_z": node.predicted_z,
                "x_hat": node.x_hat,
                "error": node.error,
                "omega": node.omega,
                "neighbors": sorted(node.neighbors),
                "parameters": asdict(node.parameters),
                "version": node.version,
            }
            for node_id, node in sorted(nodes.items())
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    @staticmethod
    def _telemetry(nodes: Dict[str, NodeState]) -> Telemetry:
        losses = [mse(node.error) for node in nodes.values()]
        latency = mean(
            [len(node.x) + len(node.z) + len(node.neighbors) for node in nodes.values()]
        )
        cost = mean(
            [
                len(node.omega) + sum(abs(value) for value in node.omega)
                for node in nodes.values()
            ]
        )
        parameters = next(iter(nodes.values())).parameters
        loss = mean(losses)
        objective = loss + parameters.latency_weight * latency + parameters.cost_weight * cost
        return Telemetry(loss, latency, cost, objective)

    @staticmethod
    def _gossip(nodes: Dict[str, NodeState]) -> None:
        snapshots = {node_id: list(node.omega) for node_id, node in nodes.items()}
        for node in nodes.values():
            gain = node.parameters.gossip_gain
            if not node.neighbors or gain == 0.0:
                continue
            width = len(node.omega)
            neighbor_mean = [
                mean(
                    [
                        LocalOperators.project_memory(snapshots[neighbor], width)[index]
                        for neighbor in node.neighbors
                    ]
                )
                for index in range(width)
            ]
            node.omega = [
                (1.0 - gain) * local + gain * remote
                for local, remote in zip(node.omega, neighbor_mean)
            ]

    @staticmethod
    def _candidate_parameters(base: Parameters) -> List[Parameters]:
        # Bounded finite-difference candidates approximate -grad J without
        # pretending the discrete cloud topology is fully differentiable.
        deltas = [
            (0.0, 0.0, 0.0),
            (-0.01, 0.0, 0.0),
            (0.01, 0.0, 0.0),
            (0.0, -0.05, 0.0),
            (0.0, 0.05, 0.0),
            (0.0, 0.0, -0.10),
            (0.0, 0.0, 0.10),
        ]
        candidates = []
        for eta_delta, rho_delta, gossip_delta in deltas:
            candidate = copy.deepcopy(base)
            candidate.learning_rate += eta_delta
            candidate.memory_decay += rho_delta
            candidate.gossip_gain += gossip_delta
            candidates.append(candidate.projected())
        return candidates

    @staticmethod
    def _shadow_ids(nodes: Dict[str, NodeState], fraction: float = 0.10) -> List[str]:
        ordered = sorted(nodes)
        count = max(1, int(math.ceil(len(ordered) * fraction)))
        return ordered[:count]

    def _simulate(
        self, source: Dict[str, NodeState], parameters: Parameters
    ) -> Dict[str, NodeState]:
        candidate = {
            node_id: LocalOperators.advance(node, parameters)
            for node_id, node in source.items()
        }
        self._gossip(candidate)
        return candidate

    def _select_parameters(self) -> Parameters:
        base = next(iter(self.nodes.values())).parameters
        shadow_ids = self._shadow_ids(self.nodes)
        scored = []
        for candidate in self._candidate_parameters(base):
            projected_all = self._simulate(self.nodes, candidate)
            projected_shadow = {
                node_id: projected_all[node_id] for node_id in shadow_ids
            }
            scored.append((self._telemetry(projected_shadow).objective, candidate))
        scored.sort(key=lambda item: item[0])
        return scored[0][1]

    def _verify_identity(
        self, previous: Dict[str, NodeState], candidate: Dict[str, NodeState]
    ) -> bool:
        if set(previous) != set(candidate):
            return False
        return all(
            previous[node_id].structural_identity()
            == candidate[node_id].structural_identity()
            for node_id in previous
        )

    def _commit(self, candidate: Dict[str, NodeState], telemetry: Telemetry) -> bool:
        previous_objective = self.last_objective
        identity_ok = self._verify_identity(self.nodes, candidate)
        objective_ok = telemetry.objective <= previous_objective + self.monotonic_tolerance
        accepted = identity_ok and objective_ok
        if accepted:
            self.nodes = candidate
            self.last_objective = telemetry.objective

        state_digest = hashlib.sha384(self._state_payload(self.nodes)).hexdigest()
        parameters = asdict(next(iter(self.nodes.values())).parameters)
        parameter_digest = hashlib.sha384(
            json.dumps(parameters, sort_keys=True).encode("utf-8")
        ).hexdigest()
        record = {
            "cycle": self.cycle,
            "accepted": accepted,
            "previous_objective": previous_objective,
            "candidate_objective": telemetry.objective,
            "state_digest": state_digest,
            "parameter_digest": parameter_digest,
        }
        previous_chain = self.chain
        self.chain = hashlib.sha384(
            bytes.fromhex(previous_chain)
            + json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self.journal.append(
            JournalEntry(
                cycle=self.cycle,
                accepted=accepted,
                previous_objective=previous_objective,
                candidate_objective=telemetry.objective,
                state_digest=state_digest,
                previous_chain=previous_chain,
                chain_digest=self.chain,
                parameter_digest=parameter_digest,
            )
        )
        return accepted

    def step(self) -> Telemetry:
        self.cycle += 1
        selected = self._select_parameters()
        candidate = self._simulate(self.nodes, selected)
        telemetry = self._telemetry(candidate)
        self._commit(candidate, telemetry)
        return telemetry


def demo_nodes() -> List[NodeState]:
    return [
        NodeState(
            "n000", (0, 0, 0), [0.1, 0.4, 0.9, 0.2, 0.7, 0.3], ["n100", "n010"]
        ),
        NodeState(
            "n100", (1, 0, 0), [0.2, 0.5, 0.8, 0.3, 0.6, 0.4], ["n000", "n110"]
        ),
        NodeState(
            "n010", (0, 1, 0), [0.3, 0.6, 0.7, 0.4, 0.5, 0.5], ["n000", "n110"]
        ),
        NodeState(
            "n110", (1, 1, 0), [0.4, 0.7, 0.6, 0.5, 0.4, 0.6], ["n100", "n010"]
        ),
    ]


def main() -> None:
    engine = CloudVMEngine(demo_nodes())
    print("=== Dr Moagi Cloud VM Master Equation ===")
    for _ in range(5):
        telemetry = engine.step()
        entry = engine.journal[-1]
        print(
            "cycle=%d accepted=%s objective=%.8f loss=%.8f chain=%s"
            % (
                engine.cycle,
                entry.accepted,
                telemetry.objective,
                telemetry.reconstruction_loss,
                entry.chain_digest[:16],
            )
        )
    assert engine.journal
    assert all(entry.chain_digest != entry.previous_chain for entry in engine.journal)
    assert engine.nodes["n000"].version >= 1
    print("Permeation complete: distributed state transition verified.")


if __name__ == "__main__":
    main()
