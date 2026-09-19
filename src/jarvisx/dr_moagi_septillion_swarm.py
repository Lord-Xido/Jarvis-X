"""Dr Moagi septillion-cubed recursive swarm autoencoder reference runtime.

The logical geometry is 10^24 sites per axis (10^72 possible addresses). Only
``active_agents`` are materialized. The runtime therefore separates logical
address-space extent from resident state and measured execution cost.

Operational invariant::

    materialize -> encode -> inward fixed-point refine -> decode -> contrast
    -> correct -> verify candidate parameter update -> remember -> evolve
    -> re-ingest -> recur

This is a deterministic software reference. It does not allocate 10^72 cells,
claim physical septillion-scale hardware, or infer trained-model capability from
virtual extent alone.
"""

from __future__ import annotations

import hashlib
import math
import random
import struct
from dataclasses import dataclass
from typing import Sequence

SEPTILLION = 10**24
VIRTUAL_SIDE = SEPTILLION
LOGICAL_SITES = SEPTILLION**3


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _mean(vectors: Sequence[Sequence[float]], width: int) -> list[float]:
    if not vectors:
        return [0.0] * width
    inv = 1.0 / len(vectors)
    return [sum(row[j] for row in vectors) * inv for j in range(width)]


def _mse(targets: Sequence[Sequence[float]], predictions: Sequence[Sequence[float]]) -> float:
    count = 0
    total = 0.0
    for target, prediction in zip(targets, predictions):
        for t, p in zip(target, prediction):
            delta = t - p
            total += delta * delta
            count += 1
    return total / max(1, count)


def _vector_l2(values: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def virtual_coordinate(index: int, axis: int, seed: int = 7) -> int:
    """Map an agent/axis tuple deterministically into ``[0, 10^24)``."""
    if index < 0:
        raise ValueError("index must be non-negative")
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2")
    payload = f"jarvisx:septillion3:{seed}:{index}:{axis}".encode("ascii")
    digest = hashlib.blake2b(payload, digest_size=16).digest()
    return int.from_bytes(digest, "big") % VIRTUAL_SIDE


def chart_coordinate(value: int) -> float:
    """Project an authoritative integer coordinate into the local ``[-1, 1]`` chart."""
    if not 0 <= value < VIRTUAL_SIDE:
        raise ValueError("coordinate outside septillion-axis domain")
    return 2.0 * (value / (VIRTUAL_SIDE - 1)) - 1.0


@dataclass(frozen=True)
class SeptillionSwarmConfig:
    active_agents: int = 128
    feature_dim: int = 4
    latent_dim: int = 8
    hidden_contractivity: float = 0.55
    collective_gain: float = 0.16
    memory_gain: float = 0.08
    relaxation: float = 0.40
    max_refine_steps: int = 32
    fixed_point_tolerance: float = 1.0e-6
    memory_beta: float = 0.92
    velocity_damping: float = 0.94
    geometric_gain: float = 0.025
    latent_motion_gain: float = 0.005
    correction_gain: float = 0.40
    recurrence_gain: float = 0.15
    learning_rate: float = 0.02
    seed: int = 7

    def __post_init__(self) -> None:
        if self.active_agents <= 0:
            raise ValueError("active_agents must be positive")
        if self.feature_dim != 4:
            raise ValueError("reference procedural feature map requires feature_dim=4")
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if not 0.0 < self.hidden_contractivity < 1.0:
            raise ValueError("hidden_contractivity must be in (0, 1)")
        if self.hidden_contractivity + self.collective_gain >= 1.0:
            raise ValueError("latent self + collective gain must remain below 1")
        for name, value in (
            ("collective_gain", self.collective_gain),
            ("memory_gain", self.memory_gain),
            ("correction_gain", self.correction_gain),
            ("recurrence_gain", self.recurrence_gain),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not 0.0 < self.relaxation <= 1.0:
            raise ValueError("relaxation must be in (0, 1]")
        if self.max_refine_steps <= 0:
            raise ValueError("max_refine_steps must be positive")
        if self.fixed_point_tolerance < 0.0:
            raise ValueError("fixed_point_tolerance must be non-negative")
        if not 0.0 <= self.memory_beta < 1.0:
            raise ValueError("memory_beta must be in [0, 1)")
        if not 0.0 <= self.velocity_damping <= 1.0:
            raise ValueError("velocity_damping must be in [0, 1]")
        if self.geometric_gain < 0.0 or self.latent_motion_gain < 0.0:
            raise ValueError("geometric gains must be non-negative")
        if self.learning_rate < 0.0:
            raise ValueError("learning_rate must be non-negative")


@dataclass
class SwarmAgent:
    address: tuple[int, int, int]
    position: list[float]
    velocity: list[float]
    features: list[float]


@dataclass(frozen=True)
class SwarmCycleReceipt:
    cycle: int
    logical_side: int
    logical_sites: int
    active_agents: int
    resident_scalars: int
    pre_adaptation_mse: float
    post_adaptation_mse: float
    corrected_mse: float
    cycle_error: float
    fixed_point_residual: float
    physical_refine_steps: int
    converged: bool
    adaptation_accepted: bool
    swarm_radius: float


class DrMoagiSeptillionSwarmEngine:
    """Bounded AE/AD swarm over a procedurally addressed septillion^3 domain."""

    def __init__(self, config: SeptillionSwarmConfig | None = None) -> None:
        self.config = config or SeptillionSwarmConfig()
        self.rng = random.Random(self.config.seed)
        self.state_dim = 6 + self.config.feature_dim
        self.agents = self._materialize_agents()
        self.w_enc = self._random_matrix(self.config.latent_dim, self.state_dim)
        self.b_enc = [0.0] * self.config.latent_dim
        self.w_latent = self._contractive_matrix(self.config.latent_dim)
        self.w_dec = self._random_matrix(self.state_dim, self.config.latent_dim)
        self.b_dec = [0.0] * self.state_dim
        self.omega = [0.0] * self.config.latent_dim
        self.cycle = 0
        self.receipts: list[SwarmCycleReceipt] = []

    def _random_matrix(self, rows: int, cols: int) -> list[list[float]]:
        scale = 1.0 / math.sqrt(max(1, cols))
        return [
            [self.rng.uniform(-scale, scale) for _ in range(cols)]
            for _ in range(rows)
        ]

    def _contractive_matrix(self, width: int) -> list[list[float]]:
        rows: list[list[float]] = []
        target = self.config.hidden_contractivity
        for _ in range(width):
            row = [self.rng.uniform(-1.0, 1.0) for _ in range(width)]
            norm = sum(abs(value) for value in row) or 1.0
            rows.append([target * value / norm for value in row])
        return rows

    def _materialize_agents(self) -> list[SwarmAgent]:
        agents: list[SwarmAgent] = []
        for index in range(self.config.active_agents):
            address = tuple(virtual_coordinate(index, axis, self.config.seed) for axis in range(3))
            position = [chart_coordinate(value) for value in address]
            velocity = [0.0, 0.0, 0.0]
            features = self._procedural_features(position, cycle=0)
            agents.append(SwarmAgent(address, position, velocity, features))
        return agents

    @staticmethod
    def _procedural_features(position: Sequence[float], cycle: int) -> list[float]:
        x, y, z = position
        phase = cycle * 0.01
        radius = math.sqrt(x * x + y * y + z * z)
        return [
            math.sin(math.pi * x + phase),
            math.cos(math.pi * y - phase),
            math.sin(math.pi * z + 0.5 * phase),
            math.tanh(radius),
        ]

    def _state_vectors(self) -> list[list[float]]:
        return [agent.position + agent.velocity + agent.features for agent in self.agents]

    def _encode_vectors(self, states: Sequence[Sequence[float]]) -> list[list[float]]:
        return [
            [math.tanh(_dot(row, state) + bias) for row, bias in zip(self.w_enc, self.b_enc)]
            for state in states
        ]

    def _refine(
        self, latent: Sequence[Sequence[float]]
    ) -> tuple[list[list[float]], float, int, bool]:
        z = [list(row) for row in latent]
        residual = math.inf
        converged = False
        steps = 0
        for steps in range(1, self.config.max_refine_steps + 1):
            collective = _mean(z, self.config.latent_dim)
            z_next: list[list[float]] = []
            residual = 0.0
            for vector in z:
                target = []
                for j, row in enumerate(self.w_latent):
                    value = (
                        _dot(row, vector)
                        + self.config.collective_gain * collective[j]
                        + self.config.memory_gain * self.omega[j]
                    )
                    target.append(math.tanh(value))
                refined = [
                    (1.0 - self.config.relaxation) * old
                    + self.config.relaxation * new
                    for old, new in zip(vector, target)
                ]
                residual = max(
                    residual,
                    max(abs(new - old) for old, new in zip(vector, refined)),
                )
                z_next.append(refined)
            z = z_next
            if residual <= self.config.fixed_point_tolerance:
                converged = True
                break
        return z, residual, steps, converged

    def _decode_vectors(self, latent: Sequence[Sequence[float]]) -> list[list[float]]:
        return [
            [math.tanh(_dot(row, vector) + bias) for row, bias in zip(self.w_dec, self.b_dec)]
            for vector in latent
        ]

    def _candidate_decoder_update(
        self,
        states: Sequence[Sequence[float]],
        latent: Sequence[Sequence[float]],
        decoded: Sequence[Sequence[float]],
    ) -> tuple[list[list[float]], list[float]]:
        grad_w = [[0.0] * self.config.latent_dim for _ in range(self.state_dim)]
        grad_b = [0.0] * self.state_dim
        normalizer = 2.0 / max(1, len(states) * self.state_dim)
        for state, z, prediction in zip(states, latent, decoded):
            for j in range(self.state_dim):
                derivative = (
                    -normalizer
                    * (state[j] - prediction[j])
                    * (1.0 - prediction[j] ** 2)
                )
                grad_b[j] += derivative
                for k in range(self.config.latent_dim):
                    grad_w[j][k] += derivative * z[k]
        candidate_w = [
            [weight - self.config.learning_rate * grad for weight, grad in zip(row, grow)]
            for row, grow in zip(self.w_dec, grad_w)
        ]
        candidate_b = [
            bias - self.config.learning_rate * grad
            for bias, grad in zip(self.b_dec, grad_b)
        ]
        return candidate_w, candidate_b

    @staticmethod
    def _decode_with(
        latent: Sequence[Sequence[float]],
        weights: Sequence[Sequence[float]],
        bias: Sequence[float],
    ) -> list[list[float]]:
        return [
            [math.tanh(_dot(row, vector) + b) for row, b in zip(weights, bias)]
            for vector in latent
        ]

    def _update_memory(self, latent: Sequence[Sequence[float]]) -> None:
        aggregate = _mean(latent, self.config.latent_dim)
        beta = self.config.memory_beta
        self.omega = [
            beta * old + (1.0 - beta) * new
            for old, new in zip(self.omega, aggregate)
        ]

    def _evolve_agents(
        self,
        corrected: Sequence[Sequence[float]],
        latent: Sequence[Sequence[float]],
    ) -> None:
        centroid = _mean([agent.position for agent in self.agents], 3)
        a = self.config.recurrence_gain
        for agent, candidate, z in zip(self.agents, corrected, latent):
            dynamic_velocity = [
                self.config.velocity_damping * agent.velocity[j]
                + self.config.geometric_gain * (centroid[j] - agent.position[j])
                + self.config.latent_motion_gain * math.tanh(z[j % self.config.latent_dim])
                for j in range(3)
            ]
            dynamic_position = [
                math.tanh(agent.position[j] + dynamic_velocity[j])
                for j in range(3)
            ]
            decoded_position = candidate[0:3]
            decoded_velocity = candidate[3:6]
            next_position = [
                (1.0 - a) * dynamic_position[j] + a * math.tanh(decoded_position[j])
                for j in range(3)
            ]
            next_velocity = [
                (1.0 - a) * dynamic_velocity[j] + a * decoded_velocity[j]
                for j in range(3)
            ]
            procedural = self._procedural_features(next_position, self.cycle + 1)
            decoded_features = candidate[6:]
            next_features = [
                (1.0 - a) * procedural[j] + a * decoded_features[j]
                for j in range(self.config.feature_dim)
            ]
            agent.position = next_position
            agent.velocity = next_velocity
            agent.features = next_features

    def step(self) -> SwarmCycleReceipt:
        states = self._state_vectors()
        latent_initial = self._encode_vectors(states)
        latent, fp_residual, refine_steps, converged = self._refine(latent_initial)
        decoded = self._decode_vectors(latent)
        pre_mse = _mse(states, decoded)

        residual = [
            [target - value for target, value in zip(state, prediction)]
            for state, prediction in zip(states, decoded)
        ]
        corrected = [
            [
                value + self.config.correction_gain * delta
                for value, delta in zip(prediction, row)
            ]
            for prediction, row in zip(decoded, residual)
        ]
        corrected_mse = _mse(states, corrected)

        cycle_latent = self._encode_vectors(corrected)
        cycle_error = _mse(latent, cycle_latent)

        candidate_w, candidate_b = self._candidate_decoder_update(states, latent, decoded)
        candidate_decoded = self._decode_with(latent, candidate_w, candidate_b)
        candidate_mse = _mse(states, candidate_decoded)
        accepted = math.isfinite(candidate_mse) and candidate_mse <= pre_mse + 1.0e-15
        if accepted:
            self.w_dec = candidate_w
            self.b_dec = candidate_b
            post_mse = candidate_mse
        else:
            post_mse = pre_mse

        self._update_memory(latent)
        self._evolve_agents(corrected, latent)
        self.cycle += 1

        radius = sum(_vector_l2(agent.position) for agent in self.agents) / len(self.agents)
        resident_scalars = self.config.active_agents * (
            3 + 3 + self.config.feature_dim + self.config.latent_dim
        )
        receipt = SwarmCycleReceipt(
            cycle=self.cycle,
            logical_side=VIRTUAL_SIDE,
            logical_sites=LOGICAL_SITES,
            active_agents=self.config.active_agents,
            resident_scalars=resident_scalars,
            pre_adaptation_mse=pre_mse,
            post_adaptation_mse=post_mse,
            corrected_mse=corrected_mse,
            cycle_error=cycle_error,
            fixed_point_residual=fp_residual,
            physical_refine_steps=refine_steps,
            converged=converged,
            adaptation_accepted=accepted,
            swarm_radius=radius,
        )
        self.receipts.append(receipt)
        return receipt

    def run(self, cycles: int) -> list[SwarmCycleReceipt]:
        if cycles < 0:
            raise ValueError("cycles must be non-negative")
        return [self.step() for _ in range(cycles)]

    def state_digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(str(self.cycle).encode("ascii"))
        for agent in self.agents:
            for coordinate in agent.address:
                digest.update(coordinate.to_bytes(10, "big"))
            for value in agent.position + agent.velocity + agent.features:
                digest.update(struct.pack("!d", value))
        for value in self.omega:
            digest.update(struct.pack("!d", value))
        for row in self.w_dec:
            for value in row:
                digest.update(struct.pack("!d", value))
        for value in self.b_dec:
            digest.update(struct.pack("!d", value))
        return digest.hexdigest()


def demo(cycles: int = 8) -> None:
    engine = DrMoagiSeptillionSwarmEngine()
    print("Dr Moagi Septillion^3 Recursive Swarm AE/AD")
    print("logical side  : 10^24")
    print("logical sites : 10^72")
    print(f"active agents : {engine.config.active_agents}")
    for receipt in engine.run(cycles):
        print(
            f"cycle={receipt.cycle:03d} "
            f"mse={receipt.pre_adaptation_mse:.8f} "
            f"adapted={receipt.post_adaptation_mse:.8f} "
            f"fp={receipt.fixed_point_residual:.3e} "
            f"k={receipt.physical_refine_steps:02d} "
            f"accept={int(receipt.adaptation_accepted)} "
            f"radius={receipt.swarm_radius:.6f}"
        )


if __name__ == "__main__":
    demo()
