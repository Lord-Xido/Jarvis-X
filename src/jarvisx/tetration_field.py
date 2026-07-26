"""Sparse brick field over a symbolic tetration address manifold."""
from __future__ import annotations

import hashlib
import json
import math
import random
import struct
from dataclasses import asdict, dataclass, replace
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

BASE = 1000
CHANNELS = 3
EDGE = 4
BRICK_SIZE = CHANNELS * EDGE * EDGE * EDGE
FACES = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


def _tower(base: int, height: int) -> str:
    if height == 1:
        return str(base)
    if height == 2:
        return f"{base}^{base}"
    return f"{base}↑↑{height}"


@dataclass(frozen=True)
class TetrationUniverse:
    height: int = 2
    base: int = BASE

    def __post_init__(self) -> None:
        if self.base < 2 or self.height < 1:
            raise ValueError("base >= 2 and height >= 1 are required")

    @property
    def coordinate_bits_if_materialisable(self) -> Optional[int]:
        if self.height == 1:
            return 3 * math.ceil(math.log2(self.base))
        if self.height == 2:
            return 3 * math.ceil(self.base * math.log2(self.base))
        return None

    @property
    def axis_expression(self) -> str:
        return _tower(self.base, self.height)

    def descriptor(self) -> Dict[str, object]:
        axis = _tower(self.base, self.height)
        if self.height == 1:
            bits = self.coordinate_bits_if_materialisable
            bits_expr = str(bits)
        elif self.height == 2:
            bits = self.coordinate_bits_if_materialisable
            bits_expr = str(bits)
        else:
            bits = None
            bits_expr = f"3*ceil(({_tower(self.base, self.height - 1)})*log2({self.base}))"
        return {
            "base": self.base,
            "tower_height": self.height,
            "axis_size": axis,
            "virtual_cells": f"({axis})^3",
            "coordinate_bits": bits_expr,
            "coordinate_bits_materialised": bits,
            "storage_model": "symbolic-address-manifold/sparse-collision-chained-field",
        }


@dataclass(frozen=True, order=True)
class TetrationAddress:
    tower_height: int
    chart: str
    x: int
    y: int
    z: int

    def __post_init__(self) -> None:
        if self.tower_height < 1 or not self.chart:
            raise ValueError("tower_height must be positive and chart non-empty")
        if not all(isinstance(v, int) for v in (self.x, self.y, self.z)):
            raise TypeError("x, y and z must be integers")

    def offset(self, dx: int = 0, dy: int = 0, dz: int = 0) -> "TetrationAddress":
        return TetrationAddress(self.tower_height, self.chart, self.x + dx, self.y + dy, self.z + dz)

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            [self.tower_height, self.chart, str(self.x), str(self.y), str(self.z)],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")


@dataclass(frozen=True)
class BrickState:
    values: Tuple[float, ...]
    omega: Tuple[float, ...]
    inactive_steps: int = 0
    revision: int = 0
    expert_index: int = 0

    def __post_init__(self) -> None:
        if len(self.values) != BRICK_SIZE or len(self.omega) != BRICK_SIZE:
            raise ValueError(f"values and omega must each contain {BRICK_SIZE} elements")


class SparseHashDirectory:
    """Finite hash table using explicit collision chains."""

    def __init__(self, bucket_count: int = 257) -> None:
        if bucket_count < 1:
            raise ValueError("bucket_count must be positive")
        self.bucket_count = bucket_count
        self._buckets: List[List[Tuple[TetrationAddress, BrickState]]] = [
            [] for _ in range(bucket_count)
        ]
        self._size = 0

    def _get_allocated_index(self, address: TetrationAddress) -> int:
        chart = int.from_bytes(
            hashlib.blake2b(address.chart.encode("utf-8"), digest_size=8).digest(), "big"
        )
        mixed = (
            address.x * 5147
            ^ address.y * 9293
            ^ address.z * 11257
            ^ address.tower_height * 131071
            ^ chart
        )
        return mixed % self.bucket_count

    def get(self, address: TetrationAddress) -> Optional[BrickState]:
        for key, value in self._buckets[self._get_allocated_index(address)]:
            if key == address:
                return value
        return None

    def set(self, address: TetrationAddress, state: BrickState) -> None:
        bucket = self._buckets[self._get_allocated_index(address)]
        for i, (key, _) in enumerate(bucket):
            if key == address:
                bucket[i] = (address, state)
                return
        bucket.append((address, state))
        self._size += 1

    def items(self) -> Iterator[Tuple[TetrationAddress, BrickState]]:
        for bucket in self._buckets:
            yield from bucket

    def to_dict(self) -> Dict[TetrationAddress, BrickState]:
        return dict(self.items())

    @classmethod
    def from_mapping(
        cls, mapping: Mapping[TetrationAddress, BrickState], bucket_count: int
    ) -> "SparseHashDirectory":
        out = cls(bucket_count)
        for address in sorted(mapping):
            out.set(address, mapping[address])
        return out

    def collision_count(self) -> int:
        return sum(max(0, len(bucket) - 1) for bucket in self._buckets)

    def __len__(self) -> int:
        return self._size


@dataclass(frozen=True)
class FieldMechanics:
    diffusion: float = 0.04
    error_gain: float = 0.45
    omega_retention: float = 0.90
    omega_rate: float = 0.08
    time_step: float = 0.10
    activation_threshold: float = 0.50
    prune_after: int = 6
    max_active_bricks: int = 128
    clip_min: float = 16.0
    clip_max: float = 235.0
    omega_limit: float = 512.0
    max_energy: float = 1.0e10

    def validate(self) -> None:
        values = tuple(asdict(self).values())
        if not all(math.isfinite(float(v)) for v in values):
            raise ValueError("mechanics must be finite")
        if self.diffusion < 0 or self.error_gain < 0 or self.omega_rate < 0:
            raise ValueError("D, K and eta must be non-negative")
        if not 0 < self.omega_retention < 1:
            raise ValueError("omega_retention must satisfy 0 < rho < 1")
        if self.time_step <= 0 or self.time_step * self.diffusion > 1 / 6:
            raise ValueError("explicit six-face diffusion requires dt > 0 and dt*D <= 1/6")
        if self.time_step * self.error_gain > 1:
            raise ValueError("residual feedback requires dt*K <= 1")
        if self.activation_threshold <= 0 or self.prune_after < 1 or self.max_active_bricks < 1:
            raise ValueError("threshold, prune_after and max_active_bricks must be positive")
        if self.clip_min >= self.clip_max or self.omega_limit <= 0 or self.max_energy <= 0:
            raise ValueError("invalid projection or energy bounds")


@dataclass(frozen=True)
class FieldStepMetrics:
    cycle: int
    materialised_bricks: int
    active_bricks: int
    frontier_bricks: int
    reconstruction_mse: float
    relative_energy: float
    expert_histogram: Tuple[int, ...]
    collisions: int
    committed: bool
    journal_hash: str
    rollback_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        out = asdict(self)
        out["expert_histogram"] = list(self.expert_histogram)
        return out


class BrickAutoencoderMoE:
    """Full 192→d→192 projection with Omega conditioning and softmax/top-1 routing."""

    def __init__(
        self,
        latent_dim: int = 16,
        expert_count: int = 4,
        seed: int = 1337,
        clip_min: float = 16.0,
        clip_max: float = 235.0,
    ) -> None:
        if latent_dim < 2 or expert_count < 1 or clip_min >= clip_max:
            raise ValueError("invalid network dimensions or projection box")
        self.latent_dim = latent_dim
        self.expert_count = expert_count
        self.mid = (clip_min + clip_max) / 2
        self.half = (clip_max - clip_min) / 2
        rng = random.Random(seed)
        enc = 1 / math.sqrt(BRICK_SIZE)
        om = 0.35 / math.sqrt(BRICK_SIZE)
        ex = 0.20 / math.sqrt(latent_dim)
        rt = 1 / math.sqrt(latent_dim)
        self.encoder = tuple(
            tuple(rng.uniform(-enc, enc) for _ in range(BRICK_SIZE))
            for _ in range(latent_dim)
        )
        self.encoder_bias = tuple(rng.uniform(-0.02, 0.02) for _ in range(latent_dim))
        self.omega_projection = tuple(
            tuple(rng.uniform(-om, om) for _ in range(BRICK_SIZE))
            for _ in range(latent_dim)
        )
        self.router = tuple(
            tuple(rng.uniform(-rt, rt) for _ in range(latent_dim))
            for _ in range(expert_count)
        )
        self.experts = tuple(
            tuple(
                tuple(rng.uniform(-ex, ex) for _ in range(latent_dim))
                for _ in range(latent_dim)
            )
            for _ in range(expert_count)
        )
        self.expert_bias = tuple(
            tuple(rng.uniform(-0.01, 0.01) for _ in range(latent_dim))
            for _ in range(expert_count)
        )
        norms = [
            max(sum(self.encoder[i][j] ** 2 for i in range(latent_dim)), 1e-9)
            for j in range(BRICK_SIZE)
        ]
        self.decoder = tuple(
            tuple(self.encoder[i][j] / norms[j] for i in range(latent_dim))
            for j in range(BRICK_SIZE)
        )

    @staticmethod
    def _dot(row: Sequence[float], vector: Sequence[float]) -> float:
        return sum(a * b for a, b in zip(row, vector))

    def encode(self, values: Sequence[float]) -> Tuple[float, ...]:
        if len(values) != BRICK_SIZE:
            raise ValueError(f"brick must contain {BRICK_SIZE} values")
        flat = tuple((float(v) - self.mid) / self.half for v in values)
        return tuple(
            math.tanh(self._dot(row, flat) + bias)
            for row, bias in zip(self.encoder, self.encoder_bias)
        )

    def condition_with_omega(
        self, latent: Sequence[float], omega: Sequence[float]
    ) -> Tuple[float, ...]:
        if len(latent) != self.latent_dim or len(omega) != BRICK_SIZE:
            raise ValueError("invalid latent or omega shape")
        omega_flat = tuple(float(v) / self.half for v in omega)
        return tuple(
            math.tanh(latent[i] + self._dot(self.omega_projection[i], omega_flat))
            for i in range(self.latent_dim)
        )

    def route(self, conditioned: Sequence[float]) -> Tuple[int, Tuple[float, ...]]:
        if len(conditioned) != self.latent_dim:
            raise ValueError("conditioned latent state has the wrong dimension")
        logits = tuple(self._dot(row, conditioned) for row in self.router)
        peak = max(logits)
        exp = tuple(math.exp(v - peak) for v in logits)
        total = sum(exp)
        gates = tuple(v / total for v in exp)
        expert = max(range(self.expert_count), key=lambda i: (gates[i], -i))
        return expert, gates

    def decode(self, latent: Sequence[float]) -> Tuple[float, ...]:
        if len(latent) != self.latent_dim:
            raise ValueError("latent state has the wrong dimension")
        return tuple(self.mid + self.half * self._dot(row, latent) for row in self.decoder)

    def forward(
        self, values: Sequence[float], omega: Sequence[float]
    ) -> Tuple[Tuple[float, ...], int, Tuple[float, ...]]:
        if len(values) != BRICK_SIZE or len(omega) != BRICK_SIZE:
            raise ValueError(f"brick and omega must each contain {BRICK_SIZE} values")
        z = self.encode(values)
        conditioned = self.condition_with_omega(z, omega)
        expert, gates = self.route(conditioned)
        evolved = tuple(
            math.tanh(
                conditioned[i]
                + self._dot(self.experts[expert][i], conditioned)
                + self.expert_bias[expert][i]
            )
            for i in range(self.latent_dim)
        )
        decoded = self.decode(evolved)
        return decoded, expert, gates


Observation = Union[float, Sequence[float]]


class TetrationFieldAutomaton:
    def __init__(
        self,
        universe: Optional[TetrationUniverse] = None,
        mechanics: Optional[FieldMechanics] = None,
        latent_dim: int = 16,
        expert_count: int = 4,
        seed: int = 1337,
        bucket_count: int = 257,
    ) -> None:
        self.universe = universe or TetrationUniverse()
        self.mechanics = mechanics or FieldMechanics()
        self.mechanics.validate()
        self.seed = seed
        self.directory = SparseHashDirectory(bucket_count)
        self.network = BrickAutoencoderMoE(
            latent_dim, expert_count, seed, self.mechanics.clip_min, self.mechanics.clip_max
        )
        self._background: Dict[TetrationAddress, Tuple[float, ...]] = {}
        self.cycle = 0
        self.journal_hash = "0" * 64
        self.last_metrics = self._metrics(True)

    def _metrics(
        self,
        committed: bool,
        frontier: int = 0,
        mse: float = 0.0,
        energy: float = 0.0,
        experts: Optional[Tuple[int, ...]] = None,
        reason: Optional[str] = None,
    ) -> FieldStepMetrics:
        return FieldStepMetrics(
            self.cycle,
            len(self.directory),
            len(self.active_addresses()),
            frontier,
            mse,
            energy,
            experts or tuple(0 for _ in range(self.network.expert_count)),
            self.directory.collision_count(),
            committed,
            self.journal_hash,
            reason,
        )

    def _check_address(self, address: TetrationAddress) -> None:
        if address.tower_height != self.universe.height:
            raise ValueError("address tower height does not match universe")

    def procedural_brick(self, address: TetrationAddress) -> Tuple[float, ...]:
        self._check_address(address)
        if address not in self._background:
            digest = hashlib.blake2b(
                address.canonical_bytes(), key=str(self.seed).encode(), digest_size=32
            ).digest()
            self._background[address] = tuple(
                self.mechanics.clip_min + 0.25 + ((digest[i % 32] / 255) - 0.5) * 0.5
                for i in range(BRICK_SIZE)
            )
        return self._background[address]

    @staticmethod
    def flat_index(channel: int, x: int, y: int, z: int) -> int:
        return (((channel * EDGE + z) * EDGE + y) * EDGE + x)

    _flat_index = flat_index

    def _resolve(
        self, address: TetrationAddress, x: int, y: int, z: int
    ) -> Tuple[TetrationAddress, int, int, int]:
        dx = dy = dz = 0
        if x < 0:
            x += EDGE
            dx = -1
        elif x >= EDGE:
            x -= EDGE
            dx = 1
        if y < 0:
            y += EDGE
            dy = -1
        elif y >= EDGE:
            y -= EDGE
            dy = 1
        if z < 0:
            z += EDGE
            dz = -1
        elif z >= EDGE:
            z -= EDGE
            dz = 1
        return address.offset(dx, dy, dz), x, y, z

    def _values(
        self, states: Mapping[TetrationAddress, BrickState], address: TetrationAddress
    ) -> Tuple[float, ...]:
        state = states.get(address)
        return state.values if state else self.procedural_brick(address)

    def _voxel(
        self,
        states: Mapping[TetrationAddress, BrickState],
        address: TetrationAddress,
        channel: int,
        x: int,
        y: int,
        z: int,
    ) -> float:
        address, x, y, z = self._resolve(address, x, y, z)
        return self._values(states, address)[self.flat_index(channel, x, y, z)]

    _voxel_value = _voxel

    def _laplacian(
        self,
        states: Mapping[TetrationAddress, BrickState],
        address: TetrationAddress,
        channel: int,
        x: int,
        y: int,
        z: int,
        centre: float,
    ) -> float:
        return (
            self._voxel(states, address, channel, x + 1, y, z)
            + self._voxel(states, address, channel, x - 1, y, z)
            + self._voxel(states, address, channel, x, y + 1, z)
            + self._voxel(states, address, channel, x, y - 1, z)
            + self._voxel(states, address, channel, x, y, z + 1)
            + self._voxel(states, address, channel, x, y, z - 1)
            - 6 * centre
        )

    def _activity(self, address: TetrationAddress, state: BrickState) -> float:
        background = self.procedural_brick(address)
        return max(
            max(abs(v - b) for v, b in zip(state.values, background)),
            max(abs(v) for v in state.omega),
        )

    def active_addresses(
        self, states: Optional[Mapping[TetrationAddress, BrickState]] = None
    ) -> Tuple[TetrationAddress, ...]:
        states = states if states is not None else self.directory.to_dict()
        return tuple(
            sorted(
                a
                for a, s in states.items()
                if self._activity(a, s) >= self.mechanics.activation_threshold
                or s.inactive_steps == 0
            )
        )

    def _frontier(
        self, states: Mapping[TetrationAddress, BrickState]
    ) -> Tuple[TetrationAddress, ...]:
        active = set(self.active_addresses(states))
        frontier = set(active)
        for address in active:
            frontier.update(address.offset(*face) for face in FACES)
        if len(frontier) <= self.mechanics.max_active_bricks:
            return tuple(sorted(frontier))
        ranked = sorted(
            frontier,
            key=lambda a: (
                -self._activity(a, states[a]) if a in states else 0.0,
                a,
            ),
        )
        return tuple(sorted(ranked[: self.mechanics.max_active_bricks]))

    def _observation(self, value: Observation) -> Tuple[float, ...]:
        if isinstance(value, (int, float)):
            return tuple(float(value) for _ in range(BRICK_SIZE))
        out = tuple(float(v) for v in value)
        if len(out) != BRICK_SIZE:
            raise ValueError(f"observation must contain {BRICK_SIZE} values")
        return out

    def _inject(
        self,
        states: Dict[TetrationAddress, BrickState],
        injections: Mapping[TetrationAddress, Observation],
    ) -> None:
        for address in sorted(injections):
            self._check_address(address)
            delta = self._observation(injections[address])
            current = states.get(address) or BrickState(
                self.procedural_brick(address), tuple(0.0 for _ in range(BRICK_SIZE))
            )
            values = tuple(
                min(self.mechanics.clip_max, max(self.mechanics.clip_min, v + u))
                for v, u in zip(current.values, delta)
            )
            states[address] = replace(
                current, values=values, inactive_steps=0, revision=current.revision + 1
            )

    def _verify(
        self, candidate: Mapping[TetrationAddress, BrickState]
    ) -> Tuple[bool, Optional[str], float]:
        if len(candidate) > self.mechanics.max_active_bricks:
            return False, "active-brick budget exceeded", math.inf
        energy = 0.0
        for address, state in candidate.items():
            self._check_address(address)
            if not 0 <= state.expert_index < self.network.expert_count:
                return False, "expert index out of range", math.inf
            for value, omega, base in zip(state.values, state.omega, self.procedural_brick(address)):
                if not math.isfinite(value) or not math.isfinite(omega):
                    return False, "non-finite field state", math.inf
                if not self.mechanics.clip_min <= value <= self.mechanics.clip_max:
                    return False, "projection bound exceeded", math.inf
                if abs(omega) > self.mechanics.omega_limit:
                    return False, "omega bound exceeded", math.inf
                energy += (value - base) ** 2 + omega**2
        if energy > self.mechanics.max_energy:
            return False, "relative energy budget exceeded", energy
        return True, None, energy

    def _journal(self, states: Mapping[TetrationAddress, BrickState], cycle: int) -> str:
        digest = hashlib.sha256(bytes.fromhex(self.journal_hash))
        digest.update(struct.pack(">Q", cycle))
        digest.update(json.dumps(asdict(self.mechanics), sort_keys=True).encode())
        digest.update(json.dumps(self.universe.descriptor(), sort_keys=True).encode())
        for address in sorted(states):
            state = states[address]
            digest.update(address.canonical_bytes())
            digest.update(struct.pack(">III", state.inactive_steps, state.revision, state.expert_index))
            for value in state.values + state.omega:
                digest.update(struct.pack(">d", value))
        return digest.hexdigest()

    def step(
        self, injections: Optional[Mapping[TetrationAddress, Observation]] = None
    ) -> FieldStepMetrics:
        working = self.directory.to_dict()
        try:
            if injections:
                self._inject(working, injections)
        except (TypeError, ValueError) as exc:
            self.last_metrics = self._metrics(False, reason=str(exc))
            return self.last_metrics
        frontier = self._frontier(working)
        updates: Dict[TetrationAddress, BrickState] = {}
        histogram = [0] * self.network.expert_count
        squared_error = 0.0
        for address in frontier:
            current = working.get(address)
            values = self._values(working, address)
            omega = current.omega if current else tuple(0.0 for _ in range(BRICK_SIZE))
            decoded, expert, _ = self.network.forward(values, omega)
            histogram[expert] += 1
            residual = tuple(p - b for p, b in zip(decoded, values))
            squared_error += sum(e * e for e in residual)
            omega_next = tuple(
                min(
                    self.mechanics.omega_limit,
                    max(
                        -self.mechanics.omega_limit,
                        self.mechanics.omega_retention * o - self.mechanics.omega_rate * e,
                    ),
                )
                for o, e in zip(omega, residual)
            )
            proposal = [0.0] * BRICK_SIZE
            for c in range(CHANNELS):
                for z in range(EDGE):
                    for y in range(EDGE):
                        for x in range(EDGE):
                            i = self.flat_index(c, x, y, z)
                            delta = self._laplacian(working, address, c, x, y, z, values[i])
                            raw = values[i] + self.mechanics.time_step * (
                                self.mechanics.diffusion * delta
                                - self.mechanics.error_gain * residual[i]
                                + omega_next[i]
                            )
                            proposal[i] = min(
                                self.mechanics.clip_max, max(self.mechanics.clip_min, raw)
                            )
            provisional = BrickState(
                tuple(proposal),
                omega_next,
                revision=(current.revision if current else 0) + 1,
                expert_index=expert,
            )
            inactive = (
                0
                if self._activity(address, provisional) >= self.mechanics.activation_threshold
                else (current.inactive_steps + 1 if current else 1)
            )
            updates[address] = replace(provisional, inactive_steps=inactive)

        candidate: Dict[TetrationAddress, BrickState] = {}
        frontier_set = set(frontier)
        for address, state in working.items():
            if address not in frontier_set:
                aged = replace(state, inactive_steps=state.inactive_steps + 1)
                if not (
                    aged.inactive_steps >= self.mechanics.prune_after
                    and self._activity(address, aged) < self.mechanics.activation_threshold
                ):
                    candidate[address] = aged
        for address, state in updates.items():
            if not (
                state.inactive_steps >= self.mechanics.prune_after
                and self._activity(address, state) < self.mechanics.activation_threshold
            ):
                candidate[address] = state
        if len(candidate) > self.mechanics.max_active_bricks:
            candidate = dict(
                sorted(
                    candidate.items(),
                    key=lambda item: (-self._activity(item[0], item[1]), item[0]),
                )[: self.mechanics.max_active_bricks]
            )
        valid, reason, energy = self._verify(candidate)
        mse = squared_error / max(1, len(frontier) * BRICK_SIZE)
        if not valid:
            self.last_metrics = self._metrics(False, len(frontier), mse, energy, reason=reason)
            return self.last_metrics
        self.cycle += 1
        self.journal_hash = self._journal(candidate, self.cycle)
        self.directory = SparseHashDirectory.from_mapping(candidate, self.directory.bucket_count)
        self.last_metrics = self._metrics(
            True, len(frontier), mse, energy, tuple(histogram)
        )
        return self.last_metrics

    def snapshot(self) -> Dict[str, object]:
        return {
            "universe": self.universe.descriptor(),
            "cycle": self.cycle,
            "materialised_bricks": len(self.directory),
            "active_bricks": len(self.active_addresses()),
            "bucket_count": self.directory.bucket_count,
            "collisions": self.directory.collision_count(),
            "brick_shape": [CHANNELS, EDGE, EDGE, EDGE],
            "brick_values": BRICK_SIZE,
            "latent_dim": self.network.latent_dim,
            "expert_count": self.network.expert_count,
            "router": "softmax/top-1",
            "mechanics": asdict(self.mechanics),
            "journal_hash": self.journal_hash,
            "last_metrics": self.last_metrics.to_dict(),
            "cost_model": "O(M_t * (192*d + d^2 + 192*6))",
        }


def make_brick_pulse(amplitude: float = 48.0) -> Tuple[float, ...]:
    if not math.isfinite(amplitude):
        raise ValueError("amplitude must be finite")
    out = [0.0] * BRICK_SIZE
    centre = (EDGE - 1) / 2
    for c in range(CHANNELS):
        for z in range(EDGE):
            for y in range(EDGE):
                for x in range(EDGE):
                    d2 = (x - centre) ** 2 + (y - centre) ** 2 + (z - centre) ** 2
                    out[TetrationFieldAutomaton.flat_index(c, x, y, z)] = (
                        amplitude * (1 - 0.15 * c) * math.exp(-0.75 * d2)
                    )
    return tuple(out)
