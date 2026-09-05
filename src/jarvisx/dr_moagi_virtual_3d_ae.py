"""Sparse virtual 3D bitstream AE/AD with bounded inward auto-optimization.

The enormous logical stream count is a virtual address-space contract. Only a finite
active tile is materialized. The optimizer is deterministic and bounded: it searches
a finite alpha/beta grid and commits a candidate only when the measured objective
improves.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass, replace
from typing import Iterable

Coord = tuple[int, int, int]
LOGICAL_STREAM_COUNT = "(10^24)^(10^24) = 10^(24 * 10^24)"
VIRTUAL_SIDE_LENGTH = "10^(8 * 10^24)"
N6 = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
POSITIVE_3 = ((1, 0, 0), (0, 1, 0), (0, 0, 1))


@dataclass(frozen=True)
class Config:
    tile: int = 8
    bits: int = 256
    latent: int = 32
    passes: int = 5
    alpha: float = 0.65
    beta: float = 0.65
    curvature: float = 1.0
    seed: int = 1337
    origin: Coord = (0, 0, 0)
    periodic: bool = True
    epsilon: float = 1e-3
    alpha_candidates: tuple[float, ...] = (0.55, 0.65, 0.80)
    beta_candidates: tuple[float, ...] = (0.35, 0.50, 0.65, 0.80)
    reconstruction_weight: float = 1.0
    spatial_weight: float = 0.25
    balance_weight: float = 0.25
    stability_weight: float = 0.10

    def __post_init__(self) -> None:
        if self.tile < 1 or self.bits < 2 or not 1 <= self.latent <= self.bits:
            raise ValueError("invalid dimensions")
        if self.passes < 1:
            raise ValueError("passes must be >= 1")
        bounded = (self.alpha, self.beta, self.curvature, *self.alpha_candidates, *self.beta_candidates)
        if any(not 0.0 <= value <= 1.0 for value in bounded):
            raise ValueError("alpha, beta, curvature and candidates must be in [0, 1]")
        if self.epsilon < 0.0:
            raise ValueError("epsilon must be >= 0")
        weights = (
            self.reconstruction_weight,
            self.spatial_weight,
            self.balance_weight,
            self.stability_weight,
        )
        if any(value < 0.0 for value in weights) or sum(weights) <= 0.0:
            raise ValueError("objective weights must be non-negative with positive total")


@dataclass(frozen=True)
class Metrics:
    pass_index: int
    reconstruction_loss: float
    cycle_loss: float
    reality_gap: float
    changed_bits: int
    active_streams: int
    seconds: float
    updates_per_second: float
    spatial_loss: float
    latent_balance_loss: float
    objective: float


@dataclass(frozen=True)
class TuningResult:
    baseline_alpha: float
    baseline_beta: float
    baseline_score: float
    alpha: float
    beta: float
    score: float
    reconstruction_loss: float
    spatial_loss: float
    latent_balance_loss: float
    mean_reality_gap: float
    passes: int
    candidates_evaluated: int
    improved: bool


def hd(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def hf(a: int, b: int, n: int) -> float:
    return hd(a, b) / n if n else 0.0


def stream(coord: Coord, n: int, seed: int) -> int:
    need = (n + 7) // 8
    raw = bytearray()
    block = 0
    while len(raw) < need:
        payload = f"{seed}|{coord[0]}|{coord[1]}|{coord[2]}|{block}".encode()
        block += 1
        raw.extend(hashlib.blake2b(payload, digest_size=64, person=b"DM3D-AE-v1").digest())
    return int.from_bytes(raw[:need], "little") & ((1 << n) - 1)


def feedback_mask(n: int, beta: float, seed: int) -> int:
    rng = random.Random(seed ^ 0xD34D1A61)
    out = 0
    for bit in rng.sample(range(n), round(beta * n)):
        out |= 1 << bit
    return out


class Codec:
    def __init__(self, n: int, d: int):
        self.n = n
        self.d = d
        self.full = (1 << n) - 1
        quotient, remainder = divmod(n, d)
        shift = 0
        self.groups: list[tuple[int, int]] = []
        for index in range(d):
            width = quotient + (index < remainder)
            mask = ((1 << width) - 1) << shift
            shift += width
            self.groups.append((width, mask))

    def encode(self, x: int) -> int:
        z = 0
        for index, (width, mask) in enumerate(self.groups):
            if (x & mask).bit_count() * 2 >= width:
                z |= 1 << index
        return z

    def decode(self, z: int) -> int:
        x = 0
        for index, (_, mask) in enumerate(self.groups):
            if z >> index & 1:
                x |= mask
        return x & self.full

    def cycle_loss(self, z: int) -> float:
        return hf(z, self.encode(self.decode(z)), self.d)


class Tile:
    def __init__(self, config: Config):
        self.c = config
        ox, oy, oz = config.origin
        n = config.tile
        self.coords = [
            (ox + x, oy + y, oz + z)
            for z in range(n)
            for y in range(n)
            for x in range(n)
        ]

    def __len__(self) -> int:
        return len(self.coords)

    def local(self, point: Coord) -> Coord:
        ox, oy, oz = self.c.origin
        return point[0] - ox, point[1] - oy, point[2] - oz

    def global_(self, point: Coord) -> Coord:
        ox, oy, oz = self.c.origin
        return point[0] + ox, point[1] + oy, point[2] + oz

    def _offsets(self, point: Coord, offsets: Iterable[Coord]) -> list[Coord]:
        x, y, z = self.local(point)
        n = self.c.tile
        out: list[Coord] = []
        for dx, dy, dz in offsets:
            q = (x + dx, y + dy, z + dz)
            if self.c.periodic:
                out.append(self.global_((q[0] % n, q[1] % n, q[2] % n)))
            elif all(0 <= value < n for value in q):
                out.append(self.global_(q))
        return out

    def neighbours(self, point: Coord) -> list[Coord]:
        return self._offsets(point, N6)

    def positive_neighbours(self, point: Coord) -> list[Coord]:
        return self._offsets(point, POSITIVE_3)


def couple(tile: Tile, latent: dict[Coord, int], d: int, alpha: float) -> dict[Coord, int]:
    """Apply synchronous six-neighbour Ising-like coupling to each latent bit."""

    if not alpha:
        return dict(latent)

    out: dict[Coord, int] = {}
    for point in tile.coords:
        neighbours = tile.neighbours(point)
        z = 0
        for bit in range(d):
            current = 1 if latent[point] >> bit & 1 else -1
            mean = (
                sum(1 if latent[q] >> bit & 1 else -1 for q in neighbours) / len(neighbours)
                if neighbours
                else current
            )
            if (1.0 - alpha) * current + alpha * mean >= 0.0:
                z |= 1 << bit
        out[point] = z
    return out


def spatial_loss(tile: Tile, latent: dict[Coord, int], d: int) -> float:
    """Mean latent Hamming disagreement along the positive x/y/z lattice edges."""

    disagreements = 0
    compared_bits = 0
    for point in tile.coords:
        for neighbour in tile.positive_neighbours(point):
            disagreements += hd(latent[point], latent[neighbour])
            compared_bits += d
    return disagreements / compared_bits if compared_bits else 0.0


def latent_balance_loss(latent: dict[Coord, int], d: int) -> float:
    """Penalize all-zero/all-one latent collapse while preserving a 50/50 bit balance."""

    total = len(latent) * d
    if not total:
        return 0.0
    p = sum(value.bit_count() for value in latent.values()) / total
    return 1.0 - 4.0 * p * (1.0 - p)


def objective(
    config: Config,
    reconstruction: float,
    spatial: float,
    balance: float,
    stability: float,
) -> float:
    return (
        config.reconstruction_weight * reconstruction
        + config.spatial_weight * spatial
        + config.balance_weight * balance
        + config.stability_weight * stability
    )


def embed(
    local: Coord,
    n: int,
    curvature: float,
    radius: float = 22.0,
    ratio: float = 1.0,
) -> tuple[float, float, float]:
    if n <= 1:
        x = y = z = 0.0
    else:
        x, y, z = ((value / (n - 1)) * 2.0 - 1.0 for value in local)
    if curvature <= 1e-12:
        return x * radius, y * radius * 0.5, z * radius * 0.5
    theta = x * math.pi * curvature
    minor = 2.5 * max(0.15, ratio)
    return (
        radius * math.sin(theta),
        y * 8.0 + math.cos(2.0 * theta) * 1.5 * curvature,
        -radius * (1.0 - math.cos(theta)) + z * minor,
    )


class DrMoagiVirtual3DAE:
    def __init__(self, config: Config | None = None):
        self.c = config or Config()
        self.tuning: TuningResult | None = None
        self.original: dict[Coord, int] = {}
        self.state: dict[Coord, int] = {}
        self.latent: dict[Coord, int] = {}
        self.coupled: dict[Coord, int] = {}
        self.decoded: dict[Coord, int] = {}
        self._rebuild()

    def _rebuild(self) -> None:
        self.tile = Tile(self.c)
        self.codec = Codec(self.c.bits, self.c.latent)
        self.full = (1 << self.c.bits) - 1
        self.mask = feedback_mask(self.c.bits, self.c.beta, self.c.seed)
        self.original.clear()
        self.state.clear()
        self.latent.clear()
        self.coupled.clear()
        self.decoded.clear()

    @property
    def active_streams(self) -> int:
        return len(self.tile)

    def materialize(self) -> None:
        for point in self.tile.coords:
            value = stream(point, self.c.bits, self.c.seed)
            self.original[point] = value
            self.state[point] = value

    def run(self) -> list[Metrics]:
        if not self.state:
            self.materialize()

        history: list[Metrics] = []
        for pass_index in range(self.c.passes):
            started = time.perf_counter()
            self.latent = {point: self.codec.encode(x) for point, x in self.state.items()}
            self.coupled = couple(self.tile, self.latent, self.c.latent, self.c.alpha)
            self.decoded = {point: self.codec.decode(z) for point, z in self.coupled.items()}

            reconstruction = sum(
                hf(self.original[point], self.decoded[point], self.c.bits)
                for point in self.tile.coords
            ) / len(self.tile)
            cycle = sum(
                self.codec.cycle_loss(self.coupled[point]) for point in self.tile.coords
            ) / len(self.tile)
            spatial = spatial_loss(self.tile, self.coupled, self.c.latent)
            balance = latent_balance_loss(self.coupled, self.c.latent)

            next_state: dict[Coord, int] = {}
            changed = 0
            anchor = self.full ^ self.mask
            for point in self.tile.coords:
                value = (
                    (self.original[point] & anchor) | (self.decoded[point] & self.mask)
                ) & self.full
                changed += hd(self.state[point], value)
                next_state[point] = value

            self.state = next_state
            gap = changed / (len(self.tile) * self.c.bits)
            elapsed = max(time.perf_counter() - started, 1e-12)
            score = objective(self.c, reconstruction, spatial, balance, gap)
            history.append(
                Metrics(
                    pass_index=pass_index,
                    reconstruction_loss=reconstruction,
                    cycle_loss=cycle,
                    reality_gap=gap,
                    changed_bits=changed,
                    active_streams=len(self.tile),
                    seconds=elapsed,
                    updates_per_second=len(self.tile) / elapsed,
                    spatial_loss=spatial,
                    latent_balance_loss=balance,
                    objective=score,
                )
            )
            if gap <= self.c.epsilon:
                break
        return history

    def _score_history(self, history: list[Metrics]) -> tuple[float, float]:
        if not history:
            return math.inf, math.inf
        mean_gap = sum(metric.reality_gap for metric in history) / len(history)
        final = history[-1]
        score = objective(
            self.c,
            final.reconstruction_loss,
            final.spatial_loss,
            final.latent_balance_loss,
            mean_gap,
        )
        return score, mean_gap

    def _evaluate(self, alpha: float, beta: float) -> tuple[float, float, list[Metrics]]:
        candidate_config = replace(self.c, alpha=alpha, beta=beta)
        candidate = DrMoagiVirtual3DAE(candidate_config)
        history = candidate.run()
        score, mean_gap = candidate._score_history(history)
        return score, mean_gap, history

    def optimize(self) -> TuningResult:
        """Search a finite deterministic alpha/beta grid and commit only improvement."""

        pairs = {(self.c.alpha, self.c.beta)}
        pairs.update(
            (alpha, beta)
            for alpha in self.c.alpha_candidates
            for beta in self.c.beta_candidates
        )

        evaluated: list[tuple[float, float, float, float, float, float, float, int]] = []
        for alpha, beta in sorted(pairs):
            score, mean_gap, history = self._evaluate(alpha, beta)
            final = history[-1]
            evaluated.append(
                (
                    score,
                    final.reconstruction_loss,
                    final.spatial_loss,
                    final.latent_balance_loss,
                    mean_gap,
                    alpha,
                    beta,
                    len(history),
                )
            )

        baseline = next(
            row
            for row in evaluated
            if row[5] == self.c.alpha and row[6] == self.c.beta
        )
        best = min(evaluated, key=lambda row: (row[0], row[1], row[2], row[5], row[6]))
        improved = best[0] < baseline[0] - 1e-12

        chosen = best if improved else baseline
        result = TuningResult(
            baseline_alpha=self.c.alpha,
            baseline_beta=self.c.beta,
            baseline_score=baseline[0],
            alpha=chosen[5],
            beta=chosen[6],
            score=chosen[0],
            reconstruction_loss=chosen[1],
            spatial_loss=chosen[2],
            latent_balance_loss=chosen[3],
            mean_reality_gap=chosen[4],
            passes=chosen[7],
            candidates_evaluated=len(evaluated),
            improved=improved,
        )

        if improved:
            self.c = replace(self.c, alpha=result.alpha, beta=result.beta)
            self._rebuild()
        self.tuning = result
        return result

    def geometry(self, count: int = 6) -> list[dict[str, object]]:
        ratio = max(0.15, (self.c.latent / self.c.bits) ** (1.0 / 3.0))
        out: list[dict[str, object]] = []
        for point in self.tile.coords[:count]:
            local = self.tile.local(point)
            out.append(
                {
                    "virtual_coord": point,
                    "input_position_3d": embed(local, self.c.tile, self.c.curvature),
                    "latent_position_3d": embed(
                        local,
                        self.c.tile,
                        self.c.curvature,
                        22.0 * ratio,
                        ratio,
                    ),
                }
            )
        return out

    def summary(self, history: list[Metrics]) -> dict[str, object]:
        return {
            "logical_stream_universe": LOGICAL_STREAM_COUNT,
            "virtual_cube_side": VIRTUAL_SIDE_LENGTH,
            "active_streams": len(self.tile),
            "codec": [self.c.bits, self.c.latent, self.c.bits],
            "compression_ratio": self.c.bits / self.c.latent,
            "alpha": self.c.alpha,
            "beta": self.c.beta,
            "curvature": self.c.curvature,
            "passes": len(history),
            "tuning": asdict(self.tuning) if self.tuning else None,
            "final": asdict(history[-1]) if history else None,
            "geometry": self.geometry(),
        }


def _origin(value: str) -> Coord:
    point = tuple(map(int, value.split(",")))
    if len(point) != 3:
        raise argparse.ArgumentTypeError("origin must be x,y,z")
    return point[0], point[1], point[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile", type=int, default=8)
    parser.add_argument("--bits", type=int, default=256)
    parser.add_argument("--latent", type=int, default=32)
    parser.add_argument("--passes", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.65)
    parser.add_argument("--beta", type=float, default=0.65)
    parser.add_argument("--curvature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--origin", type=_origin, default=(0, 0, 0))
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument("--no-periodic", action="store_true")
    parser.add_argument("--auto-optimize", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    engine = DrMoagiVirtual3DAE(
        Config(
            tile=args.tile,
            bits=args.bits,
            latent=args.latent,
            passes=args.passes,
            alpha=args.alpha,
            beta=args.beta,
            curvature=args.curvature,
            seed=args.seed,
            origin=args.origin,
            periodic=not args.no_periodic,
            epsilon=args.epsilon,
        )
    )
    if args.auto_optimize:
        tuning = engine.optimize()
        print(
            "tune "
            f"baseline={tuning.baseline_score:.6f} "
            f"score={tuning.score:.6f} "
            f"alpha={tuning.alpha:.3f} "
            f"beta={tuning.beta:.3f} "
            f"improved={tuning.improved}"
        )

    history = engine.run()
    for metric in history:
        print(
            f"pass={metric.pass_index:02d} "
            f"rec={metric.reconstruction_loss:.6f} "
            f"cycle={metric.cycle_loss:.6f} "
            f"spatial={metric.spatial_loss:.6f} "
            f"balance={metric.latent_balance_loss:.6f} "
            f"gap={metric.reality_gap:.6f} "
            f"objective={metric.objective:.6f} "
            f"changed={metric.changed_bits:,} "
            f"updates/s={metric.updates_per_second:,.0f}"
        )
    if args.json:
        print(json.dumps(engine.summary(history), indent=2))


if __name__ == "__main__":
    main()
