"""Transactional inward-looped 3D bit autoencoder/decoder reference.

The decoded 3D bit volume becomes the next encoder input. Unlike the anchored
virtual codec, the recurrence is self-referential at the authoritative state
boundary::

    X_t -> E -> C_3D -> D -> X_hat_t -> feedback -> X_{t+1} -> E -> ...

Only a finite active tile is materialized. The implementation is deterministic,
bit-width bounded and detects non-fixed synchronous cycles rather than treating
oscillation as convergence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Callable, Iterator, cast

from .candidate_contract import canonical_state_hash
from .dr_moagi_virtual_3d_ae import (
    Codec,
    Coord,
    Tile,
    couple,
    latent_balance_loss,
    spatial_loss,
    stream,
)

Gate = Callable[[dict[Coord, int], dict[Coord, int]], bool]


@dataclass(frozen=True)
class Inward3DBitConfig:
    tile: int = 8
    bits: int = 64
    latent: int = 16
    iterations: int = 32
    alpha: float = 0.65
    beta: float = 0.50
    omega_feedback: float = 0.0
    seed: int = 1337
    origin: Coord = (0, 0, 0)
    periodic: bool = True
    epsilon: float = 0.0
    detect_cycles: bool = True

    def __post_init__(self) -> None:
        if self.tile < 1 or self.bits < 2 or not 1 <= self.latent <= self.bits:
            raise ValueError("invalid dimensions")
        if self.iterations < 1:
            raise ValueError("iterations must be >= 1")
        if any(
            not 0.0 <= value <= 1.0
            for value in (self.alpha, self.beta, self.omega_feedback)
        ):
            raise ValueError("alpha, beta and omega_feedback must be in [0,1]")
        if self.epsilon < 0.0:
            raise ValueError("epsilon must be >= 0")


@dataclass(frozen=True)
class Inward3DBitMetrics:
    iteration: int
    committed: bool
    reconstruction_loss: float
    self_reconstruction_loss: float
    codec_cycle_loss: float
    spatial_loss: float
    latent_balance_loss: float
    reality_gap: float
    latent_reality_gap: float
    coupling_gap: float
    omega_density: float
    changed_bits: int
    latent_changed_bits: int
    active_cells: int
    fixed_point: bool
    cycle_detected: bool
    input_hash: str
    candidate_hash: str
    rejection_reason: str | None


def hamming_fraction(left: int, right: int, width: int) -> float:
    return (left ^ right).bit_count() / width if width else 0.0


def _exact_mask(
    point: Coord,
    width: int,
    fraction: float,
    seed: int,
    channel: str,
) -> int:
    """Select exactly round(fraction*width) deterministic bit positions per coordinate."""

    count = round(fraction * width)
    if count <= 0:
        return 0
    if count >= width:
        return (1 << width) - 1

    ranked: list[tuple[int, int]] = []
    for bit in range(width):
        payload = f"{seed}|{point[0]}|{point[1]}|{point[2]}|{channel}|{bit}".encode()
        score = int.from_bytes(
            hashlib.blake2b(payload, digest_size=8, person=b"DM-IN3D-MASK").digest(),
            "little",
        )
        ranked.append((score, bit))
    ranked.sort()

    mask = 0
    for _, bit in ranked[:count]:
        mask |= 1 << bit
    return mask


def _volume_payload(volume: dict[Coord, int]) -> list[list[int]]:
    return [[point[0], point[1], point[2], volume[point]] for point in sorted(volume)]


def _authority_hash(
    state: dict[Coord, int],
    omega: dict[Coord, int],
    latent: dict[Coord, int],
) -> str:
    return cast(
        str,
        canonical_state_hash(
            {
                "state": _volume_payload(state),
                "omega": _volume_payload(omega),
                "latent": _volume_payload(latent),
            }
        ),
    )


class Inward3DBitLoop:
    """Finite 3D bit-volume recursion with atomic candidate publication."""

    def __init__(
        self,
        config: Inward3DBitConfig | None = None,
        *,
        gate: Gate | None = None,
    ) -> None:
        self.c = config or Inward3DBitConfig()
        from .dr_moagi_virtual_3d_ae import Config as TileConfig

        self.tile = Tile(
            TileConfig(
                tile=self.c.tile,
                bits=self.c.bits,
                latent=self.c.latent,
                passes=max(1, self.c.iterations),
                alpha=self.c.alpha,
                beta=self.c.beta,
                seed=self.c.seed,
                origin=self.c.origin,
                periodic=self.c.periodic,
                epsilon=self.c.epsilon,
            )
        )
        self.codec = Codec(self.c.bits, self.c.latent)
        self.full = (1 << self.c.bits) - 1
        self.gate = gate
        self.original: dict[Coord, int] = {}
        self.state: dict[Coord, int] = {}
        self.omega: dict[Coord, int] = {}
        self.latent: dict[Coord, int] = {}
        self.iteration = 0
        self._seen: set[str] = set()
        self.materialize()

    @property
    def active_cells(self) -> int:
        return len(self.tile)

    def materialize(self) -> None:
        self.original = {
            point: stream(point, self.c.bits, self.c.seed) for point in self.tile.coords
        }
        self.state = dict(self.original)
        self.omega = {point: 0 for point in self.tile.coords}
        self.latent = self._coupled_latent(self.state)
        self.iteration = 0
        self._seen = {self.authority_hash()}

    def snapshot(self) -> dict[Coord, int]:
        return dict(self.state)

    def omega_snapshot(self) -> dict[Coord, int]:
        return dict(self.omega)

    def latent_snapshot(self) -> dict[Coord, int]:
        return dict(self.latent)

    def authority_hash(self) -> str:
        return _authority_hash(self.state, self.omega, self.latent)

    def encode_state(self, state: dict[Coord, int] | None = None) -> dict[Coord, int]:
        source = self.state if state is None else state
        return {point: self.codec.encode(source[point]) for point in self.tile.coords}

    def _coupled_latent(self, state: dict[Coord, int]) -> dict[Coord, int]:
        raw = self.encode_state(state)
        return cast(dict[Coord, int], couple(self.tile, raw, self.c.latent, self.c.alpha))

    def decode_latent(self, latent: dict[Coord, int]) -> dict[Coord, int]:
        return {point: self.codec.decode(latent[point]) for point in self.tile.coords}

    def _next_omega(
        self,
        current: dict[Coord, int],
        decoded: dict[Coord, int],
    ) -> dict[Coord, int]:
        return {
            point: (current[point] ^ decoded[point]) & self.full for point in self.tile.coords
        }

    def _feedback(
        self,
        current: dict[Coord, int],
        decoded: dict[Coord, int],
        omega: dict[Coord, int],
    ) -> dict[Coord, int]:
        out: dict[Coord, int] = {}
        for point in self.tile.coords:
            reconstruction_mask = _exact_mask(
                point,
                self.c.bits,
                self.c.beta,
                self.c.seed,
                "reconstruction",
            )
            omega_mask = _exact_mask(
                point,
                self.c.bits,
                self.c.omega_feedback,
                self.c.seed,
                "omega",
            )
            value = (
                (current[point] & (self.full ^ reconstruction_mask))
                | (decoded[point] & reconstruction_mask)
            ) & self.full
            if omega_mask:
                value ^= omega[point] & omega_mask
            out[point] = value & self.full
        return out

    def _valid_volume(self, volume: dict[Coord, int], width: int) -> bool:
        if set(volume) != set(self.tile.coords):
            return False
        maximum = (1 << width) - 1
        return all(
            isinstance(value, int)
            and not isinstance(value, bool)
            and 0 <= value <= maximum
            for value in volume.values()
        )

    def step(self) -> Inward3DBitMetrics:
        current = dict(self.state)
        current_omega = dict(self.omega)
        previous_latent = dict(self.latent)
        input_hash = _authority_hash(current, current_omega, previous_latent)

        raw_latent = self.encode_state(current)
        coupled_latent = cast(
            dict[Coord, int],
            couple(self.tile, raw_latent, self.c.latent, self.c.alpha),
        )
        decoded = self.decode_latent(coupled_latent)
        next_omega = self._next_omega(current, decoded)
        candidate = self._feedback(current, decoded, next_omega)
        candidate_latent = self._coupled_latent(candidate)
        candidate_hash = _authority_hash(candidate, next_omega, candidate_latent)

        total_source_bits = len(self.tile) * self.c.bits
        total_latent_bits = len(self.tile) * self.c.latent
        reconstruction_loss = sum(
            (self.original[point] ^ decoded[point]).bit_count() for point in self.tile.coords
        ) / total_source_bits
        self_reconstruction_loss = sum(
            (current[point] ^ decoded[point]).bit_count() for point in self.tile.coords
        ) / total_source_bits
        codec_cycle_loss = sum(
            self.codec.cycle_loss(coupled_latent[point]) for point in self.tile.coords
        ) / len(self.tile)
        changed_bits = sum(
            (current[point] ^ candidate[point]).bit_count() for point in self.tile.coords
        )
        latent_changed_bits = sum(
            (previous_latent[point] ^ candidate_latent[point]).bit_count()
            for point in self.tile.coords
        )
        coupling_changed_bits = sum(
            (raw_latent[point] ^ coupled_latent[point]).bit_count()
            for point in self.tile.coords
        )
        omega_bits = sum(value.bit_count() for value in next_omega.values())

        reality_gap = changed_bits / total_source_bits
        latent_gap = latent_changed_bits / total_latent_bits
        coupling_gap = coupling_changed_bits / total_latent_bits
        omega_density = omega_bits / total_source_bits
        full_fixed = candidate_hash == input_hash
        cycle_detected = self.c.detect_cycles and candidate_hash in self._seen and not full_fixed

        rejection_reason: str | None = None
        committed = True
        if not self._valid_volume(candidate, self.c.bits):
            committed = False
            rejection_reason = "candidate state violates bit-width/coordinate contract"
        elif not self._valid_volume(next_omega, self.c.bits):
            committed = False
            rejection_reason = "omega state violates bit-width/coordinate contract"
        elif not self._valid_volume(candidate_latent, self.c.latent):
            committed = False
            rejection_reason = "latent state violates bit-width/coordinate contract"
        elif self.gate is not None and not self.gate(candidate, next_omega):
            committed = False
            rejection_reason = "external gate rejected candidate"
        elif cycle_detected:
            committed = False
            rejection_reason = "non-fixed recursive cycle detected"

        fixed_point = committed and full_fixed and reality_gap <= self.c.epsilon
        if committed:
            self.state = candidate
            self.omega = next_omega
            self.latent = candidate_latent
            self.iteration += 1
            self._seen.add(candidate_hash)

        return Inward3DBitMetrics(
            iteration=self.iteration,
            committed=committed,
            reconstruction_loss=reconstruction_loss,
            self_reconstruction_loss=self_reconstruction_loss,
            codec_cycle_loss=codec_cycle_loss,
            spatial_loss=spatial_loss(self.tile, coupled_latent, self.c.latent),
            latent_balance_loss=latent_balance_loss(coupled_latent, self.c.latent),
            reality_gap=reality_gap,
            latent_reality_gap=latent_gap,
            coupling_gap=coupling_gap,
            omega_density=omega_density,
            changed_bits=changed_bits,
            latent_changed_bits=latent_changed_bits,
            active_cells=len(self.tile),
            fixed_point=fixed_point,
            cycle_detected=cycle_detected,
            input_hash=input_hash,
            candidate_hash=candidate_hash,
            rejection_reason=rejection_reason,
        )

    def run(self) -> Iterator[Inward3DBitMetrics]:
        for _ in range(self.c.iterations):
            metric = self.step()
            yield metric
            if not metric.committed or metric.fixed_point:
                break

    def summary(self, history: list[Inward3DBitMetrics]) -> dict[str, object]:
        return {
            "mode": "inward-recursive-3d-bit-ae-ad",
            "active_cells": self.active_cells,
            "source_bits": self.active_cells * self.c.bits,
            "latent_bits": self.active_cells * self.c.latent,
            "nominal_representation_ratio": self.c.bits / self.c.latent,
            "alpha": self.c.alpha,
            "beta": self.c.beta,
            "omega_feedback": self.c.omega_feedback,
            "iterations_committed": self.iteration,
            "authority_hash": self.authority_hash(),
            "final": asdict(history[-1]) if history else None,
        }


def _origin(value: str) -> Coord:
    point = tuple(map(int, value.split(",")))
    if len(point) != 3:
        raise argparse.ArgumentTypeError("origin must be x,y,z")
    return point[0], point[1], point[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile", type=int, default=8)
    parser.add_argument("--bits", type=int, default=64)
    parser.add_argument("--latent", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.65)
    parser.add_argument("--beta", type=float, default=0.50)
    parser.add_argument("--omega-feedback", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--origin", type=_origin, default=(0, 0, 0))
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--no-periodic", action="store_true")
    parser.add_argument("--no-cycle-detection", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    engine = Inward3DBitLoop(
        Inward3DBitConfig(
            tile=args.tile,
            bits=args.bits,
            latent=args.latent,
            iterations=args.iterations,
            alpha=args.alpha,
            beta=args.beta,
            omega_feedback=args.omega_feedback,
            seed=args.seed,
            origin=args.origin,
            periodic=not args.no_periodic,
            epsilon=args.epsilon,
            detect_cycles=not args.no_cycle_detection,
        )
    )
    history = list(engine.run())
    for metric in history:
        print(
            f"t={metric.iteration:03d} "
            f"commit={metric.committed} "
            f"rec={metric.reconstruction_loss:.6f} "
            f"self_rec={metric.self_reconstruction_loss:.6f} "
            f"dX={metric.reality_gap:.6f} "
            f"dZ={metric.latent_reality_gap:.6f} "
            f"coupling={metric.coupling_gap:.6f} "
            f"omega={metric.omega_density:.6f} "
            f"fixed={metric.fixed_point} "
            f"cycle={metric.cycle_detected}"
        )
    if args.json:
        print(json.dumps(engine.summary(history), indent=2))


if __name__ == "__main__":
    main()
