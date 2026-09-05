"""Bounded inward-looped 3D bit autoencoder/autodecoder reference runtime.

The runtime implements a deterministic discrete recurrence over a finite 3D bit
field while keeping codec, six-neighbour coupling, error memory and inward
feedback as separate operations.

The reference intentionally avoids claiming convergence for arbitrary settings.
Its full dynamical state is ``(X_t, Z_t, Omega_t)``; fixed-point detection is
therefore performed over all three components rather than ``X_t`` alone.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Mapping

Coord = tuple[int, int, int]

N6 = (
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
)


@dataclass(frozen=True)
class Config:
    """Numerical and resource contract for the inward 3D bit recurrence."""

    side: int = 8
    bits: int = 64
    latent_bits: int = 16
    iterations: int = 32
    alpha: float = 0.65
    beta: float = 0.50
    omega_gain: float = 0.15
    omega_retention: float = 0.50
    omega_capture: float = 0.50
    seed: int = 1337
    tolerance: float = 0.0
    periodic: bool = True

    def __post_init__(self) -> None:
        for name in ("side", "bits", "latent_bits", "iterations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.latent_bits > self.bits:
            raise ValueError("latent_bits must satisfy 1 <= latent_bits <= bits")

        for name in (
            "alpha",
            "beta",
            "omega_gain",
            "omega_retention",
            "omega_capture",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and within [0, 1]")

        if not math.isfinite(self.tolerance) or self.tolerance < 0.0:
            raise ValueError("tolerance must be finite and non-negative")


@dataclass(frozen=True)
class IterationMetrics:
    """Auditable telemetry for one complete inward iteration."""

    iteration: int
    local_reconstruction_loss: float
    anchor_reconstruction_loss: float
    anchor_drift: float
    latent_cycle_loss: float
    latent_coupling_loss: float
    reality_gap: float
    omega_gap: float
    full_state_gap: float
    changed_bits: int
    latent_changed_bits: int
    omega_changed_bits: int
    omega_active_bits: int
    active_cells: int
    fixed_point: bool

    @property
    def reconstruction_loss(self) -> float:
        """Backward-compatible alias for anchor reconstruction loss."""

        return self.anchor_reconstruction_loss

    @property
    def cycle_loss(self) -> float:
        """Backward-compatible alias for the algebraic latent-cycle loss."""

        return self.latent_cycle_loss


def deterministic_bits(coord: Coord, bits: int, seed: int) -> int:
    """Generate a deterministic finite-width source word for one 3D coordinate."""

    required = (bits + 7) // 8
    data = bytearray()
    counter = 0
    while len(data) < required:
        payload = f"{seed}|{coord[0]}|{coord[1]}|{coord[2]}|{counter}".encode()
        data.extend(
            hashlib.blake2b(
                payload,
                digest_size=64,
                person=b"DM-INWARD-3D",
            ).digest()
        )
        counter += 1
    return int.from_bytes(data[:required], "little") & ((1 << bits) - 1)


class Grid3D:
    """Finite cubic lattice with periodic or bounded six-neighbour topology."""

    def __init__(self, config: Config):
        self.c = config
        self.coords = [
            (x, y, z)
            for z in range(config.side)
            for y in range(config.side)
            for x in range(config.side)
        ]

    def neighbour(self, point: Coord, direction: Coord) -> Coord | None:
        x = point[0] + direction[0]
        y = point[1] + direction[1]
        z = point[2] + direction[2]

        if self.c.periodic:
            side = self.c.side
            return x % side, y % side, z % side

        if 0 <= x < self.c.side and 0 <= y < self.c.side and 0 <= z < self.c.side:
            return x, y, z
        return None

    def neighbours(self, point: Coord) -> list[Coord]:
        result: list[Coord] = []
        for direction in N6:
            neighbour = self.neighbour(point, direction)
            if neighbour is not None:
                result.append(neighbour)
        return result


class BitAutoEncoder:
    """Deterministic majority codec with tie-neutral source-bit resolution."""

    def __init__(self, source_bits: int, latent_bits: int):
        if source_bits < 1 or latent_bits < 1 or latent_bits > source_bits:
            raise ValueError("require 1 <= latent_bits <= source_bits")

        self.source_bits = source_bits
        self.latent_bits = latent_bits
        quotient, remainder = divmod(source_bits, latent_bits)
        start = 0
        self.groups: list[tuple[int, int, int]] = []

        for index in range(latent_bits):
            width = quotient + int(index < remainder)
            mask = ((1 << width) - 1) << start
            self.groups.append((start, width, mask))
            start += width

    def encode(self, source: int) -> int:
        """Compress each source group to one latent bit.

        Strict majorities map normally. For an even-width exact tie, the first
        source bit in the group resolves the tie. Under an unbiased random
        source this preserves a 0.5 latent-one probability instead of biasing
        ties toward one.
        """

        latent = 0
        for index, (start, width, mask) in enumerate(self.groups):
            ones = (source & mask).bit_count()
            tied_one = 2 * ones == width and ((source >> start) & 1)
            if 2 * ones > width or tied_one:
                latent |= 1 << index
        return latent

    def decode(self, latent: int) -> int:
        """Expand each latent bit across the corresponding source-bit group."""

        source = 0
        for index, (_, _, mask) in enumerate(self.groups):
            if (latent >> index) & 1:
                source |= mask
        return source

    def latent_cycle(self, latent: int) -> int:
        """Return E(D(z)); this codec makes the latent cycle an exact identity."""

        return self.encode(self.decode(latent))


class Inward3DBitEngine:
    """Deterministic encode -> couple -> decode -> memory -> feedback recurrence."""

    LAW_ID = "DM-vOmegaXi+-INWARD-3D-BIT"
    RECURRENCE = (
        "X[t] -> E -> Z[t] -> C_6N -> Z~[t] -> D -> X_hat[t] -> "
        "Omega[t+1] -> bounded feedback -> X[t+1]"
    )

    def __init__(self, config: Config = Config()):
        self.c = config
        self.grid = Grid3D(config)
        self.codec = BitAutoEncoder(config.bits, config.latent_bits)
        self.full_mask = (1 << config.bits) - 1
        self.state: dict[Coord, int] = {}
        self.original: dict[Coord, int] = {}
        self.omega: dict[Coord, int] = {}
        self.latent: dict[Coord, int] = {}
        self.iteration = 0
        self.reconstruction_masks: dict[Coord, int] = {}
        self.omega_injection_masks: dict[Coord, int] = {}
        self.omega_retention_masks: dict[Coord, int] = {}
        self.omega_capture_masks: dict[Coord, int] = {}
        self.materialize()

    def materialize(self) -> None:
        """Reset the deterministic source, memory, latent state and cached masks."""

        self.original = {
            point: deterministic_bits(point, self.c.bits, self.c.seed)
            for point in self.grid.coords
        }
        self.state = dict(self.original)
        self.omega = {point: 0 for point in self.grid.coords}
        self.latent = {}
        self.iteration = 0
        self._precompute_masks()

    def encode_volume(self, state: Mapping[Coord, int]) -> dict[Coord, int]:
        return {point: self.codec.encode(source) for point, source in state.items()}

    def couple_volume(self, latent: Mapping[Coord, int]) -> dict[Coord, int]:
        """Apply synchronous six-neighbour binary-spin coupling."""

        result: dict[Coord, int] = {}
        alpha = self.c.alpha

        for point in self.grid.coords:
            center = latent[point]
            neighbours = self.grid.neighbours(point)
            coupled = 0

            for bit in range(self.c.latent_bits):
                center_spin = 1 if (center >> bit) & 1 else -1
                neighbour_sum = sum(
                    1 if (latent[neighbour] >> bit) & 1 else -1
                    for neighbour in neighbours
                )
                mean_spin = neighbour_sum / len(neighbours) if neighbours else 0.0
                field = (1.0 - alpha) * center_spin + alpha * mean_spin
                if field >= 0.0:
                    coupled |= 1 << bit

            result[point] = coupled
        return result

    def decode_volume(self, latent: Mapping[Coord, int]) -> dict[Coord, int]:
        return {point: self.codec.decode(value) for point, value in latent.items()}

    def update_omega(
        self,
        state: Mapping[Coord, int],
        decoded: Mapping[Coord, int],
    ) -> dict[Coord, int]:
        """Update bounded recursive error memory.

        In the absence of new error, the rotation followed by a retention mask
        cannot increase Hamming weight. This makes memory contractive in the
        unforced case while still permitting fresh reconstruction error to enter
        through a separate capture mask.
        """

        next_omega: dict[Coord, int] = {}
        for point in self.grid.coords:
            error = state[point] ^ decoded[point]
            previous = self.omega[point]
            if self.c.bits == 1:
                rotated = previous
            else:
                rotated = (
                    (previous << 1) | (previous >> (self.c.bits - 1))
                ) & self.full_mask

            retained = rotated & self.omega_retention_masks[point]
            captured = error & self.omega_capture_masks[point]
            next_omega[point] = (retained ^ captured) & self.full_mask
        return next_omega

    def bit_selection_mask(self, coord: Coord, fraction: float, channel: str) -> int:
        """Return a deterministic finite-width mask selecting a bit fraction."""

        count = round(fraction * self.c.bits)
        if count <= 0:
            return 0
        if count >= self.c.bits:
            return self.full_mask

        scored: list[tuple[int, int]] = []
        for bit in range(self.c.bits):
            payload = f"{self.c.seed}|{coord}|{channel}|{bit}".encode()
            score = int.from_bytes(
                hashlib.blake2b(
                    payload,
                    digest_size=8,
                    person=b"DM-MASK-3D",
                ).digest(),
                "little",
            )
            scored.append((score, bit))

        scored.sort()
        mask = 0
        for _, bit in scored[:count]:
            mask |= 1 << bit
        return mask

    def _precompute_masks(self) -> None:
        """Move deterministic hashing out of the hot recurrence loop."""

        self.reconstruction_masks = {
            point: self.bit_selection_mask(point, self.c.beta, "reconstruction")
            for point in self.grid.coords
        }
        self.omega_injection_masks = {
            point: self.bit_selection_mask(point, self.c.omega_gain, "omega-injection")
            for point in self.grid.coords
        }
        self.omega_retention_masks = {
            point: self.bit_selection_mask(point, self.c.omega_retention, "omega-retention")
            for point in self.grid.coords
        }
        self.omega_capture_masks = {
            point: self.bit_selection_mask(point, self.c.omega_capture, "omega-capture")
            for point in self.grid.coords
        }

    def feedback(
        self,
        current: Mapping[Coord, int],
        decoded: Mapping[Coord, int],
        omega: Mapping[Coord, int],
    ) -> dict[Coord, int]:
        """Apply deterministic bounded reconstruction and memory feedback."""

        next_state: dict[Coord, int] = {}
        for point in self.grid.coords:
            reconstruction_mask = self.reconstruction_masks[point]
            omega_mask = self.omega_injection_masks[point]
            candidate = current[point] & ~reconstruction_mask
            candidate |= decoded[point] & reconstruction_mask
            candidate ^= omega[point] & omega_mask
            next_state[point] = candidate & self.full_mask
        return next_state

    def bitplane_field(
        self,
        bit: int,
        *,
        latent: bool = False,
        spins: bool = True,
    ) -> dict[Coord, float]:
        """Expose one state/latent bitplane as a scalar 3D field for FMDR analysis."""

        width = self.c.latent_bits if latent else self.c.bits
        if isinstance(bit, bool) or not isinstance(bit, int) or not 0 <= bit < width:
            raise ValueError(f"bit must satisfy 0 <= bit < {width}")

        source = self.latent if latent else self.state
        if latent and not source:
            raise RuntimeError("latent field is unavailable before the first step")

        off = -1.0 if spins else 0.0
        return {
            point: 1.0 if (source[point] >> bit) & 1 else off
            for point in self.grid.coords
        }

    @staticmethod
    def _hamming(left: Mapping[Coord, int], right: Mapping[Coord, int]) -> int:
        return sum((left[point] ^ right[point]).bit_count() for point in left)

    def step(self) -> IterationMetrics:
        """Execute one synchronous inward iteration and atomically commit it."""

        current = dict(self.state)
        previous_omega = dict(self.omega)

        latent_raw = self.encode_volume(current)
        latent_coupled = self.couple_volume(latent_raw)
        decoded = self.decode_volume(latent_coupled)
        omega_next = self.update_omega(current, decoded)
        state_next = self.feedback(current, decoded, omega_next)

        source_bit_count = len(self.grid.coords) * self.c.bits
        latent_bit_count = len(self.grid.coords) * self.c.latent_bits

        local_reconstruction_loss = self._hamming(current, decoded) / source_bit_count
        anchor_reconstruction_loss = self._hamming(self.original, decoded) / source_bit_count
        anchor_drift = self._hamming(self.original, state_next) / source_bit_count

        latent_cycle_bits = sum(
            (latent_coupled[point] ^ self.codec.latent_cycle(latent_coupled[point])).bit_count()
            for point in self.grid.coords
        )
        latent_cycle_loss = latent_cycle_bits / latent_bit_count
        latent_coupling_loss = self._hamming(latent_raw, latent_coupled) / latent_bit_count

        changed_bits = self._hamming(current, state_next)
        omega_changed_bits = self._hamming(previous_omega, omega_next)
        omega_active_bits = sum(value.bit_count() for value in omega_next.values())

        if self.latent:
            latent_changed_bits = self._hamming(self.latent, latent_coupled)
        else:
            latent_changed_bits = latent_bit_count

        reality_gap = changed_bits / source_bit_count
        omega_gap = omega_changed_bits / source_bit_count
        full_state_gap = (
            changed_bits + latent_changed_bits + omega_changed_bits
        ) / (2 * source_bit_count + latent_bit_count)

        pending = IterationMetrics(
            iteration=self.iteration + 1,
            local_reconstruction_loss=local_reconstruction_loss,
            anchor_reconstruction_loss=anchor_reconstruction_loss,
            anchor_drift=anchor_drift,
            latent_cycle_loss=latent_cycle_loss,
            latent_coupling_loss=latent_coupling_loss,
            reality_gap=reality_gap,
            omega_gap=omega_gap,
            full_state_gap=full_state_gap,
            changed_bits=changed_bits,
            latent_changed_bits=latent_changed_bits,
            omega_changed_bits=omega_changed_bits,
            omega_active_bits=omega_active_bits,
            active_cells=len(self.grid.coords),
            fixed_point=full_state_gap <= self.c.tolerance,
        )

        self.state = state_next
        self.omega = omega_next
        self.latent = latent_coupled
        self.iteration += 1
        return pending

    def run(self):
        """Iterate until the configured bound or a full-state fixed point."""

        for _ in range(self.c.iterations):
            metrics = self.step()
            yield metrics
            if metrics.fixed_point:
                break
