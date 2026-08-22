"""Sparse Uint64 bit-plane substrate for the Dr Moagi 3D runtime.

The logical binary volume is packed along Z in 64-bit words.  Only non-zero
words are resident, so a large logical lattice does not imply dense allocation.
The module also provides deterministic inward spatial contraction and bounded
radial attenuation for scalar sparse fields.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Mapping

from .dr_moagi_field_runtime import Coordinate, SparseField

PackedAddress = tuple[int, int, int]


@dataclass(frozen=True)
class BitPlaneMetrics:
    active_bits: int
    logical_bits: int
    packed_words: int
    logical_words: int
    density: float
    entropy: float
    phase_velocity: float
    kinetic_energy: float

    def as_dict(self) -> dict[str, object]:
        return {
            "active_bits": self.active_bits,
            "logical_bits": self.logical_bits,
            "packed_words": self.packed_words,
            "logical_words": self.logical_words,
            "density": self.density,
            "entropy": self.entropy,
            "phase_velocity": self.phase_velocity,
            "kinetic_energy": self.kinetic_energy,
        }


@dataclass(frozen=True)
class SparseBitPlane3D:
    """Sparse packed representation of a logical ``side x side x side`` bit cube."""

    side: int
    words: tuple[tuple[int, int, int, int], ...]

    def __post_init__(self) -> None:
        if isinstance(self.side, bool) or not isinstance(self.side, int) or self.side <= 0:
            raise ValueError("side must be a positive integer")
        seen: set[PackedAddress] = set()
        for x, y, q, word in self.words:
            if not (0 <= x < self.side and 0 <= y < self.side):
                raise ValueError("packed address outside logical lattice")
            if not 0 <= q < self.words_per_column:
                raise ValueError("packed Z-word index outside logical lattice")
            if not 0 < word <= 0xFFFFFFFFFFFFFFFF:
                raise ValueError("resident packed words must be non-zero uint64 values")
            address = (x, y, q)
            if address in seen:
                raise ValueError("duplicate packed word address")
            seen.add(address)

    @property
    def words_per_column(self) -> int:
        return (self.side + 63) // 64

    @property
    def logical_bits(self) -> int:
        return self.side**3

    @property
    def logical_words(self) -> int:
        return self.side * self.side * self.words_per_column

    @property
    def packed_words(self) -> int:
        return len(self.words)

    @property
    def active_bits(self) -> int:
        return sum(word.bit_count() for _, _, _, word in self.words)

    @property
    def density(self) -> float:
        return self.active_bits / self.logical_bits

    @property
    def entropy(self) -> float:
        p = self.density
        if p <= 0.0 or p >= 1.0:
            return 0.0
        return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))

    @property
    def checksum_sha256(self) -> str:
        payload = json.dumps(self.words, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def as_word_map(self) -> dict[PackedAddress, int]:
        return {(x, y, q): word for x, y, q, word in self.words}

    def hamming_fraction(self, previous: "SparseBitPlane3D") -> float:
        if self.side != previous.side:
            raise ValueError("bit-plane sides must match")
        left = self.as_word_map()
        right = previous.as_word_map()
        flips = 0
        for address in set(left) | set(right):
            flips += (left.get(address, 0) ^ right.get(address, 0)).bit_count()
        return flips / self.logical_bits

    def metrics(self, previous: "SparseBitPlane3D | None" = None) -> BitPlaneMetrics:
        velocity = 0.0 if previous is None else self.hamming_fraction(previous)
        energy = 0.5 * self.density * velocity * velocity
        return BitPlaneMetrics(
            active_bits=self.active_bits,
            logical_bits=self.logical_bits,
            packed_words=self.packed_words,
            logical_words=self.logical_words,
            density=self.density,
            entropy=self.entropy,
            phase_velocity=velocity,
            kinetic_energy=energy,
        )

    def sample_words(self, limit: int = 256) -> list[dict[str, int]]:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        return [
            {"x": x, "y": y, "q": q, "word": word}
            for x, y, q, word in self.words[:limit]
        ]

    @classmethod
    def from_scalar_field(
        cls,
        field: Mapping[Coordinate, float],
        *,
        side: int,
        activation_threshold: float = 0.5,
    ) -> "SparseBitPlane3D":
        if isinstance(side, bool) or not isinstance(side, int) or side <= 0:
            raise ValueError("side must be a positive integer")
        if not math.isfinite(float(activation_threshold)) or activation_threshold < 0.0:
            raise ValueError("activation_threshold must be finite and non-negative")

        packed: dict[PackedAddress, int] = {}
        for coordinate, raw_value in field.items():
            if len(coordinate) != 3 or any(
                isinstance(axis, bool) or not isinstance(axis, int) for axis in coordinate
            ):
                raise TypeError("field coordinates must be integer triples")
            x, y, z = coordinate
            if not (0 <= x < side and 0 <= y < side and 0 <= z < side):
                raise ValueError("field coordinate outside logical lattice")
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError("field values must be finite")
            if abs(value) < activation_threshold:
                continue
            q, bit = divmod(z, 64)
            address = (x, y, q)
            packed[address] = packed.get(address, 0) | (1 << bit)

        words = tuple((x, y, q, packed[(x, y, q)]) for x, y, q in sorted(packed))
        return cls(side=side, words=words)


def fold_and_attenuate(
    field: Mapping[Coordinate, float],
    *,
    side: int,
    contraction: float,
    attenuation: float,
    prune_epsilon: float = 0.0,
) -> SparseField:
    """Contract sparse coordinates toward the centroid and attenuate radially.

    ``contraction=0`` preserves coordinates while values may still attenuate.
    ``contraction -> 1`` moves coordinates toward the centroid.  Collisions are
    resolved by retaining the value with the greatest absolute magnitude, which
    keeps the operation deterministic and prevents collision-driven amplification.
    """

    if isinstance(side, bool) or not isinstance(side, int) or side <= 0:
        raise ValueError("side must be a positive integer")
    for name, value in (("contraction", contraction), ("attenuation", attenuation)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if not 0.0 <= contraction < 1.0:
        raise ValueError("contraction must be in [0, 1)")
    if attenuation < 0.0:
        raise ValueError("attenuation must be non-negative")
    if not math.isfinite(float(prune_epsilon)) or prune_epsilon < 0.0:
        raise ValueError("prune_epsilon must be finite and non-negative")

    center = (side - 1) / 2.0
    max_axis = max(center, (side - 1) - center)
    max_radius_sq = max(1.0, 3.0 * max_axis * max_axis)
    scale = 1.0 - contraction
    result: SparseField = {}

    for coordinate in sorted(field):
        x, y, z = coordinate
        if not (0 <= x < side and 0 <= y < side and 0 <= z < side):
            raise ValueError("field coordinate outside logical lattice")
        value = float(field[coordinate])
        if not math.isfinite(value):
            raise ValueError("field values must be finite")

        dx, dy, dz = x - center, y - center, z - center
        radius_sq = dx * dx + dy * dy + dz * dz
        radial_weight = math.exp(-attenuation * radius_sq / max_radius_sq)
        folded_value = value * radial_weight
        if abs(folded_value) <= prune_epsilon:
            continue

        target = (
            min(side - 1, max(0, int(round(center + scale * dx)))),
            min(side - 1, max(0, int(round(center + scale * dy)))),
            min(side - 1, max(0, int(round(center + scale * dz)))),
        )
        existing = result.get(target)
        if existing is None or abs(folded_value) > abs(existing):
            result[target] = folded_value

    return result
