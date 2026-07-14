"""Bit-packed topological coding engine for Jarvis-X.

JX-AAPE-Ω evolves a Boolean 3-D toroidal lattice and extracts symbolic tokens
from the resulting topology. The state belongs to F_2^N and is stored as
N/64 independent 64-bit words; no F_(2^64) field arithmetic is implied.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable, Optional, Sequence, Tuple


DEFAULT_PYTHON_VOCABULARY: Tuple[str, ...] = (
    "def",
    "class",
    "if",
    "else",
    "for",
    "while",
    "return",
    "yield",
    "import",
    "from",
    "try",
    "except",
    "with",
    "lambda",
    "async",
    "await",
    "True",
    "False",
    "None",
    "and",
    "or",
    "not",
)


@dataclass(frozen=True)
class AAPEConfig:
    """Configuration for the 64^3 Boolean-lattice reference engine."""

    side: int = 64
    word_bits: int = 64
    semantic_threshold: int = 0x4000
    kappa_initial: int = 4
    kappa_min: int = 1
    kappa_max: int = 7
    max_ca_steps: int = 3
    max_tokens: int = 256
    semantic_gap_epsilon: float = 1.0 / 262_144.0
    vocabulary: Tuple[str, ...] = DEFAULT_PYTHON_VOCABULARY

    def __post_init__(self) -> None:
        if self.side <= 0:
            raise ValueError("side must be positive")
        if self.word_bits != 64:
            raise ValueError("the reference packing format requires 64-bit words")
        if (self.side ** 3) % self.word_bits != 0:
            raise ValueError("side^3 must be divisible by 64")
        if not 0 <= self.semantic_threshold <= 0xFFFF:
            raise ValueError("semantic_threshold must fit in unsigned 16-bit range")
        if not self.kappa_min <= self.kappa_initial <= self.kappa_max:
            raise ValueError("kappa_initial must lie inside [kappa_min, kappa_max]")
        if self.max_ca_steps < 1:
            raise ValueError("max_ca_steps must be at least one")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be at least one")
        if self.semantic_gap_epsilon <= 0.0:
            raise ValueError("semantic_gap_epsilon must be positive")
        if not self.vocabulary:
            raise ValueError("vocabulary must not be empty")


@dataclass(frozen=True)
class BitLattice:
    """Immutable packed element of F_2^N."""

    side: int
    bits: int

    def __post_init__(self) -> None:
        if self.side <= 0:
            raise ValueError("side must be positive")
        if self.bits < 0:
            raise ValueError("bits must be non-negative")
        if self.bits >> self.voxel_count:
            raise ValueError("bits contain coordinates outside the lattice")

    @property
    def voxel_count(self) -> int:
        return self.side ** 3

    @property
    def word_count(self) -> int:
        return self.voxel_count // 64

    @property
    def active_count(self) -> int:
        return self.bits.bit_count()

    @property
    def density(self) -> float:
        return self.active_count / self.voxel_count

    def words(self) -> Tuple[int, ...]:
        mask = (1 << 64) - 1
        return tuple((self.bits >> (64 * index)) & mask for index in range(self.word_count))

    def to_bytes(self) -> bytes:
        return self.bits.to_bytes(self.word_count * 8, byteorder="little", signed=False)

    @classmethod
    def from_words(cls, side: int, words: Sequence[int]) -> "BitLattice":
        expected = (side ** 3) // 64
        if len(words) != expected:
            raise ValueError(f"expected {expected} words, received {len(words)}")
        bits = 0
        for index, word in enumerate(words):
            if not 0 <= int(word) <= 0xFFFFFFFFFFFFFFFF:
                raise ValueError("each word must fit in unsigned 64 bits")
            bits |= int(word) << (64 * index)
        return cls(side=side, bits=bits)


@dataclass(frozen=True)
class AAPEState:
    """Committed state after one encode/project/decode/hash cycle."""

    cycle: int
    kappa_used: int
    kappa_next: int
    encoded: BitLattice
    projected: BitLattice
    intent_mask: BitLattice
    tokens: Tuple[str, ...]
    omega_digest: str
    encoded_density: float
    projected_density: float
    convergence: str
    ca_steps: int
    semantic_gap: float
    representation_tag: str = "SIMULATION_NOT_TERRITORY"


class _ToroidalTopology:
    """Bit-parallel neighbor transforms for an L×L×L torus."""

    def __init__(self, side: int) -> None:
        self.side = side
        self.plane_bits = side * side
        self.voxel_count = side ** 3
        self.full_mask = (1 << self.voxel_count) - 1

        row_start = 0
        row_end = 0
        for row in range(side * side):
            base = row * side
            row_start |= 1 << base
            row_end |= 1 << (base + side - 1)
        self.row_start = row_start
        self.row_end = row_end

        plane_start = 0
        plane_end = 0
        row_block = (1 << side) - 1
        for z in range(side):
            base = z * self.plane_bits
            plane_start |= row_block << base
            plane_end |= row_block << (base + side * (side - 1))
        self.plane_start = plane_start
        self.plane_end = plane_end

        self.low_plane = (1 << self.plane_bits) - 1
        self.high_plane = self.low_plane << (self.plane_bits * (side - 1))

    def neighbors(self, bits: int) -> Tuple[int, int, int, int, int, int]:
        side = self.side
        plane = self.plane_bits
        full = self.full_mask

        x_plus = ((bits >> 1) & (full ^ self.row_end)) | (
            (bits & self.row_start) << (side - 1)
        )
        x_minus = (((bits << 1) & full) & (full ^ self.row_start)) | (
            (bits & self.row_end) >> (side - 1)
        )

        y_plus = ((bits >> side) & (full ^ self.plane_end)) | (
            (bits & self.plane_start) << (side * (side - 1))
        )
        y_minus = (((bits << side) & full) & (full ^ self.plane_start)) | (
            (bits & self.plane_end) >> (side * (side - 1))
        )

        z_plus = (bits >> plane) | ((bits & self.low_plane) << (plane * (side - 1)))
        z_minus = ((bits << plane) & full) | (
            (bits & self.high_plane) >> (plane * (side - 1))
        )
        return x_plus, x_minus, y_plus, y_minus, z_plus, z_minus


class JXAAPEEngine:
    """Deterministic Boolean-topology engine with bounded CA projection."""

    def __init__(self, config: Optional[AAPEConfig] = None) -> None:
        self.config = config or AAPEConfig()
        self._topology = _ToroidalTopology(self.config.side)
        self._front: Optional[AAPEState] = None
        self._back: Optional[AAPEState] = None
        self._cycle = 0
        self._kappa = self.config.kappa_initial
        self._omega = bytes(32)

    @property
    def front_state(self) -> Optional[AAPEState]:
        return self._front

    @property
    def kappa(self) -> int:
        return self._kappa

    @property
    def omega_digest(self) -> str:
        return self._omega.hex()

    def lattice(self, active_indices: Iterable[int]) -> BitLattice:
        bits = 0
        for raw_index in active_indices:
            index = int(raw_index)
            if not 0 <= index < self._topology.voxel_count:
                raise ValueError("active index lies outside the lattice")
            bits |= 1 << index
        return BitLattice(self.config.side, bits)

    def full_lattice(self) -> BitLattice:
        return BitLattice(self.config.side, self._topology.full_mask)

    def encode(self, embeddings: Sequence[int], *, kappa: Optional[int] = None) -> BitLattice:
        """Inject unsigned 16-bit embeddings through deterministic parity selection."""

        if not embeddings:
            raise ValueError("embeddings must not be empty")
        depth = self._validate_kappa(self._kappa if kappa is None else kappa)
        bits = 0
        for index, raw_value in enumerate(embeddings):
            value = int(raw_value)
            if not 0 <= value <= 0xFFFF:
                raise ValueError("embeddings must be unsigned 16-bit integers")
            key = self._feedback_key(index, depth)
            mixed = value ^ key
            if value > self.config.semantic_threshold and (mixed.bit_count() & 1):
                site = self._splitmix64((index << 16) ^ mixed ^ (depth << 56))
                bits |= 1 << (site % self._topology.voxel_count)
        return BitLattice(self.config.side, bits)

    def project_once(self, lattice: BitLattice) -> BitLattice:
        """Apply exact majority-of-seven to center plus the six axial neighbors."""

        self._validate_lattice(lattice, "lattice")
        neighbors = self._topology.neighbors(lattice.bits)
        projected = self._majority7_exact(*neighbors, lattice.bits)
        return BitLattice(self.config.side, projected & self._topology.full_mask)

    def project(
        self,
        lattice: BitLattice,
        *,
        lambda_mask: Optional[BitLattice] = None,
        semantic_anchor: Optional[BitLattice] = None,
    ) -> Tuple[BitLattice, str, int]:
        """Run a bounded synchronous CA, detecting fixed points and period-2 cycles."""

        self._validate_lattice(lattice, "lattice")
        gate = lambda_mask or self.full_lattice()
        anchor = semantic_anchor or BitLattice(self.config.side, 0)
        self._validate_lattice(gate, "lambda_mask")
        self._validate_lattice(anchor, "semantic_anchor")

        current = BitLattice(self.config.side, (lattice.bits & gate.bits) | (anchor.bits & gate.bits))
        previous_bits: Optional[int] = None
        for step in range(1, self.config.max_ca_steps + 1):
            next_state = self.project_once(current)
            next_bits = (next_state.bits & gate.bits) | (anchor.bits & gate.bits)
            if next_bits == current.bits:
                return BitLattice(self.config.side, next_bits), "fixed_point", step
            if previous_bits is not None and next_bits == previous_bits:
                return BitLattice(self.config.side, next_bits), "period_2", step
            previous_bits = current.bits
            current = BitLattice(self.config.side, next_bits)
        return current, "step_budget", self.config.max_ca_steps

    def decode(
        self,
        lattice: BitLattice,
        *,
        intent_mask: Optional[BitLattice] = None,
        max_tokens: Optional[int] = None,
    ) -> Tuple[str, ...]:
        """Extract vocabulary entries from active topology through a sparse intent gate."""

        self._validate_lattice(lattice, "lattice")
        gate = intent_mask or self.full_lattice()
        self._validate_lattice(gate, "intent_mask")
        limit = self.config.max_tokens if max_tokens is None else int(max_tokens)
        if limit < 1:
            raise ValueError("max_tokens must be at least one")

        active = lattice.bits & gate.bits
        output = []
        vocabulary = self.config.vocabulary
        side = self.config.side
        plane = side * side
        while active and len(output) < limit:
            least = active & -active
            linear = least.bit_length() - 1
            z, rem = divmod(linear, plane)
            y, x = divmod(rem, side)
            morton = self._morton3(x, y, z)
            output.append(vocabulary[morton % len(vocabulary)])
            active &= active - 1
        return tuple(output)

    def update_kappa(self, quality_signal: int) -> int:
        """Apply κ[t+1]=clip(κ[t]+(-1)^q, κ_min, κ_max).

        Convention: q=1 requests stronger coupling (decrement κ); q=0 requests
        weaker coupling (increment κ). This is a control convention, not a proof
        that parity density is monotone in κ.
        """

        q = int(quality_signal)
        if q not in (0, 1):
            raise ValueError("quality_signal must be 0 or 1")
        delta = -1 if q == 1 else 1
        self._kappa = min(max(self._kappa + delta, self.config.kappa_min), self.config.kappa_max)
        return self._kappa

    def cycle(
        self,
        embeddings: Sequence[int],
        *,
        intent_mask: Optional[BitLattice] = None,
        lambda_mask: Optional[BitLattice] = None,
        quality_signal: int = 1,
        lambda_tag: bytes = b"lambda-default",
        max_tokens: Optional[int] = None,
    ) -> AAPEState:
        """Execute encode → CA projection → sparse decode → Ω hash → buffer swap."""

        if not isinstance(lambda_tag, bytes):
            raise TypeError("lambda_tag must be bytes")
        decode_gate = intent_mask or self.full_lattice()
        semantic_anchor = intent_mask or BitLattice(self.config.side, 0)
        self._validate_lattice(decode_gate, "intent_mask")
        kappa_used = self._kappa
        encoded = self.encode(embeddings, kappa=kappa_used)
        projected, convergence, steps = self.project(
            encoded,
            lambda_mask=lambda_mask,
            semantic_anchor=semantic_anchor,
        )
        tokens = self.decode(projected, intent_mask=decode_gate, max_tokens=max_tokens)

        lambda_bytes = lambda_mask.to_bytes() if lambda_mask is not None else b""
        self._omega = hashlib.sha3_256(
            self._omega + projected.to_bytes() + lambda_tag + lambda_bytes
        ).digest()
        kappa_next = self.update_kappa(quality_signal)

        self._cycle += 1
        self._back = AAPEState(
            cycle=self._cycle,
            kappa_used=kappa_used,
            kappa_next=kappa_next,
            encoded=encoded,
            projected=projected,
            intent_mask=decode_gate,
            tokens=tokens,
            omega_digest=self._omega.hex(),
            encoded_density=encoded.density,
            projected_density=projected.density,
            convergence=convergence,
            ca_steps=steps,
            semantic_gap=self.config.semantic_gap_epsilon,
        )
        self._front, self._back = self._back, self._front
        return self._front

    def _validate_lattice(self, lattice: BitLattice, name: str) -> None:
        if not isinstance(lattice, BitLattice):
            raise TypeError(f"{name} must be a BitLattice")
        if lattice.side != self.config.side:
            raise ValueError(f"{name} side must equal {self.config.side}")

    def _validate_kappa(self, kappa: int) -> int:
        result = int(kappa)
        if not self.config.kappa_min <= result <= self.config.kappa_max:
            raise ValueError("kappa lies outside configured bounds")
        return result

    @staticmethod
    def _majority3_parts(a: int, b: int, c: int) -> Tuple[int, int]:
        ab_xor = a ^ b
        parity = ab_xor ^ c
        carry = (a & b) | (c & ab_xor)
        return parity, carry

    @classmethod
    def _majority7_exact(
        cls,
        n1: int,
        n2: int,
        n3: int,
        n4: int,
        n5: int,
        n6: int,
        center: int,
    ) -> int:
        parity_a, carry_a = cls._majority3_parts(n1, n2, n3)
        parity_b, carry_b = cls._majority3_parts(n4, n5, n6)
        low_xor = parity_a ^ parity_b
        low_majority = (parity_a & parity_b) | (center & low_xor)
        return (carry_a & carry_b) | ((carry_a ^ carry_b) & low_majority)

    @staticmethod
    def _feedback_key(index: int, kappa: int) -> int:
        base = JXAAPEEngine._splitmix64(index + 0x9E3779B97F4A7C15)
        rotation = kappa % 16
        word = base & 0xFFFF
        return ((word >> rotation) | (word << (16 - rotation))) & 0xFFFF

    @staticmethod
    def _splitmix64(value: int) -> int:
        mask = 0xFFFFFFFFFFFFFFFF
        z = (int(value) + 0x9E3779B97F4A7C15) & mask
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
        return (z ^ (z >> 31)) & mask

    @staticmethod
    def _morton3(x: int, y: int, z: int) -> int:
        result = 0
        bit = 0
        limit = max(x.bit_length(), y.bit_length(), z.bit_length(), 1)
        while bit < limit:
            result |= ((x >> bit) & 1) << (3 * bit)
            result |= ((y >> bit) & 1) << (3 * bit + 1)
            result |= ((z >> bit) & 1) << (3 * bit + 2)
            bit += 1
        return result
