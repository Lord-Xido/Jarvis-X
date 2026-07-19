"""Deterministic reference runtime for the MM3D-AED-BCE-Ω⁴ cosmogram.

The runtime implements a bounded five-stage cycle:

    encode -> policy project -> Z8 evolve -> codebook decode -> SHA3 ledger

It intentionally separates a state digest from the chained ledger digest. A digest
proves content identity; chaining and append-only storage provide tamper evidence.
"""

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable, Mapping, Optional, Sequence, Tuple


Voxel = bytes
Lattice = Tuple[int, ...]
DecodedState = Tuple[bytes, ...]


@dataclass(frozen=True)
class CosmogramConfig:
    """Static execution contract for one cosmogram instance."""

    side: int = 32
    voxel_bits: int = 384
    latent_modulus: int = 1 << 18
    reaction_modulus: int = 8
    shift: int = 1
    neighbor_weight: int = 1
    boundary: str = "bounded"
    neutral_index: int = 0
    reality_gap: str = "gamma=infinity"

    def __post_init__(self) -> None:
        if self.side <= 0:
            raise ValueError("side must be positive")
        if self.voxel_bits <= 0 or self.voxel_bits % 8 != 0:
            raise ValueError("voxel_bits must be a positive multiple of 8")
        if self.latent_modulus <= 1:
            raise ValueError("latent_modulus must exceed one")
        if self.reaction_modulus <= 1:
            raise ValueError("reaction_modulus must exceed one")
        if self.shift < 0:
            raise ValueError("shift must be non-negative")
        if self.boundary not in {"bounded", "periodic"}:
            raise ValueError("boundary must be 'bounded' or 'periodic'")
        if not 0 <= self.neutral_index < self.latent_modulus:
            raise ValueError("neutral_index must lie in the latent index range")

    @property
    def cell_count(self) -> int:
        return self.side ** 3

    @property
    def voxel_bytes(self) -> int:
        return self.voxel_bits // 8

    def fingerprint(self) -> bytes:
        payload = json.dumps(
            {
                "boundary": self.boundary,
                "latent_modulus": self.latent_modulus,
                "neighbor_weight": self.neighbor_weight,
                "neutral_index": self.neutral_index,
                "reaction_modulus": self.reaction_modulus,
                "reality_gap": self.reality_gap,
                "shift": self.shift,
                "side": self.side,
                "voxel_bits": self.voxel_bits,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha3_256(payload).digest()


@dataclass(frozen=True)
class CosmogramReceipt:
    """Complete deterministic receipt for one committed cycle."""

    cycle: int
    encoded: Lattice
    masked: Lattice
    evolved: Lattice
    decoded: DecodedState
    state_hash: str
    previous_chain_hash: str
    chain_hash: str


class MM3DCosmogram:
    """Reference implementation of the recursive 3D Dr Moagi cycle.

    The default encoder is deliberately compressive: each 384-bit voxel is mapped
    to an 18-bit index using SHA3-256. Exact reconstruction therefore requires the
    supplied codebook/domain contract; the hash projection itself is not invertible.
    """

    GENESIS_HASH = "0" * 64

    def __init__(
        self,
        codebook: Mapping[int, bytes],
        config: Optional[CosmogramConfig] = None,
        neutral_atom: bytes = b"<VOID>",
    ) -> None:
        self.config = config or CosmogramConfig()
        self.codebook = dict(codebook)
        self.neutral_atom = bytes(neutral_atom)
        self._cycle = 0
        self._chain_hash = self.GENESIS_HASH
        self._receipts = []  # type: list[CosmogramReceipt]

    @property
    def receipts(self) -> Tuple[CosmogramReceipt, ...]:
        return tuple(self._receipts)

    @property
    def chain_hash(self) -> str:
        return self._chain_hash

    def encode(self, voxels: Sequence[Voxel]) -> Lattice:
        self._validate_voxels(voxels)
        mask = self.config.latent_modulus - 1
        power_of_two = self.config.latent_modulus & mask == 0
        encoded = []
        for voxel in voxels:
            digest_value = int.from_bytes(hashlib.sha3_256(voxel).digest(), "big")
            index = digest_value & mask if power_of_two else digest_value % self.config.latent_modulus
            encoded.append(index)
        return tuple(encoded)

    def project_policy(self, encoded: Sequence[int], allow: Sequence[bool]) -> Lattice:
        self._validate_lattice(encoded, self.config.latent_modulus, "encoded")
        if len(allow) != self.config.cell_count:
            raise ValueError("policy mask length must equal side^3")
        return tuple(
            value if bool(is_allowed) else self.config.neutral_index
            for value, is_allowed in zip(encoded, allow)
        )

    def evolve(self, masked: Sequence[int]) -> Lattice:
        self._validate_lattice(masked, self.config.latent_modulus, "masked")
        side = self.config.side
        modulus = self.config.reaction_modulus
        multiplier = 1 << self.config.shift
        output = [0] * self.config.cell_count

        for z in range(side):
            for y in range(side):
                for x in range(side):
                    index = self._flat_index(x, y, z)
                    total = masked[index]
                    for nx, ny, nz in self._neighbors(x, y, z):
                        total += self.config.neighbor_weight * masked[
                            self._flat_index(nx, ny, nz)
                        ]
                    output[index] = (total * multiplier) % modulus
        return tuple(output)

    def decode(self, evolved: Sequence[int]) -> DecodedState:
        self._validate_lattice(evolved, self.config.reaction_modulus, "evolved")
        return tuple(self.codebook.get(index, self.neutral_atom) for index in evolved)

    def step(self, voxels: Sequence[Voxel], allow: Sequence[bool]) -> CosmogramReceipt:
        encoded = self.encode(voxels)
        masked = self.project_policy(encoded, allow)
        evolved = self.evolve(masked)
        decoded = self.decode(evolved)

        state_digest = hashlib.sha3_256(self._canonical_decoded(decoded)).digest()
        previous = self._chain_hash
        chain_payload = (
            bytes.fromhex(previous)
            + state_digest
            + self._cycle.to_bytes(8, "big", signed=False)
            + self.config.fingerprint()
        )
        chain_hash = hashlib.sha3_256(chain_payload).hexdigest()

        receipt = CosmogramReceipt(
            cycle=self._cycle,
            encoded=encoded,
            masked=masked,
            evolved=evolved,
            decoded=decoded,
            state_hash=state_digest.hex(),
            previous_chain_hash=previous,
            chain_hash=chain_hash,
        )
        self._receipts.append(receipt)
        self._chain_hash = chain_hash
        self._cycle += 1
        return receipt

    def verify(self) -> bool:
        previous = self.GENESIS_HASH
        expected_cycle = 0
        for receipt in self._receipts:
            if receipt.cycle != expected_cycle:
                return False
            if receipt.previous_chain_hash != previous:
                return False
            state_digest = hashlib.sha3_256(
                self._canonical_decoded(receipt.decoded)
            ).digest()
            if receipt.state_hash != state_digest.hex():
                return False
            payload = (
                bytes.fromhex(previous)
                + state_digest
                + receipt.cycle.to_bytes(8, "big", signed=False)
                + self.config.fingerprint()
            )
            expected = hashlib.sha3_256(payload).hexdigest()
            if receipt.chain_hash != expected:
                return False
            previous = expected
            expected_cycle += 1
        return previous == self._chain_hash

    def _validate_voxels(self, voxels: Sequence[Voxel]) -> None:
        if len(voxels) != self.config.cell_count:
            raise ValueError("voxel count must equal side^3")
        for voxel in voxels:
            if not isinstance(voxel, (bytes, bytearray, memoryview)):
                raise TypeError("each voxel must be bytes-like")
            if len(voxel) != self.config.voxel_bytes:
                raise ValueError(
                    "each voxel must contain exactly {} bytes".format(
                        self.config.voxel_bytes
                    )
                )

    def _validate_lattice(self, values: Sequence[int], modulus: int, label: str) -> None:
        if len(values) != self.config.cell_count:
            raise ValueError("{} lattice length must equal side^3".format(label))
        if any(not isinstance(value, int) or not 0 <= value < modulus for value in values):
            raise ValueError(
                "{} lattice values must be integers in [0, {})".format(label, modulus)
            )

    def _flat_index(self, x: int, y: int, z: int) -> int:
        side = self.config.side
        return x + side * (y + side * z)

    def _neighbors(self, x: int, y: int, z: int) -> Iterable[Tuple[int, int, int]]:
        side = self.config.side
        for dx, dy, dz in (
            (-1, 0, 0),
            (1, 0, 0),
            (0, -1, 0),
            (0, 1, 0),
            (0, 0, -1),
            (0, 0, 1),
        ):
            nx, ny, nz = x + dx, y + dy, z + dz
            if self.config.boundary == "periodic":
                yield nx % side, ny % side, nz % side
            elif 0 <= nx < side and 0 <= ny < side and 0 <= nz < side:
                yield nx, ny, nz

    @staticmethod
    def _canonical_decoded(decoded: Sequence[bytes]) -> bytes:
        payload = bytearray()
        for atom in decoded:
            raw = bytes(atom)
            payload.extend(len(raw).to_bytes(4, "big", signed=False))
            payload.extend(raw)
        return bytes(payload)
