"""Bounded text-to-quantum information reference.

This module makes one narrow claim executable: UTF-8 text can be mapped exactly
onto computational-basis qubits and manipulated by a small sparse state
simulator.  It does *not* claim quantum speedup, physical entanglement of
semantic concepts, consciousness, revelation, or any identification between a
wavefunction and a theological referent.

The implementation is intentionally dependency-free and sparse.  A basis state
for a long text stores one Python integer and one complex amplitude rather than
allocating a dense vector of size ``2**N``.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from typing import Iterable, Sequence

BitSequence = Sequence[int]
AmplitudeEntry = tuple[int, complex]


def _validate_bits(bits: Iterable[int]) -> tuple[int, ...]:
    values = tuple(bits)
    if not values:
        raise ValueError("at least one bit is required")
    for bit in values:
        if isinstance(bit, bool) or not isinstance(bit, int) or bit not in (0, 1):
            raise ValueError("bits must contain only integer 0 or 1 values")
    return values


def utf8_to_bits(text: str) -> tuple[int, ...]:
    """Encode non-empty text as big-endian bits for each UTF-8 byte."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    raw = text.encode("utf-8")
    if not raw:
        raise ValueError("text must not be empty")
    return tuple((byte >> shift) & 1 for byte in raw for shift in range(7, -1, -1))


def bits_to_utf8(bits: BitSequence) -> str:
    """Invert :func:`utf8_to_bits` for a byte-aligned bit sequence."""

    values = _validate_bits(bits)
    if len(values) % 8:
        raise ValueError("UTF-8 bit sequences must be byte aligned")

    raw = bytearray()
    for offset in range(0, len(values), 8):
        byte = 0
        for bit in values[offset : offset + 8]:
            byte = (byte << 1) | bit
        raw.append(byte)
    return bytes(raw).decode("utf-8")


def bits_to_index(bits: BitSequence) -> int:
    """Map a bit string to its computational-basis integer index."""

    values = _validate_bits(bits)
    index = 0
    for bit in values:
        index = (index << 1) | bit
    return index


def index_to_bits(index: int, qubits: int) -> tuple[int, ...]:
    """Map a basis integer back to exactly ``qubits`` big-endian bits."""

    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("index must be a non-negative integer")
    if isinstance(qubits, bool) or not isinstance(qubits, int) or qubits < 1:
        raise ValueError("qubits must be a positive integer")
    if index >= (1 << qubits):
        raise ValueError("index does not fit inside the requested qubit width")
    return tuple((index >> shift) & 1 for shift in range(qubits - 1, -1, -1))


@dataclass(frozen=True)
class BasisEncoding:
    """Exact classical text represented as one quantum computational basis state."""

    text: str
    bits: tuple[int, ...]
    basis_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        if not isinstance(self.bits, tuple):
            raise TypeError("bits must be a tuple")
        validated_bits = _validate_bits(self.bits)
        expected_bits = utf8_to_bits(self.text)
        if validated_bits != expected_bits:
            raise ValueError("bits do not match the UTF-8 encoding of text")
        if isinstance(self.basis_index, bool) or not isinstance(self.basis_index, int):
            raise TypeError("basis_index must be an integer")
        expected_index = bits_to_index(validated_bits)
        if self.basis_index != expected_index:
            raise ValueError("basis_index does not match bits")

    @classmethod
    def from_text(cls, text: str) -> "BasisEncoding":
        bits = utf8_to_bits(text)
        return cls(text=text, bits=bits, basis_index=bits_to_index(bits))

    @property
    def qubit_count(self) -> int:
        return len(self.bits)

    @property
    def ket(self) -> str:
        return "|" + "".join(str(bit) for bit in self.bits) + ">"

    def decode(self) -> str:
        return bits_to_utf8(self.bits)


def _canonicalize(
    amplitudes: dict[int, complex], tolerance: float = 1.0e-15
) -> tuple[AmplitudeEntry, ...]:
    entries = [
        (index, amplitude)
        for index, amplitude in amplitudes.items()
        if abs(amplitude) > tolerance
    ]
    entries.sort(key=lambda item: item[0])
    return tuple(entries)


@dataclass(frozen=True)
class SparseQuantumState:
    """Small sparse pure-state simulator for demonstrable quantum operations.

    ``qubit=0`` in gate methods refers to the least-significant basis bit.  The
    state is a mathematical simulator, not a hardware backend.
    """

    qubits: int
    amplitudes: tuple[AmplitudeEntry, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.qubits, bool)
            or not isinstance(self.qubits, int)
            or self.qubits < 1
        ):
            raise ValueError("qubits must be a positive integer")
        if not self.amplitudes:
            raise ValueError("at least one non-zero amplitude is required")

        seen: set[int] = set()
        norm = 0.0
        for index, amplitude in self.amplitudes:
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                raise ValueError("basis indices must be non-negative integers")
            if index >= (1 << self.qubits):
                raise ValueError("basis index exceeds the qubit width")
            if index in seen:
                raise ValueError("basis indices must be unique")
            seen.add(index)
            value = complex(amplitude)
            if not (math.isfinite(value.real) and math.isfinite(value.imag)):
                raise ValueError("amplitudes must be finite")
            norm += abs(value) ** 2

        if not math.isclose(norm, 1.0, rel_tol=1.0e-12, abs_tol=1.0e-12):
            raise ValueError("state amplitudes must be normalized")

    @classmethod
    def basis(cls, bits: BitSequence) -> "SparseQuantumState":
        values = _validate_bits(bits)
        return cls(len(values), ((bits_to_index(values), 1.0 + 0.0j),))

    @classmethod
    def from_text(cls, text: str) -> "SparseQuantumState":
        return cls.basis(utf8_to_bits(text))

    def amplitude(self, basis_index: int) -> complex:
        for index, amplitude in self.amplitudes:
            if index == basis_index:
                return amplitude
        return 0.0 + 0.0j

    def probabilities(self) -> tuple[tuple[int, float], ...]:
        """Return the exact measurement distribution over represented basis states."""

        return tuple((index, abs(amplitude) ** 2) for index, amplitude in self.amplitudes)

    def hadamard(self, qubit: int) -> "SparseQuantumState":
        """Apply a Hadamard gate to one qubit without allocating a dense state."""

        if (
            isinstance(qubit, bool)
            or not isinstance(qubit, int)
            or not 0 <= qubit < self.qubits
        ):
            raise ValueError("qubit is outside the state")

        mask = 1 << qubit
        scale = 1.0 / math.sqrt(2.0)
        updated: dict[int, complex] = {}
        for index, amplitude in self.amplitudes:
            bit_is_one = bool(index & mask)
            zero_index = index & ~mask
            one_index = zero_index | mask
            updated[zero_index] = updated.get(zero_index, 0.0j) + amplitude * scale
            sign = -1.0 if bit_is_one else 1.0
            updated[one_index] = updated.get(one_index, 0.0j) + amplitude * scale * sign
        return SparseQuantumState(self.qubits, _canonicalize(updated))

    def phase(self, basis_index: int, radians: float) -> "SparseQuantumState":
        """Apply a phase only to one represented computational-basis component."""

        if isinstance(basis_index, bool) or not isinstance(basis_index, int):
            raise ValueError("basis_index must be an integer")
        if basis_index < 0 or basis_index >= (1 << self.qubits):
            raise ValueError("basis_index is outside the state")
        if not math.isfinite(radians):
            raise ValueError("radians must be finite")

        factor = cmath.exp(1.0j * radians)
        updated = tuple(
            (index, amplitude * factor if index == basis_index else amplitude)
            for index, amplitude in self.amplitudes
        )
        return SparseQuantumState(self.qubits, updated)

    def inner_product(self, other: "SparseQuantumState") -> complex:
        """Return ``<self|other>`` for states of equal width."""

        if self.qubits != other.qubits:
            raise ValueError("states must have the same qubit width")
        right = dict(other.amplitudes)
        return sum(
            amplitude.conjugate() * right.get(index, 0.0j)
            for index, amplitude in self.amplitudes
        )


def text_basis_state(text: str) -> tuple[BasisEncoding, SparseQuantumState]:
    """Return both the reversible UTF-8 record and its exact basis-state model."""

    encoding = BasisEncoding.from_text(text)
    return encoding, SparseQuantumState.basis(encoding.bits)
