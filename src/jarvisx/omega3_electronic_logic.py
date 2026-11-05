"""Deterministic bit-level reference for the Omega3 electronic commit core.

This module maps a small, explicitly bounded part of the Omega3/Dr Moagi
research architecture onto ordinary synchronous digital-logic semantics:
fixed-width words, bit masks, signed Q1.15 adaptive memory, Hamming-distance
convergence and a fail-closed commit/rollback gate.

It is a software reference for RTL conformance. It does not claim a fabricated
chip, measured transistor performance, or autonomous source-code mutation.
"""

from __future__ import annotations

from dataclasses import dataclass

WORD_BITS = 64
WORD_MASK = (1 << WORD_BITS) - 1
XYZ_BITS = 10
XYZ_MASK = (1 << XYZ_BITS) - 1
XYZ_SIDE = 1000
REQUIRED_LAMBDA_MASK = 0xFF
Q15_SHIFT = 15
Q15_MIN = -(1 << 15)
Q15_MAX = (1 << 15) - 1


def _require_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _require_word(value: int, name: str) -> int:
    value = _require_int(value, name)
    if value < 0 or value > WORD_MASK:
        raise ValueError(f"{name} must fit in {WORD_BITS} bits")
    return value


def _require_u8(value: int, name: str) -> int:
    value = _require_int(value, name)
    if value < 0 or value > 0xFF:
        raise ValueError(f"{name} must fit in 8 bits")
    return value


def _saturate_q15(value: int) -> int:
    return max(Q15_MIN, min(Q15_MAX, int(value)))


def pack_xyz(x: int, y: int, z: int) -> int:
    """Pack a logical 1000^3 coordinate into a 30-bit integer."""

    checked = []
    for name, component in (("x", x), ("y", y), ("z", z)):
        component = _require_int(component, name)
        if component < 0 or component >= XYZ_SIDE:
            raise ValueError(f"{name} must be in [0, {XYZ_SIDE - 1}]")
        checked.append(component)
    x, y, z = checked
    return (x << 20) | (y << 10) | z


def unpack_xyz(packed: int) -> tuple[int, int, int]:
    """Unpack a 30-bit coordinate and reject codes outside the 1000^3 domain."""

    packed = _require_int(packed, "packed")
    if packed < 0 or packed >= (1 << 30):
        raise ValueError("packed coordinate must fit in 30 bits")
    x = (packed >> 20) & XYZ_MASK
    y = (packed >> 10) & XYZ_MASK
    z = packed & XYZ_MASK
    if x >= XYZ_SIDE or y >= XYZ_SIDE or z >= XYZ_SIDE:
        raise ValueError("packed coordinate is outside the logical 1000^3 lattice")
    return x, y, z


def linear_address(x: int, y: int, z: int) -> int:
    """Map a coordinate to the canonical x + N(y + Nz) linear address."""

    x, y, z = unpack_xyz(pack_xyz(x, y, z))
    return x + XYZ_SIDE * (y + XYZ_SIDE * z)


def approve_lambda(lambda_mask: int, required_mask: int = REQUIRED_LAMBDA_MASK) -> bool:
    """Return True only when every required governance bit is asserted."""

    lambda_mask = _require_u8(lambda_mask, "lambda_mask")
    required_mask = _require_u8(required_mask, "required_mask")
    return (lambda_mask & required_mask) == required_mask


def hamming_distance(left: int, right: int) -> int:
    """Return the number of differing bits between two 64-bit states."""

    left = _require_word(left, "left")
    right = _require_word(right, "right")
    return (left ^ right).bit_count()


def select_word(
    current: int,
    candidate: int,
    lambda_mask: int,
    required_mask: int = REQUIRED_LAMBDA_MASK,
) -> int:
    """Select candidate on full approval, otherwise retain current state.

    The expression mirrors a hardware AND/OR multiplexer:

        next = (candidate & mask) | (current & ~mask)

    where mask is all ones on approval and all zeros on rejection.
    """

    current = _require_word(current, "current")
    candidate = _require_word(candidate, "candidate")
    approved = approve_lambda(lambda_mask, required_mask)
    mask = WORD_MASK if approved else 0
    return ((candidate & mask) | (current & (~mask & WORD_MASK))) & WORD_MASK


def omega_candidate_q15(omega_q15: int, error_q15: int, rho_q15: int, gain_q15: int) -> int:
    """Compute one saturated signed-Q1.15 adaptive-memory candidate."""

    for name, value in (
        ("omega_q15", omega_q15),
        ("error_q15", error_q15),
        ("rho_q15", rho_q15),
        ("gain_q15", gain_q15),
    ):
        _require_int(value, name)
    if not Q15_MIN <= omega_q15 <= Q15_MAX:
        raise ValueError("omega_q15 must fit signed Q1.15")
    if not Q15_MIN <= error_q15 <= Q15_MAX:
        raise ValueError("error_q15 must fit signed Q1.15")
    if not 0 <= rho_q15 <= Q15_MAX:
        raise ValueError("rho_q15 must be a non-negative Q1.15 coefficient")
    if not 0 <= gain_q15 <= Q15_MAX:
        raise ValueError("gain_q15 must be a non-negative Q1.15 coefficient")

    retained = (rho_q15 * omega_q15) >> Q15_SHIFT
    correction = (gain_q15 * error_q15) >> Q15_SHIFT
    return _saturate_q15(retained + correction)


@dataclass(frozen=True)
class ElectronicStepReport:
    cycle: int
    current_word: int
    candidate_word: int
    next_word: int
    lambda_mask: int
    committed: bool
    hamming_delta: int
    converged: bool
    omega_before_q15: int
    omega_candidate_q15: int
    omega_after_q15: int


class Omega3ElectronicCore:
    """Clock-step reference for the bounded electronic Omega3 state register."""

    def __init__(self, initial_word: int = 0, omega_q15: int = 0) -> None:
        self._word = _require_word(initial_word, "initial_word")
        if not Q15_MIN <= _require_int(omega_q15, "omega_q15") <= Q15_MAX:
            raise ValueError("omega_q15 must fit signed Q1.15")
        self._omega_q15 = omega_q15
        self._cycle = 0

    @property
    def word(self) -> int:
        return self._word

    @property
    def omega_q15(self) -> int:
        return self._omega_q15

    @property
    def cycle(self) -> int:
        return self._cycle

    def step(
        self,
        *,
        candidate_word: int,
        lambda_mask: int,
        error_q15: int = 0,
        rho_q15: int = Q15_MAX,
        gain_q15: int = 0,
        convergence_threshold: int = 0,
    ) -> ElectronicStepReport:
        """Execute one clock-equivalent transaction.

        Candidate word and candidate Omega are computed first. The governance
        mask then controls both authoritative state registers. Rejection retains
        the previous word and previous Omega exactly.
        """

        candidate_word = _require_word(candidate_word, "candidate_word")
        convergence_threshold = _require_int(convergence_threshold, "convergence_threshold")
        if convergence_threshold < 0 or convergence_threshold > WORD_BITS:
            raise ValueError("convergence_threshold must be in [0, 64]")

        current = self._word
        omega_before = self._omega_q15
        omega_candidate = omega_candidate_q15(
            omega_before, error_q15, rho_q15, gain_q15
        )
        committed = approve_lambda(lambda_mask)
        next_word = select_word(current, candidate_word, lambda_mask)
        delta = hamming_distance(current, candidate_word)

        self._cycle += 1
        if committed:
            self._word = next_word
            self._omega_q15 = omega_candidate

        return ElectronicStepReport(
            cycle=self._cycle,
            current_word=current,
            candidate_word=candidate_word,
            next_word=self._word,
            lambda_mask=lambda_mask,
            committed=committed,
            hamming_delta=delta,
            converged=delta <= convergence_threshold,
            omega_before_q15=omega_before,
            omega_candidate_q15=omega_candidate,
            omega_after_q15=self._omega_q15,
        )
