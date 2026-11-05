from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

Q16_SCALE = 1 << 16
INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1
MASK32 = (1 << 32) - 1

LENS_Q16 = 0x0000C59A  # 0.771881103515625
ETA_Q16 = 0x00001000   # 1/16
RHO_Q16 = 0x00004000   # 1/4
EPS_Q16 = 0x00000040   # 64 raw LSBs ~= 9.765625e-4
DEFAULT_PASSES = 192


def sat_i32(value: int) -> int:
    if value > INT32_MAX:
        return INT32_MAX
    if value < INT32_MIN:
        return INT32_MIN
    return int(value)


def to_u32(value: int) -> int:
    return value & MASK32


def to_i32(value: int) -> int:
    value &= MASK32
    return value - (1 << 32) if value & (1 << 31) else value


def q16_from_float(value: float) -> int:
    return sat_i32(round(float(value) * Q16_SCALE))


def q16_to_float(value: int) -> float:
    return int(value) / Q16_SCALE


def q16_add(a: int, b: int) -> int:
    return sat_i32(int(a) + int(b))


def q16_sub(a: int, b: int) -> int:
    return sat_i32(int(a) - int(b))


def q16_mul(a: int, b: int) -> int:
    # Python integers provide the widened signed intermediate required by the
    # ROM contract. Arithmetic shift preserves signed Q16.16 semantics.
    return sat_i32((int(a) * int(b)) >> 16)


def q16_abs_sat(value: int) -> int:
    return INT32_MAX if value == INT32_MIN else abs(int(value))


def bit_and_i32(value: int, mask: int) -> int:
    return to_i32(to_u32(value) & to_u32(mask))


def bit_or_i32(a: int, b: int) -> int:
    return to_i32(to_u32(a) | to_u32(b))


def mask_le_nonnegative(value: int, threshold: int) -> int:
    """Return -1 iff 0 <= value <= threshold, otherwise 0.

    The ROM computes sign(value - (threshold + 1)). The values involved in
    the convergence path are non-negative and bounded well inside int32.
    """

    delta = int(value) - (int(threshold) + 1)
    return -(((delta >> 31) & 1))


def positive_mask_nonnegative(value: int) -> int:
    """Return -1 iff value > 0, otherwise 0 for non-negative input."""

    return ~((int(value) - 1) >> 31)


def select_mask(lock_mask: int, old: int, candidate: int) -> int:
    """Branchless bit-select: lock=-1 keeps old; lock=0 takes candidate."""

    unlocked = to_i32(~to_u32(lock_mask))
    return bit_or_i32(bit_and_i32(old, lock_mask), bit_and_i32(candidate, unlocked))


def relu_branchless(value: int) -> int:
    sign = int(value) >> 31
    mask = ~sign
    return bit_and_i32(value, mask)


def sat_l1(values: Iterable[int]) -> int:
    total = 0
    for value in values:
        total = q16_add(total, q16_abs_sat(value))
    return total


@dataclass(frozen=True)
class TraceRow:
    iteration: int
    gap: int
    lock_mask: int
    latent: int
    state: tuple[int, int, int]
    weights: tuple[int, int, int]


@dataclass(frozen=True)
class RunResult:
    anchor: tuple[int, int, int]
    state: tuple[int, int, int]
    weights: tuple[int, int, int]
    latent: int
    gap: int
    lock_mask: int
    first_lock_iteration: int | None
    trace: tuple[TraceRow, ...]

    @property
    def converged(self) -> bool:
        return self.lock_mask == -1


class InwardRecursiveROM:
    """Bit-level reference for the perfected inward recursive ROM contract.

    The implementation executes exactly ``passes`` iterations even after the
    lock becomes sticky. This mirrors the fixed-work, no-data-branch ROM model.
    """

    def __init__(
        self,
        *,
        passes: int = DEFAULT_PASSES,
        lens_q16: int = LENS_Q16,
        eta_q16: int = ETA_Q16,
        rho_q16: int = RHO_Q16,
        eps_q16: int = EPS_Q16,
    ) -> None:
        if passes <= 0:
            raise ValueError("passes must be positive")
        if eps_q16 < 0:
            raise ValueError("eps_q16 must be non-negative")
        self.passes = int(passes)
        self.lens_q16 = sat_i32(lens_q16)
        self.eta_q16 = sat_i32(eta_q16)
        self.rho_q16 = sat_i32(rho_q16)
        self.eps_q16 = sat_i32(eps_q16)

    @staticmethod
    def _vec3(values: Sequence[int]) -> list[int]:
        if len(values) != 3:
            raise ValueError("expected exactly three lanes")
        return [sat_i32(v) for v in values]

    def run_q16(self, input_xyz: Sequence[int], weights: Sequence[int]) -> RunResult:
        source = self._vec3(input_xyz)
        w = self._vec3(weights)

        # The lens is applied exactly once. The result becomes an immutable
        # anchor for the duration of the run.
        anchor = [q16_mul(v, self.lens_q16) for v in source]
        state = anchor.copy()
        lock_mask = 0
        latent = 0
        gap = INT32_MAX
        first_lock_iteration: int | None = None
        trace: list[TraceRow] = []

        for iteration in range(self.passes):
            accumulator = 0
            for x_i, w_i in zip(state, w):
                accumulator = q16_add(accumulator, q16_mul(x_i, w_i))

            latent = relu_branchless(accumulator)
            relu_mask = positive_mask_nonnegative(latent)
            reconstruction = [q16_mul(latent, w_i) for w_i in w]

            residual = [q16_sub(xhat_i, anchor_i) for xhat_i, anchor_i in zip(reconstruction, anchor)]
            gap = sat_l1(residual)

            candidate_lock = mask_le_nonnegative(gap, self.eps_q16)
            next_lock = bit_or_i32(lock_mask, candidate_lock)
            if first_lock_iteration is None and next_lock == -1:
                first_lock_iteration = iteration

            coupling = 0
            for e_i, w_i in zip(residual, w):
                coupling = q16_add(coupling, q16_mul(e_i, w_i))

            next_weights: list[int] = []
            for e_i, x_i, w_i in zip(residual, state, w):
                decoder_term = q16_mul(e_i, latent)
                encoder_term = bit_and_i32(q16_mul(coupling, x_i), relu_mask)
                gradient = q16_add(decoder_term, encoder_term)
                delta = q16_mul(self.eta_q16, gradient)
                candidate_weight = q16_sub(w_i, delta)
                next_weights.append(select_mask(next_lock, w_i, candidate_weight))

            next_state: list[int] = []
            for old_i, xhat_i, anchor_i in zip(state, reconstruction, anchor):
                relaxed = q16_add(anchor_i, q16_mul(self.rho_q16, q16_sub(xhat_i, anchor_i)))
                next_state.append(select_mask(next_lock, old_i, relaxed))

            state = next_state
            w = next_weights
            lock_mask = next_lock
            trace.append(
                TraceRow(
                    iteration=iteration,
                    gap=gap,
                    lock_mask=lock_mask,
                    latent=latent,
                    state=tuple(state),
                    weights=tuple(w),
                )
            )

        return RunResult(
            anchor=tuple(anchor),
            state=tuple(state),
            weights=tuple(w),
            latent=latent,
            gap=gap,
            lock_mask=lock_mask,
            first_lock_iteration=first_lock_iteration,
            trace=tuple(trace),
        )

    def run_float(self, input_xyz: Sequence[float], weights: Sequence[float]) -> RunResult:
        return self.run_q16(
            [q16_from_float(v) for v in input_xyz],
            [q16_from_float(v) for v in weights],
        )


def demo() -> None:
    engine = InwardRecursiveROM()
    result = engine.run_float((0.4, 0.2, 0.1), (0.7, 0.35, 0.175))
    print(f"passes={engine.passes}")
    print(f"converged={result.converged}")
    print(f"first_lock_iteration={result.first_lock_iteration}")
    print(f"gap_raw={result.gap} gap={q16_to_float(result.gap):.9f}")
    print("anchor=", tuple(round(q16_to_float(v), 9) for v in result.anchor))
    print("state =", tuple(round(q16_to_float(v), 9) for v in result.state))
    print("weights=", tuple(round(q16_to_float(v), 9) for v in result.weights))


if __name__ == "__main__":
    demo()
