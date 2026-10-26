"""GMRAL-Omega: exact semantic acceleration of bounded 3D bit transformations.

A 3D voxel lattice is flattened with Jarvis-X X:Y:Z Morton ordering.
For an affine Boolean bijection F(x) = P(x) XOR mask, composition,
inversion-free exact period derivation, and exponent reduction are exact.

This is a symbolic/Boolean reference, NOT hardware acceleration, an
arbitrary-program optimizer, or an autonomous self-modifying runtime.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

MAX_SIDE = 16
MAX_BITS = MAX_SIDE ** 3


def _check_side(side: int) -> int:
    if type(side) is not int or side < 2 or side > MAX_SIDE or side & (side - 1):
        raise ValueError("side must be a power of two between 2 and 16")
    return side


def _check_nonnegative(name: str, value: int) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def morton3d(x: int, y: int, z: int, *, side: int = 4) -> int:
    """Return the X:Y:Z-interleaved address; X is highest in each triad.

    For 20-bit-compatible coordinates, this agrees with morton60_encode.
    """
    side = _check_side(side)
    if any(type(v) is not int or v < 0 or v >= side for v in (x, y, z)):
        raise ValueError("coordinates must be integers within the cube")
    address = 0
    for bit in range(side.bit_length() - 1):
        address |= ((x >> bit) & 1) << (3 * bit + 2)
        address |= ((y >> bit) & 1) << (3 * bit + 1)
        address |= ((z >> bit) & 1) << (3 * bit)
    return address


def rotation_permutation(side: int = 4) -> tuple[int, ...]:
    """Source-to-destination map (x,y,z)->(y+1,z+1,x+1) mod side."""
    side = _check_side(side)
    permutation = [0] * (side**3)
    for x in range(side):
        for y in range(side):
            for z in range(side):
                source = morton3d(x, y, z, side=side)
                target = morton3d(
                    (y + 1) % side, (z + 1) % side, (x + 1) % side, side=side
                )
                permutation[source] = target
    return tuple(permutation)


def permute_bits(state: int, permutation: tuple[int, ...]) -> int:
    """Move each source bit to its indexed destination."""
    width = len(permutation)
    if type(state) is not int or not 0 <= state < (1 << width):
        raise ValueError("state must fit the permutation width")
    out = 0
    for source, target in enumerate(permutation):
        out |= ((state >> source) & 1) << target
    return out


def permutation_order(permutation: tuple[int, ...]) -> int:
    """Cycle decomposition: the order is the LCM of cycle lengths."""
    if not permutation or len(permutation) > MAX_BITS:
        raise ValueError("permutation width outside supported range")
    if sorted(permutation) != list(range(len(permutation))):
        raise ValueError("permutation must be bijective")
    seen = set()
    order = 1
    for start in range(len(permutation)):
        if start in seen:
            continue
        index = start
        cycle_length = 0
        while index not in seen:
            seen.add(index)
            cycle_length += 1
            index = permutation[index]
        order = math.lcm(order, cycle_length)
    return order


@dataclass(frozen=True)
class AffineBitOperator:
    """An affine Boolean bijection F(s)=P(s) XOR mask, source->dest P."""

    permutation: tuple[int, ...]
    mask: int

    def __post_init__(self) -> None:
        permutation_order(self.permutation)
        if type(self.mask) is not int or not 0 <= self.mask < (1 << self.width):
            raise ValueError("mask must be a nonnegative integer fitting width")

    @property
    def width(self) -> int:
        return len(self.permutation)

    def apply(self, state: int) -> int:
        return permute_bits(state, self.permutation) ^ self.mask

    @classmethod
    def identity(cls, width: int) -> "AffineBitOperator":
        if type(width) is not int or not 1 <= width <= MAX_BITS:
            raise ValueError("width outside supported range")
        return cls(tuple(range(width)), 0)

    def then(self, outer: "AffineBitOperator") -> "AffineBitOperator":
        """Return outer(self(s)): (Q,N)o(P,M)=(QP,Q(M) XOR N)."""
        if self.width != outer.width:
            raise ValueError("operator widths must match")
        permutation = tuple(
            outer.permutation[self.permutation[i]] for i in range(self.width)
        )
        mask = permute_bits(self.mask, outer.permutation) ^ outer.mask
        return AffineBitOperator(permutation, mask)

    def power(self, exponent: int) -> "AffineBitOperator":
        """Binary exponentiation: O(width * log(exponent)) elementary work."""
        _check_nonnegative("exponent", exponent)
        result = AffineBitOperator.identity(self.width)
        base = self
        while exponent:
            if exponent & 1:
                result = result.then(base)
            exponent >>= 1
            if exponent:
                base = base.then(base)
        return result

    def exact_period(self) -> int:
        """Minimal universal period of F=P XOR M.

        If P has order r, F**r is either the identity or an XOR translation.
        A nonzero XOR translation has order two, hence period r or 2*r.
        """
        order = permutation_order(self.permutation)
        residual = self.power(order)
        identity_perm = tuple(range(self.width))
        if residual.permutation != identity_perm:
            raise AssertionError("permutation cycle certificate invalid")
        period = order if residual.mask == 0 else 2 * order
        if self.power(period) != AffineBitOperator.identity(self.width):
            raise AssertionError("affine period certificate invalid")
        return period


@dataclass(frozen=True)
class ExactEvolutionReceipt:
    """Evidence for F**(base**exponent) calculated without that giant integer."""

    result: int
    operator_period: int
    exponent_residue: int
    width: int


def evaluate_exponential_iterations(
    operator: AffineBitOperator,
    initial_state: int,
    *,
    base: int = 1_000_000,
    exponent: int = 1_000_000,
) -> ExactEvolutionReceipt:
    """Exact reduction of F**(base**exponent) modulo its finite period.

    Proof: if F**T=identity, then F**N=F**(N mod T), regardless of N.
    Does not materialize base**exponent, even for 10**6,000,000.
    """
    _check_nonnegative("base", base)
    _check_nonnegative("exponent", exponent)
    if type(initial_state) is not int or not 0 <= initial_state < (1 << operator.width):
        raise ValueError("initial state outside operator width")
    period = operator.exact_period()
    residue = pow(base, exponent, period)
    result = operator.power(residue).apply(initial_state)
    return ExactEvolutionReceipt(result, period, residue, operator.width)


def parallel_prefix_add(a: int, b: int, *, width: int = 64) -> tuple[int, int, int]:
    """Return (sum mod 2**width, carry-out, parallel prefix rounds).

    O(width*log(width)) reference work; O(log(width)) logical carry depth.
    This pure-Python implementation does not assert a CPU runtime speedup.
    """
    if type(width) is not int or not 1 <= width <= MAX_BITS:
        raise ValueError("width outside supported range")
    for name, value in (("a", a), ("b", b)):
        if type(value) is not int or not 0 <= value < (1 << width):
            raise ValueError(f"{name} must be an unsigned width-bit integer")

    propagate = [((a >> i) ^ (b >> i)) & 1 for i in range(width)]
    pair = [(((a >> i) & (b >> i)) & 1, propagate[i]) for i in range(width)]
    rounds = 0
    distance = 1
    while distance < width:
        following = pair[:]
        for i in range(distance, width):
            upper_g, upper_p = pair[i]
            lower_g, lower_p = pair[i - distance]
            following[i] = (
                upper_g | (upper_p & lower_g), upper_p & lower_p
            )
        pair = following
        distance *= 2
        rounds += 1
    result = 0
    for i in range(width):
        carry_in = pair[i - 1][0] if i else 0
        result |= (propagate[i] ^ carry_in) << i
    return result, pair[-1][0], rounds


def seed_cube(side: int = 4) -> int:
    """Deterministic lattice field with bit on iff x==y or z==0."""
    side = _check_side(side)
    seed = 0
    for x in range(side):
        for y in range(side):
            for z in range(side):
                if x == y or z == 0:
                    seed |= 1 << morton3d(x, y, z, side=side)
    return seed


def demo() -> None:
    import json

    side = 4
    operator = AffineBitOperator(
        rotation_permutation(side), 1 << morton3d(0, 0, 0, side=side)
    )
    initial = seed_cube(side)
    receipt = evaluate_exponential_iterations(operator, initial)
    print(json.dumps({
        "engine": "GMRAL-Omega",
        "lattice": [side, side, side],
        "initial_hex": f"0x{initial:016X}",
        "final_hex": f"0x{receipt.result:016X}",
        "affine_period": receipt.operator_period,
        "giant_exponent_residue": receipt.exponent_residue,
        "virtual_iterations": "(10**6)**(10**6)",
        "physical_speed_claim": False,
    }, indent=2))


if __name__ == "__main__":
    demo()
