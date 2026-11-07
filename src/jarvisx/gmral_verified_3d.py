"""Fail-closed, arithmetic-first 3D verification for GMRAL-Omega.

Every accepted operation is a fully indexed bit-lattice transition checked
against a separate arithmetic oracle. This is runtime checking for a bounded
operation language, NOT a universal formal proof or mechanical hardware test.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Literal, cast

from jarvisx.gmral_omega import (
    AffineBitOperator,
    evaluate_exponential_iterations,
    morton3d,
    parallel_prefix_add,
    permutation_order,
)

Coordinate3D = tuple[int, int, int]
OperationName = Literal["xor", "and", "or", "not", "add", "affine", "evolve"]


class VerificationError(ValueError):
    """A candidate failed the mathematical/geometric verification contract."""


@dataclass(frozen=True)
class Grid3D:
    """A power-of-two cube using canonical X:Y:Z Morton bit ordering."""

    side: int = 4

    def __post_init__(self) -> None:
        if type(self.side) is not int or self.side < 2 or self.side > 16:
            raise ValueError("grid side must be 2..16 and power of two")
        if self.side & (self.side - 1):
            raise ValueError("grid side must be a power of two")

    @property
    def width(self) -> int:
        return self.side**3

    def address(self, xyz: Coordinate3D) -> int:
        if not isinstance(xyz, tuple) or len(xyz) != 3:
            raise ValueError("coordinate must be (x, y, z)")
        return cast(int, morton3d(*xyz, side=self.side))

    def coordinates(self, address: int) -> Coordinate3D:
        if type(address) is not int or address < 0 or address >= self.width:
            raise ValueError("address outside 3D lattice")
        xyz = [0, 0, 0]
        for bit in range(self.side.bit_length() - 1):
            for axis in range(3):
                xyz[axis] |= ((address >> (3 * bit + 2 - axis)) & 1) << bit
        result = (xyz[0], xyz[1], xyz[2])
        if self.address(result) != address:
            raise VerificationError("3D address roundtrip failed")
        return result


@dataclass(frozen=True)
class BitCube3D:
    grid: Grid3D
    bits: int

    def __post_init__(self) -> None:
        if type(self.bits) is not int or not 0 <= self.bits < (1 << self.grid.width):
            raise ValueError("bitfield exceeds 3D cube capacity")

    def at(self, xyz: Coordinate3D) -> int:
        return (self.bits >> self.grid.address(xyz)) & 1


@dataclass(frozen=True)
class Operation3D:
    """A typed request; fields not relevant to a specific operation are forbidden."""

    name: OperationName
    right: BitCube3D | None = None
    operator: AffineBitOperator | None = None
    base: int | None = None
    exponent: int | None = None

    def __post_init__(self) -> None:
        if self.name in ("xor", "and", "or", "add"):
            valid = self.right is not None and self.operator is None and self.base is None and self.exponent is None
        elif self.name == "not":
            valid = self.right is None and self.operator is None and self.base is None and self.exponent is None
        elif self.name == "affine":
            valid = self.right is None and self.operator is not None and self.base is None and self.exponent is None
        elif self.name == "evolve":
            valid = self.right is None and self.operator is not None and all(
                type(v) is int and v >= 0 for v in (self.base, self.exponent)
            )
        else:
            raise ValueError("operation is not on the admitted 3D arithmetic list")
        if not valid:
            raise ValueError("operation has missing or extraneous typed arguments")


@dataclass(frozen=True)
class VoxelEquation:
    """Exact result-bit equation at one 3D address."""

    xyz: Coordinate3D
    output_bit: int
    left_bit: int
    right_bit: int | None = None
    carry_in: int | None = None
    carry_out: int | None = None
    source_xyz: Coordinate3D | None = None


@dataclass(frozen=True)
class Verified3DReceipt:
    operation: str
    lattice_side: int
    input_hex: str
    output_hex: str
    oracle: str
    voxel_count: int
    proof_level: str
    witness_digest: str
    carry_out: int | None = None
    exact_period: int | None = None
    exponent_residue: int | None = None


@dataclass(frozen=True)
class Verified3DTransition:
    state: BitCube3D
    receipt: Verified3DReceipt
    witness: tuple[VoxelEquation, ...]


def _operator_cycles(operator: AffineBitOperator) -> list[tuple[int, ...]]:
    visited = set()
    cycles: list[tuple[int, ...]] = []
    for root in range(operator.width):
        if root in visited:
            continue
        cycle = []
        cursor = root
        while cursor not in visited:
            visited.add(cursor)
            cycle.append(cursor)
            cursor = operator.permutation[cursor]
        cycles.append(tuple(cycle))
    return cycles


def _cycle_reference(operator: AffineBitOperator, state: int, steps: int) -> int:
    """Independent O(width) per-cycle parity oracle for affine F**steps.

    In a P-cycle c[k], the destination c[k] receives x[c[k-steps]]
    XOR the parity of the latest 'steps' mask bits at c[k-j].
    Cycle-parity and doubled prefix tables avoid step-by-step execution.
    """
    if type(steps) is not int or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    output = 0
    for cycle in _operator_cycles(operator):
        length = len(cycle)
        mask = [(operator.mask >> index) & 1 for index in cycle]
        prefix = [0]
        for bit in mask + mask:
            prefix.append(prefix[-1] ^ bit)
        quotient, remainder = divmod(steps, length)
        full_parity = ((quotient & 1) * prefix[length])
        for k, destination in enumerate(cycle):
            old_index = cycle[(k - steps) % length]
            old_bit = (state >> old_index) & 1
            partial = prefix[k + length + 1] ^ prefix[k + length - remainder + 1]
            result_bit = old_bit ^ full_parity ^ partial
            output |= result_bit << destination
    return output


def _period_reference(operator: AffineBitOperator) -> int:
    r = cast(int, permutation_order(operator.permutation))
    nonzero = False
    for cycle in _operator_cycles(operator):
        parity = 0
        for address in cycle:
            parity ^= (operator.mask >> address) & 1
        if ((r // len(cycle)) & 1) and parity:
            nonzero = True
    return 2 * r if nonzero else r


def _oracle(left: BitCube3D, request: Operation3D) -> tuple[int, int | None, int | None, int | None, tuple[VoxelEquation, ...], str]:
    grid = left.grid
    width = grid.width
    bits = left.bits
    if request.right is not None and request.right.grid != grid:
        raise ValueError("operands must use the same 3D lattice")
    if request.operator is not None and request.operator.width != width:
        raise ValueError("operator width must equal lattice capacity")
    output = 0
    carry = 0
    witness: list[VoxelEquation] = []
    inverse: list[int] | None = None
    evolved_sources: list[int] | None = None
    period: int | None = None
    residue: int | None = None
    if request.operator is not None:
        inverse = [0] * width
        for src, dest in enumerate(request.operator.permutation):
            inverse[dest] = src
        if request.name == "evolve":
            period = _period_reference(request.operator)
            assert request.base is not None and request.exponent is not None
            residue = pow(request.base, request.exponent, period)
            output = _cycle_reference(request.operator, bits, residue)
            evolved_sources = [0] * width
            for cycle in _operator_cycles(request.operator):
                for k, destination in enumerate(cycle):
                    evolved_sources[destination] = cycle[(k - residue) % len(cycle)]
    for address in range(width):
        xyz = grid.coordinates(address)
        a = (bits >> address) & 1
        b = (request.right.bits >> address) & 1 if request.right is not None else None
        cin = cout = None
        source_xyz = None
        if request.name == "xor":
            assert b is not None
            result = (a + b) % 2
        elif request.name == "and":
            assert b is not None
            result = a * b
        elif request.name == "or":
            assert b is not None
            result = min(1, a + b)
        elif request.name == "not":
            result = 1 - a
        elif request.name == "add":
            assert b is not None
            cin = carry
            total = a + b + cin
            result = total % 2
            carry = total // 2
            cout = carry
        elif request.name == "affine":
            assert inverse is not None and request.operator is not None
            src = inverse[address]
            a = (bits >> src) & 1
            source_xyz = grid.coordinates(src)
            result = (a + ((request.operator.mask >> address) & 1)) % 2
        elif request.name == "evolve":
            result = (output >> address) & 1
            assert evolved_sources is not None
            src = evolved_sources[address]
            source_xyz = grid.coordinates(src)
            a = (bits >> src) & 1
            b = a ^ result  # net XOR-mask parity along the cycle path
        else:
            raise VerificationError("unknown operation")
        if request.name != "evolve":
            output |= result << address
        witness.append(VoxelEquation(xyz, result, a, b, cin, cout, source_xyz))
    if request.name == "add":
        assert request.right is not None
        expected = bits + request.right.bits
        if output != expected & ((1 << width) - 1) or carry != expected >> width:
            raise VerificationError("ripple arithmetic disagrees with integer arithmetic")
    return output, carry if request.name == "add" else None, period, residue, tuple(witness), (
        "exact per-voxel Boolean truth table" if request.name in ("xor", "and", "or", "not") else
        "ripple carry and integer sum identity" if request.name == "add" else
        "inverse 3D permutation and XOR parity" if request.name == "affine" else
        "independent permutation-cycle parity closed form"
    )


def propose(left: BitCube3D, request: Operation3D) -> tuple[int, int | None, int | None, int | None]:
    """Optimized path, distinct from the per-voxel arithmetic oracle."""
    bits, width = left.bits, left.grid.width
    mask = (1 << width) - 1
    if request.name == "not":
        return bits ^ mask, None, None, None
    if request.name in ("xor", "and", "or", "add"):
        if request.right is None or request.right.grid != left.grid:
            raise ValueError("right-hand side must be on the same grid")
        rhs = request.right.bits
        if request.name == "xor":
            return bits ^ rhs, None, None, None
        if request.name == "and":
            return bits & rhs, None, None, None
        if request.name == "or":
            return bits | rhs, None, None, None
        low, carry, _rounds = parallel_prefix_add(bits, rhs, width=width)
        return low, carry, None, None
    if request.operator is None or request.operator.width != width:
        raise ValueError("operator width must match lattice capacity")
    if request.name == "affine":
        return request.operator.apply(bits), None, None, None
    if request.name == "evolve":
        assert request.base is not None and request.exponent is not None
        receipt = evaluate_exponential_iterations(
            request.operator, bits, base=request.base, exponent=request.exponent
        )
        return receipt.result, None, receipt.operator_period, receipt.exponent_residue
    raise VerificationError("unrecognized operation")


def verify_transition(
    left: BitCube3D,
    request: Operation3D,
    candidate: int,
    *,
    carry_out: int | None = None,
    exact_period: int | None = None,
    exponent_residue: int | None = None,
) -> Verified3DTransition:
    """Independently check every output voxel and arithmetic side effect.

    No receipt is issued on mismatch. A digest records reproducible evidence;
    it does not constitute a cryptographic or formal proof of implementation.
    """
    expected, expected_carry, expected_period, expected_residue, witness, oracle = _oracle(
        left, request
    )
    if (candidate, carry_out, exact_period, exponent_residue) != (
        expected, expected_carry, expected_period, expected_residue
    ):
        raise VerificationError("candidate differs from exact 3D arithmetic oracle")
    result = BitCube3D(left.grid, candidate)
    for address, item in enumerate(witness):
        if left.grid.address(item.xyz) != address or item.output_bit != ((result.bits >> address) & 1):
            raise VerificationError("3D voxel witness mapping inconsistent")
    digest_input = {
        "operation": request.name,
        "side": left.grid.side,
        "lhs": left.bits,
        "rhs": request.right.bits if request.right is not None else None,
        "permutation": list(request.operator.permutation) if request.operator else None,
        "mask": request.operator.mask if request.operator else None,
        "base": request.base,
        "exponent": request.exponent,
        "result": result.bits,
        "carry": carry_out,
        "period": exact_period,
        "residue": exponent_residue,
        "witness": [
            (w.xyz, w.output_bit, w.left_bit, w.right_bit, w.carry_in, w.carry_out, w.source_xyz)
            for w in witness
        ],
    }
    digest = sha256(json.dumps(digest_input, separators=(",", ":"), sort_keys=True).encode()).hexdigest()
    receipt = Verified3DReceipt(
        operation=request.name,
        lattice_side=left.grid.side,
        input_hex=hex(left.bits),
        output_hex=hex(result.bits),
        oracle=oracle,
        voxel_count=left.grid.width,
        proof_level="exact bounded runtime oracle + per-voxel witness",
        witness_digest=digest,
        carry_out=carry_out,
        exact_period=exact_period,
        exponent_residue=exponent_residue,
    )
    return Verified3DTransition(result, receipt, witness)


def execute_verified(left: BitCube3D, request: Operation3D) -> Verified3DTransition:
    """Propose -> verify -> commit; raises before state can be accepted."""
    proposed, carry, period, residue = propose(left, request)
    return verify_transition(
        left, request, proposed, carry_out=carry,
        exact_period=period, exponent_residue=residue,
    )


def execute_atomic(
    initial: BitCube3D, requests: tuple[Operation3D, ...]
) -> tuple[BitCube3D, tuple[Verified3DReceipt, ...]]:
    """All-or-nothing immutable transaction; no accepted partial return on failure."""
    state = initial
    receipts = []
    for request in requests:
        transition = execute_verified(state, request)
        state = transition.state
        receipts.append(transition.receipt)
    return state, tuple(receipts)


def demo() -> None:
    grid = Grid3D(4)
    a = BitCube3D(grid, 5)
    b = BitCube3D(grid, 3)
    add = execute_verified(a, Operation3D("add", right=b))
    assert add.state.bits == 8
    print(json.dumps({"verified_add": add.receipt.__dict__}, indent=2))


if __name__ == "__main__":
    demo()
