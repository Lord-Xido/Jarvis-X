"""Bounded assembly equivalence verification with Z3.

The verifier encodes two programs in distinct symbolic namespaces, constrains
both to the same symbolic initial register state, unrolls their small-step
semantics for a finite number of steps, and asks Z3 whether any final checked
register can differ.

The result is deliberately bounded. It is suitable for small refactoring gates;
it is not an unbounded program proof or a replacement for runtime isolation.
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Set, Tuple

try:
    import z3
except Exception as exc:  # pragma: no cover - dependency error path
    raise ImportError(
        "z3-solver is required for verification. Install with: pip install z3-solver"
    ) from exc

INSTR_RE = re.compile(r"^(?P<op>[A-Za-z]+)\s*(?P<args>.*)$")
REG_RE = re.compile(r"\b([A-Z])\b")
SUPPORTED_OPS = {"SET", "ADD", "SUB", "HALT", "NOP"}


def _parse_program_lines(program: str) -> list[str]:
    lines: list[str] = []
    for raw in program.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = INSTR_RE.match(line)
        if match is None or match.group("op").upper() not in SUPPORTED_OPS:
            raise ValueError(f"unsupported instruction in bounded verifier: {line}")
        lines.append(line)
    return lines


def _collect_registers(lines: list[str]) -> Set[str]:
    registers: Set[str] = set()
    for line in lines:
        for match in REG_RE.finditer(line):
            registers.add(match.group(1))
    return registers


def _reg(namespace: str, name: str, step: int):
    return z3.Int(f"{namespace}_{name}_{step}")


def _pc(namespace: str, step: int):
    return z3.Int(f"{namespace}_pc_{step}")


def _halted(namespace: str, step: int):
    return z3.Bool(f"{namespace}_halted_{step}")


def _value(namespace: str, token: str, step: int):
    if re.fullmatch(r"[A-Z]", token):
        return _reg(namespace, token, step)
    try:
        return z3.IntVal(int(token))
    except ValueError as exc:
        raise ValueError(f"unsupported verifier operand: {token}") from exc


def _apply_step_constraints(
    solver,
    lines: list[str],
    registers: list[str],
    step: int,
    namespace: str,
) -> None:
    pc_now = _pc(namespace, step)
    pc_next = _pc(namespace, step + 1)
    halted_now = _halted(namespace, step)
    halted_next = _halted(namespace, step + 1)

    for register in registers:
        solver.add(
            z3.Implies(
                halted_now,
                _reg(namespace, register, step + 1) == _reg(namespace, register, step),
            )
        )
    solver.add(z3.Implies(halted_now, pc_next == pc_now))
    solver.add(z3.Implies(halted_now, halted_next))

    for index, instruction in enumerate(lines):
        match = INSTR_RE.match(instruction)
        if match is None:  # guarded during parsing
            raise AssertionError("validated instruction failed to parse")
        op = match.group("op").upper()
        args = match.group("args").split()
        condition = z3.And(pc_now == index, z3.Not(halted_now))

        assignments = {register: _reg(namespace, register, step) for register in registers}
        next_pc = pc_now + 1
        next_halted = z3.BoolVal(False)

        if op == "SET" and len(args) == 2:
            target, source = args
            if target not in registers:
                raise ValueError(f"invalid verifier register: {target}")
            assignments[target] = _value(namespace, source, step)
        elif op in {"ADD", "SUB"} and len(args) == 3:
            target, left, right = args
            if target not in registers:
                raise ValueError(f"invalid verifier register: {target}")
            lhs = _value(namespace, left, step)
            rhs = _value(namespace, right, step)
            assignments[target] = lhs + rhs if op == "ADD" else lhs - rhs
        elif op == "HALT" and not args:
            next_pc = pc_now
            next_halted = z3.BoolVal(True)
        elif op == "NOP" and not args:
            pass
        else:
            raise ValueError(f"malformed verifier instruction: {instruction}")

        for register, expression in assignments.items():
            solver.add(
                z3.Implies(
                    condition,
                    _reg(namespace, register, step + 1) == expression,
                )
            )
        solver.add(z3.Implies(condition, pc_next == next_pc))
        solver.add(z3.Implies(condition, halted_next == next_halted))

    outside = z3.And(
        z3.Or(pc_now < 0, pc_now >= len(lines)),
        z3.Not(halted_now),
    )
    for register in registers:
        solver.add(
            z3.Implies(
                outside,
                _reg(namespace, register, step + 1) == _reg(namespace, register, step),
            )
        )
    solver.add(z3.Implies(outside, pc_next == pc_now))
    solver.add(z3.Implies(outside, halted_next))


def check_equivalence_bounded(
    prog_a: str,
    prog_b: str,
    bound: int = 64,
    timeout_ms: int = 5000,
) -> Tuple[bool, Optional[Dict[str, int]]]:
    """Prove final-register equivalence for all shared initial states up to ``bound`` steps.

    ``False, None`` means Z3 returned ``unknown`` or timed out. A non-empty
    counterexample contains one concrete shared initial register valuation that
    permits divergent final register state within the bound.
    """

    if bound <= 0:
        raise ValueError("bound must be positive")
    if timeout_ms <= 0:
        raise ValueError("timeout_ms must be positive")

    lines_a = _parse_program_lines(prog_a)
    lines_b = _parse_program_lines(prog_b)
    registers = sorted(_collect_registers(lines_a) | _collect_registers(lines_b))
    if not registers:
        return True, None

    solver = z3.Solver()
    solver.set("timeout", timeout_ms)

    for register in registers:
        initial = z3.Int(f"initial_{register}")
        solver.add(_reg("a", register, 0) == initial)
        solver.add(_reg("b", register, 0) == initial)

    solver.add(_pc("a", 0) == 0, z3.Not(_halted("a", 0)))
    solver.add(_pc("b", 0) == 0, z3.Not(_halted("b", 0)))

    for step in range(bound):
        _apply_step_constraints(solver, lines_a, registers, step, "a")
        _apply_step_constraints(solver, lines_b, registers, step, "b")

    differences = [
        _reg("a", register, bound) != _reg("b", register, bound) for register in registers
    ]
    solver.add(z3.Or(*differences))

    result = solver.check()
    if result == z3.unsat:
        return True, None
    if result == z3.sat:
        model = solver.model()
        counterexample = {
            register: model.eval(z3.Int(f"initial_{register}"), model_completion=True).as_long()
            for register in registers
        }
        return False, counterexample
    return False, None
