"""Bounded, typed code-editing automata for Jarvis-X.

The module implements a small deterministic research optimizer for Jarvis-X
assembly.  Candidate transformations are generated inside explicit resource
bounds, checked by a composable policy layer, and committed only after a
conservative parse/shape check.  It is intentionally not an unrestricted
self-modification mechanism.
"""

from __future__ import annotations

import hashlib
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


class TransformType(Enum):
    """Classification of bounded code transformations."""

    DEAD_CODE_ELIMINATION = "dead_code_elimination"
    REGISTER_REALLOCATION = "register_reallocation"
    INSTRUCTION_FOLDING = "instruction_folding"
    CONST_PROPAGATION = "const_propagation"
    LOOP_UNROLL = "loop_unroll"
    PIPELINE_REORDER = "pipeline_reorder"
    NOP_INSERTION = "nop_insertion"


class MutationStatus(Enum):
    """Lifecycle of a candidate mutation."""

    PROPOSED = "proposed"
    VALIDATED = "validated"
    APPLIED = "applied"
    ROLLED_BACK = "rolled_back"
    REJECTED = "rejected"


@dataclass(frozen=True)
class RefactoringParameter:
    """Bounded parameters controlling one refactoring pass."""

    seed: int
    max_depth: int = 3
    max_cycles: int = 1000
    max_mutations: int = 10
    cost_model: str = "cycles"
    allow_unsafe: bool = False
    allow_heuristic: bool = False

    def validate(self) -> bool:
        """Return whether every parameter is inside the declared envelope."""

        return (
            self.seed >= 0
            and 1 <= self.max_depth <= 8
            and 1 <= self.max_cycles <= 100_000
            and 1 <= self.max_mutations <= 100
            and self.cost_model in {"cycles", "memory", "combined"}
        )


@dataclass
class Mutation:
    """One proposed assembly transformation."""

    transform_type: TransformType
    source_lines: list[int]
    original_code: str
    proposed_code: str
    estimated_improvement: float
    cycle_cost: int
    confidence: float
    status: MutationStatus = MutationStatus.PROPOSED
    policy_violations: list[str] = field(default_factory=list)
    timestamp_ns: int = 0
    hash: str = ""

    def compute_hash(self) -> str:
        """Return a deterministic content hash for the proposal."""

        payload: dict[str, object] = {
            "transform_type": self.transform_type.value,
            "source_lines": self.source_lines,
            "original_code": self.original_code,
            "proposed_code": self.proposed_code,
            "estimated_improvement": self.estimated_improvement,
            "cycle_cost": self.cycle_cost,
            "confidence": self.confidence,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "transform_type": self.transform_type.value,
            "source_lines": list(self.source_lines),
            "original_code": self.original_code,
            "proposed_code": self.proposed_code,
            "estimated_improvement": self.estimated_improvement,
            "cycle_cost": self.cycle_cost,
            "confidence": self.confidence,
            "status": self.status.value,
            "policy_violations": list(self.policy_violations),
            "timestamp_ns": self.timestamp_ns,
            "hash": self.hash,
        }


@dataclass
class RefactoringResult:
    """Receipt emitted by a bounded refactoring pass."""

    input_program: str
    output_program: str
    mutations_proposed: int
    mutations_applied: int
    mutations_rejected: int
    total_cycles_used: int
    estimated_cycles_saved: float
    estimated_memory_saved: float
    deterministic_seed: int
    journaled: bool
    mutations: list[Mutation] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable receipt."""

        return {
            "input_program": self.input_program,
            "output_program": self.output_program,
            "mutations_proposed": self.mutations_proposed,
            "mutations_applied": self.mutations_applied,
            "mutations_rejected": self.mutations_rejected,
            "total_cycles_used": self.total_cycles_used,
            "estimated_cycles_saved": self.estimated_cycles_saved,
            "estimated_memory_saved": self.estimated_memory_saved,
            "deterministic_seed": self.deterministic_seed,
            "journaled": self.journaled,
            "mutations": [mutation.to_dict() for mutation in self.mutations],
        }


class PolicyConstraint(ABC):
    """Base class for Lambda-style candidate admission policies."""

    @abstractmethod
    def validate(
        self, mutation: Mutation, context: Mapping[str, float]
    ) -> tuple[bool, str | None]:
        """Return ``(allowed, reason)`` for a mutation."""


class ConfidenceThresholdPolicy(PolicyConstraint):
    """Require a minimum declared transformation confidence."""

    def __init__(self, threshold: float = 0.7) -> None:
        self.threshold = threshold

    def validate(
        self, mutation: Mutation, context: Mapping[str, float]
    ) -> tuple[bool, str | None]:
        del context
        if mutation.confidence < self.threshold:
            return False, f"confidence {mutation.confidence:.2f} < {self.threshold}"
        return True, None


class CycleImprovementPolicy(PolicyConstraint):
    """Admit only candidates with a positive cycle improvement estimate."""

    def validate(
        self, mutation: Mutation, context: Mapping[str, float]
    ) -> tuple[bool, str | None]:
        del context
        if mutation.estimated_improvement <= 0:
            return False, f"no cycle improvement: {mutation.estimated_improvement}"
        return True, None


class MemorySafetyPolicy(PolicyConstraint):
    """Bound the estimated increase in working memory."""

    def __init__(self, max_increase_percent: float = 5.0) -> None:
        self.max_increase_percent = max_increase_percent

    def validate(
        self, mutation: Mutation, context: Mapping[str, float]
    ) -> tuple[bool, str | None]:
        del mutation
        increase = context.get("estimated_memory_increase_percent")
        if increase is not None and increase > self.max_increase_percent:
            return (
                False,
                f"memory increase {increase:.1f}% > {self.max_increase_percent}%",
            )
        return True, None


class TransformWhitelistPolicy(PolicyConstraint):
    """Admit only explicitly enabled transformation classes."""

    def __init__(self, allowed_transforms: set[TransformType]) -> None:
        self.allowed_transforms = set(allowed_transforms)

    def validate(
        self, mutation: Mutation, context: Mapping[str, float]
    ) -> tuple[bool, str | None]:
        del context
        if mutation.transform_type not in self.allowed_transforms:
            return False, f"transform {mutation.transform_type.value} not whitelisted"
        return True, None


class DeterministicCodeEditingAutomata:
    """Deterministic, bounded candidate generator and verifier."""

    def __init__(self, cycle_limit: int = 10_000, journal_path: str | None = None) -> None:
        if cycle_limit <= 0:
            raise ValueError("cycle_limit must be positive")
        self.cycle_limit = cycle_limit
        self.journal_path = journal_path
        self.cycle_counter = 0
        self.policy_constraints: list[PolicyConstraint] = []
        self.mutation_history: list[Mutation] = []

    def add_policy(self, constraint: PolicyConstraint) -> None:
        """Register a policy constraint."""

        self.policy_constraints.append(constraint)

    def _parse_assembly(self, program: str) -> list[str]:
        """Normalize assembly into non-empty, non-comment instruction lines."""

        lines: list[str] = []
        for raw_line in program.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.split():
                lines.append(line)
        return lines

    def _estimate_dead_code(self, instructions: list[str]) -> list[int]:
        """Return instruction indexes that are unconditionally after first HALT."""

        dead_lines: list[int] = []
        for index, instruction in enumerate(instructions):
            if instruction.upper().startswith("HALT"):
                dead_lines.extend(range(index + 1, len(instructions)))
                break
        return dead_lines

    def _estimate_cycle_cost(self, instruction: str) -> int:
        """Return the small reference cost assigned to an instruction."""

        tokens = instruction.split()
        if not tokens:
            return 0
        costs = {
            "SET": 1,
            "ADD": 2,
            "SUB": 2,
            "MUL": 3,
            "DIV": 4,
            "LOAD": 2,
            "STORE": 2,
            "HALT": 1,
            "JMP": 1,
            "NOP": 1,
        }
        return costs.get(tokens[0].upper(), 1)

    def _generate_dead_code_elimination(
        self, instructions: list[str], dead_lines: list[int], seed: int
    ) -> Mutation | None:
        """Create a conservative unreachable-code elimination candidate."""

        del seed
        if not dead_lines:
            return None
        original = "\n".join(instructions[index] for index in dead_lines)
        cycles_saved = sum(
            self._estimate_cycle_cost(instructions[index]) for index in dead_lines
        )
        return Mutation(
            transform_type=TransformType.DEAD_CODE_ELIMINATION,
            source_lines=list(dead_lines),
            original_code=original,
            proposed_code="# Dead code removed",
            estimated_improvement=float(cycles_saved),
            cycle_cost=5,
            confidence=0.95,
        )

    def _generate_const_propagation(
        self, instructions: list[str], seed: int
    ) -> Mutation | None:
        """Fold one deterministic straight-line ADD/SUB whose inputs are literals."""

        constants: dict[str, int] = {}
        candidates: list[tuple[int, str, str]] = []

        for index, instruction in enumerate(instructions):
            parts = instruction.split()
            if not parts:
                continue
            opcode = parts[0].upper()
            if opcode == "HALT":
                break
            if opcode == "SET" and len(parts) >= 3:
                try:
                    constants[parts[1]] = int(parts[2])
                except ValueError:
                    constants.pop(parts[1], None)
                continue
            if opcode in {"ADD", "SUB"} and len(parts) >= 4:
                destination, left, right = parts[1], parts[2], parts[3]
                if left in constants and right in constants:
                    value = (
                        constants[left] + constants[right]
                        if opcode == "ADD"
                        else constants[left] - constants[right]
                    )
                    candidates.append((index, instruction, f"SET {destination} {value}"))
                    constants[destination] = value
                else:
                    constants.pop(destination, None)
            elif len(parts) >= 2:
                constants.pop(parts[1], None)

        if not candidates:
            return None
        chosen = candidates[seed % len(candidates)]
        index, original, replacement = chosen
        return Mutation(
            transform_type=TransformType.CONST_PROPAGATION,
            source_lines=[index],
            original_code=original,
            proposed_code=replacement,
            estimated_improvement=1.0,
            cycle_cost=3,
            confidence=0.90,
        )

    def _apply_policy_checks(
        self, mutation: Mutation, context: Mapping[str, float]
    ) -> tuple[bool, list[str]]:
        """Evaluate all registered policies, failing closed on any rejection."""

        violations: list[str] = []
        for policy in self.policy_constraints:
            allowed, reason = policy.validate(mutation, context)
            if not allowed:
                violations.append(reason or "policy violation")
        return not violations, violations

    @staticmethod
    def _apply_mutation(instructions: list[str], mutation: Mutation) -> list[str]:
        """Apply a candidate to normalized instructions by immutable indexes."""

        indexes = set(mutation.source_lines)
        if mutation.transform_type is TransformType.DEAD_CODE_ELIMINATION:
            return [
                instruction
                for index, instruction in enumerate(instructions)
                if index not in indexes
            ]

        if len(mutation.source_lines) != 1:
            raise ValueError("single-instruction transform requires exactly one source index")
        index = mutation.source_lines[0]
        if not 0 <= index < len(instructions):
            raise ValueError("mutation source index is outside the current program")
        candidate = list(instructions)
        candidate[index] = mutation.proposed_code
        return candidate

    def refactor(self, program: str, params: RefactoringParameter) -> RefactoringResult:
        """Execute one deterministic bounded refactoring transaction."""

        if not params.validate():
            raise ValueError(f"Invalid refactoring parameters: {params}")

        instructions = self._parse_assembly(program)
        original_program = program
        self.cycle_counter = 0
        proposed: list[Mutation] = []
        applied: list[Mutation] = []
        rejected: list[Mutation] = []
        seen_hashes: set[str] = set()
        rng_state = params.seed

        effective_limit = min(self.cycle_limit, params.max_cycles)
        while self.cycle_counter < effective_limit and len(applied) < params.max_mutations:
            self.cycle_counter += 1
            rng_state = (rng_state * 1_103_515_245 + 12_345) % (2**31)
            transform_choice = rng_state % 3

            mutation: Mutation | None
            if transform_choice == 0:
                dead_lines = self._estimate_dead_code(instructions)
                mutation = self._generate_dead_code_elimination(
                    instructions, dead_lines, rng_state
                )
            elif transform_choice == 1:
                mutation = self._generate_const_propagation(instructions, rng_state)
            else:
                mutation = None

            if mutation is None:
                continue

            mutation.hash = mutation.compute_hash()
            if mutation.hash in seen_hashes:
                continue
            seen_hashes.add(mutation.hash)
            mutation.timestamp_ns = self.cycle_counter * 1_000_000
            proposed.append(mutation)

            allowed, violations = self._apply_policy_checks(
                mutation, {"estimated_memory_increase_percent": 0.0}
            )
            if not allowed:
                mutation.status = MutationStatus.REJECTED
                mutation.policy_violations = violations
                rejected.append(mutation)
                self.mutation_history.append(mutation)
                continue

            try:
                candidate_instructions = self._apply_mutation(instructions, mutation)
                candidate_program = "\n".join(candidate_instructions)
                if self._parse_assembly(candidate_program) != candidate_instructions:
                    raise ValueError("candidate does not round-trip through normalized parser")
            except (IndexError, ValueError) as exc:
                mutation.status = MutationStatus.REJECTED
                mutation.policy_violations = [f"candidate validation failed: {exc}"]
                rejected.append(mutation)
                self.mutation_history.append(mutation)
                continue

            mutation.status = MutationStatus.APPLIED
            applied.append(mutation)
            self.mutation_history.append(mutation)
            instructions = candidate_instructions

        output_program = "\n".join(instructions) if proposed else original_program
        result = RefactoringResult(
            input_program=original_program,
            output_program=output_program,
            mutations_proposed=len(proposed),
            mutations_applied=len(applied),
            mutations_rejected=len(rejected),
            total_cycles_used=self.cycle_counter,
            estimated_cycles_saved=sum(
                mutation.estimated_improvement for mutation in applied
            ),
            estimated_memory_saved=0.0,
            deterministic_seed=params.seed,
            journaled=self.journal_path is not None,
            mutations=proposed,
        )

        if self.journal_path is not None:
            self._journal_refactoring(result)
        return result

    def _journal_refactoring(self, result: RefactoringResult) -> None:
        """Append an integrity-tagged receipt to the configured JSONL journal."""

        if self.journal_path is None:
            return
        directory = os.path.dirname(self.journal_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        receipt = result.to_dict()
        canonical = json.dumps(receipt, sort_keys=True, separators=(",", ":"))
        entry: dict[str, object] = {
            "timestamp_ns": result.mutations[0].timestamp_ns if result.mutations else 0,
            "refactoring": receipt,
            "signature": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        }
        try:
            with open(self.journal_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, sort_keys=True) + "\n")
        except OSError as exc:
            raise RuntimeError(f"Journal write failed: {exc}") from exc

    def verify_determinism(
        self, program: str, params: RefactoringParameter, runs: int = 2
    ) -> bool:
        """Re-run the same transaction and compare output digests."""

        if runs < 1:
            raise ValueError("runs must be at least 1")
        digests: list[str] = []
        for _ in range(runs):
            result = self.refactor(program, params)
            digests.append(hashlib.sha256(result.output_program.encode("utf-8")).hexdigest())
        return len(set(digests)) == 1
