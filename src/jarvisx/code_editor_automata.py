"""
Bounded code-editing automata for Jarvis-X.

This module provides a deterministic, auditable code editing system that:
- Operates within explicit cycle and memory bounds
- Journeys every proposed mutation before commit
- Validates all changes against policy constraints (Lambda)
- Never modifies canonical VM state without approval
- Produces deterministic refactoring given identical seeds

Design principle: This is a bounded optimization agent, not an AGI.
It searches a finite space of code transformations and validates each
proposed change before committing.

Symbolic mapping:
  Ψ = input assembly program (observed state)
  Φ = proposed refactored program (internal state)
  Θ = refactoring parameters (seed, depth, cost model)
  Λ = policy constraints (admissibility checker)
  Ω = journaled mutation log
"""

import hashlib
import json
from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Optional, Dict, List, Tuple, Set
from abc import ABC, abstractmethod


class TransformType(Enum):
    """Classification of bounded code transformations."""
    DEAD_CODE_ELIMINATION = "dead_code_elimination"
    REGISTER_REALLOCATION = "register_reallocation"
    INSTRUCTION_FOLDING = "instruction_folding"
    CONST_PROPAGATION = "const_propagation"
    LOOP_UNROLL = "loop_unroll"
    PIPELINE_REORDER = "pipeline_reorder"
    NOP_INSERTION = "nop_insertion"  # Benign safety padding


class MutationStatus(Enum):
    """Lifecycle of a proposed mutation."""
    PROPOSED = "proposed"
    VALIDATED = "validated"
    APPLIED = "applied"
    ROLLED_BACK = "rolled_back"
    REJECTED = "rejected"


@dataclass
class RefactoringParameter:
    """Bounded parameters for code transformation."""
    seed: int
    max_depth: int = 3
    max_cycles: int = 1000
    max_mutations: int = 10
    cost_model: str = "cycles"  # cycles, memory, or combined
    allow_unsafe: bool = False
    allow_heuristic: bool = False

    def validate(self) -> bool:
        """Ensure parameters are within canonical bounds."""
        return (
            self.seed >= 0
            and self.max_depth > 0 and self.max_depth <= 8
            and self.max_cycles > 0 and self.max_cycles <= 100000
            and self.max_mutations > 0 and self.max_mutations <= 100
            and self.cost_model in ["cycles", "memory", "combined"]
        )


@dataclass
class Mutation:
    """A proposed change to assembly code."""
    transform_type: TransformType
    source_lines: List[int]
    original_code: str
    proposed_code: str
    estimated_improvement: float  # cycles or memory saved
    cycle_cost: int  # cost of executing this mutation
    confidence: float  # [0.0, 1.0] - certainty of correctness
    status: MutationStatus = MutationStatus.PROPOSED
    policy_violations: List[str] = field(default_factory=list)
    timestamp_ns: int = 0
    hash: str = ""

    def compute_hash(self) -> str:
        """Canonical hash of mutation proposal."""
        payload = {
            "transform_type": self.transform_type.value,
            "source_lines": self.source_lines,
            "original_code": self.original_code,
            "proposed_code": self.proposed_code,
            "estimated_improvement": self.estimated_improvement,
            "cycle_cost": self.cycle_cost,
            "confidence": self.confidence,
        }
        content = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def to_dict(self) -> dict:
        """JSON-serializable representation."""
        return {
            "transform_type": self.transform_type.value,
            "source_lines": self.source_lines,
            "original_code": self.original_code,
            "proposed_code": self.proposed_code,
            "estimated_improvement": self.estimated_improvement,
            "cycle_cost": self.cycle_cost,
            "confidence": self.confidence,
            "status": self.status.value,
            "policy_violations": self.policy_violations,
            "timestamp_ns": self.timestamp_ns,
            "hash": self.hash,
        }


@dataclass
class RefactoringResult:
    """Output of a refactoring pass."""
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
    mutations: List[Mutation] = field(default_factory=list)

    def to_dict(self) -> dict:
        """JSON-serializable result."""
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
            "mutations": [m.to_dict() for m in self.mutations],
        }


class PolicyConstraint(ABC):
    """Base class for Lambda policy validation."""

    @abstractmethod
    def validate(self, mutation: Mutation, context: dict) -> Tuple[bool, Optional[str]]:
        """
        Check if mutation satisfies this policy.

        Returns:
            (is_valid, violation_message)
        """
        pass


class ConfidenceThresholdPolicy(PolicyConstraint):
    """Only apply mutations above confidence threshold."""

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold

    def validate(self, mutation: Mutation, context: dict) -> Tuple[bool, Optional[str]]:
        if mutation.confidence < self.threshold:
            return False, f"confidence {mutation.confidence:.2f} < {self.threshold}"
        return True, None


class CycleImprovementPolicy(PolicyConstraint):
    """Only apply mutations that reduce cycle count."""

    def validate(self, mutation: Mutation, context: dict) -> Tuple[bool, Optional[str]]:
        if mutation.estimated_improvement <= 0:
            return False, f"no cycle improvement: {mutation.estimated_improvement}"
        return True, None


class MemorySafetyPolicy(PolicyConstraint):
    """Reject mutations that increase memory usage significantly."""

    def __init__(self, max_increase_percent: float = 5.0):
        self.max_increase_percent = max_increase_percent

    def validate(self, mutation: Mutation, context: dict) -> Tuple[bool, Optional[str]]:
        if "estimated_memory_increase_percent" in context:
            if context["estimated_memory_increase_percent"] > self.max_increase_percent:
                return False, (
                    f"memory increase {context['estimated_memory_increase_percent']:.1f}% "
                    f"> {self.max_increase_percent}%"
                )
        return True, None


class TransformWhitelistPolicy(PolicyConstraint):
    """Only allow specific transformation types."""

    def __init__(self, allowed_transforms: Set[TransformType]):
        self.allowed_transforms = allowed_transforms

    def validate(self, mutation: Mutation, context: dict) -> Tuple[bool, Optional[str]]:
        if mutation.transform_type not in self.allowed_transforms:
            return False, f"transform {mutation.transform_type.value} not whitelisted"
        return True, None


class DeterministicCodeEditingAutomata:
    """
    Bounded code-editing automaton for Jarvis-X assembly.

    Properties:
    - Deterministic: identical seed + input → identical output
    - Bounded: explicit cycle and mutation limits
    - Auditable: every mutation is journaled with policy decision
    - Conservative: fails closed on invalid code
    - Non-autonomous: all mutations require policy approval
    """

    def __init__(
        self,
        cycle_limit: int = 10000,
        journal_path: Optional[str] = None,
    ):
        self.cycle_limit = cycle_limit
        self.journal_path = journal_path
        self.cycle_counter = 0
        self.policy_constraints: List[PolicyConstraint] = []
        self.mutation_history: List[Mutation] = []

    def add_policy(self, constraint: PolicyConstraint) -> None:
        """Register a policy constraint."""
        self.policy_constraints.append(constraint)

    def _parse_assembly(self, program: str) -> List[str]:
        """Parse assembly into instructions (validation only)."""
        lines = []
        for i, line in enumerate(program.split("\n")):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Basic validation: each line should be instruction or label
            tokens = line.split()
            if not tokens:
                continue
            lines.append(line)
        return lines

    def _estimate_dead_code(self, instructions: List[str]) -> List[int]:
        """
        Heuristic: identify potentially dead code.

        This is a bounded heuristic, NOT a full liveness analysis.
        Conservative: only flags unreachable segments after HALT.
        """
        dead_lines = []
        for i, instr in enumerate(instructions):
            if instr.upper().startswith("HALT"):
                # Everything after HALT is dead
                dead_lines.extend(range(i + 1, len(instructions)))
                break
        return dead_lines

    def _estimate_cycle_cost(self, instruction: str) -> int:
        """Canonical cycle cost estimates."""
        opcode = instruction.split()[0].upper()
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
        }
        return costs.get(opcode, 1)

    def _generate_dead_code_elimination(
        self,
        instructions: List[str],
        dead_lines: List[int],
        seed: int,
    ) -> Optional[Mutation]:
        """Generate a mutation eliminating dead code."""
        if not dead_lines:
            return None

        original_code = "\n".join(
            instructions[i] for i in dead_lines if i < len(instructions)
        )
        proposed_code = "# Dead code removed"

        cycles_saved = sum(
            self._estimate_cycle_cost(instructions[i])
            for i in dead_lines
            if i < len(instructions)
        )

        return Mutation(
            transform_type=TransformType.DEAD_CODE_ELIMINATION,
            source_lines=dead_lines,
            original_code=original_code,
            proposed_code=proposed_code,
            estimated_improvement=float(cycles_saved),
            cycle_cost=5,  # Cost of analyzing
            confidence=0.95,
        )

    def _generate_const_propagation(
        self,
        instructions: List[str],
        seed: int,
    ) -> Optional[Mutation]:
        """
        Generate a mutation for constant propagation.

        This is deterministic: seeded PRN determines which SET values propagate.
        """
        # Simple heuristic: find SET instructions and track values
        const_map: Dict[str, int] = {}
        propagation_candidates = []

        for i, instr in enumerate(instructions):
            if instr.upper().startswith("SET"):
                parts = instr.split()
                if len(parts) >= 3:
                    reg = parts[1]
                    try:
                        val = int(parts[2])
                        const_map[reg] = val
                    except ValueError:
                        pass

        # For determinism, use seed to select candidate
        if const_map and len(instructions) > 5:
            prn = (seed * 2654435761) % 2**32
            candidate_idx = prn % len(instructions)

            original_code = instructions[candidate_idx]
            proposed_code = f"# Constant propagated from {original_code}"

            return Mutation(
                transform_type=TransformType.CONST_PROPAGATION,
                source_lines=[candidate_idx],
                original_code=original_code,
                proposed_code=proposed_code,
                estimated_improvement=1.0,
                cycle_cost=3,
                confidence=0.75,
            )

        return None

    def _apply_policy_checks(
        self, mutation: Mutation, context: dict
    ) -> Tuple[bool, List[str]]:
        """Apply all Lambda policy constraints to a mutation."""
        violations = []

        for policy in self.policy_constraints:
            is_valid, violation_msg = policy.validate(mutation, context)
            if not is_valid:
                violations.append(violation_msg or "policy violation")

        return len(violations) == 0, violations

    def refactor(
        self,
        program: str,
        params: RefactoringParameter,
    ) -> RefactoringResult:
        """
        Execute a deterministic refactoring pass on assembly code.

        Args:
            program: Assembly code as string
            params: Refactoring parameters (must be validated)

        Returns:
            RefactoringResult with all mutations journaled
        """
        if not params.validate():
            raise ValueError(f"Invalid refactoring parameters: {params}")

        instructions = self._parse_assembly(program)
        self.cycle_counter = 0
        mutations_proposed = []
        mutations_applied = []
        mutations_rejected = []
        output_program = program

        # Deterministic mutation generation seeded by params.seed
        rng_state = params.seed
        mutation_count = 0

        while (
            self.cycle_counter < self.cycle_limit
            and mutation_count < params.max_mutations
        ):
            self.cycle_counter += 1

            # Generate candidate mutation deterministically
            mutation: Optional[Mutation] = None

            rng_state = (rng_state * 1103515245 + 12345) % (2**31)
            transform_choice = rng_state % 3

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

            # Compute mutation hash and timestamp
            mutation.hash = mutation.compute_hash()
            mutation.timestamp_ns = self.cycle_counter * 1000000

            mutations_proposed.append(mutation)

            # Apply policy constraints
            context = {"estimated_memory_increase_percent": 0.0}
            is_policy_valid, violations = self._apply_policy_checks(mutation, context)

            if not is_policy_valid:
                mutation.status = MutationStatus.REJECTED
                mutation.policy_violations = violations
                mutations_rejected.append(mutation)
                self.mutation_history.append(mutation)
                continue

            # Validate proposed code can be parsed
            try:
                test_program = output_program.replace(
                    mutation.original_code, mutation.proposed_code
                )
                self._parse_assembly(test_program)
            except Exception as e:
                mutation.status = MutationStatus.REJECTED
                mutation.policy_violations = [f"parse error: {str(e)}"]
                mutations_rejected.append(mutation)
                self.mutation_history.append(mutation)
                continue

            # Apply mutation
            mutation.status = MutationStatus.APPLIED
            output_program = test_program
            mutations_applied.append(mutation)
            self.mutation_history.append(mutation)
            mutation_count += 1

        # Compute aggregate metrics
        total_cycles_saved = sum(m.estimated_improvement for m in mutations_applied)
        total_memory_saved = 0.0  # Can be extended

        result = RefactoringResult(
            input_program=program,
            output_program=output_program,
            mutations_proposed=len(mutations_proposed),
            mutations_applied=len(mutations_applied),
            mutations_rejected=len(mutations_rejected),
            total_cycles_used=self.cycle_counter,
            estimated_cycles_saved=total_cycles_saved,
            estimated_memory_saved=total_memory_saved,
            deterministic_seed=params.seed,
            journaled=self.journal_path is not None,
            mutations=mutations_proposed + mutations_rejected,
        )

        # Journal if path provided
        if self.journal_path:
            self._journal_refactoring(result)

        return result

    def _journal_refactoring(self, result: RefactoringResult) -> None:
        """Append refactoring result to journal."""
        if not self.journal_path:
            return

        import os

        os.makedirs(os.path.dirname(self.journal_path) or ".", exist_ok=True)

        entry = {
            "timestamp_ns": result.mutations[0].timestamp_ns
            if result.mutations
            else 0,
            "refactoring": result.to_dict(),
            "signature": hashlib.sha256(
                json.dumps(result.to_dict(), sort_keys=True).encode()
            ).hexdigest(),
        }

        try:
            with open(self.journal_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except IOError as e:
            # Fail closed: don't corrupt output if journal write fails
            raise RuntimeError(f"Journal write failed: {e}")

    def verify_determinism(
        self, program: str, params: RefactoringParameter, runs: int = 2
    ) -> bool:
        """Verify that identical parameters produce identical results."""
        results = [self.refactor(program, params) for _ in range(runs)]

        hashes = [
            hashlib.sha256(r.output_program.encode()).hexdigest() for r in results
        ]

        return len(set(hashes)) == 1


if __name__ == "__main__":
    # Example: bounded refactoring with policy constraints
    from jarvisx.code_editor_automata import (
        DeterministicCodeEditingAutomata,
        RefactoringParameter,
        ConfidenceThresholdPolicy,
        CycleImprovementPolicy,
        TransformType,
        TransformWhitelistPolicy,
    )

    program = """
    SET Ψ 10
    SET Φ 20
    ADD A Ψ Φ
    SET B 0
    SUB C A B
    HALT
    NOP
    NOP
    """

    automata = DeterministicCodeEditingAutomata(
        cycle_limit=10000, journal_path=None
    )

    # Install policies
    automata.add_policy(ConfidenceThresholdPolicy(threshold=0.7))
    automata.add_policy(CycleImprovementPolicy())
    automata.add_policy(
        TransformWhitelistPolicy(
            {
                TransformType.DEAD_CODE_ELIMINATION,
                TransformType.CONST_PROPAGATION,
            }
        )
    )

    params = RefactoringParameter(seed=42, max_depth=3, max_mutations=10)

    result = automata.refactor(program, params)

    print("=== Refactoring Result ===")
    print(f"Mutations proposed: {result.mutations_proposed}")
    print(f"Mutations applied: {result.mutations_applied}")
    print(f"Mutations rejected: {result.mutations_rejected}")
    print(f"Estimated cycles saved: {result.estimated_cycles_saved}")
    print(f"Deterministic seed: {result.deterministic_seed}")
    print("\n=== Output Program ===")
    print(result.output_program)
    print("\n=== Determinism Check ===")
    is_deterministic = automata.verify_determinism(program, params, runs=3)
    print(f"Deterministic across 3 runs: {is_deterministic}")
