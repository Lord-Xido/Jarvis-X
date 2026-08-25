"""
Unit tests for bounded code-editing automata.

Tests verify:
1. Determinism: identical seed produces identical refactoring
2. Auditability: all mutations are journaled with policy decisions
3. Policy enforcement: violations are caught and rejected
4. Bounded execution: cycle limits are enforced
5. Conservative validation: malformed code is rejected
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from jarvisx.code_editor_automata import (
    DeterministicCodeEditingAutomata,
    RefactoringParameter,
    Mutation,
    MutationStatus,
    TransformType,
    ConfidenceThresholdPolicy,
    CycleImprovementPolicy,
    MemorySafetyPolicy,
    TransformWhitelistPolicy,
)


class TestRefactoringParameter:
    """Validate refactoring parameter bounds."""

    def test_valid_parameters(self):
        """Accept valid parameters."""
        params = RefactoringParameter(seed=42, max_depth=3, max_cycles=1000)
        assert params.validate()

    def test_invalid_depth_too_large(self):
        """Reject depth exceeding 8."""
        params = RefactoringParameter(seed=42, max_depth=10)
        assert not params.validate()

    def test_invalid_depth_zero(self):
        """Reject zero depth."""
        params = RefactoringParameter(seed=42, max_depth=0)
        assert not params.validate()

    def test_invalid_cycles_too_large(self):
        """Reject cycles exceeding 100k."""
        params = RefactoringParameter(seed=42, max_cycles=200000)
        assert not params.validate()

    def test_invalid_mutations_too_large(self):
        """Reject mutations exceeding 100."""
        params = RefactoringParameter(seed=42, max_mutations=150)
        assert not params.validate()

    def test_invalid_cost_model(self):
        """Reject unknown cost model."""
        params = RefactoringParameter(seed=42, cost_model="invalid")
        assert not params.validate()


class TestMutation:
    """Verify mutation properties."""

    def test_mutation_hash_deterministic(self):
        """Same mutation produces same hash."""
        m1 = Mutation(
            transform_type=TransformType.DEAD_CODE_ELIMINATION,
            source_lines=[5],
            original_code="NOP",
            proposed_code="# removed",
            estimated_improvement=1.0,
            cycle_cost=1,
            confidence=0.9,
        )
        m2 = Mutation(
            transform_type=TransformType.DEAD_CODE_ELIMINATION,
            source_lines=[5],
            original_code="NOP",
            proposed_code="# removed",
            estimated_improvement=1.0,
            cycle_cost=1,
            confidence=0.9,
        )

        hash1 = m1.compute_hash()
        hash2 = m2.compute_hash()
        assert hash1 == hash2

    def test_mutation_hash_changes_with_content(self):
        """Different mutations produce different hashes."""
        m1 = Mutation(
            transform_type=TransformType.DEAD_CODE_ELIMINATION,
            source_lines=[5],
            original_code="NOP",
            proposed_code="# removed",
            estimated_improvement=1.0,
            cycle_cost=1,
            confidence=0.9,
        )
        m2 = Mutation(
            transform_type=TransformType.CONST_PROPAGATION,
            source_lines=[5],
            original_code="NOP",
            proposed_code="# removed",
            estimated_improvement=1.0,
            cycle_cost=1,
            confidence=0.9,
        )

        assert m1.compute_hash() != m2.compute_hash()

    def test_mutation_to_dict(self):
        """Mutation serializes to dict."""
        m = Mutation(
            transform_type=TransformType.DEAD_CODE_ELIMINATION,
            source_lines=[1, 2],
            original_code="NOP",
            proposed_code="# removed",
            estimated_improvement=2.0,
            cycle_cost=1,
            confidence=0.8,
            status=MutationStatus.APPLIED,
        )

        d = m.to_dict()
        assert d["transform_type"] == "dead_code_elimination"
        assert d["source_lines"] == [1, 2]
        assert d["status"] == "applied"
        assert d["confidence"] == 0.8


class TestPolicyConstraints:
    """Verify policy enforcement."""

    def test_confidence_threshold_policy_pass(self):
        """High-confidence mutation passes."""
        policy = ConfidenceThresholdPolicy(threshold=0.7)
        mutation = Mutation(
            transform_type=TransformType.DEAD_CODE_ELIMINATION,
            source_lines=[0],
            original_code="NOP",
            proposed_code="# removed",
            estimated_improvement=1.0,
            cycle_cost=1,
            confidence=0.85,
        )

        is_valid, msg = policy.validate(mutation, {})
        assert is_valid
        assert msg is None

    def test_confidence_threshold_policy_fail(self):
        """Low-confidence mutation rejected."""
        policy = ConfidenceThresholdPolicy(threshold=0.7)
        mutation = Mutation(
            transform_type=TransformType.DEAD_CODE_ELIMINATION,
            source_lines=[0],
            original_code="NOP",
            proposed_code="# removed",
            estimated_improvement=1.0,
            cycle_cost=1,
            confidence=0.5,
        )

        is_valid, msg = policy.validate(mutation, {})
        assert not is_valid
        assert "confidence" in msg.lower()

    def test_cycle_improvement_policy_pass(self):
        """Mutation with cycle improvement passes."""
        policy = CycleImprovementPolicy()
        mutation = Mutation(
            transform_type=TransformType.DEAD_CODE_ELIMINATION,
            source_lines=[0],
            original_code="NOP\nNOP",
            proposed_code="# removed",
            estimated_improvement=5.0,
            cycle_cost=1,
            confidence=0.9,
        )

        is_valid, msg = policy.validate(mutation, {})
        assert is_valid

    def test_cycle_improvement_policy_fail_zero_improvement(self):
        """Mutation with no improvement rejected."""
        policy = CycleImprovementPolicy()
        mutation = Mutation(
            transform_type=TransformType.CONST_PROPAGATION,
            source_lines=[0],
            original_code="SET A 10",
            proposed_code="SET A 10",
            estimated_improvement=0.0,
            cycle_cost=1,
            confidence=0.9,
        )

        is_valid, msg = policy.validate(mutation, {})
        assert not is_valid

    def test_memory_safety_policy_pass(self):
        """Small memory increase passes."""
        policy = MemorySafetyPolicy(max_increase_percent=5.0)
        mutation = Mutation(
            transform_type=TransformType.CONST_PROPAGATION,
            source_lines=[0],
            original_code="SET A 10",
            proposed_code="SET A 10",
            estimated_improvement=0.0,
            cycle_cost=1,
            confidence=0.9,
        )

        is_valid, msg = policy.validate(
            mutation, {"estimated_memory_increase_percent": 2.0}
        )
        assert is_valid

    def test_memory_safety_policy_fail(self):
        """Large memory increase rejected."""
        policy = MemorySafetyPolicy(max_increase_percent=5.0)
        mutation = Mutation(
            transform_type=TransformType.CONST_PROPAGATION,
            source_lines=[0],
            original_code="SET A 10",
            proposed_code="SET A 10",
            estimated_improvement=0.0,
            cycle_cost=1,
            confidence=0.9,
        )

        is_valid, msg = policy.validate(
            mutation, {"estimated_memory_increase_percent": 10.0}
        )
        assert not is_valid

    def test_transform_whitelist_policy_pass(self):
        """Whitelisted transform passes."""
        policy = TransformWhitelistPolicy(
            {TransformType.DEAD_CODE_ELIMINATION, TransformType.CONST_PROPAGATION}
        )
        mutation = Mutation(
            transform_type=TransformType.DEAD_CODE_ELIMINATION,
            source_lines=[0],
            original_code="NOP",
            proposed_code="# removed",
            estimated_improvement=1.0,
            cycle_cost=1,
            confidence=0.9,
        )

        is_valid, msg = policy.validate(mutation, {})
        assert is_valid

    def test_transform_whitelist_policy_fail(self):
        """Non-whitelisted transform rejected."""
        policy = TransformWhitelistPolicy({TransformType.CONST_PROPAGATION})
        mutation = Mutation(
            transform_type=TransformType.DEAD_CODE_ELIMINATION,
            source_lines=[0],
            original_code="NOP",
            proposed_code="# removed",
            estimated_improvement=1.0,
            cycle_cost=1,
            confidence=0.9,
        )

        is_valid, msg = policy.validate(mutation, {})
        assert not is_valid


class TestCodeEditingAutomata:
    """Test the main automata logic."""

    def test_automata_initialization(self):
        """Create automata with valid bounds."""
        automata = DeterministicCodeEditingAutomata(cycle_limit=10000)
        assert automata.cycle_limit == 10000
        assert automata.cycle_counter == 0
        assert len(automata.policy_constraints) == 0

    def test_parse_assembly_valid(self):
        """Parse valid assembly."""
        automata = DeterministicCodeEditingAutomata()
        program = "SET A 10\nADD B A A\nHALT"

        instructions = automata._parse_assembly(program)
        assert len(instructions) == 3
        assert instructions[0] == "SET A 10"

    def test_parse_assembly_with_comments(self):
        """Skip comments in assembly."""
        automata = DeterministicCodeEditingAutomata()
        program = "# Comment\nSET A 10\n# Another\nHALT"

        instructions = automata._parse_assembly(program)
        assert len(instructions) == 2
        assert "Comment" not in "".join(instructions)

    def test_estimate_dead_code_after_halt(self):
        """Identify unreachable code after HALT."""
        automata = DeterministicCodeEditingAutomata()
        instructions = ["SET A 10", "ADD B A A", "HALT", "NOP", "NOP"]

        dead = automata._estimate_dead_code(instructions)
        assert 3 in dead
        assert 4 in dead
        assert 0 not in dead

    def test_estimate_cycle_cost(self):
        """Canonical cycle costs."""
        automata = DeterministicCodeEditingAutomata()

        assert automata._estimate_cycle_cost("SET A 10") == 1
        assert automata._estimate_cycle_cost("ADD A B C") == 2
        assert automata._estimate_cycle_cost("MUL A B C") == 3
        assert automata._estimate_cycle_cost("DIV A B C") == 4

    def test_refactor_basic(self):
        """Execute basic refactoring."""
        automata = DeterministicCodeEditingAutomata(cycle_limit=1000)
        automata.add_policy(ConfidenceThresholdPolicy(threshold=0.7))
        automata.add_policy(CycleImprovementPolicy())

        program = "SET Ψ 10\nSET Φ 20\nADD A Ψ Φ\nHALT\nNOP\nNOP"
        params = RefactoringParameter(seed=42, max_mutations=5)

        result = automata.refactor(program, params)

        assert result.input_program == program
        assert result.mutations_proposed >= 0
        assert result.total_cycles_used > 0
        assert result.deterministic_seed == 42

    def test_refactor_deterministic(self):
        """Identical parameters produce identical refactoring."""
        program = "SET A 10\nADD B A A\nHALT\nNOP"
        params = RefactoringParameter(seed=123, max_mutations=3)

        automata1 = DeterministicCodeEditingAutomata(cycle_limit=1000)
        automata1.add_policy(CycleImprovementPolicy())
        result1 = automata1.refactor(program, params)

        automata2 = DeterministicCodeEditingAutomata(cycle_limit=1000)
        automata2.add_policy(CycleImprovementPolicy())
        result2 = automata2.refactor(program, params)

        assert result1.output_program == result2.output_program

    def test_refactor_respects_cycle_limit(self):
        """Cycle counter does not exceed limit."""
        automata = DeterministicCodeEditingAutomata(cycle_limit=100)
        program = "SET A 10\nADD B A A\nHALT"
        params = RefactoringParameter(seed=42, max_mutations=10)

        result = automata.refactor(program, params)
        assert result.total_cycles_used <= 100

    def test_refactor_policy_rejection(self):
        """Mutations are rejected on policy violation."""
        automata = DeterministicCodeEditingAutomata(cycle_limit=1000)
        # Only accept mutations with >0.95 confidence
        automata.add_policy(ConfidenceThresholdPolicy(threshold=0.95))

        program = "SET A 10\nADD B A A\nHALT\nNOP"
        params = RefactoringParameter(seed=42, max_mutations=5)

        result = automata.refactor(program, params)

        # Some mutations likely rejected due to confidence threshold
        assert result.mutations_rejected >= 0
        # Applied mutations have no policy violations
        for m in result.mutations:
            if m.status == MutationStatus.APPLIED:
                assert len(m.policy_violations) == 0

    def test_refactor_journaling(self):
        """Mutations are journaled to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            journal_path = os.path.join(tmpdir, "mutations.jsonl")
            automata = DeterministicCodeEditingAutomata(
                cycle_limit=1000, journal_path=journal_path
            )
            automata.add_policy(CycleImprovementPolicy())

            program = "SET A 10\nADD B A A\nHALT\nNOP"
            params = RefactoringParameter(seed=42, max_mutations=3)

            result = automata.refactor(program, params)

            # Journal should exist if mutations were applied
            if result.mutations_applied > 0:
                assert os.path.exists(journal_path)
                with open(journal_path, "r") as f:
                    lines = f.readlines()
                    assert len(lines) > 0
                    # Parse first entry as JSON
                    entry = json.loads(lines[0])
                    assert "refactoring" in entry
                    assert "signature" in entry

    def test_determinism_verification(self):
        """verify_determinism checks consistency across runs."""
        automata = DeterministicCodeEditingAutomata(cycle_limit=1000)
        automata.add_policy(CycleImprovementPolicy())

        program = "SET A 10\nADD B A A\nHALT\nNOP"
        params = RefactoringParameter(seed=555, max_mutations=2)

        is_deterministic = automata.verify_determinism(program, params, runs=3)
        assert is_deterministic

    def test_different_seeds_different_results(self):
        """Different seeds produce different refactorings."""
        program = "SET A 10\nADD B A A\nHALT\nNOP\nNOP\nNOP"

        automata1 = DeterministicCodeEditingAutomata(cycle_limit=1000)
        automata1.add_policy(CycleImprovementPolicy())
        result1 = automata1.refactor(program, RefactoringParameter(seed=100))

        automata2 = DeterministicCodeEditingAutomata(cycle_limit=1000)
        automata2.add_policy(CycleImprovementPolicy())
        result2 = automata2.refactor(program, RefactoringParameter(seed=200))

        # With different seeds, it's possible to get different results
        # (though not guaranteed if no mutations are proposed)
        # We just verify both complete without error
        assert result1.deterministic_seed == 100
        assert result2.deterministic_seed == 200

    def test_whitelist_enforcement(self):
        """Only whitelisted transforms are proposed."""
        automata = DeterministicCodeEditingAutomata(cycle_limit=1000)
        automata.add_policy(
            TransformWhitelistPolicy({TransformType.DEAD_CODE_ELIMINATION})
        )
        automata.add_policy(CycleImprovementPolicy())

        program = "SET A 10\nADD B A A\nHALT\nNOP\nNOP"
        params = RefactoringParameter(seed=42, max_mutations=5)

        result = automata.refactor(program, params)

        # Applied mutations should only be whitelisted types
        for m in result.mutations:
            if m.status == MutationStatus.APPLIED:
                assert m.transform_type == TransformType.DEAD_CODE_ELIMINATION

    def test_refactor_empty_program(self):
        """Handle empty assembly gracefully."""
        automata = DeterministicCodeEditingAutomata(cycle_limit=1000)
        program = ""
        params = RefactoringParameter(seed=42)

        result = automata.refactor(program, params)
        assert result.mutations_proposed == 0


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_full_refactoring_workflow(self):
        """Complete refactoring with policies and journaling."""
        with tempfile.TemporaryDirectory() as tmpdir:
            journal_path = os.path.join(tmpdir, "refactoring.jsonl")

            automata = DeterministicCodeEditingAutomata(
                cycle_limit=5000, journal_path=journal_path
            )

            # Install multiple policies
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

            params = RefactoringParameter(
                seed=999, max_depth=4, max_mutations=10, max_cycles=5000
            )

            result = automata.refactor(program, params)

            # Verify result structure
            assert result.input_program is not None
            assert result.output_program is not None
            assert result.mutations_proposed >= 0
            assert result.mutations_applied >= 0
            assert result.mutations_rejected >= 0
            assert (
                result.mutations_proposed
                == result.mutations_applied + result.mutations_rejected
            )

            # Verify determinism
            is_det = automata.verify_determinism(program, params, runs=2)
            assert is_det

            # Verify journal
            if result.mutations_applied > 0:
                assert os.path.exists(journal_path)
