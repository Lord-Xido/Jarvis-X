#!/usr/bin/env python3
"""
Demonstration: Bounded code-editing automata applied to real Jarvis-X bytecode.

This example shows:
1. A simple assembly program with dead code
2. Refactoring with deterministic seeding
3. Functional equivalence verification
4. Policy-constrained mutation
5. Determinism verification
"""

from jarvisx.parser import Parser
from jarvisx.assembler import Assembler
from jarvisx.core import CodexVM
from jarvisx.code_editor_automata import (
    DeterministicCodeEditingAutomata,
    RefactoringParameter,
    ConfidenceThresholdPolicy,
    CycleImprovementPolicy,
    TransformType,
    TransformWhitelistPolicy,
)


def demo_basic_refactoring():
    """Basic refactoring without policies."""
    print("=" * 70)
    print("DEMO 1: Basic Refactoring")
    print("=" * 70)

    program = """
    # Simple addition with dead code
    SET Ψ 10
    SET Φ 20
    ADD A Ψ Φ
    HALT
    NOP
    NOP
    NOP
    """

    automata = DeterministicCodeEditingAutomata(cycle_limit=5000)
    params = RefactoringParameter(seed=42, max_mutations=10)

    result = automata.refactor(program, params)

    print("\nInput program:")
    print(program)

    print("\nRefactored program:")
    print(result.output_program)

    print("\nMetrics:")
    print(f"  Mutations proposed: {result.mutations_proposed}")
    print(f"  Mutations applied: {result.mutations_applied}")
    print(f"  Mutations rejected: {result.mutations_rejected}")
    print(f"  Cycles used: {result.total_cycles_used}")
    print(f"  Estimated cycles saved: {result.estimated_cycles_saved}")
    print(f"  Deterministic seed: {result.deterministic_seed}")


def demo_policy_constrained_refactoring():
    """Refactoring with multiple policy constraints."""
    print("\n" + "=" * 70)
    print("DEMO 2: Policy-Constrained Refactoring")
    print("=" * 70)

    program = """
    SET A 10
    ADD B A A
    HALT
    NOP
    NOP
    NOP
    """

    automata = DeterministicCodeEditingAutomata(cycle_limit=10000)

    # Install policies
    automata.add_policy(ConfidenceThresholdPolicy(threshold=0.8))
    automata.add_policy(CycleImprovementPolicy())
    automata.add_policy(
        TransformWhitelistPolicy(
            {TransformType.DEAD_CODE_ELIMINATION, TransformType.CONST_PROPAGATION}
        )
    )

    params = RefactoringParameter(seed=123, max_mutations=5)
    result = automata.refactor(program, params)

    print("\nProgram:")
    print(program)

    print("\nPolicies installed:")
    print("  • ConfidenceThreshold (0.8)")
    print("  • CycleImprovement")
    print("  • TransformWhitelist (DEAD_CODE_ELIMINATION, CONST_PROPAGATION)")

    print("\nMetrics:")
    print(f"  Mutations applied (passed all policies): {result.mutations_applied}")
    print(f"  Mutations rejected (policy violation): {result.mutations_rejected}")
    print(f"  Total mutations analyzed: {result.mutations_proposed}")

    # Show rejected mutation reasons
    for i, mutation in enumerate(result.mutations):
        if mutation.status.value == "rejected":
            print(f"\n  Rejected mutation {i}:")
            print(f"    Type: {mutation.transform_type.value}")
            print(f"    Violations: {mutation.policy_violations}")


def demo_determinism_verification():
    """Verify that identical seeds produce identical results."""
    print("\n" + "=" * 70)
    print("DEMO 3: Determinism Verification")
    print("=" * 70)

    program = """
    SET X 5
    SET Y 10
    ADD Z X Y
    HALT
    NOP
    NOP
    """

    automata = DeterministicCodeEditingAutomata(cycle_limit=5000)
    automata.add_policy(CycleImprovementPolicy())

    params = RefactoringParameter(seed=999, max_mutations=5)

    print("\nProgram:")
    print(program)

    print("\nVerifying determinism across 5 independent runs...")
    is_deterministic = automata.verify_determinism(program, params, runs=5)

    print(f"\nResult: {'✓ DETERMINISTIC' if is_deterministic else '✗ NOT DETERMINISTIC'}")

    if is_deterministic:
        print("  Identical parameters produce identical outputs ✓")


def demo_functional_equivalence():
    """Verify refactored program produces same results."""
    print("\n" + "=" * 70)
    print("DEMO 4: Functional Equivalence Verification")
    print("=" * 70)

    original_program = """
    SET A 10
    SET B 20
    ADD C A B
    HALT
    NOP
    NOP
    """

    automata = DeterministicCodeEditingAutomata(cycle_limit=5000)
    automata.add_policy(CycleImprovementPolicy())

    params = RefactoringParameter(seed=42, max_mutations=10)
    result = automata.refactor(original_program, params)

    # Execute both
    print("\nExecuting original program...")
    vm1 = CodexVM()
    vm1.load(Assembler().assemble(Parser().parse(original_program)))
    state1 = vm1.run()
    print(f"  Original cycle count: {vm1.cycle_counter}")
    print(f"  Registers: {state1}")

    print("\nExecuting refactored program...")
    vm2 = CodexVM()
    vm2.load(Assembler().assemble(Parser().parse(result.output_program)))
    state2 = vm2.run()
    print(f"  Refactored cycle count: {vm2.cycle_counter}")
    print(f"  Registers: {state2}")

    if state1 == state2:
        print("\n✓ FUNCTIONALLY EQUIVALENT")
        improvement = vm1.cycle_counter - vm2.cycle_counter
        if improvement > 0:
            pct = 100 * improvement / vm1.cycle_counter
            print(f"✓ IMPROVEMENT: {improvement} cycles saved ({pct:.1f}%)")
        elif improvement < 0:
            print(f"⚠ REGRESSION: {-improvement} cycles added")
        else:
            print("⚠ NO CHANGE in cycle count")
    else:
        print("\n✗ NOT EQUIVALENT (bug in refactoring!)")


def demo_journal_inspection():
    """Demonstrate journaling of mutations."""
    print("\n" + "=" * 70)
    print("DEMO 5: Mutation Journal Inspection")
    print("=" * 70)

    import json
    import tempfile
    import os

    program = """
    SET A 10
    ADD B A A
    HALT
    NOP
    NOP
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        journal_path = os.path.join(tmpdir, "mutations.jsonl")

        automata = DeterministicCodeEditingAutomata(
            cycle_limit=5000, journal_path=journal_path
        )
        automata.add_policy(CycleImprovementPolicy())

        params = RefactoringParameter(seed=42, max_mutations=5)
        result = automata.refactor(program, params)

        print("\nProgram:")
        print(program)

        print(f"\nJournal written to: {journal_path}")

        if result.mutations_applied > 0 and os.path.exists(journal_path):
            print("\nJournal entries:")
            with open(journal_path, "r") as f:
                for line_num, line in enumerate(f, 1):
                    entry = json.loads(line)
                    refactoring = entry["refactoring"]
                    print(f"\n  Entry {line_num}:")
                    print(
                        f"    Mutations applied: {refactoring['mutations_applied']}"
                    )
                    print(
                        f"    Mutations rejected: {refactoring['mutations_rejected']}"
                    )
                    print(f"    Cycles saved (est.): {refactoring['estimated_cycles_saved']}")
                    print(
                        f"    Signature: {entry['signature'][:16]}..."
                    )

        else:
            print("(No mutations applied in this refactoring)")


def demo_comparison_different_seeds():
    """Show how different seeds produce different results."""
    print("\n" + "=" * 70)
    print("DEMO 6: Effect of Different Seeds")
    print("=" * 70)

    program = """
    SET A 10
    ADD B A A
    HALT
    NOP
    NOP
    NOP
    """

    print("\nProgram:")
    print(program)

    results = {}
    for seed in [42, 123, 999]:
        automata = DeterministicCodeEditingAutomata(cycle_limit=5000)
        automata.add_policy(CycleImprovementPolicy())

        params = RefactoringParameter(seed=seed, max_mutations=5)
        result = automata.refactor(program, params)

        results[seed] = result

    print("\nComparison of results across different seeds:")
    print(f"{'Seed':<10} {'Applied':<12} {'Rejected':<12} {'Est. Saved':<15}")
    print("-" * 50)
    for seed, result in results.items():
        print(
            f"{seed:<10} {result.mutations_applied:<12} {result.mutations_rejected:<12} {result.estimated_cycles_saved:<15.1f}"
        )


if __name__ == "__main__":
    demo_basic_refactoring()
    demo_policy_constrained_refactoring()
    demo_determinism_verification()
    demo_functional_equivalence()
    demo_journal_inspection()
    demo_comparison_different_seeds()

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)
