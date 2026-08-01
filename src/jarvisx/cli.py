import sys
import json
import argparse
from .parser import Parser
from .assembler import Assembler
from .core import CodexVM
from .api import start_api
from .web import start_web
from .node import CodexNode
from .code_editor_automata import (
    DeterministicCodeEditingAutomata,
    RefactoringParameter,
    ConfidenceThresholdPolicy,
    CycleImprovementPolicy,
    MemorySafetyPolicy,
    TransformWhitelistPolicy,
    TransformType,
)


def cmd_run(args):
    """Execute bytecode program."""
    with open(args.file) as f:
        source = f.read()
    ast = Parser().parse(source)
    bytecode = Assembler().assemble(ast)
    vm = CodexVM()
    vm.load(bytecode)
    vm.run()
    print("Registers:", vm.regs.snapshot())
    if args.ledger:
        print(f"Ledger entries: {len(vm.ledger.chain)}")
        print(f"Verified: {vm.ledger.verify()}")


def cmd_refactor(args):
    """Refactor assembly code with bounded automata."""
    with open(args.file) as f:
        program = f.read()

    automata = DeterministicCodeEditingAutomata(
        cycle_limit=args.cycle_limit,
        journal_path=args.journal if args.journal else None,
    )

    # Install policies
    if args.confidence:
        automata.add_policy(ConfidenceThresholdPolicy(args.confidence))

    if args.cycle_improvement:
        automata.add_policy(CycleImprovementPolicy())

    if args.memory_safety:
        automata.add_policy(MemorySafetyPolicy(args.memory_safety))

    if args.whitelist:
        allowed = {
            TransformType[t.upper()]
            for t in args.whitelist.split(",")
            if t.strip()
        }
        automata.add_policy(TransformWhitelistPolicy(allowed))

    params = RefactoringParameter(
        seed=args.seed,
        max_depth=args.max_depth,
        max_cycles=args.max_cycles,
        max_mutations=args.max_mutations,
        cost_model=args.cost_model,
    )

    result = automata.refactor(program, params)

    # Output results
    print(f"✓ Refactoring complete")
    print(f"  Mutations proposed: {result.mutations_proposed}")
    print(f"  Mutations applied: {result.mutations_applied}")
    print(f"  Mutations rejected: {result.mutations_rejected}")
    print(f"  Cycles used: {result.total_cycles_used}")
    print(f"  Cycles saved (est.): {result.estimated_cycles_saved}")
    print(f"  Memory saved (est.): {result.estimated_memory_saved}")
    print(f"  Deterministic seed: {result.deterministic_seed}")
    print(f"  Journaled: {result.journaled}")

    if args.output:
        with open(args.output, "w") as f:
            f.write(result.output_program)
        print(f"  Output written to: {args.output}")

    if args.json:
        output = {
            "input": result.input_program,
            "output": result.output_program,
            "metrics": {
                "mutations_proposed": result.mutations_proposed,
                "mutations_applied": result.mutations_applied,
                "mutations_rejected": result.mutations_rejected,
                "total_cycles_used": result.total_cycles_used,
                "estimated_cycles_saved": result.estimated_cycles_saved,
                "estimated_memory_saved": result.estimated_memory_saved,
                "deterministic_seed": result.deterministic_seed,
            },
        }
        with open(args.json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  Metrics written to: {args.json}")

    # Verify determinism if requested
    if args.verify_determinism:
        is_det = automata.verify_determinism(program, params, args.verify_determinism)
        status = "✓ DETERMINISTIC" if is_det else "✗ NOT DETERMINISTIC"
        print(f"\n  Determinism check ({args.verify_determinism} runs): {status}")


def cmd_verify_equivalence(args):
    """Verify refactored program is functionally equivalent."""
    with open(args.original) as f:
        original = f.read()
    with open(args.refactored) as f:
        refactored = f.read()

    # Execute both
    vm1 = CodexVM()
    vm1.load(Assembler().assemble(Parser().parse(original)))
    state1 = vm1.run()

    vm2 = CodexVM()
    vm2.load(Assembler().assemble(Parser().parse(refactored)))
    state2 = vm2.run()

    # Compare
    print(f"Original cycles: {vm1.cycle_counter}")
    print(f"Refactored cycles: {vm2.cycle_counter}")

    if state1 == state2:
        print("✓ FUNCTIONALLY EQUIVALENT")
        improvement = vm1.cycle_counter - vm2.cycle_counter
        if improvement > 0:
            print(f"✓ IMPROVEMENT: {improvement} cycles saved ({100*improvement/vm1.cycle_counter:.1f}%)")
        elif improvement < 0:
            print(f"⚠ REGRESSION: {-improvement} cycles added")
        else:
            print("⚠ NO CHANGE in cycle count")
    else:
        print("✗ NOT EQUIVALENT")
        print(f"Original state: {state1}")
        print(f"Refactored state: {state2}")


def main():
    parser = argparse.ArgumentParser(
        prog="jarvisx",
        description="Jarvis-X: Deterministic bytecode VM and bounded research platform",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # RUN command
    run_parser = subparsers.add_parser("run", help="Execute bytecode program")
    run_parser.add_argument("file", help="Assembly file to execute")
    run_parser.add_argument("--ledger", action="store_true", help="Show ledger info")
    run_parser.set_defaults(func=cmd_run)

    # REFACTOR command
    refactor_parser = subparsers.add_parser(
        "refactor", help="Refactor assembly with bounded automata"
    )
    refactor_parser.add_argument("file", help="Assembly file to refactor")
    refactor_parser.add_argument(
        "--seed", type=int, default=42, help="PRNG seed for determinism (default: 42)"
    )
    refactor_parser.add_argument(
        "--max-depth", type=int, default=3, help="Max heuristic depth (1-8, default: 3)"
    )
    refactor_parser.add_argument(
        "--max-cycles",
        type=int,
        default=1000,
        help="Cycle budget per refactoring (default: 1000)",
    )
    refactor_parser.add_argument(
        "--max-mutations",
        type=int,
        default=10,
        help="Max mutations to apply (default: 10)",
    )
    refactor_parser.add_argument(
        "--cycle-limit",
        type=int,
        default=10000,
        help="Total automata cycle limit (default: 10000)",
    )
    refactor_parser.add_argument(
        "--cost-model",
        choices=["cycles", "memory", "combined"],
        default="cycles",
        help="Optimization cost model (default: cycles)",
    )
    refactor_parser.add_argument(
        "--confidence",
        type=float,
        help="Minimum mutation confidence (0.0-1.0)",
    )
    refactor_parser.add_argument(
        "--cycle-improvement",
        action="store_true",
        help="Only apply cycle-improving mutations",
    )
    refactor_parser.add_argument(
        "--memory-safety",
        type=float,
        help="Max memory increase percent",
    )
    refactor_parser.add_argument(
        "--whitelist",
        help="Comma-separated allowed transforms (e.g. DEAD_CODE_ELIMINATION,CONST_PROPAGATION)",
    )
    refactor_parser.add_argument(
        "--output", "-o", help="Write refactored program to file"
    )
    refactor_parser.add_argument(
        "--json", "-j", help="Write metrics to JSON file"
    )
    refactor_parser.add_argument(
        "--journal", help="Write mutation journal to JSONL file"
    )
    refactor_parser.add_argument(
        "--verify-determinism",
        type=int,
        metavar="N",
        help="Verify determinism across N runs",
    )
    refactor_parser.set_defaults(func=cmd_refactor)

    # VERIFY-EQUIVALENCE command
    verify_parser = subparsers.add_parser(
        "verify-equivalence", help="Verify refactored program equivalence"
    )
    verify_parser.add_argument("original", help="Original assembly file")
    verify_parser.add_argument("refactored", help="Refactored assembly file")
    verify_parser.set_defaults(func=cmd_verify_equivalence)

    # API command
    api_parser = subparsers.add_parser("api", help="Start API server")
    api_parser.set_defaults(func=lambda args: start_api())

    # WEB command
    web_parser = subparsers.add_parser("web", help="Start web interface")
    web_parser.set_defaults(func=lambda args: start_web())

    # NODE command
    node_parser = subparsers.add_parser("node", help="Start node")
    node_parser.set_defaults(func=lambda args: CodexNode().start())

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
