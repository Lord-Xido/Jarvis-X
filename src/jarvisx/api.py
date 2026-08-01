from flask import Flask, request, jsonify
from .core import CodexVM
from .parser import Parser
from .assembler import Assembler
from .code_editor_automata import (
    DeterministicCodeEditingAutomata,
    RefactoringParameter,
    ConfidenceThresholdPolicy,
    CycleImprovementPolicy,
    MemorySafetyPolicy,
    TransformWhitelistPolicy,
    TransformType,
)

app = Flask(__name__)
vm = CodexVM()


@app.route("/run", methods=["POST"])
def run_code():
    """Execute bytecode program."""
    source = request.json.get("source", "")
    ast = Parser().parse(source)
    bytecode = Assembler().assemble(ast)
    vm.load(bytecode)
    vm.run()
    return jsonify(
        {
            "registers": vm.regs.snapshot(),
            "ledger_entries": len(vm.ledger.chain),
            "cycle_counter": vm.cycle_counter,
            "verified": vm.ledger.verify(),
        }
    )


@app.route("/refactor", methods=["POST"])
def refactor_code():
    """Refactor assembly code with bounded automata."""
    data = request.json

    program = data.get("program", "")
    seed = data.get("seed", 42)
    max_depth = data.get("max_depth", 3)
    max_cycles = data.get("max_cycles", 1000)
    max_mutations = data.get("max_mutations", 10)
    cycle_limit = data.get("cycle_limit", 10000)
    cost_model = data.get("cost_model", "cycles")

    # Create automata
    automata = DeterministicCodeEditingAutomata(cycle_limit=cycle_limit)

    # Install policies
    if data.get("confidence"):
        automata.add_policy(ConfidenceThresholdPolicy(data["confidence"]))

    if data.get("cycle_improvement"):
        automata.add_policy(CycleImprovementPolicy())

    if data.get("memory_safety"):
        automata.add_policy(MemorySafetyPolicy(data["memory_safety"]))

    if data.get("whitelist"):
        allowed = {
            TransformType[t.upper()] for t in data["whitelist"] if t.strip()
        }
        automata.add_policy(TransformWhitelistPolicy(allowed))

    # Execute refactoring
    params = RefactoringParameter(
        seed=seed,
        max_depth=max_depth,
        max_cycles=max_cycles,
        max_mutations=max_mutations,
        cost_model=cost_model,
    )

    result = automata.refactor(program, params)

    return jsonify(
        {
            "input_program": result.input_program,
            "output_program": result.output_program,
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
    )


@app.route("/verify-equivalence", methods=["POST"])
def verify_equivalence():
    """Verify functional equivalence of two programs."""
    data = request.json
    original = data.get("original", "")
    refactored = data.get("refactored", "")

    # Execute both
    vm1 = CodexVM()
    vm1.load(Assembler().assemble(Parser().parse(original)))
    state1 = vm1.run()

    vm2 = CodexVM()
    vm2.load(Assembler().assemble(Parser().parse(refactored)))
    state2 = vm2.run()

    # Compare
    equivalent = state1 == state2
    improvement = vm1.cycle_counter - vm2.cycle_counter if equivalent else None

    return jsonify(
        {
            "equivalent": equivalent,
            "original_cycles": vm1.cycle_counter,
            "refactored_cycles": vm2.cycle_counter,
            "improvement_cycles": improvement,
            "improvement_percent": (
                100 * improvement / vm1.cycle_counter if improvement and vm1.cycle_counter > 0 else None
            ),
        }
    )


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "version": "0.1.0"})


def start_api():
    app.run(host="0.0.0.0", port=8080)
