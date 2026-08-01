"""Reproducible empirical validation for canonical Jarvis-X invariants.

This module consolidates executable evidence for claims that are already implemented on
``main``.  It intentionally tests bounded software properties: deterministic replay,
journal integrity, sparse transactional behavior, closed-form geometry and numerical
invariants.  It does not infer intelligence, consciousness, physical performance or
production safety from those observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterable

from .assembler import Assembler
from .core import CodexVM
from .dr_moagi_billion_field import BillionFieldConfig, SparseBillionField
from .fractal_octree import build_fractal_octree
from .fractional_smoothing_3d import (
    FractionalHierarchyConfig,
    Grid3D,
    classical_gradient_energy,
    hierarchical_fractional_smooth,
    spectral_fractional_step,
)
from .ledger import OmegaLedger
from .parser import Parser


@dataclass(frozen=True)
class ValidationCheck:
    """One falsifiable validation protocol and its observed result."""

    name: str
    claim: str
    protocol: str
    passed: bool
    metrics: dict[str, object]
    boundary: str


@dataclass(frozen=True)
class ValidationReport:
    """Machine-readable collection of empirical checks."""

    schema_version: str
    project: str
    commit: str
    python_version: str
    platform: str
    checks: tuple[ValidationCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "project": self.project,
            "commit": self.commit,
            "environment": {
                "python_version": self.python_version,
                "platform": self.platform,
            },
            "summary": {
                "passed": self.passed,
                "checks_passed": sum(check.passed for check in self.checks),
                "checks_total": len(self.checks),
            },
            "checks": [asdict(check) for check in self.checks],
        }


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _assemble(source: str) -> list[int]:
    return Assembler().assemble(Parser().parse(source))


def _vm_replay_check(repetitions: int) -> ValidationCheck:
    source = "\n".join(
        (
            "SET Ψ 10",
            "SET Φ 20",
            "ADD A Ψ Φ",
            "SUB E Φ Ψ",
            "HALT",
        )
    )
    program = _assemble(source)
    baseline_state: dict[str, int] | None = None
    baseline_trace: tuple[tuple[int, tuple[tuple[str, int], ...]], ...] | None = None
    all_ledgers_valid = True
    all_cycles_equal = True
    started = perf_counter()

    for _ in range(repetitions):
        vm = CodexVM()
        vm.load(program)
        final_state = vm.run()
        trace = tuple(
            (int(opcode), tuple(sorted((str(key), int(value)) for key, value in state.items())))
            for opcode, state in vm.tracer.log
        )

        if baseline_state is None:
            baseline_state = final_state
            baseline_trace = trace
        else:
            all_cycles_equal = all_cycles_equal and final_state == baseline_state
            all_cycles_equal = all_cycles_equal and trace == baseline_trace

        all_ledgers_valid = all_ledgers_valid and vm.ledger.verify()
        all_cycles_equal = all_cycles_equal and vm.cycles == len(program)

    elapsed = perf_counter() - started
    assert baseline_state is not None
    assert baseline_trace is not None
    passed = (
        all_cycles_equal
        and all_ledgers_valid
        and baseline_state["A"] == 30
        and baseline_state["E"] == 10
    )
    executed_instructions = repetitions * len(program)

    return ValidationCheck(
        name="vm_deterministic_replay",
        claim="Identical canonical bytecode produces identical authoritative state and trace.",
        protocol=(
            f"Assemble one five-instruction program and execute it in {repetitions} fresh VMs; "
            "compare complete register snapshots and instruction traces and verify each ledger."
        ),
        passed=passed,
        metrics={
            "repetitions": repetitions,
            "program_words": len(program),
            "executed_instructions": executed_instructions,
            "elapsed_seconds_observed": elapsed,
            "instructions_per_second_observed": executed_instructions / elapsed if elapsed else 0.0,
            "program_digest_sha256": _canonical_digest(program),
            "final_state_digest_sha256": _canonical_digest(baseline_state),
            "trace_digest_sha256": _canonical_digest(baseline_trace),
            "ledger_integrity": all_ledgers_valid,
            "final_A": baseline_state["A"],
            "final_E": baseline_state["E"],
        },
        boundary=(
            "The timing is an observational CI measurement, not a portable performance claim. "
            "This check establishes deterministic behavior only for the exercised ISA path."
        ),
    )


def _ledger_integrity_check() -> ValidationCheck:
    timestamps = iter((100, 200, 300))
    clock: Callable[[], int] = lambda: next(timestamps)
    ledger = OmegaLedger(clock_ns=clock)
    ledger.log({"A": 1}, 1)
    ledger.log({"A": 2}, 3)
    ledger.log({"A": 3}, 4)

    valid_before_tamper = ledger.verify()
    original_tip = str(ledger.chain[-1]["hash"])
    ledger.chain[1]["state"]["A"] = 999
    tamper_detected = not ledger.verify()

    return ValidationCheck(
        name="omega_ledger_tamper_detection",
        claim="The Ω journal detects mutation of committed historical state.",
        protocol=(
            "Create a three-entry ledger with an injected deterministic clock, verify the chain, "
            "mutate the middle entry and require verification to fail."
        ),
        passed=valid_before_tamper and tamper_detected,
        metrics={
            "entries": 3,
            "valid_before_tamper": valid_before_tamper,
            "tamper_detected": tamper_detected,
            "original_tip_hash": original_tip,
        },
        boundary=(
            "A hash chain establishes integrity evidence for the serialized journal. It does not "
            "provide confidentiality, trusted timestamps or protection against deletion of all copies."
        ),
    )


def _sparse_transaction_check() -> ValidationCheck:
    observations = {
        (500, 500, 500): 1.0,
        (501, 500, 500): 0.5,
        (500, 501, 500): -0.25,
    }
    controls = {(500, 500, 500): -0.02}

    first = SparseBillionField()
    second = SparseBillionField()
    first_metrics = first.run(4, observations, controls)
    second_metrics = second.run(4, dict(reversed(tuple(observations.items()))), controls)

    insertion_order_invariant = (
        first_metrics == second_metrics
        and tuple(first.iter_active()) == tuple(second.iter_active())
        and first_metrics.journal_digest == second_metrics.journal_digest
        and first_metrics.state_digest == second_metrics.state_digest
    )

    checkpoint = first.checkpoint()
    restored = SparseBillionField.from_checkpoint(checkpoint)
    checkpoint_round_trip = (
        restored.metrics() == first.metrics()
        and tuple(restored.iter_active()) == tuple(first.iter_active())
    )

    rejected = SparseBillionField(BillionFieldConfig(residual_threshold=0.0))
    rejected_metrics = rejected.step({(10, 10, 10): 1.0})
    rejected_state = rejected.state((10, 10, 10))
    atomic_rollback = (
        not rejected_state.valid
        and rejected_state.committed == 0.0
        and rejected_state.omega == 0.0
        and rejected_metrics.rejected_cells == 1
    )

    passed = insertion_order_invariant and checkpoint_round_trip and atomic_rollback
    return ValidationCheck(
        name="sparse_field_transactionality",
        claim=(
            "The sparse billion-address reference is order-invariant, checkpoint-replayable and "
            "atomically rolls back invalid persistent updates."
        ),
        protocol=(
            "Run equivalent observations in opposite dictionary order, compare canonical state and "
            "journal digests, restore a checkpoint, then force a rejected candidate with a zero "
            "residual threshold."
        ),
        passed=passed,
        metrics={
            "virtual_cells": first_metrics.virtual_cells,
            "active_cells": first_metrics.active_cells,
            "active_ratio": first_metrics.active_ratio,
            "cycles": first_metrics.cycle,
            "mean_absolute_residual": first_metrics.mean_absolute_residual,
            "reconstruction_loss": first_metrics.reconstruction_loss,
            "coherence": first_metrics.coherence,
            "state_digest_sha256": first_metrics.state_digest,
            "journal_digest_sha256": first_metrics.journal_digest,
            "insertion_order_invariant": insertion_order_invariant,
            "checkpoint_round_trip": checkpoint_round_trip,
            "atomic_rollback": atomic_rollback,
        },
        boundary=(
            "The billion-cell figure is a virtual address-space cardinality. Only active sparse "
            "coordinates are materialized; this check does not instantiate one billion dense models."
        ),
    )


def _fractal_geometry_check(max_depth: int) -> ValidationCheck:
    rows: list[dict[str, object]] = []
    passed = True

    for depth in range(max_depth + 1):
        root = build_fractal_octree(size=1.0, max_depth=depth)
        measured = root.metrics()
        expected = root.expected_metrics()
        row_passed = measured == expected
        passed = passed and row_passed
        rows.append(
            {
                "depth": depth,
                "active_nodes": measured.active_nodes,
                "active_leaves": measured.active_leaves,
                "retained_volume": measured.retained_volume,
                "expected_active_nodes": (4 ** (depth + 1) - 1) // 3,
                "expected_active_leaves": 4**depth,
                "expected_retained_volume": 2.0 ** (-depth),
                "passed": row_passed,
            }
        )

    deepest = rows[-1]
    return ValidationCheck(
        name="fractal_octree_closed_form",
        claim="The four-survivor octree follows its exact recursive scaling laws.",
        protocol=(
            f"Materialize unit-cube octrees at every depth from 0 through {max_depth} and compare "
            "measured node count, leaf count and retained volume with the closed forms."
        ),
        passed=passed,
        metrics={
            "depths_checked": max_depth + 1,
            "max_depth": max_depth,
            "deepest_active_nodes": deepest["active_nodes"],
            "deepest_active_leaves": deepest["active_leaves"],
            "deepest_retained_volume": deepest["retained_volume"],
            "similarity_dimension": 2.0,
            "observations": rows,
        },
        boundary=(
            "This verifies a bounded geometric reference and its invariants. It does not demonstrate "
            "fractal long-memory quality or superiority over sequence-model memory architectures."
        ),
    )


def _fractional_numerics_check() -> ValidationCheck:
    field = Grid3D.impulse((4, 4, 4), (1, 1, 1), amplitude=8.0)
    config = FractionalHierarchyConfig(
        alphas=(1.0, 0.65),
        taus=(0.08, 0.20),
        coarse_blends=(0.25,),
    )
    result = hierarchical_fractional_smooth(field, config)

    split = spectral_fractional_step(field, alpha=0.7, tau=0.125)
    split = spectral_fractional_step(split, alpha=0.7, tau=0.275)
    combined = spectral_fractional_step(field, alpha=0.7, tau=0.4)
    semigroup_error = max(abs(left - right) for left, right in zip(split.values, combined.values))

    gradient_before = classical_gradient_energy(field)
    gradient_after = classical_gradient_energy(result.field)
    expected_opcodes = (
        "RESTRICT_2X2X2",
        "FRACTIONAL_HEAT_3D",
        "FRACTIONAL_HEAT_3D",
        "PROLONG_2X2X2",
        "FUSE_COARSE_FINE",
        "VERIFY_MASS_AND_UPDATE",
    )
    opcodes = tuple(instruction.opcode for instruction in result.instructions)

    mass_preserved = abs(result.mass_drift) < 1.0e-9
    variance_reduced = result.field.variance < field.variance
    gradient_reduced = gradient_after < gradient_before
    semigroup_consistent = semigroup_error < 1.0e-10
    trace_is_canonical = opcodes == expected_opcodes
    passed = (
        mass_preserved
        and variance_reduced
        and gradient_reduced
        and semigroup_consistent
        and trace_is_canonical
    )

    return ValidationCheck(
        name="fractional_smoothing_invariants",
        claim=(
            "The bounded fractional 3D reference preserves mass, dissipates variation and follows "
            "the unforced spectral semigroup within numerical tolerance."
        ),
        protocol=(
            "Smooth a 4×4×4 impulse through the two-level hierarchy; measure mass, variance and "
            "gradient energy, then compare one τ=0.4 step with split τ=0.125 and τ=0.275 steps."
        ),
        passed=passed,
        metrics={
            "shape": list(field.shape),
            "mass_before": result.mass_before,
            "mass_after": result.mass_after,
            "mass_drift": result.mass_drift,
            "variance_before": field.variance,
            "variance_after": result.field.variance,
            "gradient_energy_before": gradient_before,
            "gradient_energy_after": gradient_after,
            "semigroup_max_absolute_error": semigroup_error,
            "trace_opcodes": list(opcodes),
            "mass_preserved": mass_preserved,
            "variance_reduced": variance_reduced,
            "gradient_reduced": gradient_reduced,
            "semigroup_consistent": semigroup_consistent,
            "trace_is_canonical": trace_is_canonical,
        },
        boundary=(
            "The solver is a small-grid correctness reference using a direct DFT. These results do "
            "not establish production-scale complexity, calibrated physics or hardware acceleration."
        ),
    )


def run_validation(
    *,
    repetitions: int = 64,
    octree_max_depth: int = 6,
) -> ValidationReport:
    """Execute all canonical empirical checks and return a machine-readable report."""

    if repetitions < 2:
        raise ValueError("repetitions must be at least 2")
    if octree_max_depth < 0:
        raise ValueError("octree_max_depth must be non-negative")

    checks = (
        _vm_replay_check(repetitions),
        _ledger_integrity_check(),
        _sparse_transaction_check(),
        _fractal_geometry_check(octree_max_depth),
        _fractional_numerics_check(),
    )
    return ValidationReport(
        schema_version="jarvisx.empirical-validation.v1",
        project="Jarvis-X",
        commit=os.environ.get("GITHUB_SHA", "unknown"),
        python_version=platform.python_version(),
        platform=platform.platform(),
        checks=checks,
    )


def write_report(report: ValidationReport, output: Path) -> None:
    """Write one stable, human-readable JSON evidence artifact."""

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _render_summary(report: ValidationReport) -> str:
    lines = [
        f"Jarvis-X empirical validation: {'PASS' if report.passed else 'FAIL'}",
        f"Commit: {report.commit}",
    ]
    for check in report.checks:
        lines.append(f"- {'PASS' if check.passed else 'FAIL'}: {check.name}")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/empirical-validation.json"),
        help="JSON report destination",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=64,
        help="number of fresh VM replay trials",
    )
    parser.add_argument(
        "--octree-max-depth",
        type=int,
        default=6,
        help="largest materialized octree depth",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    report = run_validation(
        repetitions=args.repetitions,
        octree_max_depth=args.octree_max_depth,
    )
    write_report(report, args.output)
    print(_render_summary(report))
    print(f"Evidence artifact: {args.output}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
