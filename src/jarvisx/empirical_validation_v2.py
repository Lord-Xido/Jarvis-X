"""System-wide empirical evidence for Jarvis-X canonical and adaptive invariants.

Version 2 composes the original canonical validation harness with adaptive,
transactional and precision checks. It remains deliberately bounded: passing
this suite establishes the declared software invariants only, not intelligence,
production safety, physical performance or superiority over external baselines.
"""

from __future__ import annotations

import argparse
import math
import os
import platform
import sys
from collections.abc import Iterable
from pathlib import Path

from .candidate_adapters import virtual_3d_tuning_receipt
from .candidate_contract import (
    CandidateDecision,
    CandidateProposal,
    ConstraintResult,
    ResourceEnvelope,
    ResourceUsage,
    admit_candidate,
    canonical_state_hash,
)
from .dr_moagi_deep_distiller import (
    DeepDistiller,
    DeepDistillerConfig,
    DeepDistillerTheta,
)
from .dr_moagi_field_runtime import (
    DrMoagiFieldConfig,
    DrMoagiFieldRuntime,
    IdentityFieldCodec,
)
from .dr_moagi_virtual_3d_ae import Config as Virtual3DConfig
from .dr_moagi_virtual_3d_ae import DrMoagiVirtual3DAE
from .empirical_validation import (
    ValidationCheck,
    ValidationReport,
    run_validation as run_core_validation,
    write_report,
)
from .orthogonal_quantization import (
    dct2_orthonormal_basis,
    orthogonal_quantization_trace,
)


def _candidate_contract_check() -> ValidationCheck:
    parent_hash = canonical_state_hash({"value": 1})
    candidate_hash = canonical_state_hash({"value": 2})
    accepted_proposal = CandidateProposal(
        subsystem="empirical-fixture",
        candidate_id="candidate-1",
        operator_version="fixture-v1",
        parent_state_hash=parent_hash,
        candidate_state_hash=candidate_hash,
        objective_before=1.0,
        objective_after=0.5,
        constraints=(ConstraintResult("finite", True),),
        resource_envelope=ResourceEnvelope(10, 10, 10),
        resource_usage=ResourceUsage(1, 1, 1),
    )
    rejected_proposal = CandidateProposal(
        subsystem=accepted_proposal.subsystem,
        candidate_id=accepted_proposal.candidate_id,
        operator_version=accepted_proposal.operator_version,
        parent_state_hash=accepted_proposal.parent_state_hash,
        candidate_state_hash=accepted_proposal.candidate_state_hash,
        objective_before=1.0,
        objective_after=0.0,
        constraints=(ConstraintResult("hard_guard", False),),
        resource_envelope=accepted_proposal.resource_envelope,
        resource_usage=accepted_proposal.resource_usage,
    )
    accepted = admit_candidate(accepted_proposal)
    rejected = admit_candidate(rejected_proposal)
    repeated = admit_candidate(accepted_proposal)
    passed = (
        accepted.decision is CandidateDecision.COMMIT
        and rejected.decision is CandidateDecision.ROLLBACK
        and accepted.verify()
        and rejected.verify()
        and accepted.receipt_hash == repeated.receipt_hash
        and "constraint:hard_guard" in rejected.rejection_reasons
    )
    return ValidationCheck(
        name="shared_candidate_admission_contract",
        claim=(
            "Jarvis-X candidate receipts are deterministic and hard constraints "
            "cannot be overridden by a better objective."
        ),
        protocol=(
            "Evaluate one admissible improving proposal twice and one larger-improvement "
            "proposal that violates a hard constraint; verify deterministic hashes and "
            "require the violating proposal to roll back."
        ),
        passed=passed,
        metrics={
            "accepted_decision": accepted.decision.value,
            "rejected_decision": rejected.decision.value,
            "deterministic_receipt": accepted.receipt_hash == repeated.receipt_hash,
            "accepted_receipt_hash": accepted.receipt_hash,
            "rejected_receipt_hash": rejected.receipt_hash,
        },
        boundary=(
            "This validates the common software admission primitive. It does not prove "
            "that every subsystem has already been migrated to the primitive."
        ),
    )


def _field_runtime_check() -> ValidationCheck:
    field = {(3, 3, 3): 0.75, (4, 3, 3): -0.25}
    config = DrMoagiFieldConfig(
        side=8,
        alpha=1.0,
        lambda_residual=0.1,
        eta=0.1,
        dt=0.02,
        max_active_cells=64,
    )
    first = DrMoagiFieldRuntime(IdentityFieldCodec(), config)
    second = DrMoagiFieldRuntime(IdentityFieldCodec(), config)
    first.load(field)
    second.load(dict(reversed(tuple(field.items()))))
    first_metrics = first.step()
    second_metrics = second.step()

    rejected = DrMoagiFieldRuntime(IdentityFieldCodec(), config)
    rejected.load(field)
    before = rejected.snapshot()
    rejection = rejected.step(validator=lambda candidate, metrics: False)

    deterministic = first.snapshot() == second.snapshot() and first_metrics == second_metrics
    rollback = not rejection.committed and rejected.snapshot() == before
    bounded = first_metrics.support_cells <= config.max_active_cells
    passed = deterministic and rollback and bounded and first_metrics.committed
    return ValidationCheck(
        name="field_runtime_candidate_transaction",
        claim=(
            "The sparse 3D field transition is deterministic, support-bounded and "
            "retains the authoritative state when its candidate validator rejects."
        ),
        protocol=(
            "Run one field step from identical states supplied in opposite insertion order, "
            "compare metrics/state, then force validator rejection and require exact rollback."
        ),
        passed=passed,
        metrics={
            "deterministic": deterministic,
            "rollback_atomic": rollback,
            "support_cells": first_metrics.support_cells,
            "max_active_cells": config.max_active_cells,
            "reconstruction_mse": first_metrics.reconstruction_mse,
            "anchor_mse": first_metrics.anchor_mse,
        },
        boundary=(
            "The identity codec fixture verifies sparse transition semantics only; it does not "
            "establish stability for arbitrary learned codecs or production-scale throughput."
        ),
    )


def _deep_distiller_check() -> ValidationCheck:
    field = {
        (0, 0, 0): 1.0,
        (1, 0, 0): 0.5,
        (0, 1, 0): -0.25,
    }
    config = DeepDistillerConfig(
        logical_side=8,
        max_active_cells=16,
        max_latent_cells=16,
        max_iterations=8,
        learning_rate=0.05,
        omega_gain=0.25,
        rho=0.5,
    )
    first = DeepDistiller(config, theta=DeepDistillerTheta(0.8, 0.8))
    second = DeepDistiller(config, theta=DeepDistillerTheta(0.8, 0.8))
    first.load(field)
    second.load(field)
    first_report = first.step()
    second_report = second.step()
    deterministic = (
        first_report.committed
        and second_report.committed
        and first.snapshot() == second.snapshot()
        and first.omega_snapshot() == second.omega_snapshot()
        and first.theta == second.theta
        and math.isclose(first_report.loss, second_report.loss, rel_tol=0.0, abs_tol=0.0)
    )

    rejecting = DeepDistiller(
        config,
        theta=DeepDistillerTheta(0.8, 0.8),
        gate=lambda candidate: getattr(candidate, "latent_cells") == 0,
    )
    original_state = rejecting.load(field)
    original_omega = rejecting.omega_snapshot()
    original_theta = rejecting.theta
    rejected_report = rejecting.step()
    atomic = (
        not rejected_report.committed
        and rejecting.snapshot() == original_state
        and rejecting.omega_snapshot() == original_omega
        and rejecting.theta == original_theta
    )
    return ValidationCheck(
        name="deep_distiller_atomic_adaptation",
        claim=(
            "DM-DD deterministically couples state, residual memory and learnable parameters, "
            "and rolls all three back together when Pi_Lambda rejects a proposal."
        ),
        protocol=(
            "Execute identical residual-learning steps in two fresh engines, compare state/memory/"
            "parameters, then reject a proposal after initial admission and require tuple-level rollback."
        ),
        passed=deterministic and atomic,
        metrics={
            "deterministic": deterministic,
            "atomic_rollback": atomic,
            "residual_rms": first_report.residual_rms,
            "encoder_gain_after": first.theta.encoder_gain,
            "decoder_gain_after": first.theta.decoder_gain,
            "rejected_iteration": rejected_report.iteration,
        },
        boundary=(
            "This checks the bounded scalar-gain reference learner, not high-capacity neural model "
            "quality or convergence for arbitrary adaptive codecs."
        ),
    )


def _virtual_3d_optimizer_check() -> ValidationCheck:
    config = Virtual3DConfig(
        tile=3,
        bits=24,
        latent=6,
        passes=4,
        alpha=0.65,
        beta=0.65,
        alpha_candidates=(0.55, 0.65, 0.80),
        beta_candidates=(0.35, 0.50, 0.65),
        epsilon=0.0,
    )
    first_engine = DrMoagiVirtual3DAE(config)
    first = first_engine.optimize()
    second = DrMoagiVirtual3DAE(config).optimize()
    receipt = virtual_3d_tuning_receipt(config, first)
    repeated_receipt = virtual_3d_tuning_receipt(config, second)
    history = first_engine.run()

    expected = CandidateDecision.COMMIT if first.improved else CandidateDecision.ROLLBACK
    passed = (
        first == second
        and first.score <= first.baseline_score
        and receipt.decision is expected
        and receipt.verify()
        and receipt.receipt_hash == repeated_receipt.receipt_hash
        and history[-1].reality_gap == 0.0
    )
    return ValidationCheck(
        name="virtual_3d_optimizer_admission",
        claim=(
            "The virtual 3D AE search is deterministic, non-regressive, agrees with the shared "
            "candidate-admission contract and preserves fixed-point closure."
        ),
        protocol=(
            "Optimize the same bounded alpha/beta lattice twice, adapt the result into the global "
            "candidate receipt, compare hashes/decisions and run the promoted engine to zero reality gap."
        ),
        passed=passed,
        metrics={
            "baseline_score": first.baseline_score,
            "selected_score": first.score,
            "selected_alpha": first.alpha,
            "selected_beta": first.beta,
            "candidates_evaluated": first.candidates_evaluated,
            "receipt_decision": receipt.decision.value,
            "receipt_hash": receipt.receipt_hash,
            "final_reality_gap": history[-1].reality_gap,
        },
        boundary=(
            "This is bounded scalar parameter search for a deterministic bitstream codec laboratory; "
            "it does not establish gradient-trained representation learning or unrestricted self-modification."
        ),
    )


def _orthogonal_precision_check() -> ValidationCheck:
    values = (0.25, -0.75, 1.5, 0.125, -0.5, 0.9, 0.0, 0.33)
    basis = dct2_orthonormal_basis(len(values))
    trace = orthogonal_quantization_trace(values, basis, 0.125)
    passed = (
        trace.committed
        and trace.gate_ratio <= 1.0 + 1.0e-12
        and trace.orthogonality_error <= 1.0e-10
        and trace.residual_norm <= trace.deterministic_bound + 1.0e-12
    )
    return ValidationCheck(
        name="orthogonal_quantization_precision_gate",
        claim=(
            "The declared orthonormal transform obeys the deterministic nearest-neighbour "
            "quantization error envelope before transpose-as-inverse reconstruction is admitted."
        ),
        protocol=(
            "Build an 8-point orthonormal DCT-II basis, quantize with a fixed step and verify "
            "orthogonality plus the spatial L2 residual bound."
        ),
        passed=passed,
        metrics={
            "dimension": len(values),
            "orthogonality_error": trace.orthogonality_error,
            "residual_norm": trace.residual_norm,
            "deterministic_bound": trace.deterministic_bound,
            "gate_ratio": trace.gate_ratio,
        },
        boundary=(
            "This is a deterministic transform/quantization correctness check, not a production "
            "video-codec benchmark or perceptual-quality claim."
        ),
    )


def run_validation(
    *,
    repetitions: int = 64,
    octree_max_depth: int = 6,
) -> ValidationReport:
    """Run core v1 evidence plus adaptive/system-wide v2 checks."""

    core = run_core_validation(
        repetitions=repetitions,
        octree_max_depth=octree_max_depth,
    )
    checks = (
        *core.checks,
        _candidate_contract_check(),
        _field_runtime_check(),
        _deep_distiller_check(),
        _virtual_3d_optimizer_check(),
        _orthogonal_precision_check(),
    )
    return ValidationReport(
        schema_version="jarvisx.empirical-validation.v2",
        project="Jarvis-X",
        commit=os.environ.get("GITHUB_SHA", "unknown"),
        python_version=platform.python_version(),
        platform=platform.platform(),
        checks=checks,
    )


def _render_summary(report: ValidationReport) -> str:
    lines = [
        f"Jarvis-X empirical validation v2: {'PASS' if report.passed else 'FAIL'}",
        f"Commit: {report.commit}",
        f"Checks: {sum(check.passed for check in report.checks)}/{len(report.checks)}",
    ]
    lines.extend(
        f"- {'PASS' if check.passed else 'FAIL'}: {check.name}" for check in report.checks
    )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/empirical-validation.json"),
        help="JSON report destination",
    )
    parser.add_argument("--repetitions", type=int, default=64)
    parser.add_argument("--octree-max-depth", type=int, default=6)
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
