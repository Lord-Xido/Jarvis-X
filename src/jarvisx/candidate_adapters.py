"""Adapters from existing Jarvis-X optimizers into the shared candidate contract."""

from __future__ import annotations

from .candidate_contract import (
    AdmissionPolicy,
    CandidateProposal,
    CandidateReceipt,
    ConstraintResult,
    ResourceEnvelope,
    ResourceUsage,
    admit_candidate,
    canonical_state_hash,
    normalize_metrics,
)
from .dr_moagi_virtual_3d_ae import Config, TuningResult


def virtual_3d_tuning_receipt(config: Config, result: TuningResult) -> CandidateReceipt:
    """Translate a virtual-3D tuning result into the global admission receipt.

    The adapter deliberately re-evaluates the optimizer's promotion decision from
    objective values and resource bounds. This lets system evidence detect drift
    between a subsystem-local commit rule and the repository-wide contract.
    """

    pairs = {(config.alpha, config.beta)}
    pairs.update(
        (alpha, beta)
        for alpha in config.alpha_candidates
        for beta in config.beta_candidates
    )
    candidate_count = len(pairs)

    parent = {
        "subsystem": "dr_moagi_virtual_3d_ae",
        "alpha": config.alpha,
        "beta": config.beta,
        "seed": config.seed,
        "tile": config.tile,
        "bits": config.bits,
        "latent": config.latent,
    }
    candidate = {
        **parent,
        "alpha": result.alpha,
        "beta": result.beta,
        "score": result.score,
    }

    constraints = (
        ConstraintResult(
            "candidate_budget",
            result.candidates_evaluated <= candidate_count,
            observed=result.candidates_evaluated,
            limit=candidate_count,
        ),
        ConstraintResult(
            "bounded_alpha",
            0.0 <= result.alpha <= 1.0,
            observed=result.alpha,
            limit="[0,1]",
        ),
        ConstraintResult(
            "bounded_beta",
            0.0 <= result.beta <= 1.0,
            observed=result.beta,
            limit="[0,1]",
        ),
    )
    proposal = CandidateProposal(
        subsystem="dr_moagi_virtual_3d_ae",
        candidate_id=f"alpha={result.alpha:.17g};beta={result.beta:.17g}",
        operator_version="virtual-3d-ae-v2",
        parent_state_hash=canonical_state_hash(parent),
        candidate_state_hash=canonical_state_hash(candidate),
        objective_before=float(result.baseline_score),
        objective_after=float(result.score),
        metrics=normalize_metrics(
            {
                "reconstruction_loss": result.reconstruction_loss,
                "spatial_loss": result.spatial_loss,
                "latent_balance_loss": result.latent_balance_loss,
                "mean_reality_gap": result.mean_reality_gap,
            }
        ),
        constraints=constraints,
        resource_envelope=ResourceEnvelope(
            max_work_units=candidate_count,
            max_resident_units=config.tile**3,
            max_output_bytes=0,
        ),
        resource_usage=ResourceUsage(
            work_units=result.candidates_evaluated,
            resident_units=config.tile**3,
            output_bytes=0,
        ),
    )
    return admit_candidate(
        proposal,
        policy=AdmissionPolicy(min_improvement=0.0, improvement_epsilon=1.0e-12),
    )
