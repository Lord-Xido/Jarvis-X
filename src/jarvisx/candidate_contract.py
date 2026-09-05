"""Shared deterministic candidate/admission contract for Jarvis-X research runtimes.

The contract is intentionally backend-neutral. A subsystem may use grid search,
gradient descent, evolutionary search, symbolic planning or another bounded
proposal mechanism, but promotion always follows the same authority boundary:

    parent state -> candidate -> hard constraints -> objective gate
                 -> COMMIT or ROLLBACK -> deterministic receipt

Hard constraints are never converted into soft objective penalties.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Iterable, Mapping


class CandidateDecision(str, Enum):
    COMMIT = "commit"
    ROLLBACK = "rollback"


@dataclass(frozen=True)
class ResourceEnvelope:
    """Declared upper bounds for one candidate evaluation."""

    max_work_units: int
    max_resident_units: int
    max_output_bytes: int

    def __post_init__(self) -> None:
        for name in ("max_work_units", "max_resident_units", "max_output_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True)
class ResourceUsage:
    """Observed deterministic resource counters for a candidate."""

    work_units: int = 0
    resident_units: int = 0
    output_bytes: int = 0

    def __post_init__(self) -> None:
        for name in ("work_units", "resident_units", "output_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True)
class ConstraintResult:
    """One hard admission predicate evaluated before objective comparison."""

    name: str
    passed: bool
    observed: float | int | str | bool | None = None
    limit: float | int | str | bool | None = None
    detail: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("constraint name cannot be empty")


@dataclass(frozen=True)
class CandidateProposal:
    """Backend-neutral identity and evidence for one proposed state."""

    subsystem: str
    candidate_id: str
    operator_version: str
    parent_state_hash: str
    candidate_state_hash: str
    objective_before: float
    objective_after: float
    metrics: tuple[tuple[str, float], ...] = ()
    constraints: tuple[ConstraintResult, ...] = ()
    resource_envelope: ResourceEnvelope | None = None
    resource_usage: ResourceUsage = ResourceUsage()

    def __post_init__(self) -> None:
        for name in ("subsystem", "candidate_id", "operator_version"):
            value = getattr(self, name)
            if not str(value).strip():
                raise ValueError(f"{name} cannot be empty")
        for name in ("parent_state_hash", "candidate_state_hash"):
            value = getattr(self, name)
            if not _looks_like_digest(str(value)):
                raise ValueError(f"{name} must be a lowercase hexadecimal digest")
        for name in ("objective_before", "objective_after"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        seen: set[str] = set()
        for key, value in self.metrics:
            if not key or key in seen:
                raise ValueError("metric names must be non-empty and unique")
            seen.add(key)
            if not math.isfinite(float(value)):
                raise ValueError(f"metric {key!r} must be finite")


@dataclass(frozen=True)
class AdmissionPolicy:
    """Global lower-is-better objective gate.

    ``min_improvement`` is an absolute objective decrease required for promotion.
    Hard constraints and resource limits are checked before this objective gate.
    """

    min_improvement: float = 0.0
    improvement_epsilon: float = 1.0e-12

    def __post_init__(self) -> None:
        if not math.isfinite(self.min_improvement) or self.min_improvement < 0.0:
            raise ValueError("min_improvement must be finite and non-negative")
        if not math.isfinite(self.improvement_epsilon) or self.improvement_epsilon < 0.0:
            raise ValueError("improvement_epsilon must be finite and non-negative")


@dataclass(frozen=True)
class CandidateReceipt:
    """Deterministic admission result suitable for journaling or evidence artifacts."""

    schema_version: str
    proposal: CandidateProposal
    decision: CandidateDecision
    improvement: float
    rejection_reasons: tuple[str, ...]
    receipt_hash: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "proposal": _proposal_payload(self.proposal),
            "decision": self.decision.value,
            "improvement": self.improvement,
            "rejection_reasons": list(self.rejection_reasons),
            "receipt_hash": self.receipt_hash,
        }

    def verify(self) -> bool:
        payload = self.to_dict()
        receipt_hash = str(payload.pop("receipt_hash"))
        return receipt_hash == _sha256(payload)


SCHEMA_VERSION = "jarvisx.candidate-receipt.v1"


def canonical_state_hash(value: object) -> str:
    """Hash a JSON-native authoritative state using canonical JSON."""

    return _sha256(value)


def normalize_metrics(
    metrics: Mapping[str, float] | Iterable[tuple[str, float]],
) -> tuple[tuple[str, float], ...]:
    """Return metrics in deterministic lexicographic order."""

    items = metrics.items() if isinstance(metrics, Mapping) else metrics
    materialized = [(str(key), float(value)) for key, value in items]
    return tuple(sorted(materialized))


def admit_candidate(
    proposal: CandidateProposal,
    *,
    policy: AdmissionPolicy | None = None,
) -> CandidateReceipt:
    """Apply hard constraints, resource bounds and a non-regressive objective gate."""

    resolved = policy or AdmissionPolicy()
    reasons: list[str] = []

    for constraint in proposal.constraints:
        if not constraint.passed:
            reasons.append(f"constraint:{constraint.name}")

    envelope = proposal.resource_envelope
    usage = proposal.resource_usage
    if envelope is not None:
        if usage.work_units > envelope.max_work_units:
            reasons.append("resource:max_work_units")
        if usage.resident_units > envelope.max_resident_units:
            reasons.append("resource:max_resident_units")
        if usage.output_bytes > envelope.max_output_bytes:
            reasons.append("resource:max_output_bytes")

    improvement = float(proposal.objective_before) - float(proposal.objective_after)
    threshold = resolved.min_improvement + resolved.improvement_epsilon
    if improvement <= threshold:
        reasons.append("objective:no_material_improvement")

    decision = CandidateDecision.ROLLBACK if reasons else CandidateDecision.COMMIT
    body = {
        "schema_version": SCHEMA_VERSION,
        "proposal": _proposal_payload(proposal),
        "decision": decision.value,
        "improvement": improvement,
        "rejection_reasons": reasons,
    }
    return CandidateReceipt(
        schema_version=SCHEMA_VERSION,
        proposal=proposal,
        decision=decision,
        improvement=improvement,
        rejection_reasons=tuple(reasons),
        receipt_hash=_sha256(body),
    )


def _proposal_payload(proposal: CandidateProposal) -> dict[str, object]:
    return {
        "subsystem": proposal.subsystem,
        "candidate_id": proposal.candidate_id,
        "operator_version": proposal.operator_version,
        "parent_state_hash": proposal.parent_state_hash,
        "candidate_state_hash": proposal.candidate_state_hash,
        "objective_before": proposal.objective_before,
        "objective_after": proposal.objective_after,
        "metrics": {key: value for key, value in proposal.metrics},
        "constraints": [asdict(item) for item in proposal.constraints],
        "resource_envelope": (
            asdict(proposal.resource_envelope) if proposal.resource_envelope is not None else None
        ),
        "resource_usage": asdict(proposal.resource_usage),
    }


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _looks_like_digest(value: str) -> bool:
    return (
        len(value) >= 16
        and value == value.lower()
        and all(char in "0123456789abcdef" for char in value)
    )
