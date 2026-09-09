"""Backend-neutral candidate admission receipts for Jarvis-X.

The contract separates integrity, numerical validity, epistemic evidence and
execution authority.  It intentionally hashes only the declared state scope;
a receipt must never be interpreted as proof of external truth merely because
its candidate converged or its digest verifies.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

VerificationStatus = Literal["PASS", "FAIL", "NOT_APPLICABLE"]
JsonScalar = str | int | float | bool | None


@dataclass(frozen=True)
class VerificationVector:
    integrity: VerificationStatus
    numerical: VerificationStatus
    epistemic: VerificationStatus
    authority: VerificationStatus

    def as_dict(self) -> dict[str, VerificationStatus]:
        return {
            "integrity": self.integrity,
            "numerical": self.numerical,
            "epistemic": self.epistemic,
            "authority": self.authority,
        }


@dataclass(frozen=True)
class CandidateReceipt:
    subsystem: str
    operator: str
    state_scope: str
    parent_state_hash: str
    candidate_state_hash: str
    after_state_hash: str
    objective_before: float
    candidate_objective: float
    objective_after: float
    metrics: tuple[tuple[str, JsonScalar], ...]
    hard_constraints: tuple[tuple[str, bool], ...]
    verification: VerificationVector
    committed: bool
    rejection_reason: str | None
    receipt_hash: str

    def as_dict(self) -> dict[str, object]:
        return {
            "subsystem": self.subsystem,
            "operator": self.operator,
            "state_scope": self.state_scope,
            "parent_state_hash": self.parent_state_hash,
            "candidate_state_hash": self.candidate_state_hash,
            "after_state_hash": self.after_state_hash,
            "objective_before": self.objective_before,
            "candidate_objective": self.candidate_objective,
            "objective_after": self.objective_after,
            "metrics": dict(self.metrics),
            "hard_constraints": dict(self.hard_constraints),
            "verification": self.verification.as_dict(),
            "committed": self.committed,
            "rejection_reason": self.rejection_reason,
            "receipt_hash": self.receipt_hash,
        }


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _finite_float(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _normalize_metrics(metrics: Mapping[str, JsonScalar]) -> tuple[tuple[str, JsonScalar], ...]:
    normalized: list[tuple[str, JsonScalar]] = []
    for key in sorted(metrics):
        value = metrics[key]
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"metric {key!r} must be finite")
        normalized.append((str(key), value))
    return tuple(normalized)


def hash_sparse_field(field: Mapping[tuple[int, int], Sequence[float]]) -> str:
    """Hash a finite sparse 2D-coordinate/vector state deterministically."""

    records: list[list[object]] = []
    for coordinate in sorted(field):
        if len(coordinate) != 2 or any(isinstance(axis, bool) or not isinstance(axis, int) for axis in coordinate):
            raise TypeError("sparse-field coordinates must be integer pairs")
        values: list[float] = []
        for raw in field[coordinate]:
            value = float(raw)
            if not math.isfinite(value):
                raise ValueError("sparse-field values must be finite")
            values.append(value)
        records.append([coordinate[0], coordinate[1], values])
    return hashlib.sha256(_canonical_json(records)).hexdigest()


def build_candidate_receipt(
    *,
    subsystem: str,
    operator: str,
    state_scope: str,
    parent_state_hash: str,
    candidate_state_hash: str,
    after_state_hash: str,
    objective_before: float,
    candidate_objective: float,
    objective_after: float,
    metrics: Mapping[str, JsonScalar],
    hard_constraints: Mapping[str, bool],
    verification: VerificationVector,
    committed: bool,
    rejection_reason: str | None = None,
) -> CandidateReceipt:
    """Build a deterministic receipt while enforcing commit/rollback identity.

    A commit must publish the candidate state.  A rollback must leave the
    declared authoritative state scope equal to its parent state.
    """

    if not subsystem or not operator or not state_scope:
        raise ValueError("subsystem, operator and state_scope must be non-empty")
    if committed:
        if after_state_hash != candidate_state_hash:
            raise ValueError("committed receipt must publish the candidate state")
        if rejection_reason is not None:
            raise ValueError("committed receipt cannot contain a rejection reason")
    else:
        if after_state_hash != parent_state_hash:
            raise ValueError("rollback receipt must preserve the parent state")
        if rejection_reason is None:
            rejection_reason = "candidate rejected"

    objective_before_f = _finite_float("objective_before", objective_before)
    candidate_objective_f = _finite_float("candidate_objective", candidate_objective)
    objective_after_f = _finite_float("objective_after", objective_after)
    normalized_metrics = _normalize_metrics(metrics)
    normalized_constraints = tuple((str(key), bool(hard_constraints[key])) for key in sorted(hard_constraints))

    body = {
        "subsystem": subsystem,
        "operator": operator,
        "state_scope": state_scope,
        "parent_state_hash": parent_state_hash,
        "candidate_state_hash": candidate_state_hash,
        "after_state_hash": after_state_hash,
        "objective_before": objective_before_f,
        "candidate_objective": candidate_objective_f,
        "objective_after": objective_after_f,
        "metrics": dict(normalized_metrics),
        "hard_constraints": dict(normalized_constraints),
        "verification": verification.as_dict(),
        "committed": committed,
        "rejection_reason": rejection_reason,
    }
    receipt_hash = hashlib.sha256(_canonical_json(body)).hexdigest()
    return CandidateReceipt(
        subsystem=subsystem,
        operator=operator,
        state_scope=state_scope,
        parent_state_hash=parent_state_hash,
        candidate_state_hash=candidate_state_hash,
        after_state_hash=after_state_hash,
        objective_before=objective_before_f,
        candidate_objective=candidate_objective_f,
        objective_after=objective_after_f,
        metrics=normalized_metrics,
        hard_constraints=normalized_constraints,
        verification=verification,
        committed=committed,
        rejection_reason=rejection_reason,
        receipt_hash=receipt_hash,
    )
