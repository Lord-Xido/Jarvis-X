"""Commit-time authorization and proof-carrying policy enforcement."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, Optional, Tuple


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class AuthorityWitness:
    witness_id: str
    tenant_id: str
    subject: str
    action: str
    resource: str
    state_version: int
    issued_at_ns: int
    expires_at_ns: int
    approval_epoch: int
    bindings_hash: str
    roles: FrozenSet[str] = field(default_factory=frozenset)

    @classmethod
    def issue(
        cls,
        *,
        tenant_id: str,
        subject: str,
        action: str,
        resource: str,
        state_version: int,
        approval_epoch: int,
        bindings: Dict[str, object],
        roles: Iterable[str],
        ttl_seconds: int = 300,
        now_ns: Optional[int] = None,
    ) -> "AuthorityWitness":
        issued = now_ns or time.time_ns()
        return cls(
            witness_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            subject=subject,
            action=action,
            resource=resource,
            state_version=state_version,
            issued_at_ns=issued,
            expires_at_ns=issued + ttl_seconds * 1_000_000_000,
            approval_epoch=approval_epoch,
            bindings_hash=_digest(bindings),
            roles=frozenset(roles),
        )


@dataclass(frozen=True)
class CommitRequest:
    tenant_id: str
    subject: str
    action: str
    resource: str
    state_version: int
    approval_epoch: int
    bindings: Dict[str, object]
    estimated_cost_minor: int
    risk: float
    required_roles: FrozenSet[str] = field(default_factory=frozenset)
    required_approvals: int = 0
    approvals: Tuple[str, ...] = ()


@dataclass(frozen=True)
class CommitDecision:
    allowed: bool
    reason: str
    proof_hash: str


class CommitPolicyEngine:
    """Fail-closed authorization checked at the durability boundary."""

    def __init__(
        self, *, max_risk: float = 0.35, max_cost_minor: int = 10_000_000
    ) -> None:
        if not 0 <= max_risk <= 1:
            raise ValueError("max_risk must be inside [0, 1]")
        self.max_risk = max_risk
        self.max_cost_minor = max_cost_minor

    def decide(
        self,
        request: CommitRequest,
        witness: AuthorityWitness,
        *,
        now_ns: Optional[int] = None,
    ) -> CommitDecision:
        now = now_ns or time.time_ns()
        checks = {
            "tenant": witness.tenant_id == request.tenant_id,
            "subject": witness.subject == request.subject,
            "action": witness.action == request.action,
            "resource": witness.resource == request.resource,
            "fresh": witness.issued_at_ns <= now <= witness.expires_at_ns,
            "state_version": witness.state_version == request.state_version,
            "approval_epoch": witness.approval_epoch == request.approval_epoch,
            "binding": witness.bindings_hash == _digest(request.bindings),
            "roles": request.required_roles.issubset(witness.roles),
            "approvals": len(set(request.approvals)) >= request.required_approvals,
            "risk": 0 <= request.risk <= self.max_risk,
            "cost": 0 <= request.estimated_cost_minor <= self.max_cost_minor,
        }
        failed = tuple(name for name, passed in checks.items() if not passed)
        proof = {
            "request": {
                "tenant_id": request.tenant_id,
                "subject": request.subject,
                "action": request.action,
                "resource": request.resource,
                "state_version": request.state_version,
                "approval_epoch": request.approval_epoch,
                "bindings_hash": _digest(request.bindings),
                "estimated_cost_minor": request.estimated_cost_minor,
                "risk": request.risk,
            },
            "witness_id": witness.witness_id,
            "checks": checks,
        }
        return CommitDecision(
            allowed=not failed,
            reason="authorized" if not failed else "failed:" + ",".join(failed),
            proof_hash=_digest(proof),
        )
