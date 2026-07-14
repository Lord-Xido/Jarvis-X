"""Commit-time authorization with signed witnesses and approvals."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Dict, FrozenSet, Iterable, Optional, Tuple, Union

Key = Union[str, bytes]


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode((value + padding).encode("ascii"))


def _key_bytes(value: Optional[Key]) -> bytes:
    if value is None:
        return secrets.token_bytes(32)
    encoded = value.encode("utf-8") if isinstance(value, str) else bytes(value)
    if len(encoded) < 32:
        raise ValueError("authority signing key must contain at least 32 bytes")
    return encoded


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


@dataclass(frozen=True)
class ApprovalGrant:
    approval_id: str
    tenant_id: str
    approver: str
    action: str
    resource: str
    state_version: int
    issued_at_ns: int
    expires_at_ns: int
    approval_epoch: int
    bindings_hash: str


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
    approval_tokens: Tuple[str, ...] = ()


@dataclass(frozen=True)
class CommitDecision:
    allowed: bool
    reason: str
    proof_hash: str
    witness_id: Optional[str] = None
    approval_ids: Tuple[str, ...] = ()


class AuthorityTokenService:
    """Issue and verify compact HMAC-SHA256 authority tokens."""

    VERSION = 1

    def __init__(
        self,
        signing_key: Optional[Key] = None,
        *,
        issuer: str = "dr-moagi",
    ) -> None:
        self._key = _key_bytes(signing_key)
        self.issuer = issuer

    @property
    def key_fingerprint(self) -> str:
        return hashlib.sha256(self._key).hexdigest()[:16]

    def _encode(self, kind: str, claims: Dict[str, object]) -> str:
        envelope = {
            "version": self.VERSION,
            "issuer": self.issuer,
            "kind": kind,
            "claims": claims,
        }
        body = json.dumps(
            envelope,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        encoded = _b64encode(body)
        signature = hmac.new(
            self._key,
            encoded.encode("ascii"),
            hashlib.sha256,
        ).digest()
        return encoded + "." + _b64encode(signature)

    def _decode(self, token: str, expected_kind: str) -> Dict[str, object]:
        try:
            encoded, supplied_signature = token.split(".", 1)
            signature = _b64decode(supplied_signature)
            expected_signature = hmac.new(
                self._key,
                encoded.encode("ascii"),
                hashlib.sha256,
            ).digest()
            if not hmac.compare_digest(signature, expected_signature):
                raise ValueError("invalid token signature")
            envelope = json.loads(_b64decode(encoded).decode("utf-8"))
        except (
            ValueError,
            TypeError,
            json.JSONDecodeError,
            UnicodeDecodeError,
        ) as exc:
            raise ValueError("invalid authority token") from exc
        if envelope.get("version") != self.VERSION:
            raise ValueError("unsupported authority token version")
        if envelope.get("issuer") != self.issuer:
            raise ValueError("authority token issuer mismatch")
        if envelope.get("kind") != expected_kind:
            raise ValueError("authority token kind mismatch")
        claims = envelope.get("claims")
        if not isinstance(claims, dict):
            raise ValueError("invalid authority token claims")
        return claims

    def issue_witness(
        self,
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
    ) -> str:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        issued = time.time_ns() if now_ns is None else now_ns
        witness = AuthorityWitness(
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
        claims = asdict(witness)
        claims["roles"] = sorted(witness.roles)
        return self._encode("witness", claims)

    def issue_approval(
        self,
        *,
        tenant_id: str,
        approver: str,
        action: str,
        resource: str,
        state_version: int,
        approval_epoch: int,
        bindings: Dict[str, object],
        ttl_seconds: int = 300,
        now_ns: Optional[int] = None,
    ) -> str:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        issued = time.time_ns() if now_ns is None else now_ns
        approval = ApprovalGrant(
            approval_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            approver=approver,
            action=action,
            resource=resource,
            state_version=state_version,
            issued_at_ns=issued,
            expires_at_ns=issued + ttl_seconds * 1_000_000_000,
            approval_epoch=approval_epoch,
            bindings_hash=_digest(bindings),
        )
        return self._encode("approval", asdict(approval))

    def verify_witness(self, token: str) -> AuthorityWitness:
        claims = self._decode(token, "witness")
        try:
            roles = claims.get("roles", [])
            if not isinstance(roles, list):
                raise ValueError("invalid witness roles")
            return AuthorityWitness(
                witness_id=str(claims["witness_id"]),
                tenant_id=str(claims["tenant_id"]),
                subject=str(claims["subject"]),
                action=str(claims["action"]),
                resource=str(claims["resource"]),
                state_version=int(claims["state_version"]),
                issued_at_ns=int(claims["issued_at_ns"]),
                expires_at_ns=int(claims["expires_at_ns"]),
                approval_epoch=int(claims["approval_epoch"]),
                bindings_hash=str(claims["bindings_hash"]),
                roles=frozenset(str(role) for role in roles),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid witness claims") from exc

    def verify_approval(self, token: str) -> ApprovalGrant:
        claims = self._decode(token, "approval")
        try:
            return ApprovalGrant(
                approval_id=str(claims["approval_id"]),
                tenant_id=str(claims["tenant_id"]),
                approver=str(claims["approver"]),
                action=str(claims["action"]),
                resource=str(claims["resource"]),
                state_version=int(claims["state_version"]),
                issued_at_ns=int(claims["issued_at_ns"]),
                expires_at_ns=int(claims["expires_at_ns"]),
                approval_epoch=int(claims["approval_epoch"]),
                bindings_hash=str(claims["bindings_hash"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid approval claims") from exc


class CommitPolicyEngine:
    """Fail-closed authorization checked at the durability boundary."""

    def __init__(
        self,
        *,
        max_risk: float = 0.35,
        max_cost_minor: int = 10_000_000,
        signing_key: Optional[Key] = None,
        issuer: str = "dr-moagi",
        allow_self_approval: bool = False,
    ) -> None:
        if not 0 <= max_risk <= 1:
            raise ValueError("max_risk must be inside [0, 1]")
        if max_cost_minor < 0:
            raise ValueError("max_cost_minor must be non-negative")
        self.max_risk = max_risk
        self.max_cost_minor = max_cost_minor
        self.allow_self_approval = allow_self_approval
        self.tokens = AuthorityTokenService(signing_key, issuer=issuer)

    def issue_witness(
        self,
        request: CommitRequest,
        *,
        roles: Iterable[str],
        ttl_seconds: int = 300,
        now_ns: Optional[int] = None,
    ) -> str:
        return self.tokens.issue_witness(
            tenant_id=request.tenant_id,
            subject=request.subject,
            action=request.action,
            resource=request.resource,
            state_version=request.state_version,
            approval_epoch=request.approval_epoch,
            bindings=request.bindings,
            roles=roles,
            ttl_seconds=ttl_seconds,
            now_ns=now_ns,
        )

    def issue_approval(
        self,
        request: CommitRequest,
        *,
        approver: str,
        ttl_seconds: int = 300,
        now_ns: Optional[int] = None,
    ) -> str:
        return self.tokens.issue_approval(
            tenant_id=request.tenant_id,
            approver=approver,
            action=request.action,
            resource=request.resource,
            state_version=request.state_version,
            approval_epoch=request.approval_epoch,
            bindings=request.bindings,
            ttl_seconds=ttl_seconds,
            now_ns=now_ns,
        )

    def decide(
        self,
        request: CommitRequest,
        witness_token: str,
        *,
        now_ns: Optional[int] = None,
    ) -> CommitDecision:
        now = time.time_ns() if now_ns is None else now_ns
        failed = []
        try:
            witness = self.tokens.verify_witness(witness_token)
        except ValueError:
            proof = {
                "request": self._proof_request(request),
                "witness": "invalid",
            }
            return CommitDecision(
                False,
                "failed:witness_signature",
                _digest(proof),
            )

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
            "risk": 0 <= request.risk <= self.max_risk,
            "cost": 0 <= request.estimated_cost_minor <= self.max_cost_minor,
        }
        failed.extend(name for name, passed in checks.items() if not passed)

        valid_approvals = []
        approval_rejections = []
        for token in request.approval_tokens:
            try:
                approval = self.tokens.verify_approval(token)
            except ValueError:
                approval_rejections.append("signature")
                continue
            approval_checks = (
                approval.tenant_id == request.tenant_id,
                approval.action == request.action,
                approval.resource == request.resource,
                approval.state_version == request.state_version,
                approval.approval_epoch == request.approval_epoch,
                approval.bindings_hash == _digest(request.bindings),
                approval.issued_at_ns <= now <= approval.expires_at_ns,
                self.allow_self_approval or approval.approver != request.subject,
            )
            if all(approval_checks):
                valid_approvals.append(approval)
            else:
                approval_rejections.append("scope")
        distinct = {approval.approver: approval for approval in valid_approvals}
        approval_ok = len(distinct) >= request.required_approvals
        checks["approvals"] = approval_ok
        if not approval_ok:
            failed.append("approvals")

        proof = {
            "request": self._proof_request(request),
            "witness_id": witness.witness_id,
            "approval_ids": sorted(
                item.approval_id for item in distinct.values()
            ),
            "approval_rejections": approval_rejections,
            "checks": checks,
        }
        return CommitDecision(
            allowed=not failed,
            reason="authorized" if not failed else "failed:" + ",".join(failed),
            proof_hash=_digest(proof),
            witness_id=witness.witness_id,
            approval_ids=tuple(
                sorted(item.approval_id for item in distinct.values())
            ),
        )

    @staticmethod
    def _proof_request(request: CommitRequest) -> Dict[str, object]:
        return {
            "tenant_id": request.tenant_id,
            "subject": request.subject,
            "action": request.action,
            "resource": request.resource,
            "state_version": request.state_version,
            "approval_epoch": request.approval_epoch,
            "bindings_hash": _digest(request.bindings),
            "estimated_cost_minor": request.estimated_cost_minor,
            "risk": request.risk,
            "required_roles": sorted(request.required_roles),
            "required_approvals": request.required_approvals,
        }
