from __future__ import annotations

import hashlib
import hmac
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

CONTROL_PROTOCOL = "jarvisx.control-plane.v1"
GENESIS_RECEIPT_HASH = "0" * 64
Decision = Literal["commit", "rollback"]


def _normalize_json(value: object) -> object:
    """Normalize a value into deterministic JSON-native data.

    The unified control plane deliberately refuses implicit stringification.
    Subsystems must declare how domain-specific state is represented before it
    can become part of authoritative evidence.
    """

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("control-plane evidence cannot contain non-finite floats")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("control-plane mapping keys must be strings")
            normalized[key] = _normalize_json(item)
        return normalized
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize_json(item) for item in value]
    raise TypeError(f"unsupported control-plane evidence type: {type(value).__name__}")


def canonical_json(value: object) -> str:
    return json.dumps(
        _normalize_json(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def digest_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class StateEnvelope:
    """Typed digest of one subsystem state at a transaction boundary."""

    state_type: str
    state_version: int
    dimensions: tuple[int, ...]
    payload_digest: str
    authoritative: bool
    protocol: str = CONTROL_PROTOCOL

    def __post_init__(self) -> None:
        if self.protocol != CONTROL_PROTOCOL:
            raise ValueError("unsupported control-plane protocol")
        if not self.state_type:
            raise ValueError("state_type must be non-empty")
        if self.state_version < 1:
            raise ValueError("state_version must be positive")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.dimensions
        ):
            raise ValueError("state dimensions must be non-negative integers")
        if len(self.payload_digest) != 64 or any(
            character not in "0123456789abcdef" for character in self.payload_digest
        ):
            raise ValueError("payload_digest must be 64 lowercase hexadecimal characters")

    @classmethod
    def from_payload(
        cls,
        *,
        state_type: str,
        state_version: int,
        dimensions: Sequence[int],
        payload: object,
        authoritative: bool,
    ) -> "StateEnvelope":
        return cls(
            state_type=state_type,
            state_version=state_version,
            dimensions=tuple(dimensions),
            payload_digest=digest_json(payload),
            authoritative=bool(authoritative),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol": self.protocol,
            "state_type": self.state_type,
            "state_version": self.state_version,
            "dimensions": list(self.dimensions),
            "payload_digest": self.payload_digest,
            "authoritative": self.authoritative,
        }


@dataclass(frozen=True, slots=True)
class TransactionReceipt:
    """Deterministic evidence for one candidate-first state transition."""

    sequence: int
    subsystem: str
    operation: str
    transaction_id: str
    decision: Decision
    reason: str | None
    before: StateEnvelope
    candidate: StateEnvelope
    after: StateEnvelope
    metrics: Mapping[str, object]
    previous_hash: str
    receipt_hash: str
    protocol: str = CONTROL_PROTOCOL

    @classmethod
    def build(
        cls,
        *,
        sequence: int,
        subsystem: str,
        operation: str,
        decision: Decision,
        reason: str | None,
        before: StateEnvelope,
        candidate: StateEnvelope,
        after: StateEnvelope,
        metrics: Mapping[str, object] | None,
        previous_hash: str,
    ) -> "TransactionReceipt":
        if sequence < 0:
            raise ValueError("receipt sequence must be non-negative")
        if not subsystem or not operation:
            raise ValueError("subsystem and operation must be non-empty")
        if decision not in ("commit", "rollback"):
            raise ValueError("decision must be commit or rollback")
        if before.state_type != candidate.state_type or before.state_type != after.state_type:
            raise ValueError("transaction state types must match")
        if not before.authoritative or candidate.authoritative or not after.authoritative:
            raise ValueError("transaction authority flags are inconsistent")
        if decision == "commit" and candidate.payload_digest != after.payload_digest:
            raise ValueError("committed after-state must equal the admitted candidate")
        if decision == "rollback" and before.payload_digest != after.payload_digest:
            raise ValueError("rollback must preserve the authoritative before-state")
        if len(previous_hash) != 64 or any(
            character not in "0123456789abcdef" for character in previous_hash
        ):
            raise ValueError("previous_hash must be 64 lowercase hexadecimal characters")

        normalized_metrics = _normalize_json(dict(metrics or {}))
        assert isinstance(normalized_metrics, dict)
        transaction_id = digest_json(
            {
                "protocol": CONTROL_PROTOCOL,
                "sequence": sequence,
                "subsystem": subsystem,
                "operation": operation,
                "before": before.to_dict(),
                "candidate": candidate.to_dict(),
            }
        )
        body = {
            "protocol": CONTROL_PROTOCOL,
            "sequence": sequence,
            "subsystem": subsystem,
            "operation": operation,
            "transaction_id": transaction_id,
            "decision": decision,
            "reason": reason,
            "before": before.to_dict(),
            "candidate": candidate.to_dict(),
            "after": after.to_dict(),
            "metrics": normalized_metrics,
            "previous_hash": previous_hash,
        }
        return cls(
            sequence=sequence,
            subsystem=subsystem,
            operation=operation,
            transaction_id=transaction_id,
            decision=decision,
            reason=reason,
            before=before,
            candidate=candidate,
            after=after,
            metrics=normalized_metrics,
            previous_hash=previous_hash,
            receipt_hash=digest_json(body),
        )

    def body(self) -> dict[str, object]:
        return {
            "protocol": self.protocol,
            "sequence": self.sequence,
            "subsystem": self.subsystem,
            "operation": self.operation,
            "transaction_id": self.transaction_id,
            "decision": self.decision,
            "reason": self.reason,
            "before": self.before.to_dict(),
            "candidate": self.candidate.to_dict(),
            "after": self.after.to_dict(),
            "metrics": dict(self.metrics),
            "previous_hash": self.previous_hash,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.body(), "receipt_hash": self.receipt_hash}

    def verify(self) -> bool:
        expected_transaction_id = digest_json(
            {
                "protocol": self.protocol,
                "sequence": self.sequence,
                "subsystem": self.subsystem,
                "operation": self.operation,
                "before": self.before.to_dict(),
                "candidate": self.candidate.to_dict(),
            }
        )
        if not hmac.compare_digest(self.transaction_id, expected_transaction_id):
            return False
        if self.decision == "commit":
            if self.candidate.payload_digest != self.after.payload_digest:
                return False
        elif self.decision == "rollback":
            if self.before.payload_digest != self.after.payload_digest:
                return False
        else:
            return False
        return hmac.compare_digest(self.receipt_hash, digest_json(self.body()))


class OmegaEvidenceChain:
    """Hash-chained subsystem-neutral transaction evidence.

    This chain complements subsystem-specific telemetry. It does not attest
    external truth and it does not replace OS/process isolation.
    """

    def __init__(self) -> None:
        self.chain: list[TransactionReceipt] = []

    def checkpoint(self) -> int:
        return len(self.chain)

    def restore(self, checkpoint: int) -> None:
        if (
            isinstance(checkpoint, bool)
            or not isinstance(checkpoint, int)
            or checkpoint < 0
            or checkpoint > len(self.chain)
        ):
            raise ValueError("evidence checkpoint is invalid")
        del self.chain[checkpoint:]

    def reset(self) -> None:
        self.chain.clear()

    def append(
        self,
        *,
        subsystem: str,
        operation: str,
        decision: Decision,
        before: StateEnvelope,
        candidate: StateEnvelope,
        after: StateEnvelope,
        metrics: Mapping[str, object] | None = None,
        reason: str | None = None,
    ) -> TransactionReceipt:
        previous_hash = self.chain[-1].receipt_hash if self.chain else GENESIS_RECEIPT_HASH
        receipt = TransactionReceipt.build(
            sequence=len(self.chain),
            subsystem=subsystem,
            operation=operation,
            decision=decision,
            reason=reason,
            before=before,
            candidate=candidate,
            after=after,
            metrics=metrics,
            previous_hash=previous_hash,
        )
        self.chain.append(receipt)
        return receipt

    def verify(self) -> bool:
        previous_hash = GENESIS_RECEIPT_HASH
        for sequence, receipt in enumerate(self.chain):
            if receipt.sequence != sequence:
                return False
            if receipt.previous_hash != previous_hash:
                return False
            if not receipt.verify():
                return False
            previous_hash = receipt.receipt_hash
        return True

    def snapshot(self) -> list[dict[str, object]]:
        return [receipt.to_dict() for receipt in self.chain]
