"""End-to-end transactional orchestration fabric for Jarvis-X.

This module composes the canonical authority boundary into one bounded pipeline:

OBSERVE -> NORMALIZE -> PROVENANCE -> AUTHORIZE -> ENCODE -> TRANSFORM
-> RECONSTRUCT -> MEASURE -> PROPOSE -> VERIFY -> PI_LAMBDA
-> COMMIT/ROLLBACK -> OMEGA RECEIPT.

The fabric is deliberately adapter-driven. Research subsystems may propose state but
cannot become authoritative without passing the transaction gate.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, MutableMapping, Protocol, Sequence

from .m3_acme import RuntimeReport, process_records


class FabricError(RuntimeError):
    """Base error for transaction fabric failures."""


class AdmissionRejected(FabricError):
    """Raised when a candidate cannot cross Pi_Lambda."""


class TransactionAdapter(Protocol):
    """Adapter contract for optional bounded research transforms."""

    name: str
    version: str

    def transform(self, state: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def validate(self, before: Mapping[str, Any], after: Mapping[str, Any]) -> Sequence[str]: ...


@dataclass(frozen=True)
class FabricLimits:
    max_records: int = 1024
    max_state_bytes: int = 8 * 1024 * 1024
    max_adapters: int = 16
    max_numeric_abs: float = 1e12
    require_finite: bool = True


@dataclass(frozen=True)
class CandidateDecision:
    adapter: str
    version: str
    accepted: bool
    reasons: tuple[str, ...]
    before_digest: str
    after_digest: str | None


@dataclass(frozen=True)
class FabricReceipt:
    transaction_id: str
    observed_at: str
    committed: bool
    input_digest: str
    output_digest: str
    accepted_records: int
    rejected_records: int
    decisions: tuple[CandidateDecision, ...]
    metrics: Mapping[str, float]
    authoritative_state: Mapping[str, Any]
    omega_entry: Mapping[str, Any]


@dataclass
class OmegaChain:
    """Dependency-free append-only hash chain for fabric receipts."""

    entries: list[dict[str, Any]] = field(default_factory=list)

    def append(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        previous_hash = self.entries[-1]["hash"] if self.entries else "0" * 64
        canonical_payload = _canonical_bytes(payload)
        digest = hashlib.sha256(previous_hash.encode("ascii") + canonical_payload).hexdigest()
        entry = dict(payload)
        entry["previous_hash"] = previous_hash
        entry["hash"] = digest
        self.entries.append(entry)
        return entry

    def verify(self) -> bool:
        previous_hash = "0" * 64
        for entry in self.entries:
            payload = {k: v for k, v in entry.items() if k not in {"previous_hash", "hash"}}
            expected = hashlib.sha256(previous_hash.encode("ascii") + _canonical_bytes(payload)).hexdigest()
            if entry.get("previous_hash") != previous_hash or entry.get("hash") != expected:
                return False
            previous_hash = expected
        return True


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _state_size(value: Any) -> int:
    return len(_canonical_bytes(value))


def _validate_numeric_tree(value: Any, limits: FabricLimits, path: str = "$.") -> list[str]:
    failures: list[str] = []
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return failures
    if isinstance(value, (int, float)):
        numeric = float(value)
        if limits.require_finite and not math.isfinite(numeric):
            failures.append(f"{path}: non-finite numeric value")
        elif abs(numeric) > limits.max_numeric_abs:
            failures.append(f"{path}: numeric magnitude exceeds limit")
        return failures
    if isinstance(value, Mapping):
        for key, item in value.items():
            failures.extend(_validate_numeric_tree(item, limits, f"{path}{key}."))
        return failures
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            failures.extend(_validate_numeric_tree(item, limits, f"{path}{index}."))
    return failures


def _pi_lambda(
    before: Mapping[str, Any],
    candidate: Mapping[str, Any],
    adapter: TransactionAdapter,
    limits: FabricLimits,
) -> tuple[str, ...]:
    failures: list[str] = []
    size = _state_size(candidate)
    if size > limits.max_state_bytes:
        failures.append(f"candidate state exceeds max_state_bytes ({size} > {limits.max_state_bytes})")
    failures.extend(_validate_numeric_tree(candidate, limits))
    failures.extend(str(reason) for reason in adapter.validate(before, candidate))
    return tuple(failures)


def _report_to_state(report: RuntimeReport) -> dict[str, Any]:
    accepted = [
        {
            "index": record.index,
            "latent_digest": record.latent_digest,
            "compressed_bytes": record.compressed_bytes,
            "abstraction": asdict(record.abstraction),
            "reconstructed": record.reconstructed,
            "loss": asdict(record.loss),
        }
        for record in report.accepted
    ]
    rejected = [asdict(record) for record in report.rejected]
    return {
        "m3_acme": {
            "accepted": accepted,
            "rejected": rejected,
            "accepted_count": report.accepted_count,
            "rejected_count": report.rejected_count,
        }
    }


def execute_transaction(
    records: Sequence[Mapping[str, Any]],
    *,
    adapters: Sequence[TransactionAdapter] = (),
    limits: FabricLimits | None = None,
    omega: OmegaChain | None = None,
    now: datetime | None = None,
) -> FabricReceipt:
    """Execute one complete bounded Jarvis-X research transaction.

    M3-ACME performs the admission/encode/abstract/decode/loss boundary. Optional adapters
    then receive isolated candidate state. Each adapter must pass Pi_Lambda before its state
    can become authoritative. A rejected adapter is rolled back atomically to the previous
    authoritative state and recorded in the receipt.
    """

    effective_limits = limits or FabricLimits()
    chain = omega or OmegaChain()
    timestamp = now or datetime.now(timezone.utc)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    if len(records) > effective_limits.max_records:
        raise AdmissionRejected("record count exceeds configured max_records")
    if len(adapters) > effective_limits.max_adapters:
        raise AdmissionRejected("adapter count exceeds configured max_adapters")

    input_digest = _digest(records)
    report = process_records(records, now=timestamp)
    authoritative: MutableMapping[str, Any] = _report_to_state(report)
    decisions: list[CandidateDecision] = []

    for adapter in adapters:
        before = json.loads(json.dumps(authoritative))
        before_digest = _digest(before)
        try:
            candidate = dict(adapter.transform(before))
            failures = _pi_lambda(before, candidate, adapter, effective_limits)
        except Exception as exc:  # fail closed at the adapter boundary
            candidate = None
            failures = (f"adapter exception: {type(exc).__name__}: {exc}",)

        if failures:
            decisions.append(
                CandidateDecision(
                    adapter=adapter.name,
                    version=adapter.version,
                    accepted=False,
                    reasons=tuple(failures),
                    before_digest=before_digest,
                    after_digest=None if candidate is None else _digest(candidate),
                )
            )
            continue

        authoritative = candidate
        decisions.append(
            CandidateDecision(
                adapter=adapter.name,
                version=adapter.version,
                accepted=True,
                reasons=(),
                before_digest=before_digest,
                after_digest=_digest(candidate),
            )
        )

    output_digest = _digest(authoritative)
    losses = [record.loss.total for record in report.accepted]
    metrics = {
        "accepted_records": float(report.accepted_count),
        "rejected_records": float(report.rejected_count),
        "adapter_commits": float(sum(1 for item in decisions if item.accepted)),
        "adapter_rollbacks": float(sum(1 for item in decisions if not item.accepted)),
        "mean_moagi_loss": float(sum(losses) / len(losses)) if losses else 0.0,
        "authoritative_state_bytes": float(_state_size(authoritative)),
    }

    committed = report.rejected_count == 0
    transaction_id = hashlib.sha256(
        f"{input_digest}:{output_digest}:{timestamp.isoformat()}".encode("utf-8")
    ).hexdigest()
    omega_payload = {
        "transaction_id": transaction_id,
        "observed_at": timestamp.isoformat(),
        "committed": committed,
        "input_digest": input_digest,
        "output_digest": output_digest,
        "accepted_records": report.accepted_count,
        "rejected_records": report.rejected_count,
        "adapter_decisions": [asdict(item) for item in decisions],
        "metrics": metrics,
    }
    omega_entry = chain.append(omega_payload)

    return FabricReceipt(
        transaction_id=transaction_id,
        observed_at=timestamp.isoformat(),
        committed=committed,
        input_digest=input_digest,
        output_digest=output_digest,
        accepted_records=report.accepted_count,
        rejected_records=report.rejected_count,
        decisions=tuple(decisions),
        metrics=metrics,
        authoritative_state=dict(authoritative),
        omega_entry=omega_entry,
    )
