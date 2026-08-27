"""Canonical kinetic transaction runtime for Jarvis-X.

This module operationalizes the engine's own execution kinetics as one reusable
candidate-first state transition protocol:

    snapshot -> observe -> encode -> propose -> shadow -> verify
             -> commit | rollback -> receipt -> re-enter

The runtime is deliberately backend-neutral. It does not decide how a model,
field, VM, scheduler, or optimizer computes a candidate. It only decides when a
candidate may become authoritative and emits a deterministic receipt for every
attempt.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from typing import Callable, Generic, Iterable, Mapping, TypeVar

StateT = TypeVar("StateT")
ObservationT = TypeVar("ObservationT")
EncodedT = TypeVar("EncodedT")
CandidateT = TypeVar("CandidateT")


class KineticStage(str, Enum):
    SNAPSHOT = "snapshot"
    OBSERVE = "observe"
    ENCODE = "encode"
    PROPOSE = "propose"
    SHADOW = "shadow"
    VERIFY = "verify"
    COMMIT = "commit"
    ROLLBACK = "rollback"
    JOURNAL = "journal"
    REENTER = "reenter"


@dataclass(frozen=True)
class ValidatorResult:
    name: str
    passed: bool
    metrics: Mapping[str, object]
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "passed": self.passed,
            "metrics": dict(self.metrics),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class KineticReceipt:
    schema_version: str
    transaction_id: str
    parent_state_hash: str
    candidate_hash: str
    resulting_state_hash: str
    decision: str
    stages: tuple[str, ...]
    validators: tuple[ValidatorResult, ...]
    telemetry: Mapping[str, object]
    previous_receipt_hash: str
    receipt_hash: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "transaction_id": self.transaction_id,
            "parent_state_hash": self.parent_state_hash,
            "candidate_hash": self.candidate_hash,
            "resulting_state_hash": self.resulting_state_hash,
            "decision": self.decision,
            "stages": list(self.stages),
            "validators": [item.to_dict() for item in self.validators],
            "telemetry": dict(self.telemetry),
            "previous_receipt_hash": self.previous_receipt_hash,
            "receipt_hash": self.receipt_hash,
        }


@dataclass(frozen=True)
class KineticResult(Generic[StateT, CandidateT]):
    state: StateT
    candidate: CandidateT
    committed: bool
    receipt: KineticReceipt


def _canonical_value(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _canonical_value(asdict(value))
    if isinstance(value, Mapping):
        items = sorted(value.items(), key=lambda item: str(item[0]))
        return {str(key): _canonical_value(item) for key, item in items}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def canonical_hash(value: object) -> str:
    payload = json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class KineticTransactionEngine(Generic[StateT, ObservationT, EncodedT, CandidateT]):
    """Execute one bounded self-observing candidate transaction.

    ``snapshot`` must return an isolated authoritative snapshot. ``shadow`` must
    evaluate the candidate without mutating the authoritative state. ``commit``
    is called only after every validator passes. ``rollback`` receives the
    original snapshot and is called for rejected candidates.
    """

    def __init__(
        self,
        *,
        snapshot: Callable[[StateT], StateT],
        observe: Callable[[StateT], ObservationT],
        encode: Callable[[StateT, ObservationT], EncodedT],
        propose: Callable[[StateT, ObservationT, EncodedT], CandidateT],
        shadow: Callable[[StateT, CandidateT], Mapping[str, object]],
        validators: Iterable[Callable[[StateT, CandidateT], ValidatorResult]],
        commit: Callable[[StateT, CandidateT], StateT],
        rollback: Callable[[StateT], StateT],
        state_identity: Callable[[StateT], object] | None = None,
        candidate_identity: Callable[[CandidateT], object] | None = None,
    ) -> None:
        self._snapshot = snapshot
        self._observe = observe
        self._encode = encode
        self._propose = propose
        self._shadow = shadow
        self._validators = tuple(validators)
        if not self._validators:
            raise ValueError("at least one validator is required")
        self._commit = commit
        self._rollback = rollback
        self._state_identity = state_identity or (lambda value: value)
        self._candidate_identity = candidate_identity or (lambda value: value)
        self._previous_receipt_hash = "0" * 64
        self._logical_time = 0

    @property
    def previous_receipt_hash(self) -> str:
        return self._previous_receipt_hash

    def step(self, authoritative_state: StateT) -> KineticResult[StateT, CandidateT]:
        stages: list[str] = []
        snapshot = self._snapshot(authoritative_state)
        stages.append(KineticStage.SNAPSHOT.value)
        parent_hash = canonical_hash(self._state_identity(snapshot))

        observation = self._observe(snapshot)
        stages.append(KineticStage.OBSERVE.value)
        encoded = self._encode(snapshot, observation)
        stages.append(KineticStage.ENCODE.value)
        candidate = self._propose(snapshot, observation, encoded)
        stages.append(KineticStage.PROPOSE.value)
        candidate_hash = canonical_hash(self._candidate_identity(candidate))

        telemetry = dict(self._shadow(snapshot, candidate))
        stages.append(KineticStage.SHADOW.value)

        results = tuple(validator(snapshot, candidate) for validator in self._validators)
        stages.append(KineticStage.VERIFY.value)
        committed = all(result.passed for result in results)

        if committed:
            state = self._commit(snapshot, candidate)
            stages.append(KineticStage.COMMIT.value)
        else:
            state = self._rollback(snapshot)
            stages.append(KineticStage.ROLLBACK.value)

        resulting_hash = canonical_hash(self._state_identity(state))
        self._logical_time += 1
        tx_seed = {
            "logical_time": self._logical_time,
            "parent_state_hash": parent_hash,
            "candidate_hash": candidate_hash,
            "decision": "commit" if committed else "rollback",
            "previous_receipt_hash": self._previous_receipt_hash,
        }
        transaction_id = canonical_hash(tx_seed)[:24]

        receipt_stages = (*stages, KineticStage.JOURNAL.value, KineticStage.REENTER.value)
        receipt_body = {
            "schema_version": "jarvisx.kinetic-receipt.v1",
            "transaction_id": transaction_id,
            "parent_state_hash": parent_hash,
            "candidate_hash": candidate_hash,
            "resulting_state_hash": resulting_hash,
            "decision": "commit" if committed else "rollback",
            "stages": list(receipt_stages),
            "validators": [item.to_dict() for item in results],
            "telemetry": telemetry,
            "previous_receipt_hash": self._previous_receipt_hash,
        }
        receipt_hash = canonical_hash(receipt_body)
        receipt = KineticReceipt(
            schema_version="jarvisx.kinetic-receipt.v1",
            transaction_id=transaction_id,
            parent_state_hash=parent_hash,
            candidate_hash=candidate_hash,
            resulting_state_hash=resulting_hash,
            decision="commit" if committed else "rollback",
            stages=receipt_stages,
            validators=results,
            telemetry=telemetry,
            previous_receipt_hash=self._previous_receipt_hash,
            receipt_hash=receipt_hash,
        )
        self._previous_receipt_hash = receipt_hash
        return KineticResult(
            state=state,
            candidate=candidate,
            committed=committed,
            receipt=receipt,
        )
