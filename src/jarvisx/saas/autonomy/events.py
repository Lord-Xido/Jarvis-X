"""Causal, tenant-scoped, tamper-evident enterprise event ledger."""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True)
class EventEnvelope:
    event_id: str
    tenant_id: str
    stream: str
    sequence: int
    event_type: str
    payload: Dict[str, Any]
    actor: str
    occurred_at_ns: int
    correlation_id: str
    causation_id: Optional[str]
    state_version: int
    vector_clock: Dict[str, int]
    previous_hash: str
    event_hash: str

    def verification_body(self) -> Dict[str, Any]:
        body = asdict(self)
        body.pop("event_hash", None)
        return body


class CausalEventLedger:
    """Append-only ledger with per-tenant stream ordering and causal clocks."""

    GENESIS = "0" * 64

    def __init__(self) -> None:
        self._events: List[EventEnvelope] = []
        self._stream_seq: Dict[Tuple[str, str], int] = {}
        self._tenant_version: Dict[str, int] = {}
        self._clocks: Dict[str, Dict[str, int]] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _hash(body: Dict[str, Any]) -> str:
        return hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest()

    def append(
        self,
        *,
        tenant_id: str,
        stream: str,
        event_type: str,
        payload: Dict[str, Any],
        actor: str,
        correlation_id: Optional[str] = None,
        causation_id: Optional[str] = None,
        node_id: str = "primary",
        expected_version: Optional[int] = None,
        occurred_at_ns: Optional[int] = None,
    ) -> EventEnvelope:
        if not tenant_id or not stream or not event_type or not actor:
            raise ValueError("tenant, stream, event_type, and actor are required")
        with self._lock:
            version = self._tenant_version.get(tenant_id, 0)
            if expected_version is not None and expected_version != version:
                raise RuntimeError(
                    "optimistic concurrency conflict: expected %d, found %d"
                    % (expected_version, version)
                )
            sequence = self._stream_seq.get((tenant_id, stream), 0) + 1
            clock = dict(self._clocks.get(tenant_id, {}))
            clock[node_id] = clock.get(node_id, 0) + 1
            previous_hash = (
                self._events[-1].event_hash if self._events else self.GENESIS
            )
            body = {
                "event_id": str(uuid.uuid4()),
                "tenant_id": tenant_id,
                "stream": stream,
                "sequence": sequence,
                "event_type": event_type,
                "payload": payload,
                "actor": actor,
                "occurred_at_ns": occurred_at_ns or time.time_ns(),
                "correlation_id": correlation_id or str(uuid.uuid4()),
                "causation_id": causation_id,
                "state_version": version + 1,
                "vector_clock": clock,
                "previous_hash": previous_hash,
            }
            event = EventEnvelope(event_hash=self._hash(body), **body)
            self._events.append(event)
            self._stream_seq[(tenant_id, stream)] = sequence
            self._tenant_version[tenant_id] = version + 1
            self._clocks[tenant_id] = clock
            return event

    def events(
        self,
        tenant_id: str,
        *,
        stream: Optional[str] = None,
        after_version: int = 0,
    ) -> Tuple[EventEnvelope, ...]:
        return tuple(
            event
            for event in self._events
            if event.tenant_id == tenant_id
            and event.state_version > after_version
            and (stream is None or event.stream == stream)
        )

    def version(self, tenant_id: str) -> int:
        return self._tenant_version.get(tenant_id, 0)

    def verify(self) -> bool:
        previous_hash = self.GENESIS
        tenant_versions: Dict[str, int] = {}
        stream_sequences: Dict[Tuple[str, str], int] = {}
        for event in self._events:
            if event.previous_hash != previous_hash:
                return False
            if self._hash(event.verification_body()) != event.event_hash:
                return False
            expected_version = tenant_versions.get(event.tenant_id, 0) + 1
            if event.state_version != expected_version:
                return False
            key = (event.tenant_id, event.stream)
            expected_sequence = stream_sequences.get(key, 0) + 1
            if event.sequence != expected_sequence:
                return False
            tenant_versions[event.tenant_id] = expected_version
            stream_sequences[key] = expected_sequence
            previous_hash = event.event_hash
        return True

    def replay(self, tenant_id: str, reducer, initial: Any) -> Any:
        state = initial
        for event in self.events(tenant_id):
            state = reducer(state, event)
        return state

    def merkle_root(self, tenant_id: str) -> str:
        hashes = [event.event_hash for event in self.events(tenant_id)]
        if not hashes:
            return self.GENESIS
        while len(hashes) > 1:
            if len(hashes) % 2:
                hashes.append(hashes[-1])
            hashes = [
                hashlib.sha256((hashes[i] + hashes[i + 1]).encode("ascii")).hexdigest()
                for i in range(0, len(hashes), 2)
            ]
        return hashes[0]

    def export(self, tenant_id: str) -> List[Dict[str, Any]]:
        return [asdict(event) for event in self.events(tenant_id)]
