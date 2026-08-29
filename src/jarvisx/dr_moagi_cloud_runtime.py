"""Transactional cloud control plane for bounded Jarvis-X execution.

The coordinator turns a request into an auditable state-machine transaction:

    RECEIVED -> VALIDATED -> PLANNED -> DISPATCHED -> RUNNING
             -> VERIFIED -> COMMITTED

or a terminal REJECTED/FAILED state. Executors remain non-authoritative:
results are committed only after verifier and policy approval. The durable
store uses canonical JSON, atomic replacement, and a hash-chained event log so
a job can be replayed and independently integrity-checked by job_id.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence, cast

JsonObject = dict[str, Any]
Verifier = Callable[[Mapping[str, Any], Mapping[str, Any]], tuple[bool, str]]

PROTOCOL = "jarvisx.dr-moagi-cloud.v1"
GENESIS_DIGEST = "0" * 64


class JobState(str, Enum):
    RECEIVED = "RECEIVED"
    VALIDATED = "VALIDATED"
    PLANNED = "PLANNED"
    DISPATCHED = "DISPATCHED"
    RUNNING = "RUNNING"
    VERIFIED = "VERIFIED"
    COMMITTED = "COMMITTED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"


TERMINAL_STATES = {JobState.COMMITTED, JobState.REJECTED, JobState.FAILED}


class Executor(Protocol):
    """Bounded execution adapter selected by operation name."""

    def execute(self, payload: Mapping[str, Any], limits: "ResourceLimits") -> Mapping[str, Any]:
        ...


@dataclass(frozen=True, slots=True)
class ResourceLimits:
    max_input_bytes: int = 1_000_000
    max_output_bytes: int = 2_000_000
    max_runtime_ms: int = 5_000

    def __post_init__(self) -> None:
        for name in ("max_input_bytes", "max_output_bytes", "max_runtime_ms"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class JobPolicy:
    allowed_operations: frozenset[str]
    limits: ResourceLimits = ResourceLimits()

    def __post_init__(self) -> None:
        if not self.allowed_operations:
            raise ValueError("at least one operation must be allowed")
        if any(not op or len(op) > 128 for op in self.allowed_operations):
            raise ValueError("operation names must be non-empty and <= 128 characters")


class EchoExecutor:
    """Deterministic reference executor used to prove the transaction spine."""

    def execute(self, payload: Mapping[str, Any], limits: ResourceLimits) -> Mapping[str, Any]:
        del limits
        return {"echo": _json_compatible(payload)}


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_hex(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _json_compatible(value: object) -> Any:
    """Canonicalize a value through JSON serialization.

    This low-level helper intentionally returns ``Any`` because ``json.loads``
    does. Public typed boundaries validate the resulting container shape before
    returning it.
    """

    return json.loads(canonical_json_bytes(value).decode("utf-8"))


def _json_object(value: object, *, name: str = "value") -> JsonObject:
    converted = _json_compatible(value)
    if not isinstance(converted, dict):
        raise TypeError(f"{name} must canonicalize to a JSON object")
    return cast(JsonObject, converted)


def _json_event_sequence(value: object) -> Sequence[Mapping[str, Any]]:
    converted = _json_compatible(value)
    if not isinstance(converted, list) or any(not isinstance(item, dict) for item in converted):
        raise TypeError("events must canonicalize to a list of JSON objects")
    return cast(list[Mapping[str, Any]], converted)


def default_verifier(result: Mapping[str, Any], context: Mapping[str, Any]) -> tuple[bool, str]:
    del context
    try:
        canonical_json_bytes(result)
    except (TypeError, ValueError):
        return False, "result is not finite canonical JSON"
    return True, "verified"


class AtomicJobStore:
    """Durable job documents with atomic replacement and hash-chained events."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    @staticmethod
    def _validate_job_id(job_id: str) -> None:
        try:
            parsed = uuid.UUID(job_id)
        except (ValueError, AttributeError) as error:
            raise ValueError("job_id must be a canonical UUID") from error
        if str(parsed) != job_id:
            raise ValueError("job_id must be a canonical UUID")

    def _path(self, job_id: str) -> Path:
        self._validate_job_id(job_id)
        return self.root / f"{job_id}.json"

    def create(self, envelope: Mapping[str, Any]) -> None:
        job_id = str(envelope["job_id"])
        destination = self._path(job_id)
        with self._lock:
            if destination.exists():
                raise RuntimeError("job already exists")
            self._write(destination, envelope)

    def replace(self, envelope: Mapping[str, Any]) -> None:
        job_id = str(envelope["job_id"])
        destination = self._path(job_id)
        with self._lock:
            if not destination.exists():
                raise FileNotFoundError(job_id)
            self._write(destination, envelope)

    def get(self, job_id: str) -> JsonObject:
        path = self._path(job_id)
        with self._lock:
            value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("stored job must be a JSON object")
        return cast(JsonObject, value)

    def count(self) -> int:
        return sum(1 for path in self.root.glob("*.json") if path.is_file())

    def ready(self) -> bool:
        probe = self.root / ".ready"
        try:
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except OSError:
            return False
        return True

    def _write(self, destination: Path, envelope: Mapping[str, Any]) -> None:
        encoded = canonical_json_bytes(envelope)
        temporary = destination.with_suffix(f".tmp-{os.getpid()}-{threading.get_ident()}")
        with temporary.open("wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)


class DrMoagiCloudCoordinator:
    """Synchronous reference control plane with fail-closed commit semantics."""

    def __init__(
        self,
        *,
        executors: Mapping[str, Executor],
        policy: JobPolicy,
        store: AtomicJobStore,
        verifier: Verifier = default_verifier,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        missing = policy.allowed_operations.difference(executors)
        if missing:
            raise ValueError(f"no executor registered for: {sorted(missing)}")
        self.executors = dict(executors)
        self.policy = policy
        self.store = store
        self.verifier = verifier
        self.clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)

    def submit(
        self,
        *,
        principal: str,
        operation: str,
        payload: Mapping[str, Any],
        request_id: str | None = None,
    ) -> JsonObject:
        principal = self._validate_principal(principal)
        operation = self._validate_operation(operation)
        payload_json = _json_object(payload, name="payload")
        input_bytes = canonical_json_bytes(payload_json)
        if len(input_bytes) > self.policy.limits.max_input_bytes:
            raise ValueError("input exceeds max_input_bytes")

        job_id = str(uuid.uuid4())
        now = self.clock_ms()
        envelope: JsonObject = {
            "protocol": PROTOCOL,
            "job_id": job_id,
            "request_id": request_id,
            "principal": principal,
            "operation": operation,
            "state": JobState.RECEIVED.value,
            "created_at_ms": now,
            "updated_at_ms": now,
            "input": payload_json,
            "input_digest": sha256_hex(payload_json),
            "plan": None,
            "result": None,
            "result_digest": None,
            "verification": None,
            "resource_usage": None,
            "events": [],
            "envelope_digest": None,
        }
        self._append_event(envelope, JobState.RECEIVED, {"request_id": request_id})
        self._seal(envelope)
        self.store.create(envelope)

        try:
            self._transition(envelope, JobState.VALIDATED, {"input_bytes": len(input_bytes)})
            plan = {
                "operation": operation,
                "executor": type(self.executors[operation]).__name__,
                "limits": {
                    "max_input_bytes": self.policy.limits.max_input_bytes,
                    "max_output_bytes": self.policy.limits.max_output_bytes,
                    "max_runtime_ms": self.policy.limits.max_runtime_ms,
                },
            }
            envelope["plan"] = plan
            self._transition(envelope, JobState.PLANNED, {"plan_digest": sha256_hex(plan)})
            self._transition(envelope, JobState.DISPATCHED, {})
            self._transition(envelope, JobState.RUNNING, {})

            started_ns = time.perf_counter_ns()
            result = self.executors[operation].execute(payload_json, self.policy.limits)
            runtime_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
            result_json = _json_object(result, name="executor result")
            output_bytes = canonical_json_bytes(result_json)
            resource_usage = {
                "input_bytes": len(input_bytes),
                "output_bytes": len(output_bytes),
                "runtime_ms": runtime_ms,
            }
            envelope["resource_usage"] = resource_usage
            if runtime_ms > self.policy.limits.max_runtime_ms:
                return self._reject(envelope, "runtime limit exceeded")
            if len(output_bytes) > self.policy.limits.max_output_bytes:
                return self._reject(envelope, "output exceeds max_output_bytes")

            context = {
                "job_id": job_id,
                "principal": principal,
                "operation": operation,
                "input_digest": envelope["input_digest"],
                "plan": plan,
                "resource_usage": resource_usage,
            }
            verified, reason = self.verifier(result_json, context)
            envelope["verification"] = {"passed": bool(verified), "reason": str(reason)}
            if not verified:
                return self._reject(envelope, f"verification failed: {reason}")

            envelope["result"] = result_json
            envelope["result_digest"] = sha256_hex(result_json)
            self._transition(
                envelope,
                JobState.VERIFIED,
                {"result_digest": envelope["result_digest"]},
            )
            self._transition(envelope, JobState.COMMITTED, {"committed": True})
            return _json_object(envelope, name="job envelope")
        except Exception as error:
            self._fail(envelope, error)
            raise

    def get(self, job_id: str) -> JsonObject:
        return self.store.get(job_id)

    def verify_job(self, job_id: str) -> JsonObject:
        envelope = self.store.get(job_id)
        ok, reason = self._verify_envelope(envelope)
        return {
            "job_id": job_id,
            "verified": ok,
            "reason": reason,
            "state": envelope.get("state"),
            "event_count": (
                len(envelope.get("events", []))
                if isinstance(envelope.get("events"), list)
                else 0
            ),
        }

    def events(self, job_id: str) -> Sequence[Mapping[str, Any]]:
        envelope = self.store.get(job_id)
        events = envelope.get("events")
        if not isinstance(events, list):
            raise ValueError("job events are invalid")
        return _json_event_sequence(events)

    def _transition(
        self, envelope: JsonObject, state: JobState, details: Mapping[str, Any]
    ) -> None:
        envelope["state"] = state.value
        envelope["updated_at_ms"] = self.clock_ms()
        self._append_event(envelope, state, details)
        self._seal(envelope)
        self.store.replace(envelope)

    def _reject(self, envelope: JsonObject, reason: str) -> JsonObject:
        envelope["verification"] = envelope.get("verification") or {
            "passed": False,
            "reason": reason,
        }
        self._transition(envelope, JobState.REJECTED, {"reason": reason})
        return _json_object(envelope, name="job envelope")

    def _fail(self, envelope: JsonObject, error: Exception) -> None:
        if envelope.get("state") in {state.value for state in TERMINAL_STATES}:
            return
        envelope["verification"] = {"passed": False, "reason": "execution failure"}
        self._transition(
            envelope,
            JobState.FAILED,
            {"error_type": type(error).__name__, "reason": str(error)[:512]},
        )

    def _append_event(
        self, envelope: JsonObject, state: JobState, details: Mapping[str, Any]
    ) -> None:
        events = envelope["events"]
        if not isinstance(events, list):
            raise TypeError("events must be a list")
        previous = events[-1]["event_digest"] if events else GENESIS_DIGEST
        core = {
            "sequence": len(events),
            "state": state.value,
            "timestamp_ms": self.clock_ms(),
            "details": _json_object(details, name="event details"),
            "previous_digest": previous,
        }
        event = dict(core)
        event["event_digest"] = sha256_hex(core)
        events.append(event)

    @staticmethod
    def _seal(envelope: JsonObject) -> None:
        unsigned = {key: value for key, value in envelope.items() if key != "envelope_digest"}
        envelope["envelope_digest"] = sha256_hex(unsigned)

    @staticmethod
    def _verify_envelope(envelope: Mapping[str, Any]) -> tuple[bool, str]:
        if envelope.get("protocol") != PROTOCOL:
            return False, "unsupported protocol"
        expected = envelope.get("envelope_digest")
        if not isinstance(expected, str):
            return False, "envelope digest missing"
        unsigned = {key: value for key, value in envelope.items() if key != "envelope_digest"}
        if expected != sha256_hex(unsigned):
            return False, "envelope digest mismatch"
        events = envelope.get("events")
        if not isinstance(events, list) or not events:
            return False, "event journal missing"
        previous = GENESIS_DIGEST
        for index, event in enumerate(events):
            if not isinstance(event, dict):
                return False, f"event {index} invalid"
            digest = event.get("event_digest")
            core = {key: value for key, value in event.items() if key != "event_digest"}
            if event.get("sequence") != index:
                return False, f"event {index} sequence mismatch"
            if event.get("previous_digest") != previous:
                return False, f"event {index} chain mismatch"
            if digest != sha256_hex(core):
                return False, f"event {index} digest mismatch"
            previous = str(digest)
        if envelope.get("state") != events[-1].get("state"):
            return False, "terminal state does not match event journal"
        result = envelope.get("result")
        result_digest = envelope.get("result_digest")
        if result is not None and result_digest != sha256_hex(result):
            return False, "result digest mismatch"
        return True, "verified"

    def _validate_operation(self, operation: str) -> str:
        if not isinstance(operation, str) or not operation or len(operation) > 128:
            raise ValueError("operation must be a non-empty string <= 128 characters")
        if operation not in self.policy.allowed_operations:
            raise PermissionError("operation is not allowed by policy")
        return operation

    @staticmethod
    def _validate_principal(principal: str) -> str:
        if not isinstance(principal, str) or not principal.strip() or len(principal) > 256:
            raise ValueError("principal must be a non-empty string <= 256 characters")
        return principal.strip()
