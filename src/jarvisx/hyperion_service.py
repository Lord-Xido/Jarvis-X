"""Operational FastAPI surface and durable replay store for Hyperion.

The service freezes one Hyperion configuration and score model per process,
accepts bounded observation batches, writes deterministic evidence bundles with
atomic replacement, and verifies stored reports by replaying the committed
inputs. It does not acquire sensor data or attest source truth.
"""

from __future__ import annotations

import hmac
import json
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

from fastapi import Depends, FastAPI, Header, HTTPException, Response, status
from pydantic import BaseModel, ConfigDict, Field

from .hyperion import (
    AuditReport,
    HyperionConfig,
    HyperionEngine,
    Observation,
    ScoreModel,
    sha256_hex,
)

SERVICE_PROTOCOL = "jarvisx.hyperion-service.v1"
DEFAULT_DATA_DIR = Path("state/hyperion")


class ObservationPayload(BaseModel):
    """Strict JSON representation of one Hyperion observation."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(min_length=1, max_length=128)
    timestamp_ms: int = Field(ge=0)
    value: float
    quantity: str = Field(min_length=1, max_length=128)
    unit: str = Field(min_length=1, max_length=64)
    correlation_id: str | None = Field(default=None, max_length=256)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    available: bool = True
    label: str | None = Field(default=None, max_length=512)
    metadata: dict[str, object] = Field(default_factory=dict)

    def to_observation(self) -> Observation:
        return Observation(
            source=self.source,
            timestamp_ms=self.timestamp_ms,
            value=self.value,
            quantity=self.quantity,
            unit=self.unit,
            correlation_id=self.correlation_id,
            confidence=self.confidence,
            available=self.available,
            label=self.label,
            metadata=self.metadata,
        )


class AuditRequest(BaseModel):
    """Bounded audit request accepted by the service."""

    model_config = ConfigDict(extra="forbid")

    observations: list[ObservationPayload] = Field(min_length=1)
    request_id: str | None = Field(default=None, max_length=256)


@dataclass(frozen=True, slots=True)
class HyperionRuntimeSettings:
    """Operational settings that do not alter audit mathematics."""

    data_dir: Path = DEFAULT_DATA_DIR
    max_observations: int = 50_000
    api_key: str | None = None
    require_api_key: bool = False

    def __post_init__(self) -> None:
        if self.max_observations < 1:
            raise ValueError("max_observations must be positive")
        if self.require_api_key and not self.api_key:
            raise ValueError("HYPERION_API_KEY is required when authentication is enforced")

    @classmethod
    def from_env(cls) -> "HyperionRuntimeSettings":
        raw_limit = os.getenv("HYPERION_MAX_OBSERVATIONS", "50000")
        try:
            max_observations = int(raw_limit)
        except ValueError as error:
            raise ValueError("HYPERION_MAX_OBSERVATIONS must be an integer") from error
        require_api_key = os.getenv("HYPERION_REQUIRE_API_KEY", "false").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        return cls(
            data_dir=Path(os.getenv("HYPERION_DATA_DIR", str(DEFAULT_DATA_DIR))),
            max_observations=max_observations,
            api_key=os.getenv("HYPERION_API_KEY") or None,
            require_api_key=require_api_key,
        )


class AtomicReportStore:
    """Filesystem evidence store using deterministic JSON and atomic replace."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._lock = threading.RLock()
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _validate_digest(report_digest: str) -> None:
        if len(report_digest) != 64 or any(
            character not in "0123456789abcdef" for character in report_digest
        ):
            raise ValueError("report digest must be 64 lowercase hexadecimal characters")

    def _path(self, report_digest: str) -> Path:
        self._validate_digest(report_digest)
        return self.root / f"{report_digest}.json"

    def put(self, bundle: Mapping[str, object]) -> Path:
        report_digest = bundle.get("report_digest")
        if not isinstance(report_digest, str):
            raise ValueError("bundle report_digest is missing")
        destination = self._path(report_digest)
        encoded = json.dumps(
            bundle,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        with self._lock:
            if destination.exists():
                current = destination.read_bytes()
                if current != encoded:
                    raise RuntimeError("immutable report digest collision")
                return destination
            temporary = destination.with_suffix(
                f".tmp-{os.getpid()}-{threading.get_ident()}"
            )
            with temporary.open("wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        return destination

    def get(self, report_digest: str) -> dict[str, object]:
        path = self._path(report_digest)
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("stored report bundle must be a JSON object")
        return cast(dict[str, object], value)

    def ready(self) -> bool:
        probe = self.root / ".ready"
        try:
            probe.write_text("ready", encoding="utf-8")
            probe.unlink()
        except OSError:
            return False
        return True

    def count(self) -> int:
        return sum(1 for path in self.root.glob("*.json") if path.is_file())


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return cast(dict[str, Any], value)


def load_hyperion_config(path: Path | None) -> HyperionConfig:
    if path is None:
        return HyperionConfig()
    return HyperionConfig(**_json_object(path))


def load_score_model(path: Path | None) -> ScoreModel:
    if path is None:
        return ScoreModel()
    payload = _json_object(path)
    weights = payload.get("weights")
    if isinstance(weights, list):
        payload["weights"] = tuple(float(value) for value in weights)
    return ScoreModel(**payload)


def engine_from_env() -> HyperionEngine:
    config_path = os.getenv("HYPERION_CONFIG_FILE")
    model_path = os.getenv("HYPERION_MODEL_FILE")
    return HyperionEngine(
        config=load_hyperion_config(Path(config_path) if config_path else None),
        model=load_score_model(Path(model_path) if model_path else None),
    )


def _json_compatible(value: object) -> object:
    return json.loads(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    )


def _required_int(value: object, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as error:
            raise ValueError(f"{field_name} must be an integer") from error
    raise ValueError(f"{field_name} must be an integer")


def _required_float(value: object, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric")
    if isinstance(value, (int, float, str)):
        try:
            converted = float(value)
        except ValueError as error:
            raise ValueError(f"{field_name} must be numeric") from error
        if not converted == converted or converted in (float("inf"), float("-inf")):
            raise ValueError(f"{field_name} must be finite")
        return converted
    raise ValueError(f"{field_name} must be numeric")


def report_to_dict(report: AuditReport) -> dict[str, object]:
    value = _json_compatible(asdict(report))
    if not isinstance(value, dict):
        raise TypeError("serialized report must be an object")
    return cast(dict[str, object], value)


def observation_to_dict(observation: Observation) -> dict[str, object]:
    payload = cast(dict[str, object], asdict(observation))
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        payload["metadata"] = dict(metadata)
    return payload


def observation_from_dict(payload: Mapping[str, object]) -> Observation:
    return Observation(
        source=str(payload["source"]),
        timestamp_ms=_required_int(payload["timestamp_ms"], "timestamp_ms"),
        value=_required_float(payload["value"], "value"),
        quantity=str(payload["quantity"]),
        unit=str(payload["unit"]),
        correlation_id=(
            str(payload["correlation_id"])
            if payload.get("correlation_id") is not None
            else None
        ),
        confidence=_required_float(payload.get("confidence", 1.0), "confidence"),
        available=bool(payload.get("available", True)),
        label=str(payload["label"]) if payload.get("label") is not None else None,
        metadata=(
            cast(Mapping[str, object], payload.get("metadata", {}))
            if isinstance(payload.get("metadata", {}), Mapping)
            else {}
        ),
    )


def build_evidence_bundle(
    engine: HyperionEngine,
    observations: Sequence[Observation],
) -> dict[str, object]:
    report = engine.audit(observations)
    if not report.verify():
        raise RuntimeError("Hyperion generated a report that failed internal verification")
    canonical_observations = sorted(
        observations,
        key=lambda item: (
            item.timestamp_ms,
            item.correlation_id or "",
            item.source,
            item.value,
            item.confidence,
        ),
    )
    core: dict[str, object] = {
        "protocol": SERVICE_PROTOCOL,
        "model_hash": report.model_hash,
        "configuration_hash": report.configuration_hash,
        "observations": [observation_to_dict(item) for item in canonical_observations],
        "report": report_to_dict(report),
        "report_digest": report.report_digest,
    }
    core["bundle_digest"] = sha256_hex(core)
    return core


def verify_evidence_bundle(
    bundle: Mapping[str, object],
    engine: HyperionEngine,
) -> tuple[bool, str]:
    expected_bundle_digest = bundle.get("bundle_digest")
    if not isinstance(expected_bundle_digest, str):
        return False, "bundle digest missing"
    unsigned = {key: value for key, value in bundle.items() if key != "bundle_digest"}
    if not hmac.compare_digest(expected_bundle_digest, sha256_hex(unsigned)):
        return False, "bundle digest mismatch"
    if bundle.get("protocol") != SERVICE_PROTOCOL:
        return False, "unsupported protocol"
    if bundle.get("configuration_hash") != engine.configuration_hash:
        return False, "configuration hash mismatch"
    if bundle.get("model_hash") != engine.model.model_hash:
        return False, "model hash mismatch"
    raw_observations = bundle.get("observations")
    if not isinstance(raw_observations, list):
        return False, "observations missing"
    observations: list[Observation] = []
    for payload in raw_observations:
        if not isinstance(payload, dict):
            return False, "invalid observation payload"
        observations.append(
            observation_from_dict(cast(Mapping[str, object], payload))
        )
    replay = engine.audit(observations)
    if not replay.verify():
        return False, "replayed report failed verification"
    if bundle.get("report_digest") != replay.report_digest:
        return False, "report digest mismatch"
    if bundle.get("report") != report_to_dict(replay):
        return False, "stored report differs from deterministic replay"
    return True, "verified"


class HyperionRuntime:
    """Thread-safe operational coordinator around a frozen Hyperion engine."""

    def __init__(
        self,
        engine: HyperionEngine,
        settings: HyperionRuntimeSettings,
        store: AtomicReportStore | None = None,
    ) -> None:
        self.engine = engine
        self.settings = settings
        self.store = store or AtomicReportStore(settings.data_dir)
        self._metrics_lock = threading.Lock()
        self._audits_total = 0
        self._failures_total = 0
        self._observations_total = 0
        self._critical_events_total = 0
        self._last_success_time = 0.0

    def audit(
        self,
        observations: Sequence[Observation],
        *,
        request_id: str | None = None,
    ) -> dict[str, object]:
        if not observations:
            raise ValueError("at least one observation is required")
        if len(observations) > self.settings.max_observations:
            raise ValueError(
                f"observation count exceeds limit {self.settings.max_observations}"
            )
        try:
            bundle = build_evidence_bundle(self.engine, observations)
            self.store.put(bundle)
        except Exception:
            with self._metrics_lock:
                self._failures_total += 1
            raise
        report = cast(Mapping[str, object], bundle["report"])
        raw_points = report.get("points", [])
        critical_count = 0
        if isinstance(raw_points, list):
            critical_count = sum(
                1
                for point in raw_points
                if isinstance(point, Mapping) and bool(point.get("critical"))
            )
        with self._metrics_lock:
            self._audits_total += 1
            self._observations_total += len(observations)
            self._critical_events_total += critical_count
            self._last_success_time = time.time()
        return {
            "request_id": request_id,
            "report_digest": bundle["report_digest"],
            "bundle_digest": bundle["bundle_digest"],
            "verified": True,
            "geometric_health_score": report["geometric_health_score"],
            "critical_events": critical_count,
            "event_count": len(raw_points) if isinstance(raw_points, list) else 0,
            "model_hash": bundle["model_hash"],
            "configuration_hash": bundle["configuration_hash"],
        }

    def get(self, report_digest: str) -> dict[str, object]:
        return self.store.get(report_digest)

    def verify(self, report_digest: str) -> dict[str, object]:
        bundle = self.store.get(report_digest)
        verified, reason = verify_evidence_bundle(bundle, self.engine)
        return {
            "report_digest": report_digest,
            "verified": verified,
            "reason": reason,
            "model_hash": self.engine.model.model_hash,
            "configuration_hash": self.engine.configuration_hash,
        }

    def metrics_text(self) -> str:
        with self._metrics_lock:
            values = {
                "hyperion_audits_total": self._audits_total,
                "hyperion_failures_total": self._failures_total,
                "hyperion_observations_total": self._observations_total,
                "hyperion_critical_events_total": self._critical_events_total,
                "hyperion_last_success_unixtime_seconds": self._last_success_time,
            }
        values["hyperion_reports_stored"] = self.store.count()
        lines = [
            "# TYPE hyperion_audits_total counter",
            f"hyperion_audits_total {values['hyperion_audits_total']}",
            "# TYPE hyperion_failures_total counter",
            f"hyperion_failures_total {values['hyperion_failures_total']}",
            "# TYPE hyperion_observations_total counter",
            f"hyperion_observations_total {values['hyperion_observations_total']}",
            "# TYPE hyperion_critical_events_total counter",
            f"hyperion_critical_events_total {values['hyperion_critical_events_total']}",
            "# TYPE hyperion_reports_stored gauge",
            f"hyperion_reports_stored {values['hyperion_reports_stored']}",
            "# TYPE hyperion_last_success_unixtime_seconds gauge",
            (
                "hyperion_last_success_unixtime_seconds "
                f"{values['hyperion_last_success_unixtime_seconds']}"
            ),
        ]
        return "\n".join(lines) + "\n"

    def manifest(self) -> dict[str, object]:
        return {
            "protocol": SERVICE_PROTOCOL,
            "maturity": "operational-experimental",
            "model_hash": self.engine.model.model_hash,
            "model_version": self.engine.model.version,
            "configuration_hash": self.engine.configuration_hash,
            "target_quantity": self.engine.config.target_quantity,
            "target_unit": self.engine.config.target_unit,
            "max_observations": self.settings.max_observations,
            "authentication_required": self.settings.require_api_key,
            "persistence": "atomic-local-filesystem",
            "proof_boundary": "deterministic-computation-over-committed-inputs",
        }


def create_hyperion_app(
    *,
    settings: HyperionRuntimeSettings | None = None,
    engine: HyperionEngine | None = None,
    store: AtomicReportStore | None = None,
) -> FastAPI:
    runtime_settings = settings or HyperionRuntimeSettings.from_env()
    runtime = HyperionRuntime(engine or engine_from_env(), runtime_settings, store)
    application = FastAPI(
        title="Jarvis-X Hyperion Audit Service",
        version="1.0.0",
        docs_url="/docs",
        redoc_url=None,
    )
    application.state.hyperion_runtime = runtime

    def authorize(x_hyperion_key: str | None = Header(default=None)) -> None:
        if not runtime_settings.require_api_key:
            return
        expected = runtime_settings.api_key
        if expected is None or x_hyperion_key is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="unauthorized",
            )
        if not hmac.compare_digest(x_hyperion_key, expected):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="unauthorized",
            )

    @application.get("/healthz", include_in_schema=False)
    def healthz() -> dict[str, str]:
        return {"status": "ok", "protocol": SERVICE_PROTOCOL}

    @application.get("/readyz", include_in_schema=False)
    def readyz() -> dict[str, object]:
        ready = runtime.store.ready()
        if not ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="report store is not writable",
            )
        return {
            "status": "ready",
            "model_hash": runtime.engine.model.model_hash,
            "configuration_hash": runtime.engine.configuration_hash,
        }

    @application.get("/v1/hyperion/manifest", dependencies=[Depends(authorize)])
    def manifest() -> dict[str, object]:
        return runtime.manifest()

    @application.post(
        "/v1/hyperion/audits",
        status_code=status.HTTP_201_CREATED,
        dependencies=[Depends(authorize)],
    )
    def create_audit(request: AuditRequest) -> dict[str, object]:
        observations = [payload.to_observation() for payload in request.observations]
        try:
            return runtime.audit(observations, request_id=request.request_id)
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(error),
            ) from error
        except OSError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="report persistence failed",
            ) from error

    @application.get(
        "/v1/hyperion/audits/{report_digest}",
        dependencies=[Depends(authorize)],
    )
    def get_audit(report_digest: str) -> dict[str, object]:
        try:
            return runtime.get(report_digest)
        except (FileNotFoundError, ValueError) as error:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="not found",
            ) from error

    @application.post(
        "/v1/hyperion/audits/{report_digest}/verify",
        dependencies=[Depends(authorize)],
    )
    def verify_audit(report_digest: str) -> dict[str, object]:
        try:
            result = runtime.verify(report_digest)
        except (FileNotFoundError, ValueError) as error:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="not found",
            ) from error
        if not result["verified"]:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=result,
            )
        return result

    @application.get(
        "/metrics",
        dependencies=[Depends(authorize)],
        include_in_schema=False,
    )
    def metrics_endpoint() -> Response:
        return Response(runtime.metrics_text(), media_type="text/plain; version=0.0.4")

    return application


def app_factory() -> FastAPI:
    """Create the environment-configured ASGI application for Uvicorn."""

    return create_hyperion_app()
