"""FastAPI control surface for the Dr Moagi Cloud transactional runtime."""

from __future__ import annotations

import hmac
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from fastapi import Depends, FastAPI, Header, HTTPException, Response, status
from pydantic import BaseModel, ConfigDict, Field

from .dr_moagi_cloud_operations import DrMoagiFieldStepExecutor
from .dr_moagi_cloud_runtime import (
    AtomicJobStore,
    DrMoagiCloudCoordinator,
    EchoExecutor,
    JobPolicy,
    JsonObject,
    ResourceLimits,
)

DEFAULT_DATA_DIR = Path("state/dr-moagi-cloud")


class JobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: str = Field(min_length=1, max_length=128)
    input: dict[str, Any] = Field(default_factory=dict)
    request_id: str | None = Field(default=None, max_length=256)


@dataclass(frozen=True, slots=True)
class CloudServiceSettings:
    data_dir: Path = DEFAULT_DATA_DIR
    api_key: str | None = None
    require_api_key: bool = False
    max_input_bytes: int = 1_000_000
    max_output_bytes: int = 2_000_000
    max_runtime_ms: int = 5_000

    def __post_init__(self) -> None:
        if self.require_api_key and not self.api_key:
            raise ValueError("DR_MOAGI_CLOUD_API_KEY is required when authentication is enforced")
        ResourceLimits(
            max_input_bytes=self.max_input_bytes,
            max_output_bytes=self.max_output_bytes,
            max_runtime_ms=self.max_runtime_ms,
        )

    @classmethod
    def from_env(cls) -> "CloudServiceSettings":
        return cls(
            data_dir=Path(os.getenv("DR_MOAGI_CLOUD_DATA_DIR", str(DEFAULT_DATA_DIR))),
            api_key=os.getenv("DR_MOAGI_CLOUD_API_KEY") or None,
            require_api_key=_env_bool("DR_MOAGI_CLOUD_REQUIRE_API_KEY", False),
            max_input_bytes=_env_int("DR_MOAGI_CLOUD_MAX_INPUT_BYTES", 1_000_000),
            max_output_bytes=_env_int("DR_MOAGI_CLOUD_MAX_OUTPUT_BYTES", 2_000_000),
            max_runtime_ms=_env_int("DR_MOAGI_CLOUD_MAX_RUNTIME_MS", 5_000),
        )


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def build_default_coordinator(settings: CloudServiceSettings) -> DrMoagiCloudCoordinator:
    limits = ResourceLimits(
        max_input_bytes=settings.max_input_bytes,
        max_output_bytes=settings.max_output_bytes,
        max_runtime_ms=settings.max_runtime_ms,
    )
    return DrMoagiCloudCoordinator(
        executors={
            "echo.v1": EchoExecutor(),
            "dr-moagi-field-step.v1": DrMoagiFieldStepExecutor(),
        },
        policy=JobPolicy(
            allowed_operations=frozenset({"echo.v1", "dr-moagi-field-step.v1"}),
            limits=limits,
        ),
        store=AtomicJobStore(settings.data_dir),
    )


def create_app(
    settings: CloudServiceSettings | None = None,
    coordinator: DrMoagiCloudCoordinator | None = None,
) -> FastAPI:
    settings = settings or CloudServiceSettings.from_env()
    runtime = coordinator or build_default_coordinator(settings)

    app = FastAPI(
        title="Dr Moagi Cloud Runtime",
        version="1.0.0",
        description=(
            "Transactional Jarvis-X execution control plane. Executors cannot commit results "
            "until verification and policy gates pass."
        ),
    )
    app.state.settings = settings
    app.state.runtime = runtime

    def authenticated_principal(
        x_dr_moagi_key: str | None = Header(default=None, alias="X-Dr-Moagi-Key"),
        x_dr_moagi_principal: str | None = Header(default=None, alias="X-Dr-Moagi-Principal"),
    ) -> str:
        if settings.require_api_key:
            if x_dr_moagi_key is None or settings.api_key is None or not hmac.compare_digest(
                x_dr_moagi_key, settings.api_key
            ):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="invalid API key",
                )
        default_principal = "api-key" if settings.require_api_key else "local-dev"
        principal = (x_dr_moagi_principal or default_principal).strip()
        if not principal or len(principal) > 256:
            raise HTTPException(status_code=400, detail="invalid principal")
        return principal

    @app.get("/health/live")
    def live() -> Mapping[str, object]:
        return {"status": "live", "protocol": "jarvisx.dr-moagi-cloud.v1"}

    @app.get("/health/ready")
    def ready(response: Response) -> Mapping[str, object]:
        is_ready = runtime.store.ready()
        if not is_ready:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "ready" if is_ready else "not-ready", "store_ready": is_ready}

    @app.post("/api/v1/jobs", status_code=status.HTTP_201_CREATED)
    def create_job(
        request: JobRequest,
        principal: str = Depends(authenticated_principal),
    ) -> JsonObject:
        try:
            return runtime.submit(
                principal=principal,
                operation=request.operation,
                payload=request.input,
                request_id=request.request_id,
            )
        except PermissionError as error:
            raise HTTPException(status_code=403, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get("/api/v1/jobs/{job_id}")
    def get_job(
        job_id: str,
        principal: str = Depends(authenticated_principal),
    ) -> JsonObject:
        del principal
        try:
            return runtime.get(job_id)
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail="job not found") from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/api/v1/jobs/{job_id}/events")
    def get_events(
        job_id: str,
        principal: str = Depends(authenticated_principal),
    ) -> JsonObject:
        del principal
        try:
            return {"job_id": job_id, "events": runtime.events(job_id)}
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail="job not found") from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/api/v1/jobs/{job_id}/verify")
    def verify_job(
        job_id: str,
        principal: str = Depends(authenticated_principal),
    ) -> JsonObject:
        del principal
        try:
            return runtime.verify_job(job_id)
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail="job not found") from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/metrics", response_class=Response)
    def metrics(principal: str = Depends(authenticated_principal)) -> Response:
        del principal
        ready_value = 1 if runtime.store.ready() else 0
        body = "\n".join(
            [
                "# TYPE dr_moagi_cloud_jobs_stored gauge",
                f"dr_moagi_cloud_jobs_stored {runtime.store.count()}",
                "# TYPE dr_moagi_cloud_store_ready gauge",
                f"dr_moagi_cloud_store_ready {ready_value}",
                "",
            ]
        )
        return Response(content=body, media_type="text/plain; version=0.0.4; charset=utf-8")

    return app


app = create_app()
