"""FastAPI control plane for the bounded JARVIS-X unified runtime."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .unified_runtime import UnifiedRuntimeConfig, UnifiedRuntimeRegistry

app = FastAPI(
    title="JARVIS-X Unified Runtime",
    version="1.0.0",
    description=(
        "Bounded shared Psi-Phi-Lambda-Omega-Theta coordination state for the "
        "Dr Moagi ANN IDE."
    ),
)
registry = UnifiedRuntimeRegistry(max_sessions=16)


class RuntimeCreateRequest(BaseModel):
    latent_block_size: int = Field(2, ge=1, le=32)
    omega_retention: float = Field(0.85, ge=0.0, le=1.0)
    psi_memory_gain: float = Field(0.25, ge=0.0, le=1.0)
    theta_gain: float = Field(0.05, gt=0.0, le=1.0)
    theta_limit: float = Field(1.0, gt=0.0, le=16.0)
    stability_epsilon: float = Field(1.0e-6, gt=0.0, le=1.0)
    max_dimensions: int = Field(256, ge=1, le=4096)
    value_limit: float = Field(1.0e6, gt=0.0, le=1.0e12)
    resource_weight: float = Field(1.0e-3, gt=0.0, le=1.0)

    def config(self) -> UnifiedRuntimeConfig:
        return UnifiedRuntimeConfig(
            latent_block_size=self.latent_block_size,
            omega_retention=self.omega_retention,
            psi_memory_gain=self.psi_memory_gain,
            theta_gain=self.theta_gain,
            theta_limit=self.theta_limit,
            stability_epsilon=self.stability_epsilon,
            max_dimensions=self.max_dimensions,
            value_limit=self.value_limit,
            resource_weight=self.resource_weight,
        )


class RuntimeTickRequest(BaseModel):
    values: list[float] = Field(min_length=1, max_length=4096)


def fail(exc: Exception) -> HTTPException:
    if isinstance(exc, KeyError):
        return HTTPException(404, "runtime session not found")
    if isinstance(exc, (TypeError, ValueError)):
        return HTTPException(422, str(exc))
    return HTTPException(409, str(exc))


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "jarvis-x-unified-runtime",
        "version": app.version,
        "contract": "S[t+1] = M(S[t], X[t])",
        "authoritative_execution": False,
    }


@app.get("/v1/capabilities")
def capabilities() -> dict[str, Any]:
    return {
        "state": [
            "psi",
            "phi",
            "lambda",
            "omega",
            "theta",
            "latent",
            "reconstruction",
            "error",
        ],
        "metrics": [
            "reconstruction_mse",
            "latent_cycle_mse",
            "state_delta",
            "h_mmm",
        ],
        "invariants": {
            "bounded_dimensions": True,
            "finite_values": True,
            "deterministic": True,
            "state_hashed": True,
            "arbitrary_code_execution": False,
        },
        "role": (
            "coordination telemetry above canonical CodexVM/ANN/policy components; "
            "not a replacement authority"
        ),
    }


@app.post("/v1/sessions")
def create_session(req: RuntimeCreateRequest) -> dict[str, Any]:
    try:
        return registry.create(req.config())
    except Exception as exc:
        raise fail(exc) from exc


@app.get("/v1/sessions/{session_id}")
def session_status(session_id: str) -> dict[str, Any]:
    try:
        return registry.status(session_id)
    except Exception as exc:
        raise fail(exc) from exc


@app.post("/v1/sessions/{session_id}/tick")
def tick(session_id: str, req: RuntimeTickRequest) -> dict[str, Any]:
    try:
        return registry.step(session_id, req.values)
    except Exception as exc:
        raise fail(exc) from exc


@app.post("/v1/sessions/{session_id}/reset")
def reset(session_id: str) -> dict[str, Any]:
    try:
        return registry.reset(session_id)
    except Exception as exc:
        raise fail(exc) from exc


@app.delete("/v1/sessions/{session_id}")
def delete_session(session_id: str) -> dict[str, Any]:
    if not registry.delete(session_id):
        raise HTTPException(404, "runtime session not found")
    return {"deleted": True, "session_id": session_id}
