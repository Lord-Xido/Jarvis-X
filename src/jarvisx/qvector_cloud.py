"""Cloud scheduling and inward optimization for Q16.16x3 vector fields."""

from __future__ import annotations

import hashlib
import json
import math
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

from .cloud_os import DrMoagiCloudOS
from .qvector3d import QVectorAutoencoder3D, QVectorField3D, QVectorRoundTrip3D

QVECTOR_CLOUD_PROTOCOL = "jarvisx.dr-moagi-qvector-cloud.v1"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _volume(shape: Sequence[int]) -> int:
    if len(shape) != 3:
        raise ValueError("shape must contain exactly three dimensions")
    x, y, z = (int(axis) for axis in shape)
    if min(x, y, z) < 1:
        raise ValueError("shape dimensions must be positive")
    return x * y * z


def _shape3(shape: Sequence[int]) -> tuple[int, int, int]:
    _volume(shape)
    return (int(shape[0]), int(shape[1]), int(shape[2]))


@dataclass(slots=True)
class QVectorCloudJob:
    job_id: str
    request_id: str
    node_id: str
    operation: str
    status: str = "queued"
    result: dict[str, object] | None = None
    error: str | None = None


class DrMoagiQVectorCloudEngine3D:
    """Bounded cloud executor for 96-bit Q16.16x3 vector cells."""

    def __init__(
        self,
        cloud: DrMoagiCloudOS | None = None,
        *,
        ledger_path: Path | None = None,
        default_node_cells: int = 3_000_000,
    ) -> None:
        if cloud is not None and ledger_path is not None:
            raise ValueError("ledger_path cannot be supplied with an existing cloud runtime")
        if default_node_cells < 3:
            raise ValueError("default_node_cells must be at least three scalar lanes")
        self.cloud = cloud or DrMoagiCloudOS(ledger_path=ledger_path)
        if not self.cloud.nodes:
            self.cloud.register_node(
                "qvector-local",
                max_cells=default_node_cells,
                max_concurrency=1,
            )
        self.encoder = QVectorAutoencoder3D()
        self.jobs: dict[str, QVectorCloudJob] = {}
        self.request_index: dict[str, str] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _candidate_shapes(shape: tuple[int, int, int]) -> tuple[tuple[int, int, int], ...]:
        half = (
            max(1, shape[0] // 2),
            max(1, shape[1] // 2),
            max(1, shape[2] // 2),
        )
        quarter = (
            max(1, shape[0] // 4),
            max(1, shape[1] // 4),
            max(1, shape[2] // 4),
        )
        candidates: set[tuple[int, int, int]] = {shape, half, quarter, (1, 1, 1)}
        return tuple(sorted(candidates, key=lambda item: (_volume(item), item)))

    @staticmethod
    def _job_id(request_id: str, operation: str, field: QVectorField3D) -> str:
        return _sha256(
            {
                "protocol": QVECTOR_CLOUD_PROTOCOL,
                "request_id": request_id,
                "operation": operation,
                "field_digest": field.digest,
            }
        )[:24]

    @staticmethod
    def _roundtrip_payload(round_trip: QVectorRoundTrip3D) -> dict[str, object]:
        return {
            "latent": round_trip.latent.raw_payload(),
            "reconstruction": round_trip.reconstruction.raw_payload(),
            "axis_mse": list(round_trip.axis_mse),
            "component_mse": round_trip.component_mse,
            "vector_mse": round_trip.vector_mse,
            "compression_ratio": round_trip.compression_ratio,
        }

    def _run_job(
        self,
        *,
        operation: str,
        request_id: str,
        field: QVectorField3D,
        execute: Callable[[], dict[str, object]],
    ) -> QVectorCloudJob:
        with self._lock, self.cloud._lock:
            existing_job_id = self.request_index.get(request_id)
            if existing_job_id is not None:
                existing = self.jobs[existing_job_id]
                if existing.operation != operation:
                    raise ValueError("request_id was already used for a different vector operation")
                return existing

            node = self.cloud._select_node(field.scalar_lanes)
            job_id = self._job_id(request_id, operation, field)
            job = QVectorCloudJob(
                job_id=job_id,
                request_id=request_id,
                node_id=node.node_id,
                operation=operation,
            )
            self.jobs[job_id] = job
            self.request_index[request_id] = job_id
            node.active_jobs += 1
            self.cloud.ledger.append(
                "qvector.job.started",
                {
                    "protocol": QVECTOR_CLOUD_PROTOCOL,
                    "job_id": job_id,
                    "request_id": request_id,
                    "node_id": node.node_id,
                    "operation": operation,
                    "vector_cells": field.cells,
                    "scalar_lanes": field.scalar_lanes,
                    "field_digest": field.digest,
                },
            )

        try:
            result = execute()
            result_digest = _sha256(result)
            committed = {**result, "result_digest": result_digest}
            with self._lock:
                job.status = "succeeded"
                job.result = committed
                self.cloud.ledger.append(
                    "qvector.job.committed",
                    {
                        "protocol": QVECTOR_CLOUD_PROTOCOL,
                        "job_id": job.job_id,
                        "node_id": job.node_id,
                        "result_digest": result_digest,
                    },
                )
        except Exception as error:
            with self._lock:
                job.status = "failed"
                job.error = f"{type(error).__name__}: {error}"
                self.cloud.ledger.append(
                    "qvector.job.failed",
                    {
                        "protocol": QVECTOR_CLOUD_PROTOCOL,
                        "job_id": job.job_id,
                        "node_id": job.node_id,
                        "error": job.error,
                    },
                )
            raise
        finally:
            with self.cloud._lock:
                self.cloud.nodes[job.node_id].active_jobs -= 1
        return job

    def round_trip(
        self,
        field: QVectorField3D,
        latent_shape: Sequence[int],
        *,
        request_id: str,
    ) -> QVectorCloudJob:
        latent_shape3 = _shape3(latent_shape)

        def execute() -> dict[str, object]:
            result = self.encoder.round_trip(field, latent_shape3)
            return {
                "selected_latent_shape": list(latent_shape3),
                **self._roundtrip_payload(result),
            }

        return self._run_job(
            operation="qvector_round_trip",
            request_id=request_id,
            field=field,
            execute=execute,
        )

    def auto_optimize(
        self,
        field: QVectorField3D,
        *,
        request_id: str,
        complexity_weight: float = 0.01,
        candidates: Sequence[Sequence[int]] | None = None,
    ) -> QVectorCloudJob:
        if not math.isfinite(complexity_weight) or complexity_weight < 0.0:
            raise ValueError("complexity_weight must be finite and non-negative")

        if candidates is None:
            candidate_shapes = self._candidate_shapes(field.shape)
        else:
            candidate_shapes = tuple(_shape3(candidate) for candidate in candidates)
            if not candidate_shapes:
                raise ValueError("at least one candidate latent shape is required")

        for candidate in candidate_shapes:
            if any(axis > source for axis, source in zip(candidate, field.shape)):
                raise ValueError("candidate latent dimensions cannot exceed source dimensions")

        def execute() -> dict[str, object]:
            evaluated: list[tuple[float, int, tuple[int, int, int], QVectorRoundTrip3D]] = []
            for candidate in candidate_shapes:
                round_trip = self.encoder.round_trip(field, candidate)
                objective = (
                    round_trip.component_mse
                    + complexity_weight * round_trip.compression_ratio
                )
                evaluated.append((objective, round_trip.latent.cells, candidate, round_trip))

            objective, _, selected_shape, selected = min(
                evaluated,
                key=lambda item: (item[0], item[1], item[2]),
            )
            return {
                "objective": objective,
                "complexity_weight": complexity_weight,
                "selected_latent_shape": list(selected_shape),
                "candidates": [
                    {
                        "latent_shape": list(candidate),
                        "objective": score,
                        "axis_mse": list(result.axis_mse),
                        "component_mse": result.component_mse,
                        "vector_mse": result.vector_mse,
                        "compression_ratio": result.compression_ratio,
                    }
                    for score, _, candidate, result in evaluated
                ],
                **self._roundtrip_payload(selected),
            }

        return self._run_job(
            operation="qvector_auto_optimize",
            request_id=request_id,
            field=field,
            execute=execute,
        )

    def job_snapshot(self, job_id: str) -> dict[str, object]:
        with self._lock:
            try:
                job = self.jobs[job_id]
            except KeyError as error:
                raise KeyError(f"unknown qvector job {job_id!r}") from error
            return dict(asdict(job))


__all__ = [
    "QVECTOR_CLOUD_PROTOCOL",
    "DrMoagiQVectorCloudEngine3D",
    "QVectorCloudJob",
]
