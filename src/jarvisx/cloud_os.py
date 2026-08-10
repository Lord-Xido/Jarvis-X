"""Deterministic 3D auto-encoding cloud control plane for Jarvis-X.

This module is deliberately a user-space runtime rather than a kernel or
hypervisor.  It turns the Dr Moagi inward-loop concepts into bounded,
auditable software primitives:

field -> encode -> decode -> score -> select -> journal -> commit.

All core operations are dependency-free, deterministic for the same inputs,
bounded by registered node capacity, and safe to replay from the ledger.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

CLOUD_PROTOCOL = "jarvisx.dr-moagi-cloud-os.v1"


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
    if x < 1 or y < 1 or z < 1:
        raise ValueError("shape dimensions must be positive")
    return x * y * z


def _shape3(shape: Sequence[int]) -> tuple[int, int, int]:
    _volume(shape)
    return (int(shape[0]), int(shape[1]), int(shape[2]))


@dataclass(frozen=True, slots=True)
class Field3D:
    """Finite scalar field over a dense virtual 3D lattice."""

    values: tuple[float, ...]
    shape: tuple[int, int, int]

    def __post_init__(self) -> None:
        expected = _volume(self.shape)
        if len(self.values) != expected:
            raise ValueError(f"field contains {len(self.values)} values; expected {expected}")
        if any(not math.isfinite(value) for value in self.values):
            raise ValueError("field values must all be finite")

    @classmethod
    def from_values(cls, values: Iterable[float], shape: Sequence[int]) -> "Field3D":
        parsed = tuple(float(value) for value in values)
        return cls(values=parsed, shape=_shape3(shape))

    @property
    def cells(self) -> int:
        return len(self.values)


@dataclass(frozen=True, slots=True)
class RoundTrip:
    """One deterministic 3D encode/decode result."""

    latent: Field3D
    reconstruction: Field3D
    mse: float
    compression_ratio: float


class GeometricAutoencoder3D:
    """Dependency-free block-partition autoencoder for scalar 3D fields.

    Each input cell maps to exactly one latent cell by integer coordinate
    projection.  The latent value is the mean of all source cells assigned to
    that region; decoding broadcasts each latent value back to the same region.
    This intentionally simple transform is deterministic and exactly preserves
    constant fields.
    """

    @staticmethod
    def _latent_index(
        x: int,
        y: int,
        z: int,
        source_shape: tuple[int, int, int],
        latent_shape: tuple[int, int, int],
    ) -> int:
        sx, sy, sz = source_shape
        lx, ly, lz = latent_shape
        ix = min(lx - 1, (x * lx) // sx)
        iy = min(ly - 1, (y * ly) // sy)
        iz = min(lz - 1, (z * lz) // sz)
        return ix + lx * (iy + ly * iz)

    @staticmethod
    def _source_index(x: int, y: int, z: int, shape: tuple[int, int, int]) -> int:
        sx, sy, _ = shape
        return x + sx * (y + sy * z)

    def encode(self, field: Field3D, latent_shape: Sequence[int]) -> Field3D:
        latent_shape3 = _shape3(latent_shape)
        if any(latent > source for latent, source in zip(latent_shape3, field.shape)):
            raise ValueError("latent dimensions cannot exceed source dimensions")

        latent_cells = _volume(latent_shape3)
        sums = [0.0] * latent_cells
        counts = [0] * latent_cells
        sx, sy, sz = field.shape

        for z in range(sz):
            for y in range(sy):
                for x in range(sx):
                    source_index = self._source_index(x, y, z, field.shape)
                    latent_index = self._latent_index(x, y, z, field.shape, latent_shape3)
                    sums[latent_index] += field.values[source_index]
                    counts[latent_index] += 1

        values = tuple(
            sums[index] / counts[index] if counts[index] else 0.0
            for index in range(latent_cells)
        )
        return Field3D(values=values, shape=latent_shape3)

    def decode(self, latent: Field3D, output_shape: Sequence[int]) -> Field3D:
        output_shape3 = _shape3(output_shape)
        if any(latent_axis > output for latent_axis, output in zip(latent.shape, output_shape3)):
            raise ValueError("latent dimensions cannot exceed output dimensions")

        sx, sy, sz = output_shape3
        values = [0.0] * _volume(output_shape3)
        for z in range(sz):
            for y in range(sy):
                for x in range(sx):
                    source_index = self._source_index(x, y, z, output_shape3)
                    latent_index = self._latent_index(x, y, z, output_shape3, latent.shape)
                    values[source_index] = latent.values[latent_index]

        return Field3D(values=tuple(values), shape=output_shape3)

    def round_trip(self, field: Field3D, latent_shape: Sequence[int]) -> RoundTrip:
        latent = self.encode(field, latent_shape)
        reconstruction = self.decode(latent, field.shape)
        squared_error = sum(
            (source - decoded) ** 2
            for source, decoded in zip(field.values, reconstruction.values)
        )
        mse = squared_error / field.cells
        return RoundTrip(
            latent=latent,
            reconstruction=reconstruction,
            mse=mse,
            compression_ratio=latent.cells / field.cells,
        )


@dataclass(slots=True)
class CloudNode:
    node_id: str
    max_cells: int
    max_concurrency: int = 1
    active_jobs: int = 0
    healthy: bool = True

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ValueError("node_id cannot be empty")
        if self.max_cells < 1:
            raise ValueError("max_cells must be positive")
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")

    def can_run(self, cells: int) -> bool:
        return self.healthy and cells <= self.max_cells and self.active_jobs < self.max_concurrency


@dataclass(slots=True)
class CloudJob:
    job_id: str
    request_id: str
    node_id: str
    operation: str
    status: str = "queued"
    result: dict[str, object] | None = None
    error: str | None = None


class HashChainLedger:
    """Append-only deterministic hash-chain ledger."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.records: list[dict[str, object]] = []
        self._lock = threading.RLock()
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists():
                self._load()

    def _load(self) -> None:
        assert self.path is not None
        loaded: list[dict[str, object]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("ledger records must be JSON objects")
                loaded.append(value)
        self.records = loaded
        if not self.verify():
            raise ValueError("ledger hash chain verification failed")

    def append(self, event: str, payload: Mapping[str, object]) -> dict[str, object]:
        with self._lock:
            previous_digest = str(self.records[-1]["digest"]) if self.records else "0" * 64
            core: dict[str, object] = {
                "protocol": CLOUD_PROTOCOL,
                "sequence": len(self.records),
                "event": event,
                "payload": dict(payload),
                "previous_digest": previous_digest,
            }
            record = {**core, "digest": _sha256(core)}
            self.records.append(record)
            if self.path is not None:
                encoded = _canonical_json(record) + b"\n"
                with self.path.open("ab") as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
            return dict(record)

    def verify(self) -> bool:
        previous_digest = "0" * 64
        for sequence, record in enumerate(self.records):
            digest = record.get("digest")
            core = {
                "protocol": record.get("protocol"),
                "sequence": record.get("sequence"),
                "event": record.get("event"),
                "payload": record.get("payload"),
                "previous_digest": record.get("previous_digest"),
            }
            if record.get("protocol") != CLOUD_PROTOCOL:
                return False
            if record.get("sequence") != sequence:
                return False
            if record.get("previous_digest") != previous_digest:
                return False
            if digest != _sha256(core):
                return False
            previous_digest = str(digest)
        return True


class DrMoagiCloudOS:
    """Synchronous reference cloud control plane.

    Jobs are scheduled onto registered virtual nodes, executed as deterministic
    3D auto-encoding cycles, verified, journaled, and committed to the job
    table. The API is synchronous by design so the reference implementation
    has no hidden background workers or unbounded queues.
    """

    def __init__(self, ledger_path: Path | None = None) -> None:
        self.encoder = GeometricAutoencoder3D()
        self.nodes: dict[str, CloudNode] = {}
        self.jobs: dict[str, CloudJob] = {}
        self.request_index: dict[str, str] = {}
        self.ledger = HashChainLedger(ledger_path)
        self._lock = threading.RLock()

    def register_node(self, node_id: str, max_cells: int, max_concurrency: int = 1) -> CloudNode:
        with self._lock:
            if node_id in self.nodes:
                raise ValueError(f"node {node_id!r} is already registered")
            node = CloudNode(
                node_id=node_id,
                max_cells=max_cells,
                max_concurrency=max_concurrency,
            )
            self.nodes[node_id] = node
            self.ledger.append(
                "node.registered",
                {
                    "node_id": node.node_id,
                    "max_cells": node.max_cells,
                    "max_concurrency": node.max_concurrency,
                },
            )
            return node

    def _select_node(self, cells: int) -> CloudNode:
        candidates = [node for node in self.nodes.values() if node.can_run(cells)]
        if not candidates:
            raise RuntimeError("no healthy node has capacity for this field")
        return min(
            candidates,
            key=lambda node: (
                node.active_jobs / node.max_concurrency,
                node.max_cells,
                node.node_id,
            ),
        )

    @staticmethod
    def _candidate_shapes(shape: tuple[int, int, int]) -> tuple[tuple[int, int, int], ...]:
        candidates: set[tuple[int, int, int]] = {
            shape,
            _shape3(tuple(max(1, axis // 2) for axis in shape)),
            _shape3(tuple(max(1, axis // 4) for axis in shape)),
            (1, 1, 1),
        }
        return tuple(sorted(candidates, key=lambda item: (_volume(item), item)))

    @staticmethod
    def _job_id(request_id: str, operation: str, field: Field3D) -> str:
        return _sha256(
            {
                "request_id": request_id,
                "operation": operation,
                "shape": field.shape,
                "values": field.values,
            }
        )[:24]

    @staticmethod
    def _roundtrip_payload(round_trip: RoundTrip) -> dict[str, object]:
        return {
            "latent": {
                "shape": list(round_trip.latent.shape),
                "values": list(round_trip.latent.values),
            },
            "reconstruction": {
                "shape": list(round_trip.reconstruction.shape),
                "values": list(round_trip.reconstruction.values),
            },
            "mse": round_trip.mse,
            "compression_ratio": round_trip.compression_ratio,
        }

    def _run_job(
        self,
        *,
        operation: str,
        request_id: str,
        field: Field3D,
        execute: object,
    ) -> CloudJob:
        with self._lock:
            existing_job_id = self.request_index.get(request_id)
            if existing_job_id is not None:
                existing = self.jobs[existing_job_id]
                if existing.operation != operation:
                    raise ValueError("request_id was already used for a different operation")
                return existing

            node = self._select_node(field.cells)
            job_id = self._job_id(request_id, operation, field)
            job = CloudJob(
                job_id=job_id,
                request_id=request_id,
                node_id=node.node_id,
                operation=operation,
            )
            self.jobs[job_id] = job
            self.request_index[request_id] = job_id
            node.active_jobs += 1
            self.ledger.append(
                "job.started",
                {
                    "job_id": job_id,
                    "request_id": request_id,
                    "node_id": node.node_id,
                    "operation": operation,
                    "cells": field.cells,
                },
            )

        try:
            if not callable(execute):
                raise TypeError("execute must be callable")
            result = execute()
            if not isinstance(result, dict):
                raise TypeError("job result must be a dictionary")
            result_digest = _sha256(result)
            result = {**result, "result_digest": result_digest}
            with self._lock:
                job.status = "succeeded"
                job.result = result
                self.ledger.append(
                    "job.committed",
                    {
                        "job_id": job.job_id,
                        "node_id": job.node_id,
                        "result_digest": result_digest,
                    },
                )
        except Exception as error:
            with self._lock:
                job.status = "failed"
                job.error = f"{type(error).__name__}: {error}"
                self.ledger.append(
                    "job.failed",
                    {
                        "job_id": job.job_id,
                        "node_id": job.node_id,
                        "error": job.error,
                    },
                )
            raise
        finally:
            with self._lock:
                self.nodes[job.node_id].active_jobs -= 1
        return job

    def round_trip(
        self,
        field: Field3D,
        latent_shape: Sequence[int],
        *,
        request_id: str,
    ) -> CloudJob:
        latent_shape3 = _shape3(latent_shape)

        def execute() -> dict[str, object]:
            result = self.encoder.round_trip(field, latent_shape3)
            return {
                "selected_latent_shape": list(latent_shape3),
                **self._roundtrip_payload(result),
            }

        return self._run_job(
            operation="round_trip",
            request_id=request_id,
            field=field,
            execute=execute,
        )

    def auto_optimize(
        self,
        field: Field3D,
        *,
        request_id: str,
        complexity_weight: float = 0.01,
        candidates: Sequence[Sequence[int]] | None = None,
    ) -> CloudJob:
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
            evaluated: list[tuple[float, int, tuple[int, int, int], RoundTrip]] = []
            for candidate in candidate_shapes:
                round_trip = self.encoder.round_trip(field, candidate)
                objective = round_trip.mse + complexity_weight * round_trip.compression_ratio
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
                        "mse": result.mse,
                        "compression_ratio": result.compression_ratio,
                    }
                    for score, _, candidate, result in evaluated
                ],
                **self._roundtrip_payload(selected),
            }

        return self._run_job(
            operation="auto_optimize",
            request_id=request_id,
            field=field,
            execute=execute,
        )

    def job_snapshot(self, job_id: str) -> dict[str, object]:
        with self._lock:
            try:
                job = self.jobs[job_id]
            except KeyError as error:
                raise KeyError(f"unknown job {job_id!r}") from error
            return {
                "job_id": job.job_id,
                "request_id": job.request_id,
                "node_id": job.node_id,
                "operation": job.operation,
                "status": job.status,
                "result": job.result,
                "error": job.error,
            }

    def node_snapshots(self) -> list[dict[str, object]]:
        with self._lock:
            return [
                {
                    "node_id": node.node_id,
                    "max_cells": node.max_cells,
                    "max_concurrency": node.max_concurrency,
                    "active_jobs": node.active_jobs,
                    "healthy": node.healthy,
                }
                for node in (self.nodes[node_id] for node_id in sorted(self.nodes))
            ]
