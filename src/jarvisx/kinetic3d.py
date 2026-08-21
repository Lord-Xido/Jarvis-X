from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from math import isfinite
from struct import pack
from time import perf_counter
from typing import Sequence

from .kinetic3d_backend import resolve_backend

Shape3D = tuple[int, int, int]
Block3D = tuple[int, int, int]
MAX_KINETIC_VOXELS = 1_048_576


class KineticOp(str, Enum):
    OBSERVE = "OBSERVE"
    PREDICT = "PREDICT"
    RESIDUAL = "RESIDUAL"
    ACTIVE_SET = "ACTIVE_SET"
    ENCODE_COARSE = "ENCODE_COARSE"
    LATENT_WRITE = "LATENT_WRITE"
    REFINE = "REFINE"
    DECODE = "DECODE"
    VERIFY = "VERIFY"
    COMMIT = "COMMIT"
    TELEMETRY = "TELEMETRY"
    EMIT = "EMIT"
    HALT = "HALT"


@dataclass(frozen=True)
class SpatialIRNode:
    op: KineticOp
    x: int
    y: int
    z: int
    stage: str
    dtype: str = "fp64-reference"

    def as_payload(self) -> dict[str, object]:
        return {
            "op": self.op.value,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "stage": self.stage,
            "dtype": self.dtype,
        }


@dataclass(frozen=True)
class CoarseLatent:
    block: Block3D
    residual: float
    active_count: int

    def as_payload(self) -> dict[str, object]:
        return {
            "block": list(self.block),
            "residual": self.residual,
            "active_count": self.active_count,
        }


@dataclass(frozen=True)
class FineCorrection:
    index: int
    correction: float

    def as_payload(self) -> dict[str, object]:
        return {"index": self.index, "correction": self.correction}


@dataclass(frozen=True)
class PathAssignment:
    path_id: int
    block: Block3D
    active_cells: int
    resource: str
    estimated_cost: float

    def as_payload(self) -> dict[str, object]:
        return {
            "path_id": self.path_id,
            "block": list(self.block),
            "active_cells": self.active_cells,
            "resource": self.resource,
            "estimated_cost": self.estimated_cost,
        }


@dataclass(frozen=True)
class KineticVerification:
    mse: float
    max_abs_error: float
    tolerance: float
    passed: bool
    checksum_sha256: str

    def as_payload(self) -> dict[str, object]:
        return {
            "mse": self.mse,
            "max_abs_error": self.max_abs_error,
            "tolerance": self.tolerance,
            "passed": self.passed,
            "checksum_sha256": self.checksum_sha256,
        }


@dataclass(frozen=True)
class KineticTelemetry:
    cycles: int
    total_cells: int
    active_cells: int
    active_fraction: float
    coarse_latent_cells: int
    fine_corrections: int
    latent_values: int
    value_compression_ratio: float
    estimated_bytes_moved: int
    elapsed_ms: float
    backend: str

    def as_payload(self) -> dict[str, object]:
        return {
            "cycles": self.cycles,
            "total_cells": self.total_cells,
            "active_cells": self.active_cells,
            "active_fraction": self.active_fraction,
            "coarse_latent_cells": self.coarse_latent_cells,
            "fine_corrections": self.fine_corrections,
            "latent_values": self.latent_values,
            "value_compression_ratio": self.value_compression_ratio,
            "estimated_bytes_moved": self.estimated_bytes_moved,
            "elapsed_ms": self.elapsed_ms,
            "backend": self.backend,
        }


@dataclass(frozen=True)
class Kinetic3DResult:
    shape: Shape3D
    prediction: tuple[float, ...]
    residual: tuple[float, ...]
    active_indices: tuple[int, ...]
    coarse_latent: tuple[CoarseLatent, ...]
    fine_corrections: tuple[FineCorrection, ...]
    reconstructed: tuple[float, ...]
    schedule: tuple[PathAssignment, ...]
    spatial_ir: tuple[SpatialIRNode, ...]
    verification: KineticVerification
    telemetry: KineticTelemetry
    committed: bool
    epoch_before: int
    epoch_after: int

    def as_payload(self) -> dict[str, object]:
        return {
            "shape": list(self.shape),
            "prediction": list(self.prediction),
            "residual": list(self.residual),
            "active_indices": list(self.active_indices),
            "coarse_latent": [item.as_payload() for item in self.coarse_latent],
            "fine_corrections": [item.as_payload() for item in self.fine_corrections],
            "reconstructed": list(self.reconstructed),
            "schedule": [item.as_payload() for item in self.schedule],
            "spatial_ir": [item.as_payload() for item in self.spatial_ir],
            "verification": self.verification.as_payload(),
            "telemetry": self.telemetry.as_payload(),
            "committed": self.committed,
            "epoch_before": self.epoch_before,
            "epoch_after": self.epoch_after,
        }


def _voxel_count(shape: Shape3D) -> int:
    sx, sy, sz = shape
    return sx * sy * sz


def _validate_shape(shape: Shape3D, max_voxels: int) -> int:
    if len(shape) != 3 or any(dimension < 1 for dimension in shape):
        raise ValueError("shape must contain exactly three positive dimensions")
    count = _voxel_count(shape)
    if count > max_voxels:
        raise ValueError(f"voxel count {count} exceeds runtime limit {max_voxels}")
    return count


def _validate_vector(name: str, values: Sequence[float], count: int) -> list[float]:
    if len(values) != count:
        raise ValueError(f"{name} length {len(values)} does not match voxel count {count}")
    vector = [float(value) for value in values]
    if not all(isfinite(value) for value in vector):
        raise ValueError(f"{name} values must all be finite")
    return vector


def _xyz(shape: Shape3D, index: int) -> tuple[int, int, int]:
    sx, sy, _ = shape
    plane = sx * sy
    z, rem = divmod(index, plane)
    y, x = divmod(rem, sx)
    return x, y, z


def _block_for(shape: Shape3D, index: int, factor: int) -> Block3D:
    x, y, z = _xyz(shape, index)
    return x // factor, y // factor, z // factor


def _checksum(values: Sequence[float]) -> str:
    digest = sha256()
    for value in values:
        digest.update(pack("<d", value))
    return digest.hexdigest()


def compile_kinetic_ir() -> tuple[SpatialIRNode, ...]:
    """Compile the kinetic inward/outward execution wave into typed spatial IR."""

    stages = (
        (KineticOp.OBSERVE, 0, "observation-ingress"),
        (KineticOp.PREDICT, 1, "world-prediction"),
        (KineticOp.RESIDUAL, 2, "prediction-error"),
        (KineticOp.ACTIVE_SET, 3, "sparse-active-volume"),
        (KineticOp.ENCODE_COARSE, 4, "hierarchical-coarse-encode"),
        (KineticOp.LATENT_WRITE, 5, "latent-core"),
        (KineticOp.REFINE, 4, "selective-refinement"),
        (KineticOp.DECODE, 3, "sparse-reconstruction"),
        (KineticOp.VERIFY, 2, "transaction-verify"),
        (KineticOp.COMMIT, 1, "world-state-commit"),
        (KineticOp.TELEMETRY, 1, "runtime-telemetry"),
        (KineticOp.EMIT, 0, "result-egress"),
        (KineticOp.HALT, 0, "halt"),
    )
    return tuple(
        SpatialIRNode(op=op, x=tick, y=tick, z=depth, stage=stage)
        for tick, (op, depth, stage) in enumerate(stages)
    )


class Kinetic3DRuntime:
    """Transactional predictive 3D runtime with sparse residual execution.

    The state/IR/verification contract is backend-neutral. ``auto`` uses the native
    C++ active-volume kernel when it is explicitly installed and falls back to the
    pure Python correctness oracle otherwise. Future Triton/CUDA backends can enter
    through the same boundary without changing the observable kinetic semantics.
    """

    def __init__(self, *, max_voxels: int = MAX_KINETIC_VOXELS, backend: str = "auto") -> None:
        if max_voxels < 1:
            raise ValueError("max_voxels must be positive")
        self.max_voxels = max_voxels
        self.backend = backend
        self._committed_world: tuple[float, ...] | None = None
        self._committed_shape: Shape3D | None = None
        self._epoch = 0

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def committed_shape(self) -> Shape3D | None:
        return self._committed_shape

    @property
    def committed_world(self) -> tuple[float, ...] | None:
        return self._committed_world

    def reset(self) -> None:
        self._committed_world = None
        self._committed_shape = None
        self._epoch = 0

    def _prediction(
        self, previous: Sequence[float] | None, shape: Shape3D, count: int
    ) -> list[float]:
        if previous is not None:
            return _validate_vector("previous", previous, count)
        if self._committed_shape == shape and self._committed_world is not None:
            return list(self._committed_world)
        return [0.0] * count

    def execute(
        self,
        current: Sequence[float],
        shape: Shape3D,
        *,
        previous: Sequence[float] | None = None,
        active_threshold: float = 0.0,
        coarse_factor: int = 2,
        refine_threshold: float = 0.0,
        tolerance: float = 0.0,
        backend: str | None = None,
    ) -> Kinetic3DResult:
        count = _validate_shape(shape, self.max_voxels)
        observed = _validate_vector("current", current, count)
        if not isfinite(active_threshold) or active_threshold < 0:
            raise ValueError("active_threshold must be finite and non-negative")
        if not 1 <= coarse_factor <= 32:
            raise ValueError("coarse_factor must be between 1 and 32")
        if not isfinite(refine_threshold) or refine_threshold < 0:
            raise ValueError("refine_threshold must be finite and non-negative")
        if not isfinite(tolerance) or tolerance < 0:
            raise ValueError("tolerance must be finite and non-negative")

        started = perf_counter()
        epoch_before = self._epoch
        ir = compile_kinetic_ir()
        prediction = self._prediction(previous, shape, count)
        selected_backend = resolve_backend(backend or self.backend)
        backend_result = selected_backend.step(
            observed,
            prediction,
            shape,
            active_threshold=active_threshold,
            coarse_factor=coarse_factor,
            refine_threshold=refine_threshold,
        )
        residual = list(backend_result.residual)
        active_indices = backend_result.active_indices
        reconstructed = list(backend_result.reconstructed)
        coarse_values = dict(backend_result.coarse_values)

        grouped: dict[Block3D, list[int]] = {}
        for index in active_indices:
            grouped.setdefault(_block_for(shape, index, coarse_factor), []).append(index)

        coarse_latent: list[CoarseLatent] = []
        schedule: list[PathAssignment] = []
        for path_id, block in enumerate(sorted(grouped)):
            indices = grouped[block]
            value = coarse_values[block]
            coarse_latent.append(
                CoarseLatent(block=block, residual=value, active_count=len(indices))
            )
            schedule.append(
                PathAssignment(
                    path_id=path_id,
                    block=block,
                    active_cells=len(indices),
                    resource=f"{selected_backend.name}:0",
                    estimated_cost=float(len(indices)),
                )
            )

        fine_corrections = [
            FineCorrection(index=index, correction=correction)
            for index, correction in backend_result.fine_corrections
        ]

        errors = [source - target for source, target in zip(observed, reconstructed)]
        mse = sum(error * error for error in errors) / count
        max_abs_error = max((abs(error) for error in errors), default=0.0)
        passed = max_abs_error <= tolerance
        verification = KineticVerification(
            mse=mse,
            max_abs_error=max_abs_error,
            tolerance=tolerance,
            passed=passed,
            checksum_sha256=_checksum(reconstructed),
        )

        if passed:
            self._committed_world = tuple(reconstructed)
            self._committed_shape = shape
            self._epoch += 1

        latent_values = len(coarse_latent) + len(fine_corrections)
        value_compression_ratio = count / max(1, latent_values)
        estimated_bytes_moved = 8 * (
            len(observed)
            + len(prediction)
            + len(residual)
            + len(reconstructed)
            + latent_values
        )
        telemetry = KineticTelemetry(
            cycles=len(ir),
            total_cells=count,
            active_cells=len(active_indices),
            active_fraction=len(active_indices) / count,
            coarse_latent_cells=len(coarse_latent),
            fine_corrections=len(fine_corrections),
            latent_values=latent_values,
            value_compression_ratio=value_compression_ratio,
            estimated_bytes_moved=estimated_bytes_moved,
            elapsed_ms=(perf_counter() - started) * 1000.0,
            backend=selected_backend.name,
        )

        return Kinetic3DResult(
            shape=shape,
            prediction=tuple(prediction),
            residual=tuple(residual),
            active_indices=active_indices,
            coarse_latent=tuple(coarse_latent),
            fine_corrections=tuple(fine_corrections),
            reconstructed=tuple(reconstructed),
            schedule=tuple(schedule),
            spatial_ir=ir,
            verification=verification,
            telemetry=telemetry,
            committed=passed,
            epoch_before=epoch_before,
            epoch_after=self._epoch,
        )
