from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

Shape3D = tuple[int, int, int]
Block3D = tuple[int, int, int]


class BackendUnavailable(ValueError):
    """Raised when an explicitly requested execution backend cannot be loaded."""


@dataclass(frozen=True)
class BackendStepResult:
    residual: tuple[float, ...]
    active_indices: tuple[int, ...]
    coarse_values: tuple[tuple[Block3D, float], ...]
    fine_corrections: tuple[tuple[int, float], ...]
    reconstructed: tuple[float, ...]


class KineticBackend(Protocol):
    name: str

    def step(
        self,
        current: Sequence[float],
        prediction: Sequence[float],
        shape: Shape3D,
        *,
        active_threshold: float,
        coarse_factor: int,
        refine_threshold: float,
    ) -> BackendStepResult: ...


def _xyz(shape: Shape3D, index: int) -> tuple[int, int, int]:
    sx, sy, _ = shape
    plane = sx * sy
    z, rem = divmod(index, plane)
    y, x = divmod(rem, sx)
    return x, y, z


def _block_for(shape: Shape3D, index: int, factor: int) -> Block3D:
    x, y, z = _xyz(shape, index)
    return x // factor, y // factor, z // factor


class ReferenceBackend:
    """Pure Python semantics reference used as the correctness oracle."""

    name = "cpu-reference"

    def step(
        self,
        current: Sequence[float],
        prediction: Sequence[float],
        shape: Shape3D,
        *,
        active_threshold: float,
        coarse_factor: int,
        refine_threshold: float,
    ) -> BackendStepResult:
        residual = [value - predicted for value, predicted in zip(current, prediction)]
        active_indices = tuple(
            index for index, error in enumerate(residual) if abs(error) > active_threshold
        )

        grouped: dict[Block3D, list[int]] = {}
        for index in active_indices:
            grouped.setdefault(_block_for(shape, index, coarse_factor), []).append(index)

        coarse_values: dict[Block3D, float] = {}
        for block in sorted(grouped):
            indices = grouped[block]
            coarse_values[block] = sum(residual[index] for index in indices) / len(indices)

        reconstructed = list(prediction)
        fine_corrections: list[tuple[int, float]] = []
        for index in active_indices:
            block = _block_for(shape, index, coarse_factor)
            coarse = coarse_values[block]
            reconstructed[index] += coarse
            correction = residual[index] - coarse
            if abs(correction) > refine_threshold:
                reconstructed[index] += correction
                fine_corrections.append((index, correction))

        return BackendStepResult(
            residual=tuple(residual),
            active_indices=active_indices,
            coarse_values=tuple((block, coarse_values[block]) for block in sorted(coarse_values)),
            fine_corrections=tuple(fine_corrections),
            reconstructed=tuple(reconstructed),
        )


class NativeBackend:
    """ctypes bridge to the compiled C++ active-volume backend."""

    name = "native-cpu"

    def __init__(self, library_path: str | os.PathLike[str]) -> None:
        self.library_path = str(library_path)
        try:
            self._library = ctypes.CDLL(self.library_path)
        except OSError as exc:
            raise BackendUnavailable(f"unable to load native backend: {self.library_path}") from exc

        function = self._library.jx_kinetic3d_step
        double_ptr = ctypes.POINTER(ctypes.c_double)
        byte_ptr = ctypes.POINTER(ctypes.c_uint8)
        function.argtypes = [
            double_ptr,
            double_ptr,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.c_double,
            double_ptr,
            double_ptr,
            byte_ptr,
            double_ptr,
            double_ptr,
        ]
        function.restype = ctypes.c_int
        self._step_function = function

    def step(
        self,
        current: Sequence[float],
        prediction: Sequence[float],
        shape: Shape3D,
        *,
        active_threshold: float,
        coarse_factor: int,
        refine_threshold: float,
    ) -> BackendStepResult:
        count = len(current)
        current_array = (ctypes.c_double * count)(*current)
        prediction_array = (ctypes.c_double * count)(*prediction)
        residual_array = (ctypes.c_double * count)()
        reconstructed_array = (ctypes.c_double * count)()
        active_mask = (ctypes.c_uint8 * count)()
        coarse_per_cell = (ctypes.c_double * count)()
        fine_per_cell = (ctypes.c_double * count)()
        sx, sy, sz = shape

        status = self._step_function(
            current_array,
            prediction_array,
            count,
            sx,
            sy,
            sz,
            active_threshold,
            coarse_factor,
            refine_threshold,
            residual_array,
            reconstructed_array,
            active_mask,
            coarse_per_cell,
            fine_per_cell,
        )
        if status != 0:
            raise RuntimeError(f"native kinetic backend failed with status {status}")

        active_indices = tuple(index for index in range(count) if active_mask[index])
        coarse_values: dict[Block3D, float] = {}
        for index in active_indices:
            block = _block_for(shape, index, coarse_factor)
            coarse_values.setdefault(block, coarse_per_cell[index])

        fine_corrections = tuple(
            (index, fine_per_cell[index])
            for index in active_indices
            if fine_per_cell[index] != 0.0
        )
        return BackendStepResult(
            residual=tuple(residual_array[index] for index in range(count)),
            active_indices=active_indices,
            coarse_values=tuple((block, coarse_values[block]) for block in sorted(coarse_values)),
            fine_corrections=fine_corrections,
            reconstructed=tuple(reconstructed_array[index] for index in range(count)),
        )


def _native_library_candidate() -> Path | None:
    configured = os.environ.get("JARVISX_KINETIC3D_NATIVE_LIB")
    if configured:
        return Path(configured)
    return None


def native_backend_available() -> bool:
    candidate = _native_library_candidate()
    if candidate is None or not candidate.is_file():
        return False
    try:
        NativeBackend(candidate)
    except BackendUnavailable:
        return False
    return True


def available_backends() -> tuple[str, ...]:
    names = [ReferenceBackend.name]
    if native_backend_available():
        names.append(NativeBackend.name)
    return tuple(names)


def resolve_backend(name: str) -> KineticBackend:
    normalized = name.strip().lower()
    if normalized == "cpu-reference":
        return ReferenceBackend()
    if normalized == "native-cpu":
        candidate = _native_library_candidate()
        if candidate is None or not candidate.is_file():
            raise BackendUnavailable(
                "native-cpu backend requested but JARVISX_KINETIC3D_NATIVE_LIB is unavailable"
            )
        return NativeBackend(candidate)
    if normalized == "auto":
        candidate = _native_library_candidate()
        if candidate is not None and candidate.is_file():
            try:
                return NativeBackend(candidate)
            except BackendUnavailable:
                pass
        return ReferenceBackend()
    raise ValueError("backend must be one of: auto, cpu-reference, native-cpu")
