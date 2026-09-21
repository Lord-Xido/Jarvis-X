from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from struct import Struct, pack, unpack_from
from typing import Sequence

from .kinetic3d_backend import ReferenceBackend

Shape3D = tuple[int, int, int]
Block3D = tuple[int, int, int]

MAGIC = b"JXK2"
VERSION = 1
MAX_CAPSULE_VOXELS = 1_048_576
_HEADER = Struct("<4sBBBBIIIdddIIII32s")
_CHECKSUM_BYTES = 32


class CapsuleError(ValueError):
    """Raised when a kinetic capsule is malformed or cannot be verified."""


@dataclass(frozen=True)
class PlanCandidate:
    active_threshold: float
    coarse_factor: int
    refine_threshold: float
    max_abs_error: float
    mse: float
    active_cells: int
    coarse_values: int
    fine_corrections: int
    capsule_bytes: int
    raw_bytes: int

    @property
    def wire_compression_ratio(self) -> float:
        return self.raw_bytes / self.capsule_bytes

    def as_payload(self) -> dict[str, object]:
        return {
            "active_threshold": self.active_threshold,
            "coarse_factor": self.coarse_factor,
            "refine_threshold": self.refine_threshold,
            "max_abs_error": self.max_abs_error,
            "mse": self.mse,
            "active_cells": self.active_cells,
            "coarse_values": self.coarse_values,
            "fine_corrections": self.fine_corrections,
            "capsule_bytes": self.capsule_bytes,
            "raw_bytes": self.raw_bytes,
            "wire_compression_ratio": self.wire_compression_ratio,
        }


@dataclass(frozen=True)
class AdaptivePlan:
    tolerance: float
    selected: PlanCandidate
    candidates: tuple[PlanCandidate, ...]

    def as_payload(self) -> dict[str, object]:
        return {
            "tolerance": self.tolerance,
            "selected": self.selected.as_payload(),
            "candidate_count": len(self.candidates),
            "candidates": [candidate.as_payload() for candidate in self.candidates],
        }


@dataclass(frozen=True)
class KineticCapsule:
    shape: Shape3D
    active_threshold: float
    coarse_factor: int
    refine_threshold: float
    tolerance: float
    active_indices: tuple[int, ...]
    coarse_values: tuple[tuple[Block3D, float], ...]
    fine_corrections: tuple[tuple[int, float], ...]
    predictor_checksum_sha256: str
    checksum_sha256: str

    def decode(self, prediction: Sequence[float]) -> tuple[float, ...]:
        count = _validate_shape(self.shape)
        predictor = _validate_vector("prediction", prediction, count)
        if _checksum_f64(predictor) != self.predictor_checksum_sha256:
            raise CapsuleError("prediction checksum does not match capsule predictor")

        coarse_map = dict(self.coarse_values)
        reconstructed = list(predictor)
        active_set = set(self.active_indices)
        for index in self.active_indices:
            block = _block_for(self.shape, index, self.coarse_factor)
            try:
                reconstructed[index] += coarse_map[block]
            except KeyError as exc:
                raise CapsuleError("active index references a missing coarse block") from exc

        seen_fine: set[int] = set()
        for index, correction in self.fine_corrections:
            if index not in active_set:
                raise CapsuleError("fine correction references an inactive index")
            if index in seen_fine:
                raise CapsuleError("duplicate fine correction index")
            seen_fine.add(index)
            reconstructed[index] += correction

        return tuple(reconstructed)

    def as_payload(self) -> dict[str, object]:
        return {
            "format": MAGIC.decode("ascii"),
            "version": VERSION,
            "shape": list(self.shape),
            "active_threshold": self.active_threshold,
            "coarse_factor": self.coarse_factor,
            "refine_threshold": self.refine_threshold,
            "tolerance": self.tolerance,
            "active_cells": len(self.active_indices),
            "coarse_values": len(self.coarse_values),
            "fine_corrections": len(self.fine_corrections),
            "predictor_checksum_sha256": self.predictor_checksum_sha256,
            "checksum_sha256": self.checksum_sha256,
        }


def _voxel_count(shape: Shape3D) -> int:
    sx, sy, sz = shape
    return sx * sy * sz


def _validate_shape(shape: Shape3D) -> int:
    if len(shape) != 3 or any(not isinstance(dimension, int) or dimension < 1 for dimension in shape):
        raise ValueError("shape must contain exactly three positive integer dimensions")
    count = _voxel_count(shape)
    if count > MAX_CAPSULE_VOXELS:
        raise ValueError(f"voxel count {count} exceeds capsule limit {MAX_CAPSULE_VOXELS}")
    return count


def _validate_vector(name: str, values: Sequence[float], count: int) -> list[float]:
    if len(values) != count:
        raise ValueError(f"{name} length {len(values)} does not match voxel count {count}")
    result = [float(value) for value in values]
    if not all(isfinite(value) for value in result):
        raise ValueError(f"{name} values must all be finite")
    return result


def _xyz(shape: Shape3D, index: int) -> tuple[int, int, int]:
    sx, sy, _ = shape
    plane = sx * sy
    z, rem = divmod(index, plane)
    y, x = divmod(rem, sx)
    return x, y, z


def _block_for(shape: Shape3D, index: int, factor: int) -> Block3D:
    x, y, z = _xyz(shape, index)
    return x // factor, y // factor, z // factor


def _checksum_f64(values: Sequence[float]) -> str:
    digest = sha256()
    for value in values:
        digest.update(pack("<d", value))
    return digest.hexdigest()


def _encode_uvarint(value: int) -> bytes:
    if value < 0:
        raise ValueError("uvarint value must be non-negative")
    output = bytearray()
    while value >= 0x80:
        output.append((value & 0x7F) | 0x80)
        value >>= 7
    output.append(value)
    return bytes(output)


def _decode_uvarint(data: bytes, offset: int, limit: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while offset < limit:
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7
        if shift > 63:
            raise CapsuleError("uvarint exceeds 64-bit range")
    raise CapsuleError("truncated uvarint")


def _active_runs(indices: Sequence[int]) -> tuple[tuple[int, int], ...]:
    if not indices:
        return ()
    runs: list[tuple[int, int]] = []
    start = previous = indices[0]
    for index in indices[1:]:
        if index == previous + 1:
            previous = index
            continue
        runs.append((start, previous - start + 1))
        start = previous = index
    runs.append((start, previous - start + 1))
    return tuple(runs)


def build_capsule(
    *,
    shape: Shape3D,
    prediction: Sequence[float],
    active_threshold: float,
    coarse_factor: int,
    refine_threshold: float,
    tolerance: float,
    active_indices: Sequence[int],
    coarse_values: Sequence[tuple[Block3D, float]],
    fine_corrections: Sequence[tuple[int, float]],
) -> bytes:
    count = _validate_shape(shape)
    predictor = _validate_vector("prediction", prediction, count)
    if not isfinite(active_threshold) or active_threshold < 0:
        raise ValueError("active_threshold must be finite and non-negative")
    if not 1 <= coarse_factor <= 32:
        raise ValueError("coarse_factor must be between 1 and 32")
    if not isfinite(refine_threshold) or refine_threshold < 0:
        raise ValueError("refine_threshold must be finite and non-negative")
    if not isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")

    active = tuple(int(index) for index in active_indices)
    if any(index < 0 or index >= count for index in active):
        raise ValueError("active index is outside the volume")
    if tuple(sorted(set(active))) != active:
        raise ValueError("active indices must be strictly increasing and unique")

    coarse = tuple(sorted(((tuple(block), float(value)) for block, value in coarse_values)))
    coarse_blocks = [block for block, _ in coarse]
    if len(set(coarse_blocks)) != len(coarse_blocks):
        raise ValueError("coarse blocks must be unique")
    sx, sy, sz = shape
    bx_count = (sx + coarse_factor - 1) // coarse_factor
    by_count = (sy + coarse_factor - 1) // coarse_factor
    bz_count = (sz + coarse_factor - 1) // coarse_factor
    for block, value in coarse:
        if len(block) != 3 or not all(isinstance(axis, int) for axis in block):
            raise ValueError("coarse block must contain three integer coordinates")
        bx, by, bz = block
        if not (0 <= bx < bx_count and 0 <= by < by_count and 0 <= bz < bz_count):
            raise ValueError("coarse block lies outside the volume")
        if not isfinite(value):
            raise ValueError("coarse values must be finite")

    fine = tuple((int(index), float(value)) for index, value in fine_corrections)
    if tuple(sorted(index for index, _ in fine)) != tuple(index for index, _ in fine):
        raise ValueError("fine correction indices must be sorted")
    if len({index for index, _ in fine}) != len(fine):
        raise ValueError("fine correction indices must be unique")
    active_set = set(active)
    for index, value in fine:
        if index not in active_set:
            raise ValueError("fine correction references an inactive index")
        if not isfinite(value):
            raise ValueError("fine corrections must be finite")

    required_blocks = {_block_for(shape, index, coarse_factor) for index in active}
    if required_blocks != set(coarse_blocks):
        raise ValueError("coarse block set must exactly cover active indices")

    runs = _active_runs(active)
    predictor_digest = bytes.fromhex(_checksum_f64(predictor))
    body = bytearray(
        _HEADER.pack(
            MAGIC,
            VERSION,
            0,
            coarse_factor,
            0,
            sx,
            sy,
            sz,
            active_threshold,
            refine_threshold,
            tolerance,
            len(active),
            len(runs),
            len(coarse),
            len(fine),
            predictor_digest,
        )
    )

    previous_end = -1
    for start, length in runs:
        baseline = previous_end + 1
        body.extend(_encode_uvarint(start - baseline))
        body.extend(_encode_uvarint(length))
        previous_end = start + length - 1

    for (bx, by, bz), value in coarse:
        body.extend(_encode_uvarint(bx))
        body.extend(_encode_uvarint(by))
        body.extend(_encode_uvarint(bz))
        body.extend(pack("<d", value))

    previous_fine = -1
    for index, correction in fine:
        body.extend(_encode_uvarint(index - previous_fine - 1))
        body.extend(pack("<d", correction))
        previous_fine = index

    digest = sha256(body).digest()
    body.extend(digest)
    return bytes(body)


def parse_capsule(data: bytes) -> KineticCapsule:
    if len(data) < _HEADER.size + _CHECKSUM_BYTES:
        raise CapsuleError("capsule is truncated")
    payload_limit = len(data) - _CHECKSUM_BYTES
    expected_digest = data[payload_limit:]
    actual_digest = sha256(data[:payload_limit]).digest()
    if actual_digest != expected_digest:
        raise CapsuleError("capsule checksum mismatch")

    (
        magic,
        version,
        flags,
        coarse_factor,
        reserved,
        sx,
        sy,
        sz,
        active_threshold,
        refine_threshold,
        tolerance,
        active_count,
        run_count,
        coarse_count,
        fine_count,
        predictor_digest,
    ) = _HEADER.unpack_from(data, 0)
    if magic != MAGIC:
        raise CapsuleError("unknown capsule magic")
    if version != VERSION:
        raise CapsuleError(f"unsupported capsule version {version}")
    if flags != 0 or reserved != 0:
        raise CapsuleError("unsupported capsule flags")
    if not 1 <= coarse_factor <= 32:
        raise CapsuleError("invalid coarse factor")
    if not all(isfinite(value) and value >= 0 for value in (active_threshold, refine_threshold, tolerance)):
        raise CapsuleError("invalid threshold metadata")

    shape: Shape3D = (sx, sy, sz)
    count = _validate_shape(shape)
    if active_count > count or run_count > active_count or fine_count > active_count:
        raise CapsuleError("capsule count metadata exceeds volume bounds")

    offset = _HEADER.size
    active: list[int] = []
    previous_end = -1
    for _ in range(run_count):
        start_delta, offset = _decode_uvarint(data, offset, payload_limit)
        length, offset = _decode_uvarint(data, offset, payload_limit)
        if length < 1:
            raise CapsuleError("active run length must be positive")
        start = previous_end + 1 + start_delta
        end = start + length - 1
        if start <= previous_end or end >= count:
            raise CapsuleError("active run is outside the volume or overlaps")
        active.extend(range(start, end + 1))
        previous_end = end
    if len(active) != active_count:
        raise CapsuleError("active run count does not match active cell count")

    coarse: list[tuple[Block3D, float]] = []
    bx_count = (sx + coarse_factor - 1) // coarse_factor
    by_count = (sy + coarse_factor - 1) // coarse_factor
    bz_count = (sz + coarse_factor - 1) // coarse_factor
    for _ in range(coarse_count):
        bx, offset = _decode_uvarint(data, offset, payload_limit)
        by, offset = _decode_uvarint(data, offset, payload_limit)
        bz, offset = _decode_uvarint(data, offset, payload_limit)
        if offset + 8 > payload_limit:
            raise CapsuleError("truncated coarse value")
        (value,) = unpack_from("<d", data, offset)
        offset += 8
        if not isfinite(value):
            raise CapsuleError("non-finite coarse value")
        if not (bx < bx_count and by < by_count and bz < bz_count):
            raise CapsuleError("coarse block lies outside the volume")
        coarse.append(((bx, by, bz), value))
    coarse_blocks = [block for block, _ in coarse]
    if len(set(coarse_blocks)) != len(coarse_blocks):
        raise CapsuleError("duplicate coarse block")

    fine: list[tuple[int, float]] = []
    previous_fine = -1
    active_set = set(active)
    for _ in range(fine_count):
        delta, offset = _decode_uvarint(data, offset, payload_limit)
        index = previous_fine + 1 + delta
        if offset + 8 > payload_limit:
            raise CapsuleError("truncated fine correction")
        (correction,) = unpack_from("<d", data, offset)
        offset += 8
        if index <= previous_fine or index not in active_set:
            raise CapsuleError("invalid fine correction index")
        if not isfinite(correction):
            raise CapsuleError("non-finite fine correction")
        fine.append((index, correction))
        previous_fine = index

    if offset != payload_limit:
        raise CapsuleError("capsule contains trailing payload bytes")

    required_blocks = {_block_for(shape, index, coarse_factor) for index in active}
    if required_blocks != set(coarse_blocks):
        raise CapsuleError("coarse blocks do not exactly cover active indices")

    return KineticCapsule(
        shape=shape,
        active_threshold=active_threshold,
        coarse_factor=coarse_factor,
        refine_threshold=refine_threshold,
        tolerance=tolerance,
        active_indices=tuple(active),
        coarse_values=tuple(coarse),
        fine_corrections=tuple(fine),
        predictor_checksum_sha256=predictor_digest.hex(),
        checksum_sha256=expected_digest.hex(),
    )


def decode_capsule(data: bytes, prediction: Sequence[float]) -> tuple[float, ...]:
    return parse_capsule(data).decode(prediction)


def _candidate_thresholds(tolerance: float) -> tuple[float, ...]:
    if tolerance == 0.0:
        return (0.0,)
    return tuple(sorted({0.0, tolerance * 0.25, tolerance * 0.5, tolerance}))


def plan_rate_distortion(
    current: Sequence[float],
    prediction: Sequence[float],
    shape: Shape3D,
    *,
    tolerance: float,
    coarse_factors: Sequence[int] = (1, 2, 4, 8),
) -> AdaptivePlan:
    count = _validate_shape(shape)
    observed = _validate_vector("current", current, count)
    predictor = _validate_vector("prediction", prediction, count)
    if not isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")

    factors = tuple(sorted(set(int(factor) for factor in coarse_factors)))
    if not factors or any(not 1 <= factor <= 32 for factor in factors):
        raise ValueError("coarse_factors must contain values between 1 and 32")

    backend = ReferenceBackend()
    thresholds = _candidate_thresholds(tolerance)
    raw_bytes = count * 8
    candidates: list[PlanCandidate] = []

    for active_threshold in thresholds:
        for coarse_factor in factors:
            for refine_threshold in thresholds:
                step = backend.step(
                    observed,
                    predictor,
                    shape,
                    active_threshold=active_threshold,
                    coarse_factor=coarse_factor,
                    refine_threshold=refine_threshold,
                )
                errors = [source - target for source, target in zip(observed, step.reconstructed)]
                mse = sum(error * error for error in errors) / count
                max_abs_error = max((abs(error) for error in errors), default=0.0)
                if max_abs_error > tolerance:
                    continue
                capsule = build_capsule(
                    shape=shape,
                    prediction=predictor,
                    active_threshold=active_threshold,
                    coarse_factor=coarse_factor,
                    refine_threshold=refine_threshold,
                    tolerance=tolerance,
                    active_indices=step.active_indices,
                    coarse_values=step.coarse_values,
                    fine_corrections=step.fine_corrections,
                )
                candidates.append(
                    PlanCandidate(
                        active_threshold=active_threshold,
                        coarse_factor=coarse_factor,
                        refine_threshold=refine_threshold,
                        max_abs_error=max_abs_error,
                        mse=mse,
                        active_cells=len(step.active_indices),
                        coarse_values=len(step.coarse_values),
                        fine_corrections=len(step.fine_corrections),
                        capsule_bytes=len(capsule),
                        raw_bytes=raw_bytes,
                    )
                )

    if not candidates:
        raise RuntimeError("no rate-distortion candidate satisfies the requested tolerance")

    ordered = tuple(
        sorted(
            candidates,
            key=lambda candidate: (
                candidate.capsule_bytes,
                candidate.max_abs_error,
                candidate.active_cells,
                candidate.fine_corrections,
                candidate.coarse_values,
                candidate.coarse_factor,
                candidate.refine_threshold,
                candidate.active_threshold,
            ),
        )
    )
    return AdaptivePlan(tolerance=tolerance, selected=ordered[0], candidates=ordered)
