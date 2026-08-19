"""Operational 3D spatial codec and telemetry for the Dr Moagi field runtime.

The module keeps spatial execution explicit. Integer ``(x, y, z)`` lattice
coordinates are encoded into reversible Morton (Z-order) addresses and scalar
field values are quantized independently. The resulting latent representation
therefore retains a deterministic spatial address instead of treating the 3D
coordinates as UI-only metadata.

This is a bounded reference implementation. It does not claim that Morton
ordering is an optimal learned latent geometry or that the reported packed-byte
estimate is Python object memory usage.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .auto_codec_loop import AutoCodecLoop, AutoCodecRunSummary, digest_field
from .dr_moagi_field_runtime import Coordinate, SparseField, Validator

MORTON_BITS_PER_AXIS = 21
MORTON_MAX_COORDINATE = (1 << MORTON_BITS_PER_AXIS) - 1


def _coordinate_component(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0 or value > MORTON_MAX_COORDINATE:
        raise ValueError(
            f"{name} must be within [0, {MORTON_MAX_COORDINATE}] for the 63-bit Morton key"
        )
    return value


def morton_encode_3d(x: int, y: int, z: int) -> int:
    """Interleave 21 bits from each axis into one reversible 63-bit key."""

    x = _coordinate_component(x, "x")
    y = _coordinate_component(y, "y")
    z = _coordinate_component(z, "z")
    code = 0
    for bit in range(MORTON_BITS_PER_AXIS):
        code |= ((x >> bit) & 1) << (3 * bit)
        code |= ((y >> bit) & 1) << (3 * bit + 1)
        code |= ((z >> bit) & 1) << (3 * bit + 2)
    return code


def morton_decode_3d(code: int) -> Coordinate:
    """Recover ``(x, y, z)`` from a 63-bit Morton key."""

    if isinstance(code, bool) or not isinstance(code, int):
        raise TypeError("Morton code must be an integer")
    if code < 0 or code >= 1 << (3 * MORTON_BITS_PER_AXIS):
        raise ValueError("Morton code is outside the supported 63-bit range")
    x = y = z = 0
    for bit in range(MORTON_BITS_PER_AXIS):
        x |= ((code >> (3 * bit)) & 1) << bit
        y |= ((code >> (3 * bit + 1)) & 1) << bit
        z |= ((code >> (3 * bit + 2)) & 1) << bit
    return x, y, z


@dataclass(frozen=True)
class MortonQuantizedLatent3D:
    """Sparse 3D latent: reversible Morton address -> signed quantized code."""

    step: float
    entries: dict[int, int]


class MortonQuantizedFieldCodec3D:
    """Deterministic spatial codec implementing the field-runtime codec contract."""

    def __init__(
        self,
        step: float = 0.05,
        *,
        side: int = 1000,
        prune_zero_codes: bool = True,
    ) -> None:
        if isinstance(step, bool) or not isinstance(step, (int, float)):
            raise TypeError("step must be numeric")
        step = float(step)
        if not math.isfinite(step) or step <= 0.0:
            raise ValueError("step must be finite and positive")
        if isinstance(side, bool) or not isinstance(side, int) or side <= 0:
            raise ValueError("side must be a positive integer")
        if side - 1 > MORTON_MAX_COORDINATE:
            raise ValueError("side exceeds the supported Morton coordinate range")
        self.step = step
        self.side = side
        self.prune_zero_codes = bool(prune_zero_codes)

    def encode(self, field: Mapping[Coordinate, float]) -> MortonQuantizedLatent3D:
        entries: dict[int, int] = {}
        for coordinate, raw_value in field.items():
            self._validate_coordinate(coordinate)
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise TypeError("field values must be numeric")
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError("field values must be finite")
            code = int(round(value / self.step))
            if code != 0 or not self.prune_zero_codes:
                key = morton_encode_3d(*coordinate)
                if key in entries:
                    raise RuntimeError("Morton address collision detected")
                entries[key] = code
        return MortonQuantizedLatent3D(step=self.step, entries=entries)

    def decode(
        self,
        latent: MortonQuantizedLatent3D,
        support: Sequence[Coordinate],
    ) -> Mapping[Coordinate, float]:
        if not isinstance(latent, MortonQuantizedLatent3D):
            raise TypeError("latent must be a MortonQuantizedLatent3D")
        if latent.step != self.step:
            raise ValueError("latent quantization step does not match codec")
        result: SparseField = {}
        for coordinate in support:
            self._validate_coordinate(coordinate)
            key = morton_encode_3d(*coordinate)
            result[coordinate] = float(latent.entries.get(key, 0)) * self.step
        return result

    def _validate_coordinate(self, coordinate: Coordinate) -> Coordinate:
        if (
            not isinstance(coordinate, tuple)
            or len(coordinate) != 3
            or any(isinstance(value, bool) or not isinstance(value, int) for value in coordinate)
        ):
            raise TypeError("coordinates must be integer (x, y, z) tuples")
        if any(value < 0 or value >= self.side for value in coordinate):
            raise ValueError("coordinate is outside the configured 3D side length")
        return coordinate


@dataclass(frozen=True)
class SpatialMetrics3D:
    active_cells: int
    bounds_min: Coordinate | None
    bounds_max: Coordinate | None
    centroid: tuple[float, float, float] | None
    weighted_centroid: tuple[float, float, float] | None
    rms_radius: float
    l1_energy: float
    l2_energy: float
    six_face_links: int
    occupancy_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_cells": self.active_cells,
            "bounds_min": list(self.bounds_min) if self.bounds_min is not None else None,
            "bounds_max": list(self.bounds_max) if self.bounds_max is not None else None,
            "centroid": list(self.centroid) if self.centroid is not None else None,
            "weighted_centroid": (
                list(self.weighted_centroid) if self.weighted_centroid is not None else None
            ),
            "rms_radius": self.rms_radius,
            "l1_energy": self.l1_energy,
            "l2_energy": self.l2_energy,
            "six_face_links": self.six_face_links,
            "occupancy_ratio": self.occupancy_ratio,
        }


def measure_spatial_field(field: Mapping[Coordinate, float], *, side: int) -> SpatialMetrics3D:
    """Measure geometry directly from the active sparse 3D state."""

    if isinstance(side, bool) or not isinstance(side, int) or side <= 0:
        raise ValueError("side must be a positive integer")
    if not field:
        return SpatialMetrics3D(0, None, None, None, None, 0.0, 0.0, 0.0, 0, 0.0)

    coordinates = tuple(field)
    for coordinate in coordinates:
        if (
            not isinstance(coordinate, tuple)
            or len(coordinate) != 3
            or any(isinstance(value, bool) or not isinstance(value, int) for value in coordinate)
        ):
            raise TypeError("coordinates must be integer (x, y, z) tuples")
        if any(value < 0 or value >= side for value in coordinate):
            raise ValueError("coordinate is outside the configured 3D side length")

    xs = tuple(coordinate[0] for coordinate in coordinates)
    ys = tuple(coordinate[1] for coordinate in coordinates)
    zs = tuple(coordinate[2] for coordinate in coordinates)
    count = len(coordinates)
    centroid = (sum(xs) / count, sum(ys) / count, sum(zs) / count)

    weights = tuple(abs(float(field[coordinate])) for coordinate in coordinates)
    if not all(math.isfinite(weight) for weight in weights):
        raise ValueError("field values must be finite")
    weight_sum = sum(weights)
    weighted_centroid = (
        (
            sum(coordinate[0] * weight for coordinate, weight in zip(coordinates, weights))
            / weight_sum,
            sum(coordinate[1] * weight for coordinate, weight in zip(coordinates, weights))
            / weight_sum,
            sum(coordinate[2] * weight for coordinate, weight in zip(coordinates, weights))
            / weight_sum,
        )
        if weight_sum > 0.0
        else centroid
    )

    rms_radius = math.sqrt(
        sum(
            (coordinate[0] - centroid[0]) ** 2
            + (coordinate[1] - centroid[1]) ** 2
            + (coordinate[2] - centroid[2]) ** 2
            for coordinate in coordinates
        )
        / count
    )
    values = tuple(float(field[coordinate]) for coordinate in coordinates)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("field values must be finite")
    l1_energy = sum(abs(value) for value in values)
    l2_energy = sum(value * value for value in values)

    support = set(coordinates)
    links = 0
    for x, y, z in coordinates:
        for neighbor in ((x + 1, y, z), (x, y + 1, z), (x, y, z + 1)):
            if neighbor in support:
                links += 1

    return SpatialMetrics3D(
        active_cells=count,
        bounds_min=(min(xs), min(ys), min(zs)),
        bounds_max=(max(xs), max(ys), max(zs)),
        centroid=centroid,
        weighted_centroid=weighted_centroid,
        rms_radius=rms_radius,
        l1_energy=l1_energy,
        l2_energy=l2_energy,
        six_face_links=links,
        occupancy_ratio=count / float(side**3),
    )


@dataclass(frozen=True)
class SpatialFrame3D:
    cycle: int
    state_digest: str
    metrics: SpatialMetrics3D
    points: tuple[dict[str, int | float], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle": self.cycle,
            "state_digest": self.state_digest,
            "metrics": self.metrics.to_dict(),
            "points": list(self.points),
        }


def spatial_frame(
    field: Mapping[Coordinate, float],
    *,
    side: int,
    cycle: int,
    max_render_points: int = 4096,
) -> SpatialFrame3D:
    """Create a deterministic bounded point-cloud frame from authoritative state."""

    if isinstance(max_render_points, bool) or not isinstance(max_render_points, int):
        raise TypeError("max_render_points must be an integer")
    if max_render_points <= 0:
        raise ValueError("max_render_points must be positive")

    ordered = sorted(field.items(), key=lambda item: morton_encode_3d(*item[0]))
    if len(ordered) > max_render_points:
        stride = math.ceil(len(ordered) / max_render_points)
        ordered = ordered[::stride][:max_render_points]
    points = tuple(
        {"x": x, "y": y, "z": z, "value": float(value)}
        for (x, y, z), value in ordered
    )
    return SpatialFrame3D(
        cycle=cycle,
        state_digest=digest_field(field),
        metrics=measure_spatial_field(field, side=side),
        points=points,
    )


def digest_latent(latent: MortonQuantizedLatent3D) -> str:
    payload = {
        "step": latent.step,
        "entries": [[key, value] for key, value in sorted(latent.entries.items())],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class SpatialAutoCodec3DSummary:
    """Machine-readable result for one bounded spatial auto-codec execution."""

    codec: str
    latent_entries: int
    latent_digest: str
    packed_latent_bytes_estimate: int
    input_frame: SpatialFrame3D
    final_frame: SpatialFrame3D
    frames: tuple[SpatialFrame3D, ...]
    loop: AutoCodecRunSummary

    def to_dict(self) -> dict[str, Any]:
        result = self.loop.to_dict()
        result.update(
            {
                "spatial_mode": "3d-morton-quantized",
                "codec": self.codec,
                "latent_entries": self.latent_entries,
                "latent_digest": self.latent_digest,
                "packed_latent_bytes_estimate": self.packed_latent_bytes_estimate,
                "input_frame": self.input_frame.to_dict(),
                "final_frame": self.final_frame.to_dict(),
                "frames": [frame.to_dict() for frame in self.frames],
            }
        )
        return result


class SpatialAutoCodec3DSystem:
    """Coordinate-aware 3D wrapper around the bounded auto-codec loop."""

    def __init__(
        self,
        loop: AutoCodecLoop,
        codec: MortonQuantizedFieldCodec3D,
        *,
        side: int,
        frame_stride: int = 1,
        max_render_points: int = 4096,
        max_frames: int = 128,
    ) -> None:
        if loop.runtime.codec is not codec:
            raise ValueError("loop runtime must use the same spatial codec instance")
        for name, value in (
            ("frame_stride", frame_stride),
            ("max_render_points", max_render_points),
            ("max_frames", max_frames),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if side != codec.side:
            raise ValueError("system side must match codec side")
        self.loop = loop
        self.codec = codec
        self.side = side
        self.frame_stride = frame_stride
        self.max_render_points = max_render_points
        self.max_frames = max_frames

    def run(
        self,
        field: Mapping[Coordinate, float],
        *,
        validator: Validator | None = None,
    ) -> SpatialAutoCodec3DSummary:
        self.loop.load(field)
        input_frame = spatial_frame(
            self.loop.runtime.snapshot(),
            side=self.side,
            cycle=self.loop.runtime.cycle,
            max_render_points=self.max_render_points,
        )
        frames: list[SpatialFrame3D] = [input_frame]

        def capture(cycle: int, state: SparseField) -> None:
            if cycle % self.frame_stride != 0:
                return
            frame = spatial_frame(
                state,
                side=self.side,
                cycle=cycle,
                max_render_points=self.max_render_points,
            )
            if len(frames) < self.max_frames:
                frames.append(frame)
            elif frames:
                frames[-1] = frame

        summary = self.loop.run(validator=validator, on_cycle=capture)
        final_state = self.loop.runtime.snapshot()
        final_frame = spatial_frame(
            final_state,
            side=self.side,
            cycle=self.loop.runtime.cycle,
            max_render_points=self.max_render_points,
        )
        if frames[-1].state_digest != final_frame.state_digest:
            if len(frames) < self.max_frames:
                frames.append(final_frame)
            else:
                frames[-1] = final_frame

        latent = self.codec.encode(final_state)
        # A portable packed record would use uint64 Morton key + int32 value code.
        packed_bytes = len(latent.entries) * 12
        return SpatialAutoCodec3DSummary(
            codec="MortonQuantizedFieldCodec3D",
            latent_entries=len(latent.entries),
            latent_digest=digest_latent(latent),
            packed_latent_bytes_estimate=packed_bytes,
            input_frame=input_frame,
            final_frame=final_frame,
            frames=tuple(frames),
            loop=summary,
        )
