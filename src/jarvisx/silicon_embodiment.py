"""Finite reference for silicon embodiment plus 3D inward encode/decode.

The module deliberately separates two ideas:

* materialize_geometry is a small abstraction of a slow geometry-to-material
  manufacturing map. It is not a semiconductor device simulator.
* run_cycle is a bounded runtime encode, inward fixed point, decode, contrast,
  and Omega update over a finite 3D scalar field.

The reference encoder preserves an explicit residual shell, so decoding the
unrefined latent is exactly reconstructible up to floating-point arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Sequence


@dataclass(frozen=True)
class EncodedLevel:
    """One 2x2x2 contraction level plus residual side information."""

    source_side: int
    latent_side: int
    latent: tuple[float, ...]
    residual: tuple[float, ...]


@dataclass(frozen=True)
class FixedPointResult:
    """Result of a bounded contractive latent refinement."""

    latent: tuple[float, ...]
    iterations: int
    delta: float


@dataclass(frozen=True)
class SiliconLoopConfig:
    """Bounded parameters for one runtime cycle."""

    contraction: float = 0.5
    omega_gain: float = 0.1
    omega_decay: float = 0.9
    tolerance: float = 1e-9
    max_steps: int = 32


@dataclass(frozen=True)
class SiliconLoopResult:
    """Observable state emitted by one finite loop cycle."""

    encoded: EncodedLevel
    refined_latent: tuple[float, ...]
    reconstruction: tuple[float, ...]
    error: tuple[float, ...]
    omega_next: tuple[float, ...]
    iterations: int
    fixed_point_delta: float
    mse: float


def _cube_side(length: int) -> int:
    if length <= 0:
        raise ValueError("volume must contain at least one value")

    side = round(length ** (1.0 / 3.0))
    for candidate in range(max(1, side - 2), side + 3):
        if candidate**3 == length:
            return candidate
    raise ValueError("volume length must be a perfect cube")


def _finite(values: Sequence[float], name: str) -> tuple[float, ...]:
    out = tuple(float(value) for value in values)
    if not all(isfinite(value) for value in out):
        raise ValueError(f"{name} must contain only finite values")
    return out


def materialize_geometry(
    mask: Sequence[int], *, off_conductivity: float = 0.0, on_conductivity: float = 1.0
) -> tuple[float, ...]:
    """Map a binary design mask to a simple scalar material/conductivity field."""

    if not isfinite(off_conductivity) or not isfinite(on_conductivity):
        raise ValueError("conductivity values must be finite")

    material: list[float] = []
    for value in mask:
        if value not in (0, 1):
            raise ValueError("mask values must be binary")
        material.append(on_conductivity if value else off_conductivity)
    return tuple(material)


def _index(side: int, x: int, y: int, z: int) -> int:
    return (z * side + y) * side + x


def encode_level(volume: Sequence[float]) -> EncodedLevel:
    """Contract a cubic field by 2 in every axis and preserve residuals."""

    source = _finite(volume, "volume")
    side = _cube_side(len(source))
    if side < 2 or side % 2:
        raise ValueError("source side must be an even integer >= 2")

    latent_side = side // 2
    latent = [0.0] * (latent_side**3)
    residual = [0.0] * len(source)

    for lz in range(latent_side):
        for ly in range(latent_side):
            for lx in range(latent_side):
                coords: list[int] = []
                values: list[float] = []
                for dz in (0, 1):
                    for dy in (0, 1):
                        for dx in (0, 1):
                            x = 2 * lx + dx
                            y = 2 * ly + dy
                            z = 2 * lz + dz
                            idx = _index(side, x, y, z)
                            coords.append(idx)
                            values.append(source[idx])

                mean = sum(values) / 8.0
                latent_idx = _index(latent_side, lx, ly, lz)
                latent[latent_idx] = mean
                for idx, value in zip(coords, values):
                    residual[idx] = value - mean

    return EncodedLevel(
        source_side=side,
        latent_side=latent_side,
        latent=tuple(latent),
        residual=tuple(residual),
    )


def decode_level(
    encoded: EncodedLevel, *, latent_override: Sequence[float] | None = None
) -> tuple[float, ...]:
    """Decode one level using the stored residual shell."""

    latent = encoded.latent if latent_override is None else _finite(latent_override, "latent")
    if len(latent) != encoded.latent_side**3:
        raise ValueError("latent size does not match encoded geometry")
    if len(encoded.residual) != encoded.source_side**3:
        raise ValueError("residual size does not match encoded geometry")

    out = [0.0] * len(encoded.residual)
    for lz in range(encoded.latent_side):
        for ly in range(encoded.latent_side):
            for lx in range(encoded.latent_side):
                latent_idx = _index(encoded.latent_side, lx, ly, lz)
                centre = latent[latent_idx]
                for dz in (0, 1):
                    for dy in (0, 1):
                        for dx in (0, 1):
                            x = 2 * lx + dx
                            y = 2 * ly + dy
                            z = 2 * lz + dz
                            idx = _index(encoded.source_side, x, y, z)
                            out[idx] = centre + encoded.residual[idx]
    return tuple(out)


def fixed_point_refine(
    latent: Sequence[float],
    *,
    omega: Sequence[float] | None = None,
    contraction: float = 0.5,
    omega_gain: float = 0.0,
    tolerance: float = 1e-9,
    max_steps: int = 32,
) -> FixedPointResult:
    """Apply an explicit contractive inward recurrence."""

    state = _finite(latent, "latent")
    if not state:
        raise ValueError("latent must not be empty")
    if not 0.0 <= contraction < 1.0:
        raise ValueError("contraction must satisfy 0 <= contraction < 1")
    if tolerance < 0.0 or not isfinite(tolerance):
        raise ValueError("tolerance must be finite and non-negative")
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")
    if not isfinite(omega_gain):
        raise ValueError("omega_gain must be finite")

    if omega is None:
        memory = (0.0,) * len(state)
    else:
        memory = _finite(omega, "omega")
        if len(memory) != len(state):
            raise ValueError("omega must have the same size as latent")

    centre = sum(state) / len(state)
    target = tuple(centre + omega_gain * value for value in memory)

    current = state
    delta = float("inf")
    iterations = 0
    for step in range(1, max_steps + 1):
        next_state = tuple(
            target_value + contraction * (value - target_value)
            for value, target_value in zip(current, target)
        )
        delta = max(abs(a - b) for a, b in zip(next_state, current))
        current = next_state
        iterations = step
        if delta <= tolerance:
            break

    return FixedPointResult(latent=current, iterations=iterations, delta=delta)


def _latent_block_mean(values: Sequence[float], source_side: int) -> tuple[float, ...]:
    encoded = encode_level(values)
    if encoded.source_side != source_side:
        raise ValueError("source geometry changed unexpectedly")
    return encoded.latent


def run_cycle(
    world: Sequence[float],
    *,
    omega: Sequence[float] | None = None,
    config: SiliconLoopConfig = SiliconLoopConfig(),
) -> SiliconLoopResult:
    """Run one finite 3D encode/refine/decode/contrast/memory cycle."""

    source = _finite(world, "world")
    encoded = encode_level(source)

    if not 0.0 <= config.omega_decay <= 1.0:
        raise ValueError("omega_decay must satisfy 0 <= omega_decay <= 1")

    memory = (0.0,) * len(encoded.latent) if omega is None else _finite(omega, "omega")
    if len(memory) != len(encoded.latent):
        raise ValueError("omega must have the same size as the encoded latent")

    refined = fixed_point_refine(
        encoded.latent,
        omega=memory,
        contraction=config.contraction,
        omega_gain=config.omega_gain,
        tolerance=config.tolerance,
        max_steps=config.max_steps,
    )

    reconstruction = decode_level(encoded, latent_override=refined.latent)
    error = tuple(x - x_hat for x, x_hat in zip(source, reconstruction))
    error_latent = _latent_block_mean(error, encoded.source_side)
    omega_next = tuple(
        config.omega_decay * old + (1.0 - config.omega_decay) * err
        for old, err in zip(memory, error_latent)
    )
    mse = sum(value * value for value in error) / len(error)

    return SiliconLoopResult(
        encoded=encoded,
        refined_latent=refined.latent,
        reconstruction=reconstruction,
        error=error,
        omega_next=omega_next,
        iterations=refined.iterations,
        fixed_point_delta=refined.delta,
        mse=mse,
    )
