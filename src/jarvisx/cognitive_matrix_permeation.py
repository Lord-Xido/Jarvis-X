"""Bounded Cognitive Matrix stochastic projection and permeation reference.

This module gives the Cognitive Matrix visualization a deterministic numerical
companion. The logical domain may be astronomically large, but only a bounded
particle ensemble and a bounded radial permeation field are materialized.

It is deliberately non-authoritative: receipts are evidence for CTR / Pi_Lambda.
The module does not claim physical energy transport, consciousness, AGI, or
physical residency of the virtual address space.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable, Sequence

Vector3 = tuple[float, float, float]


class CognitiveMatrixError(ValueError):
    """Raised when Cognitive Matrix inputs violate the bounded contract."""


def _finite(value: float | int, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CognitiveMatrixError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise CognitiveMatrixError(f"{name} must be finite")
    return result


def _vec3(value: Sequence[float | int], *, name: str) -> Vector3:
    if len(value) != 3:
        raise CognitiveMatrixError(f"{name} must contain exactly three values")
    return (
        _finite(value[0], name=f"{name}[0]"),
        _finite(value[1], name=f"{name}[1]"),
        _finite(value[2], name=f"{name}[2]"),
    )


def _smoothstep(value: float) -> float:
    u = min(1.0, max(0.0, value))
    return u * u * (3.0 - 2.0 * u)


def _lerp(left: Vector3, right: Vector3, weight: float) -> Vector3:
    return (
        left[0] + (right[0] - left[0]) * weight,
        left[1] + (right[1] - left[1]) * weight,
        left[2] + (right[2] - left[2]) * weight,
    )


def _norm(value: Vector3) -> float:
    return math.sqrt(value[0] ** 2 + value[1] ** 2 + value[2] ** 2)


@dataclass(frozen=True, slots=True)
class CognitiveMatrixConfig:
    """Resource and numerical limits for the bounded reference."""

    virtual_axis: int = 10**24
    bits_per_virtual_site: int = 1
    particle_count: int = 256
    max_particles: int = 4096
    seed: int = 7
    encoder_scale: float = 0.5
    torus_major_radius: float = 2.5
    torus_minor_radius: float = 0.8
    boundary_radius: float = 6.0
    radial_cells: int = 64
    field_diffusion: float = 0.12
    field_decay: float = 0.08
    field_source: float = 0.8
    field_source_sigma: float = 1.25
    particle_mobility: float = 0.08
    particle_diffusion: float = 0.015
    dt: float = 1.0 / 60.0
    entropy_bins_per_axis: int = 6
    flux_reference: float = 1.0
    max_reconstruction_mse: float = 1e-10

    def __post_init__(self) -> None:
        if not isinstance(self.virtual_axis, int) or self.virtual_axis <= 0:
            raise CognitiveMatrixError("virtual_axis must be a positive integer")
        if not isinstance(self.bits_per_virtual_site, int) or self.bits_per_virtual_site <= 0:
            raise CognitiveMatrixError("bits_per_virtual_site must be a positive integer")
        if not isinstance(self.particle_count, int) or self.particle_count <= 0:
            raise CognitiveMatrixError("particle_count must be a positive integer")
        if not isinstance(self.max_particles, int) or self.max_particles <= 0:
            raise CognitiveMatrixError("max_particles must be a positive integer")
        if self.particle_count > self.max_particles:
            raise CognitiveMatrixError("particle_count exceeds max_particles")
        if not isinstance(self.seed, int) or self.seed < 0 or self.seed > 0xFFFFFFFF:
            raise CognitiveMatrixError("seed must be a uint32")
        if not isinstance(self.radial_cells, int) or self.radial_cells < 8:
            raise CognitiveMatrixError("radial_cells must be an integer >= 8")
        if not isinstance(self.entropy_bins_per_axis, int) or not (
            2 <= self.entropy_bins_per_axis <= 32
        ):
            raise CognitiveMatrixError("entropy_bins_per_axis must be in [2, 32]")
        for name in (
            "encoder_scale",
            "torus_major_radius",
            "torus_minor_radius",
            "boundary_radius",
            "field_diffusion",
            "field_decay",
            "field_source",
            "field_source_sigma",
            "particle_mobility",
            "particle_diffusion",
            "dt",
            "flux_reference",
            "max_reconstruction_mse",
        ):
            value = _finite(getattr(self, name), name=name)
            if value < 0.0:
                raise CognitiveMatrixError(f"{name} must be non-negative")
        if self.encoder_scale == 0.0:
            raise CognitiveMatrixError("encoder_scale must be positive")
        if self.boundary_radius == 0.0:
            raise CognitiveMatrixError("boundary_radius must be positive")
        if self.field_source_sigma == 0.0:
            raise CognitiveMatrixError("field_source_sigma must be positive")
        if self.dt == 0.0:
            raise CognitiveMatrixError("dt must be positive")
        if self.flux_reference == 0.0:
            raise CognitiveMatrixError("flux_reference must be positive")


@dataclass(frozen=True, slots=True)
class CognitiveMatrixReceipt:
    """Non-authoritative evidence emitted by one bounded simulation step."""

    virtual_axis: int
    virtual_sites: int
    virtual_bits: int
    active_particles: int
    phase: float
    reconstruction_mse: float
    spatial_entropy_bits: float
    permeation_flux: float
    permeation_flux_percent: float
    max_field_value: float
    finite: bool
    accepted: bool
    claim_status: str = "bounded_stochastic_projection"


class CognitiveMatrixPermeation:
    """Deterministic sampled Cognitive Matrix with a bounded radial PDE field."""

    def __init__(self, config: CognitiveMatrixConfig | None = None) -> None:
        self.config = config or CognitiveMatrixConfig()
        self._rng = random.Random(self.config.seed)
        self._dr = self.config.boundary_radius / (self.config.radial_cells - 1)
        self._field = [0.0] * self.config.radial_cells
        self._time = 0.0
        self._input = self._sample_input_cloud()
        self._encoded = tuple(self.encode(position) for position in self._input)
        self._reconstructed = tuple(self.decode(latent) for latent in self._encoded)
        self._latent_geometry = tuple(
            self._latent_to_torus(latent) for latent in self._encoded
        )
        self._output_geometry = self._make_output_lattice(self.config.particle_count)
        self._offsets = [(0.0, 0.0, 0.0) for _ in range(self.config.particle_count)]

    @property
    def virtual_sites(self) -> int:
        return self.config.virtual_axis**3

    @property
    def virtual_bits(self) -> int:
        return self.virtual_sites * self.config.bits_per_virtual_site

    @property
    def time(self) -> float:
        return self._time

    def _sample_input_cloud(self) -> tuple[Vector3, ...]:
        scale = min(1.0, self.config.boundary_radius / 6.0)
        return tuple(
            (
                self._rng.gauss(0.0, scale),
                self._rng.gauss(0.0, scale),
                self._rng.gauss(0.0, scale),
            )
            for _ in range(self.config.particle_count)
        )

    def encode(self, position: Sequence[float | int]) -> Vector3:
        x, y, z = _vec3(position, name="position")
        scale = self.config.encoder_scale
        return (
            math.tanh(scale * x),
            math.tanh(scale * y),
            math.tanh(scale * z),
        )

    def decode(self, latent: Sequence[float | int]) -> Vector3:
        z0, z1, z2 = _vec3(latent, name="latent")
        scale = self.config.encoder_scale

        def inverse(value: float) -> float:
            clipped = min(1.0 - 1e-15, max(-1.0 + 1e-15, value))
            return math.atanh(clipped) / scale

        return (inverse(z0), inverse(z1), inverse(z2))

    def _latent_to_torus(self, latent: Vector3) -> Vector3:
        theta = math.pi * (latent[0] + 1.0)
        phi = math.pi * (latent[1] + 1.0)
        radius_scale = 0.75 + 0.25 * (latent[2] + 1.0) / 2.0
        minor = self.config.torus_minor_radius * radius_scale
        major = self.config.torus_major_radius
        cos_phi = math.cos(phi)
        return (
            (major + minor * cos_phi) * math.cos(theta),
            minor * math.sin(phi),
            (major + minor * cos_phi) * math.sin(theta),
        )

    @staticmethod
    def _make_output_lattice(count: int) -> tuple[Vector3, ...]:
        side = math.ceil(count ** (1.0 / 3.0))
        spacing = 0.35
        center = (side - 1) / 2.0
        points: list[Vector3] = []
        for index in range(count):
            x = index % side
            y = (index // side) % side
            z = index // (side * side)
            points.append(
                (
                    (x - center) * spacing,
                    (y - center) * spacing,
                    (z - center) * spacing,
                )
            )
        return tuple(points)

    def reconstruction_mse(self) -> float:
        total = 0.0
        for original, reconstructed in zip(self._input, self._reconstructed):
            total += sum((a - b) ** 2 for a, b in zip(original, reconstructed))
        return total / (3.0 * self.config.particle_count)

    def _source(self, radius: float) -> float:
        sigma = self.config.field_source_sigma
        return self.config.field_source * math.exp(-(radius**2) / (2.0 * sigma**2))

    def _advance_field(self, dt: float) -> None:
        diffusion = self.config.field_diffusion
        if diffusion == 0.0:
            stable_dt = dt
        else:
            stable_dt = 0.45 * self._dr * self._dr / max(diffusion, 1e-15)
        substeps = max(1, math.ceil(dt / stable_dt))
        h = dt / substeps

        for _ in range(substeps):
            old = self._field
            new = old.copy()
            new[0] = old[0] + h * (
                diffusion * 6.0 * (old[1] - old[0]) / (self._dr**2)
                + self._source(0.0)
                - self.config.field_decay * old[0]
            )
            for index in range(1, self.config.radial_cells - 1):
                radius = index * self._dr
                second = (old[index + 1] - 2.0 * old[index] + old[index - 1]) / (
                    self._dr**2
                )
                first = (old[index + 1] - old[index - 1]) / (2.0 * self._dr)
                laplacian = second + 2.0 * first / radius
                new[index] = old[index] + h * (
                    diffusion * laplacian
                    + self._source(radius)
                    - self.config.field_decay * old[index]
                )
                if new[index] < 0.0 and new[index] > -1e-15:
                    new[index] = 0.0
            new[-1] = 0.0
            if not all(math.isfinite(value) and value >= 0.0 for value in new):
                raise CognitiveMatrixError("permeation field became non-finite or negative")
            self._field = new

    def _field_gradient_radial(self, radius: float) -> float:
        clamped = min(self.config.boundary_radius, max(0.0, radius))
        position = clamped / self._dr
        index = min(self.config.radial_cells - 2, int(position))
        if index == 0:
            return (self._field[1] - self._field[0]) / self._dr
        return (self._field[index + 1] - self._field[index - 1]) / (2.0 * self._dr)

    def permeation_flux(self) -> float:
        gradient = (self._field[-1] - self._field[-2]) / self._dr
        current_radial = -self.config.field_diffusion * gradient
        return 4.0 * math.pi * self.config.boundary_radius**2 * current_radial

    def phase_positions(self, phase: float) -> tuple[Vector3, ...]:
        value = _finite(phase, name="phase")
        if not 0.0 <= value <= 2.0:
            raise CognitiveMatrixError("phase must be in [0, 2]")
        if value <= 1.0:
            weight = _smoothstep(value)
            base = tuple(
                _lerp(source, latent, weight)
                for source, latent in zip(self._input, self._latent_geometry)
            )
        else:
            weight = _smoothstep(value - 1.0)
            base = tuple(
                _lerp(latent, output, weight)
                for latent, output in zip(self._latent_geometry, self._output_geometry)
            )
        return tuple(
            (
                point[0] + offset[0],
                point[1] + offset[1],
                point[2] + offset[2],
            )
            for point, offset in zip(base, self._offsets)
        )

    def _advance_particles(self, base_positions: Sequence[Vector3], dt: float) -> None:
        next_offsets: list[Vector3] = []
        noise_scale = math.sqrt(2.0 * self.config.particle_diffusion * dt)
        for point, offset in zip(base_positions, self._offsets):
            current = (
                point[0] + offset[0],
                point[1] + offset[1],
                point[2] + offset[2],
            )
            radius = _norm(current)
            if radius > 1e-15:
                gradient = self._field_gradient_radial(radius)
                direction = (
                    current[0] / radius,
                    current[1] / radius,
                    current[2] / radius,
                )
                drift = tuple(
                    self.config.particle_mobility * gradient * component * dt
                    for component in direction
                )
            else:
                drift = (0.0, 0.0, 0.0)
            jitter = (
                noise_scale * self._rng.gauss(0.0, 1.0),
                noise_scale * self._rng.gauss(0.0, 1.0),
                noise_scale * self._rng.gauss(0.0, 1.0),
            )
            candidate = (
                offset[0] + drift[0] + jitter[0],
                offset[1] + drift[1] + jitter[1],
                offset[2] + drift[2] + jitter[2],
            )
            candidate_radius = _norm(
                (
                    point[0] + candidate[0],
                    point[1] + candidate[1],
                    point[2] + candidate[2],
                )
            )
            if candidate_radius > self.config.boundary_radius:
                shrink = self.config.boundary_radius / candidate_radius
                candidate = (
                    (point[0] + candidate[0]) * shrink - point[0],
                    (point[1] + candidate[1]) * shrink - point[1],
                    (point[2] + candidate[2]) * shrink - point[2],
                )
            next_offsets.append(candidate)
        self._offsets = next_offsets

    def spatial_entropy_bits(self, positions: Iterable[Vector3]) -> float:
        bins = self.config.entropy_bins_per_axis
        counts = [0] * (bins**3)
        radius = self.config.boundary_radius
        count = 0
        for point in positions:
            count += 1
            indices = []
            for component in point:
                normalized = (component + radius) / (2.0 * radius)
                index = min(bins - 1, max(0, int(normalized * bins)))
                indices.append(index)
            flat = indices[0] + bins * (indices[1] + bins * indices[2])
            counts[flat] += 1
        if count == 0:
            return 0.0
        entropy = 0.0
        for value in counts:
            if value:
                probability = value / count
                entropy -= probability * math.log2(probability)
        return entropy

    def step(self, phase: float, *, dt: float | None = None) -> CognitiveMatrixReceipt:
        step_dt = self.config.dt if dt is None else _finite(dt, name="dt")
        if step_dt <= 0.0:
            raise CognitiveMatrixError("dt must be positive")
        phase_value = _finite(phase, name="phase")
        base = self.phase_positions(phase_value)
        self._advance_field(step_dt)
        self._advance_particles(base, step_dt)
        rendered = self.phase_positions(phase_value)
        self._time += step_dt

        mse = self.reconstruction_mse()
        entropy = self.spatial_entropy_bits(rendered)
        flux = self.permeation_flux()
        flux_percent = min(100.0, 100.0 * abs(flux) / self.config.flux_reference)
        maximum = max(self._field)
        finite = all(
            math.isfinite(value)
            for value in (mse, entropy, flux, flux_percent, maximum)
        )
        accepted = (
            finite
            and self.config.particle_count <= self.config.max_particles
            and mse <= self.config.max_reconstruction_mse
        )
        return CognitiveMatrixReceipt(
            virtual_axis=self.config.virtual_axis,
            virtual_sites=self.virtual_sites,
            virtual_bits=self.virtual_bits,
            active_particles=self.config.particle_count,
            phase=phase_value,
            reconstruction_mse=mse,
            spatial_entropy_bits=entropy,
            permeation_flux=flux,
            permeation_flux_percent=flux_percent,
            max_field_value=maximum,
            finite=finite,
            accepted=accepted,
        )


__all__ = [
    "CognitiveMatrixConfig",
    "CognitiveMatrixError",
    "CognitiveMatrixPermeation",
    "CognitiveMatrixReceipt",
    "Vector3",
]
