"""Deterministic reference solver for periodic fractional diffusion in 3D.

The implementation uses the exact spectrum of the six-neighbour discrete
Laplacian and a dependency-free separable DFT. It is designed for correctness,
small fixtures and auditable experiments, not production-scale FFT workloads.
"""

from __future__ import annotations

import cmath
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

Shape3D = tuple[int, int, int]
Coordinate3D = tuple[int, int, int]


def _index(x: int, y: int, z: int, shape: Shape3D) -> int:
    nx, ny, _ = shape
    return x + nx * (y + ny * z)


def _rms(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum(value * value for value in values) / len(values))


def _validate_alpha(alpha: float) -> None:
    if not math.isfinite(alpha) or not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be finite and in (0, 1]")


@dataclass(frozen=True)
class Grid3D:
    """Immutable scalar field with x as the fastest-moving coordinate."""

    shape: Shape3D
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        nx, ny, nz = self.shape
        if nx < 1 or ny < 1 or nz < 1:
            raise ValueError("all grid dimensions must be positive")
        if len(self.values) != nx * ny * nz:
            raise ValueError("value count does not match grid shape")
        if not all(math.isfinite(value) for value in self.values):
            raise ValueError("grid values must be finite")

    @classmethod
    def constant(cls, shape: Shape3D, value: float) -> "Grid3D":
        if not math.isfinite(value):
            raise ValueError("constant value must be finite")
        nx, ny, nz = shape
        return cls(shape, (float(value),) * (nx * ny * nz))

    @classmethod
    def impulse(
        cls,
        shape: Shape3D,
        coordinate: Coordinate3D,
        amplitude: float = 1.0,
    ) -> "Grid3D":
        nx, ny, nz = shape
        x, y, z = coordinate
        if not (0 <= x < nx and 0 <= y < ny and 0 <= z < nz):
            raise ValueError("impulse coordinate lies outside the grid")
        if not math.isfinite(amplitude):
            raise ValueError("impulse amplitude must be finite")
        values = [0.0] * (nx * ny * nz)
        values[_index(x, y, z, shape)] = float(amplitude)
        return cls(shape, tuple(values))

    def at(self, x: int, y: int, z: int) -> float:
        nx, ny, nz = self.shape
        if not (0 <= x < nx and 0 <= y < ny and 0 <= z < nz):
            raise IndexError("grid coordinate out of range")
        return self.values[_index(x, y, z, self.shape)]

    @property
    def mean(self) -> float:
        return sum(self.values) / len(self.values)

    @property
    def mass(self) -> float:
        return sum(self.values)

    @property
    def variance(self) -> float:
        mean = self.mean
        return sum((value - mean) ** 2 for value in self.values) / len(self.values)


def grid_from_values(shape: Shape3D, values: Iterable[float]) -> Grid3D:
    return Grid3D(shape, tuple(float(value) for value in values))


def _transform_x(values: Sequence[complex], shape: Shape3D, inverse: bool) -> list[complex]:
    nx, ny, nz = shape
    output = [0j] * len(values)
    sign = 1.0 if inverse else -1.0
    scale = 1.0 / nx if inverse else 1.0
    roots = [[cmath.exp(sign * 2j * math.pi * k * x / nx) for x in range(nx)] for k in range(nx)]
    for z in range(nz):
        for y in range(ny):
            for k in range(nx):
                total = sum(values[_index(x, y, z, shape)] * roots[k][x] for x in range(nx))
                output[_index(k, y, z, shape)] = total * scale
    return output


def _transform_y(values: Sequence[complex], shape: Shape3D, inverse: bool) -> list[complex]:
    nx, ny, nz = shape
    output = [0j] * len(values)
    sign = 1.0 if inverse else -1.0
    scale = 1.0 / ny if inverse else 1.0
    roots = [[cmath.exp(sign * 2j * math.pi * k * y / ny) for y in range(ny)] for k in range(ny)]
    for z in range(nz):
        for x in range(nx):
            for k in range(ny):
                total = sum(values[_index(x, y, z, shape)] * roots[k][y] for y in range(ny))
                output[_index(x, k, z, shape)] = total * scale
    return output


def _transform_z(values: Sequence[complex], shape: Shape3D, inverse: bool) -> list[complex]:
    nx, ny, nz = shape
    output = [0j] * len(values)
    sign = 1.0 if inverse else -1.0
    scale = 1.0 / nz if inverse else 1.0
    roots = [[cmath.exp(sign * 2j * math.pi * k * z / nz) for z in range(nz)] for k in range(nz)]
    for y in range(ny):
        for x in range(nx):
            for k in range(nz):
                total = sum(values[_index(x, y, z, shape)] * roots[k][z] for z in range(nz))
                output[_index(x, y, k, shape)] = total * scale
    return output


def forward_dft3(field: Grid3D) -> tuple[complex, ...]:
    """Return the unnormalised separable 3D DFT of ``field``."""

    values = [complex(value, 0.0) for value in field.values]
    transformed = _transform_x(values, field.shape, False)
    transformed = _transform_y(transformed, field.shape, False)
    transformed = _transform_z(transformed, field.shape, False)
    return tuple(transformed)


def inverse_dft3(values: Sequence[complex], shape: Shape3D) -> Grid3D:
    """Invert a spectrum using one ``1/N`` factor per transformed axis."""

    nx, ny, nz = shape
    if len(values) != nx * ny * nz:
        raise ValueError("spectrum size does not match shape")
    transformed = _transform_z(values, shape, True)
    transformed = _transform_y(transformed, shape, True)
    transformed = _transform_x(transformed, shape, True)
    return Grid3D(shape, tuple(value.real for value in transformed))


def laplacian_eigenvalue(kx: int, ky: int, kz: int, shape: Shape3D) -> float:
    """Eigenvalue of the periodic six-neighbour operator ``(-Delta)``."""

    nx, ny, nz = shape
    if not (0 <= kx < nx and 0 <= ky < ny and 0 <= kz < nz):
        raise ValueError("mode coordinate lies outside the spectrum")
    return (
        6.0
        - 2.0 * math.cos(2.0 * math.pi * kx / nx)
        - 2.0 * math.cos(2.0 * math.pi * ky / ny)
        - 2.0 * math.cos(2.0 * math.pi * kz / nz)
    )


def _zero_mean(field: Grid3D) -> Grid3D:
    mean = field.mean
    return Grid3D(field.shape, tuple(value - mean for value in field.values))


def fractional_laplacian(field: Grid3D, alpha: float) -> Grid3D:
    """Return ``(-Delta)^alpha field`` on the periodic lattice."""

    _validate_alpha(alpha)
    spectrum = list(forward_dft3(field))
    nx, ny, nz = field.shape
    for kz in range(nz):
        for ky in range(ny):
            for kx in range(nx):
                index = _index(kx, ky, kz, field.shape)
                eigenvalue = laplacian_eigenvalue(kx, ky, kz, field.shape)
                multiplier = eigenvalue**alpha if eigenvalue > 0.0 else 0.0
                spectrum[index] *= multiplier
    return inverse_dft3(spectrum, field.shape)


def spectral_fractional_step(
    field: Grid3D,
    alpha: float,
    tau: float,
    diffusivity: float = 1.0,
    omega: Grid3D | None = None,
    zero_mean_omega: bool = True,
) -> Grid3D:
    """Advance ``du/dt = -D(-Delta)^alpha u + omega`` exactly per mode."""

    _validate_alpha(alpha)
    if not math.isfinite(tau) or tau < 0.0:
        raise ValueError("tau must be finite and non-negative")
    if not math.isfinite(diffusivity) or diffusivity < 0.0:
        raise ValueError("diffusivity must be finite and non-negative")
    if omega is not None and omega.shape != field.shape:
        raise ValueError("omega shape must match field shape")
    if tau == 0.0:
        return field

    forcing = omega
    if forcing is not None and zero_mean_omega:
        forcing = _zero_mean(forcing)
    if diffusivity == 0.0:
        forcing_values = forcing.values if forcing is not None else (0.0,) * len(field.values)
        return Grid3D(
            field.shape,
            tuple(
                value + tau * forcing_value
                for value, forcing_value in zip(field.values, forcing_values)
            ),
        )

    spectrum = list(forward_dft3(field))
    forcing_spectrum = list(forward_dft3(forcing)) if forcing is not None else None
    nx, ny, nz = field.shape
    for kz in range(nz):
        for ky in range(ny):
            for kx in range(nx):
                index = _index(kx, ky, kz, field.shape)
                eigenvalue = laplacian_eigenvalue(kx, ky, kz, field.shape)
                rate = diffusivity * (eigenvalue**alpha if eigenvalue > 0.0 else 0.0)
                decay = math.exp(-tau * rate)
                updated = decay * spectrum[index]
                if forcing_spectrum is not None:
                    if rate > 0.0:
                        updated += (1.0 - decay) * forcing_spectrum[index] / rate
                    else:
                        updated += tau * forcing_spectrum[index]
                spectrum[index] = updated
    return inverse_dft3(spectrum, field.shape)


def classical_gradient_energy(field: Grid3D) -> float:
    """Mean squared forward difference over the three periodic axes."""

    nx, ny, nz = field.shape
    total = 0.0
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                center = field.at(x, y, z)
                dx = field.at((x + 1) % nx, y, z) - center
                dy = field.at(x, (y + 1) % ny, z) - center
                dz = field.at(x, y, (z + 1) % nz) - center
                total += dx * dx + dy * dy + dz * dz
    return total / (3.0 * len(field.values))


def equilibrium_residual(
    field: Grid3D,
    alpha: float,
    diffusivity: float = 1.0,
    omega: Grid3D | None = None,
    zero_mean_omega: bool = True,
) -> float:
    """Return the RMS residual of ``-D(-Delta)^alpha u + omega``."""

    if not math.isfinite(diffusivity) or diffusivity < 0.0:
        raise ValueError("diffusivity must be finite and non-negative")
    if omega is not None and omega.shape != field.shape:
        raise ValueError("omega shape must match field shape")
    operator = fractional_laplacian(field, alpha)
    forcing = omega
    if forcing is not None and zero_mean_omega:
        forcing = _zero_mean(forcing)
    forcing_values = forcing.values if forcing is not None else (0.0,) * len(field.values)
    return _rms(
        tuple(
            -diffusivity * operator_value + forcing_value
            for operator_value, forcing_value in zip(operator.values, forcing_values)
        )
    )


def restrict_half(field: Grid3D) -> Grid3D:
    """Average each ``2×2×2`` block into one coarse cell."""

    nx, ny, nz = field.shape
    if nx % 2 or ny % 2 or nz % 2:
        raise ValueError("all dimensions must be even for half restriction")
    coarse_shape = (nx // 2, ny // 2, nz // 2)
    values: list[float] = []
    for cz in range(coarse_shape[2]):
        for cy in range(coarse_shape[1]):
            for cx in range(coarse_shape[0]):
                total = sum(
                    field.at(2 * cx + dx, 2 * cy + dy, 2 * cz + dz)
                    for dz in range(2)
                    for dy in range(2)
                    for dx in range(2)
                )
                values.append(total / 8.0)
    return Grid3D(coarse_shape, tuple(values))


def prolong_double(field: Grid3D) -> Grid3D:
    """Replicate each coarse cell across one ``2×2×2`` fine block."""

    nx, ny, nz = field.shape
    fine_shape = (2 * nx, 2 * ny, 2 * nz)
    fine = [0.0] * (fine_shape[0] * fine_shape[1] * fine_shape[2])
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                value = field.at(x, y, z)
                for dz in range(2):
                    for dy in range(2):
                        for dx in range(2):
                            fine[_index(2 * x + dx, 2 * y + dy, 2 * z + dz, fine_shape)] = value
    return Grid3D(fine_shape, tuple(fine))


def round_trip_error(coarse: Grid3D) -> float:
    reconstructed = restrict_half(prolong_double(coarse))
    return _rms(tuple(left - right for left, right in zip(reconstructed.values, coarse.values)))


@dataclass(frozen=True)
class FractionalHierarchyConfig:
    alphas: tuple[float, ...] = (1.0, 0.75, 0.5)
    taus: tuple[float, ...] = (0.05, 0.1, 0.2)
    coarse_blends: tuple[float, ...] = (0.2, 0.35)
    diffusivity: float = 1.0
    zero_mean_omega: bool = True

    def validate(self) -> None:
        if not self.alphas:
            raise ValueError("at least one hierarchy level is required")
        if len(self.taus) != len(self.alphas):
            raise ValueError("taus must contain one value per hierarchy level")
        if len(self.coarse_blends) != len(self.alphas) - 1:
            raise ValueError("coarse_blends must contain one value per fusion boundary")
        for alpha in self.alphas:
            _validate_alpha(alpha)
        if not all(math.isfinite(tau) and tau >= 0.0 for tau in self.taus):
            raise ValueError("all smoothing times must be finite and non-negative")
        if not all(math.isfinite(weight) and 0.0 <= weight <= 1.0 for weight in self.coarse_blends):
            raise ValueError("all coarse blends must lie in [0, 1]")
        if not math.isfinite(self.diffusivity) or self.diffusivity < 0.0:
            raise ValueError("diffusivity must be finite and non-negative")


@dataclass(frozen=True)
class MechanisticInstruction:
    sequence: int
    opcode: str
    level: int
    detail: str


@dataclass(frozen=True)
class LevelTrace:
    level: int
    shape: Shape3D
    alpha: float
    tau: float
    mean_before: float
    mean_after: float
    variance_before: float
    variance_after: float
    gradient_energy_before: float
    gradient_energy_after: float
    equilibrium_residual_after: float


@dataclass(frozen=True)
class HierarchicalSmoothingResult:
    field: Grid3D
    traces: tuple[LevelTrace, ...]
    instructions: tuple[MechanisticInstruction, ...]
    mass_before: float
    mass_after: float
    mass_drift: float
    update_rms: float


@dataclass(frozen=True)
class EquilibriumResult:
    field: Grid3D
    converged: bool
    cycles: int
    update_history: tuple[float, ...]
    residual_history: tuple[float, ...]


def _validate_hierarchy_shape(shape: Shape3D, levels: int) -> None:
    divisor = 2 ** (levels - 1)
    if any(dimension % divisor for dimension in shape):
        raise ValueError("each dimension must be divisible by 2**(levels - 1)")


def _blend_fields(local: Grid3D, projected: Grid3D, weight: float) -> Grid3D:
    if local.shape != projected.shape:
        raise ValueError("fields must share a shape before blending")
    return Grid3D(
        local.shape,
        tuple(
            (1.0 - weight) * local_value + weight * projected_value
            for local_value, projected_value in zip(local.values, projected.values)
        ),
    )


def hierarchical_fractional_smooth(
    field: Grid3D,
    config: FractionalHierarchyConfig | None = None,
    omega: Grid3D | None = None,
) -> HierarchicalSmoothingResult:
    """Execute one fine-to-coarse-to-fine smoothing transaction."""

    active = config or FractionalHierarchyConfig()
    active.validate()
    levels = len(active.alphas)
    _validate_hierarchy_shape(field.shape, levels)
    if omega is not None and omega.shape != field.shape:
        raise ValueError("omega shape must match field shape")

    grids = [field]
    omega_grids: list[Grid3D | None] = [omega]
    instructions: list[MechanisticInstruction] = []
    sequence = 0
    for level in range(1, levels):
        grids.append(restrict_half(grids[-1]))
        parent_omega = omega_grids[-1]
        omega_grids.append(restrict_half(parent_omega) if parent_omega is not None else None)
        instructions.append(
            MechanisticInstruction(
                sequence,
                "RESTRICT_2X2X2",
                level,
                f"{grids[level - 1].shape} -> {grids[level].shape}",
            )
        )
        sequence += 1

    local_results: list[Grid3D] = []
    traces: list[LevelTrace] = []
    for level, source in enumerate(grids):
        result = spectral_fractional_step(
            source,
            active.alphas[level],
            active.taus[level],
            active.diffusivity,
            omega_grids[level],
            active.zero_mean_omega,
        )
        local_results.append(result)
        traces.append(
            LevelTrace(
                level=level,
                shape=source.shape,
                alpha=active.alphas[level],
                tau=active.taus[level],
                mean_before=source.mean,
                mean_after=result.mean,
                variance_before=source.variance,
                variance_after=result.variance,
                gradient_energy_before=classical_gradient_energy(source),
                gradient_energy_after=classical_gradient_energy(result),
                equilibrium_residual_after=equilibrium_residual(
                    result,
                    active.alphas[level],
                    active.diffusivity,
                    omega_grids[level],
                    active.zero_mean_omega,
                ),
            )
        )
        instructions.append(
            MechanisticInstruction(
                sequence,
                "FRACTIONAL_HEAT_3D",
                level,
                f"alpha={active.alphas[level]:.6f} "
                f"tau={active.taus[level]:.6f} shape={source.shape}",
            )
        )
        sequence += 1

    fused = local_results[-1]
    for level in range(levels - 2, -1, -1):
        projected = prolong_double(fused)
        instructions.append(
            MechanisticInstruction(
                sequence,
                "PROLONG_2X2X2",
                level,
                f"{fused.shape} -> {projected.shape}",
            )
        )
        sequence += 1
        weight = active.coarse_blends[level]
        fused = _blend_fields(local_results[level], projected, weight)
        instructions.append(
            MechanisticInstruction(
                sequence,
                "FUSE_COARSE_FINE",
                level,
                f"coarse_weight={weight:.6f}",
            )
        )
        sequence += 1

    update_rms = _rms(tuple(after - before for after, before in zip(fused.values, field.values)))
    mass_before = field.mass
    mass_after = fused.mass
    instructions.append(
        MechanisticInstruction(
            sequence,
            "VERIFY_MASS_AND_UPDATE",
            0,
            f"mass_drift={mass_after - mass_before:.12e} update_rms={update_rms:.12e}",
        )
    )
    return HierarchicalSmoothingResult(
        field=fused,
        traces=tuple(traces),
        instructions=tuple(instructions),
        mass_before=mass_before,
        mass_after=mass_after,
        mass_drift=mass_after - mass_before,
        update_rms=update_rms,
    )


def run_to_equilibrium(
    field: Grid3D,
    config: FractionalHierarchyConfig | None = None,
    omega: Grid3D | None = None,
    tolerance: float = 1.0e-6,
    max_cycles: int = 100,
) -> EquilibriumResult:
    """Repeat hierarchical transactions until update RMS reaches tolerance."""

    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    if max_cycles < 1:
        raise ValueError("max_cycles must be positive")
    active = config or FractionalHierarchyConfig()
    active.validate()

    current = field
    updates: list[float] = []
    residuals: list[float] = []
    for cycle in range(1, max_cycles + 1):
        result = hierarchical_fractional_smooth(current, active, omega)
        current = result.field
        updates.append(result.update_rms)
        residuals.append(
            equilibrium_residual(
                current,
                active.alphas[0],
                active.diffusivity,
                omega,
                active.zero_mean_omega,
            )
        )
        if result.update_rms <= tolerance:
            return EquilibriumResult(current, True, cycle, tuple(updates), tuple(residuals))
    return EquilibriumResult(current, False, max_cycles, tuple(updates), tuple(residuals))
