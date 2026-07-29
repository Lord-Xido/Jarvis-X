"""Hierarchical three-dimensional fractional geometric smoothing.

The module implements a deterministic reference solver for periodic 3D scalar
fields.  It uses the exact eigenvalues of the six-neighbour discrete Laplacian
and a separable discrete Fourier transform, so the fractional heat update is
performed mode-by-mode without external numerical dependencies.

This is a correctness-oriented CPU prototype.  It is not intended to replace
FFT libraries or production sparse-grid solvers.
"""
from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple


Shape3D = Tuple[int, int, int]


def _index(x: int, y: int, z: int, shape: Shape3D) -> int:
    nx, ny, _ = shape
    return x + nx * (y + ny * z)


def _rms(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum(value * value for value in values) / len(values))


@dataclass(frozen=True)
class Grid3D:
    """Dense scalar field stored with x as the fastest-moving coordinate."""

    shape: Shape3D
    values: Tuple[float, ...]

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
        coordinate: Tuple[int, int, int],
        amplitude: float = 1.0,
    ) -> "Grid3D":
        nx, ny, nz = shape
        x, y, z = coordinate
        if not (0 <= x < nx and 0 <= y < ny and 0 <= z < nz):
            raise ValueError("impulse coordinate lies outside the grid")
        if not math.isfinite(amplitude):
            raise ValueError("impulse amplitude must be finite")
        values = [0.0] * (nx * ny * nz)
        values[_index(x, y, z, shape)] = amplitude
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

    @property
    def l2_rms(self) -> float:
        return _rms(self.values)


def _transform_x(
    values: Sequence[complex], shape: Shape3D, inverse: bool
) -> List[complex]:
    nx, ny, nz = shape
    output = [0j] * len(values)
    sign = 1.0 if inverse else -1.0
    scale = 1.0 / nx if inverse else 1.0
    roots = [
        [cmath.exp(sign * 2j * math.pi * k * x / nx) for x in range(nx)]
        for k in range(nx)
    ]
    for z in range(nz):
        for y in range(ny):
            for k in range(nx):
                total = 0j
                for x in range(nx):
                    total += values[_index(x, y, z, shape)] * roots[k][x]
                output[_index(k, y, z, shape)] = total * scale
    return output


def _transform_y(
    values: Sequence[complex], shape: Shape3D, inverse: bool
) -> List[complex]:
    nx, ny, nz = shape
    output = [0j] * len(values)
    sign = 1.0 if inverse else -1.0
    scale = 1.0 / ny if inverse else 1.0
    roots = [
        [cmath.exp(sign * 2j * math.pi * k * y / ny) for y in range(ny)]
        for k in range(ny)
    ]
    for z in range(nz):
        for x in range(nx):
            for k in range(ny):
                total = 0j
                for y in range(ny):
                    total += values[_index(x, y, z, shape)] * roots[k][y]
                output[_index(x, k, z, shape)] = total * scale
    return output


def _transform_z(
    values: Sequence[complex], shape: Shape3D, inverse: bool
) -> List[complex]:
    nx, ny, nz = shape
    output = [0j] * len(values)
    sign = 1.0 if inverse else -1.0
    scale = 1.0 / nz if inverse else 1.0
    roots = [
        [cmath.exp(sign * 2j * math.pi * k * z / nz) for z in range(nz)]
        for k in range(nz)
    ]
    for y in range(ny):
        for x in range(nx):
            for k in range(nz):
                total = 0j
                for z in range(nz):
                    total += values[_index(x, y, z, shape)] * roots[k][z]
                output[_index(x, y, k, shape)] = total * scale
    return output


def _dft3(values: Sequence[float]) -> List[complex]:
    raise RuntimeError("_dft3 requires a shape; use _forward_dft3")


def _forward_dft3(grid: Grid3D) -> List[complex]:
    values = [complex(value, 0.0) for value in grid.values]
    return _transform_z(_transform_y(_transform_x(values, grid.shape, False), grid.shape, False), grid.shape, False)


def _inverse_dft3(values: Sequence[complex], shape: Shape3D) -> Grid3D:
    transformed = _transform_x(
        _transform_y(_transform_z(values, shape, True), shape, True),
        shape,
        True,
    )
    return Grid3D(shape, tuple(value.real for value in transformed))


def _laplacian_eigenvalue(kx: int, ky: int, kz: int, shape: Shape3D) -> float:
    nx, ny, nz = shape
    return (
        6.0
        - 2.0 * math.cos(2.0 * math.pi * kx / nx)
        - 2.0 * math.cos(2.0 * math.pi * ky / ny)
        - 2.0 * math.cos(2.0 * math.pi * kz / nz)
    )


def _prepare_omega(omega: Grid3D, zero_mean: bool) -> Grid3D:
    if not zero_mean:
        return omega
    mean = omega.mean
    return Grid3D(omega.shape, tuple(value - mean for value in omega.values))


def fractional_laplacian(field: Grid3D, alpha: float) -> Grid3D:
    """Return ``(-Delta)^alpha field`` on a periodic six-neighbour lattice."""

    if not math.isfinite(alpha) or not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be finite and in (0, 1]")
    spectrum = _forward_dft3(field)
    nx, ny, nz = field.shape
    for kz in range(nz):
        for ky in range(ny):
            for kx in range(nx):
                index = _index(kx, ky, kz, field.shape)
                eigenvalue = _laplacian_eigenvalue(kx, ky, kz, field.shape)
                spectrum[index] *= eigenvalue**alpha if eigenvalue > 0.0 else 0.0
    return _inverse_dft3(spectrum, field.shape)


def spectral_fractional_step(
    field: Grid3D,
    alpha: float,
    tau: float,
    diffusivity: float = 1.0,
    omega: Optional[Grid3D] = None,
    zero_mean_omega: bool = True,
) -> Grid3D:
    """Advance ``du/dt = -D(-Delta)^alpha u + omega`` exactly per mode.

    The forcing field is treated as constant over the time interval ``tau``.
    With no forcing, the constant Fourier mode is preserved exactly and every
    non-constant mode is attenuated by ``exp(-tau * D * lambda**alpha)``.
    """

    if not math.isfinite(alpha) or not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be finite and in (0, 1]")
    if not math.isfinite(tau) or tau < 0.0:
        raise ValueError("tau must be finite and non-negative")
    if not math.isfinite(diffusivity) or diffusivity < 0.0:
        raise ValueError("diffusivity must be finite and non-negative")
    if omega is not None and omega.shape != field.shape:
        raise ValueError("omega shape must match field shape")

    spectrum = _forward_dft3(field)
    omega_spectrum: Optional[List[complex]] = None
    if omega is not None:
        omega_spectrum = _forward_dft3(_prepare_omega(omega, zero_mean_omega))

    nx, ny, nz = field.shape
    for kz in range(nz):
        for ky in range(ny):
            for kx in range(nx):
                index = _index(kx, ky, kz, field.shape)
                eigenvalue = _laplacian_eigenvalue(kx, ky, kz, field.shape)
                rate = diffusivity * (eigenvalue**alpha if eigenvalue > 0.0 else 0.0)
                decay = math.exp(-tau * rate)
                updated = decay * spectrum[index]
                if omega_spectrum is not None:
                    if rate > 0.0:
                        updated += (1.0 - decay) * omega_spectrum[index] / rate
                    else:
                        updated += tau * omega_spectrum[index]
                spectrum[index] = updated

    return _inverse_dft3(spectrum, field.shape)


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
    omega: Optional[Grid3D] = None,
    zero_mean_omega: bool = True,
) -> float:
    """RMS of ``-D(-Delta)^alpha u + omega``."""

    operator = fractional_laplacian(field, alpha)
    forcing = (0.0,) * len(field.values)
    if omega is not None:
        if omega.shape != field.shape:
            raise ValueError("omega shape must match field shape")
        forcing = _prepare_omega(omega, zero_mean_omega).values
    return _rms(
        tuple(
            -diffusivity * operator_value + forcing_value
            for operator_value, forcing_value in zip(operator.values, forcing)
        )
    )


def restrict_half(field: Grid3D) -> Grid3D:
    """Average every 2x2x2 block into one coarse voxel."""

    nx, ny, nz = field.shape
    if nx % 2 or ny % 2 or nz % 2:
        raise ValueError("all dimensions must be even for half restriction")
    coarse_shape = (nx // 2, ny // 2, nz // 2)
    coarse = [0.0] * (coarse_shape[0] * coarse_shape[1] * coarse_shape[2])
    for cz in range(coarse_shape[2]):
        for cy in range(coarse_shape[1]):
            for cx in range(coarse_shape[0]):
                total = 0.0
                for dz in range(2):
                    for dy in range(2):
                        for dx in range(2):
                            total += field.at(2 * cx + dx, 2 * cy + dy, 2 * cz + dz)
                coarse[_index(cx, cy, cz, coarse_shape)] = total / 8.0
    return Grid3D(coarse_shape, tuple(coarse))


def prolong_double(field: Grid3D) -> Grid3D:
    """Replicate each coarse voxel into a 2x2x2 fine block.

    This constant prolongation obeys ``restrict_half(prolong_double(u)) == u``
    up to floating-point roundoff, making the coarse-to-fine-to-coarse loop an
    identity on the coarse subspace.
    """

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
                            fine[
                                _index(2 * x + dx, 2 * y + dy, 2 * z + dz, fine_shape)
                            ] = value
    return Grid3D(fine_shape, tuple(fine))


def round_trip_error(coarse: Grid3D) -> float:
    reconstructed = restrict_half(prolong_double(coarse))
    return _rms(
        tuple(left - right for left, right in zip(reconstructed.values, coarse.values))
    )


@dataclass(frozen=True)
class FractionalHierarchyConfig:
    """Fine-to-coarse schedule for hierarchical fractional smoothing."""

    alphas: Tuple[float, ...] = (1.0, 0.75, 0.50)
    taus: Tuple[float, ...] = (0.05, 0.10, 0.20)
    coarse_blends: Tuple[float, ...] = (0.20, 0.35)
    diffusivity: float = 1.0
    zero_mean_omega: bool = True

    def validate(self) -> None:
        if not self.alphas:
            raise ValueError("at least one hierarchy level is required")
        if len(self.taus) != len(self.alphas):
            raise ValueError("taus must contain one value per hierarchy level")
        if len(self.coarse_blends) != len(self.alphas) - 1:
            raise ValueError("coarse_blends must contain one value per fusion boundary")
        if not all(math.isfinite(alpha) and 0.0 < alpha <= 1.0 for alpha in self.alphas):
            raise ValueError("all fractional orders must lie in (0, 1]")
        if not all(math.isfinite(tau) and tau >= 0.0 for tau in self.taus):
            raise ValueError("all smoothing times must be non-negative")
        if not all(
            math.isfinite(blend) and 0.0 <= blend <= 1.0
            for blend in self.coarse_blends
        ):
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
    traces: Tuple[LevelTrace, ...]
    instructions: Tuple[MechanisticInstruction, ...]
    mass_before: float
    mass_after: float
    mass_drift: float
    update_rms: float


@dataclass(frozen=True)
class EquilibriumResult:
    field: Grid3D
    converged: bool
    cycles: int
    update_history: Tuple[float, ...]
    residual_history: Tuple[float, ...]


def _validate_hierarchy_shape(shape: Shape3D, levels: int) -> None:
    nx, ny, nz = shape
    divisor = 2 ** (levels - 1)
    if nx % divisor or ny % divisor or nz % divisor:
        raise ValueError(
            "each dimension must be divisible by 2**(levels - 1)"
        )


def _blend_fields(local: Grid3D, coarse_projection: Grid3D, weight: float) -> Grid3D:
    if local.shape != coarse_projection.shape:
        raise ValueError("fields must share a shape before blending")
    return Grid3D(
        local.shape,
        tuple(
            (1.0 - weight) * left + weight * right
            for left, right in zip(local.values, coarse_projection.values)
        ),
    )


def hierarchical_fractional_smooth(
    field: Grid3D,
    config: Optional[FractionalHierarchyConfig] = None,
    omega: Optional[Grid3D] = None,
) -> HierarchicalSmoothingResult:
    """Run one full fine-to-coarse-to-fine smoothing transaction."""

    active_config = config or FractionalHierarchyConfig()
    active_config.validate()
    levels = len(active_config.alphas)
    _validate_hierarchy_shape(field.shape, levels)
    if omega is not None and omega.shape != field.shape:
        raise ValueError("omega shape must match field shape")

    grids = [field]
    omega_grids: List[Optional[Grid3D]] = [omega]
    instructions: List[MechanisticInstruction] = []
    sequence = 0

    for level in range(1, levels):
        grids.append(restrict_half(grids[-1]))
        omega_grids.append(
            restrict_half(omega_grids[-1]) if omega_grids[-1] is not None else None
        )
        instructions.append(
            MechanisticInstruction(
                sequence,
                "RESTRICT_2X2X2",
                level,
                "%s -> %s" % (grids[level - 1].shape, grids[level].shape),
            )
        )
        sequence += 1

    local_results: List[Grid3D] = []
    traces: List[LevelTrace] = []
    for level, source in enumerate(grids):
        alpha = active_config.alphas[level]
        tau = active_config.taus[level]
        result = spectral_fractional_step(
            source,
            alpha,
            tau,
            active_config.diffusivity,
            omega_grids[level],
            active_config.zero_mean_omega,
        )
        local_results.append(result)
        traces.append(
            LevelTrace(
                level,
                source.shape,
                alpha,
                tau,
                source.mean,
                result.mean,
                source.variance,
                result.variance,
                classical_gradient_energy(source),
                classical_gradient_energy(result),
                equilibrium_residual(
                    result,
                    alpha,
                    active_config.diffusivity,
                    omega_grids[level],
                    active_config.zero_mean_omega,
                ),
            )
        )
        instructions.append(
            MechanisticInstruction(
                sequence,
                "FRACTIONAL_HEAT_3D",
                level,
                "alpha=%.6f tau=%.6f shape=%s" % (alpha, tau, source.shape),
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
                "%s -> %s" % (fused.shape, projected.shape),
            )
        )
        sequence += 1
        weight = active_config.coarse_blends[level]
        fused = _blend_fields(local_results[level], projected, weight)
        instructions.append(
            MechanisticInstruction(
                sequence,
                "FUSE_COARSE_FINE",
                level,
                "coarse_weight=%.6f" % weight,
            )
        )
        sequence += 1

    update_rms = _rms(
        tuple(after - before for after, before in zip(fused.values, field.values))
    )
    mass_before = field.mass
    mass_after = fused.mass
    instructions.append(
        MechanisticInstruction(
            sequence,
            "VERIFY_MASS_AND_UPDATE",
            0,
            "mass_drift=%.12e update_rms=%.12e"
            % (mass_after - mass_before, update_rms),
        )
    )

    return HierarchicalSmoothingResult(
        fused,
        tuple(traces),
        tuple(instructions),
        mass_before,
        mass_after,
        mass_after - mass_before,
        update_rms,
    )


def run_to_equilibrium(
    field: Grid3D,
    config: Optional[FractionalHierarchyConfig] = None,
    omega: Optional[Grid3D] = None,
    tolerance: float = 1.0e-6,
    max_cycles: int = 100,
) -> EquilibriumResult:
    """Repeat hierarchical smoothing until the field update becomes negligible."""

    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    if max_cycles < 1:
        raise ValueError("max_cycles must be positive")
    active_config = config or FractionalHierarchyConfig()
    active_config.validate()

    current = field
    updates: List[float] = []
    residuals: List[float] = []
    for cycle in range(1, max_cycles + 1):
        result = hierarchical_fractional_smooth(current, active_config, omega)
        current = result.field
        updates.append(result.update_rms)
        residuals.append(
            equilibrium_residual(
                current,
                active_config.alphas[0],
                active_config.diffusivity,
                omega,
                active_config.zero_mean_omega,
            )
        )
        if result.update_rms <= tolerance:
            return EquilibriumResult(
                current,
                True,
                cycle,
                tuple(updates),
                tuple(residuals),
            )

    return EquilibriumResult(
        current,
        False,
        max_cycles,
        tuple(updates),
        tuple(residuals),
    )


def grid_from_values(shape: Shape3D, values: Iterable[float]) -> Grid3D:
    """Convenience constructor that normalises an iterable to immutable storage."""

    return Grid3D(shape, tuple(float(value) for value in values))
