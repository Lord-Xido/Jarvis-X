import cmath
import math

import pytest

from jarvisx.fractional_smoothing_3d import (
    FractionalHierarchyConfig,
    Grid3D,
    classical_gradient_energy,
    equilibrium_residual,
    forward_dft3,
    fractional_laplacian,
    grid_from_values,
    hierarchical_fractional_smooth,
    inverse_dft3,
    laplacian_eigenvalue,
    prolong_double,
    restrict_half,
    round_trip_error,
    run_to_equilibrium,
    spectral_fractional_step,
)


def direct_dft3(field: Grid3D) -> tuple[complex, ...]:
    nx, ny, nz = field.shape
    spectrum: list[complex] = []
    for kz in range(nz):
        for ky in range(ny):
            for kx in range(nx):
                total = 0j
                for z in range(nz):
                    for y in range(ny):
                        for x in range(nx):
                            phase = -2j * math.pi * (
                                kx * x / nx + ky * y / ny + kz * z / nz
                            )
                            total += field.at(x, y, z) * cmath.exp(phase)
                spectrum.append(total)
    return tuple(spectrum)


def spatial_negative_laplacian(field: Grid3D) -> Grid3D:
    nx, ny, nz = field.shape
    values: list[float] = []
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                center = field.at(x, y, z)
                neighbours = (
                    field.at((x - 1) % nx, y, z)
                    + field.at((x + 1) % nx, y, z)
                    + field.at(x, (y - 1) % ny, z)
                    + field.at(x, (y + 1) % ny, z)
                    + field.at(x, y, (z - 1) % nz)
                    + field.at(x, y, (z + 1) % nz)
                )
                values.append(6.0 * center - neighbours)
    return Grid3D(field.shape, tuple(values))


def test_separable_dft_matches_independent_direct_definition() -> None:
    field = grid_from_values((2, 2, 2), range(8))

    assert forward_dft3(field) == pytest.approx(direct_dft3(field), abs=1.0e-12)


def test_dft_round_trip_recovers_non_cubic_field() -> None:
    field = grid_from_values((2, 3, 2), (math.sin(index) for index in range(12)))

    restored = inverse_dft3(forward_dft3(field), field.shape)

    assert restored.values == pytest.approx(field.values, abs=1.0e-12)


def test_alpha_one_matches_periodic_spatial_stencil() -> None:
    field = grid_from_values(
        (3, 2, 2),
        (math.sin(index * 0.7) + 0.25 * index for index in range(12)),
    )

    spectral = fractional_laplacian(field, alpha=1.0)
    spatial = spatial_negative_laplacian(field)

    assert spectral.values == pytest.approx(spatial.values, abs=1.0e-10)


def test_laplacian_mode_bounds_and_zero_mode() -> None:
    assert laplacian_eigenvalue(0, 0, 0, (4, 4, 4)) == pytest.approx(0.0)
    assert laplacian_eigenvalue(2, 2, 2, (4, 4, 4)) == pytest.approx(12.0)
    with pytest.raises(ValueError, match="outside"):
        laplacian_eigenvalue(4, 0, 0, (4, 4, 4))


def test_constant_field_is_fractional_equilibrium() -> None:
    field = Grid3D.constant((4, 4, 4), 3.25)

    result = spectral_fractional_step(field, alpha=0.6, tau=1.0)

    assert result.values == pytest.approx(field.values, abs=1.0e-12)
    assert equilibrium_residual(result, alpha=0.6) == pytest.approx(0.0, abs=1.0e-12)


def test_fractional_step_has_semigroup_property_without_forcing() -> None:
    field = Grid3D.impulse((4, 4, 4), (1, 2, 3), amplitude=2.0)

    split = spectral_fractional_step(field, alpha=0.7, tau=0.125)
    split = spectral_fractional_step(split, alpha=0.7, tau=0.275)
    combined = spectral_fractional_step(field, alpha=0.7, tau=0.4)

    assert split.values == pytest.approx(combined.values, abs=1.0e-11)


def test_fractional_step_preserves_mass_and_reduces_variance() -> None:
    field = Grid3D.impulse((4, 4, 4), (1, 1, 1), amplitude=8.0)

    result = spectral_fractional_step(field, alpha=0.7, tau=0.2)

    assert result.mass == pytest.approx(field.mass, abs=1.0e-10)
    assert result.variance < field.variance
    assert max(result.values) < max(field.values)
    assert min(result.values) > -1.0e-12


def test_zero_diffusivity_applies_forcing_in_physical_space() -> None:
    field = Grid3D.constant((2, 2, 2), 1.0)
    omega = grid_from_values((2, 2, 2), range(8))

    result = spectral_fractional_step(
        field,
        alpha=0.5,
        tau=0.25,
        diffusivity=0.0,
        omega=omega,
        zero_mean_omega=False,
    )

    expected = tuple(1.0 + 0.25 * value for value in omega.values)
    assert result.values == pytest.approx(expected)


def test_zero_mean_omega_preserves_total_mass() -> None:
    field = Grid3D.constant((4, 4, 4), 1.0)
    omega = Grid3D(
        (4, 4, 4),
        tuple(1.0 if index == 0 else -1.0 / 63.0 for index in range(64)),
    )

    result = spectral_fractional_step(
        field,
        alpha=0.75,
        tau=0.1,
        omega=omega,
        zero_mean_omega=True,
    )

    assert result.mass == pytest.approx(field.mass, abs=1.0e-9)
    assert result.values != pytest.approx(field.values, abs=1.0e-12)


def test_tau_zero_returns_authoritative_field_object() -> None:
    field = Grid3D.impulse((2, 2, 2), (0, 0, 0))

    assert spectral_fractional_step(field, alpha=0.5, tau=0.0) is field


def test_coarse_expansion_contraction_is_identity() -> None:
    coarse = grid_from_values((2, 2, 2), range(8))

    expanded = prolong_double(coarse)
    reconstructed = restrict_half(expanded)

    assert reconstructed.values == pytest.approx(coarse.values, abs=1.0e-12)
    assert round_trip_error(coarse) == pytest.approx(0.0, abs=1.0e-12)


def test_hierarchy_emits_ordered_mechanistic_trace() -> None:
    field = Grid3D.impulse((4, 4, 4), (0, 0, 0), amplitude=4.0)
    config = FractionalHierarchyConfig(
        alphas=(1.0, 0.5),
        taus=(0.05, 0.2),
        coarse_blends=(0.3,),
    )

    result = hierarchical_fractional_smooth(field, config)
    opcodes = tuple(instruction.opcode for instruction in result.instructions)
    sequences = tuple(instruction.sequence for instruction in result.instructions)

    assert len(result.traces) == 2
    assert result.traces[0].shape == (4, 4, 4)
    assert result.traces[1].shape == (2, 2, 2)
    assert opcodes == (
        "RESTRICT_2X2X2",
        "FRACTIONAL_HEAT_3D",
        "FRACTIONAL_HEAT_3D",
        "PROLONG_2X2X2",
        "FUSE_COARSE_FINE",
        "VERIFY_MASS_AND_UPDATE",
    )
    assert sequences == tuple(range(len(sequences)))


def test_hierarchy_preserves_mass_and_reduces_gradient_energy() -> None:
    values = tuple(
        math.sin(2.0 * math.pi * x / 4.0)
        + 0.5 * math.cos(2.0 * math.pi * y / 4.0)
        + (1.0 if (x + y + z) % 2 else -1.0)
        for z in range(4)
        for y in range(4)
        for x in range(4)
    )
    field = Grid3D((4, 4, 4), values)
    config = FractionalHierarchyConfig(
        alphas=(1.0, 0.6),
        taus=(0.08, 0.2),
        coarse_blends=(0.25,),
    )

    result = hierarchical_fractional_smooth(field, config)

    assert result.mass_after == pytest.approx(result.mass_before, abs=1.0e-9)
    assert abs(result.mass_drift) < 1.0e-9
    assert classical_gradient_energy(result.field) < classical_gradient_energy(field)
    assert result.update_rms > 0.0


def test_repeated_smoothing_moves_toward_equilibrium() -> None:
    field = Grid3D.impulse((4, 4, 4), (1, 2, 3), amplitude=1.0)
    config = FractionalHierarchyConfig(
        alphas=(1.0, 0.65),
        taus=(0.2, 0.35),
        coarse_blends=(0.25,),
    )

    result = run_to_equilibrium(
        field,
        config,
        tolerance=1.0e-5,
        max_cycles=80,
    )

    assert result.converged
    assert result.update_history[-1] <= 1.0e-5
    assert result.residual_history[-1] < result.residual_history[0]
    assert result.field.mass == pytest.approx(field.mass, abs=1.0e-8)


def test_validation_rejects_invalid_geometry_and_parameters() -> None:
    with pytest.raises(ValueError, match="positive"):
        Grid3D((0, 1, 1), ())
    with pytest.raises(ValueError, match="value count"):
        Grid3D((2, 2, 2), (0.0,))
    with pytest.raises(ValueError, match="finite"):
        Grid3D.constant((1, 1, 1), math.inf)
    with pytest.raises(ValueError, match="outside"):
        Grid3D.impulse((2, 2, 2), (2, 0, 0))
    with pytest.raises(IndexError, match="out of range"):
        Grid3D.constant((1, 1, 1), 0.0).at(1, 0, 0)
    with pytest.raises(ValueError, match="alpha"):
        spectral_fractional_step(Grid3D.constant((1, 1, 1), 0.0), 0.0, 0.1)
    with pytest.raises(ValueError, match="tau"):
        spectral_fractional_step(Grid3D.constant((1, 1, 1), 0.0), 0.5, -0.1)
    with pytest.raises(ValueError, match="diffusivity"):
        equilibrium_residual(Grid3D.constant((1, 1, 1), 0.0), 0.5, -1.0)
    with pytest.raises(ValueError, match="even"):
        restrict_half(Grid3D.constant((3, 2, 2), 0.0))
    with pytest.raises(ValueError, match="spectrum size"):
        inverse_dft3((0j,), (2, 2, 2))


def test_hierarchy_configuration_and_convergence_bounds_are_enforced() -> None:
    field = Grid3D.constant((4, 4, 4), 0.0)

    with pytest.raises(ValueError, match="one value per hierarchy"):
        FractionalHierarchyConfig(alphas=(1.0, 0.5), taus=(0.1,)).validate()
    with pytest.raises(ValueError, match="fusion boundary"):
        FractionalHierarchyConfig(
            alphas=(1.0, 0.5),
            taus=(0.1, 0.2),
            coarse_blends=(),
        ).validate()
    with pytest.raises(ValueError, match="divisible"):
        hierarchical_fractional_smooth(
            Grid3D.constant((3, 4, 4), 0.0),
            FractionalHierarchyConfig(
                alphas=(1.0, 0.5),
                taus=(0.1, 0.2),
                coarse_blends=(0.5,),
            ),
        )
    with pytest.raises(ValueError, match="tolerance"):
        run_to_equilibrium(field, tolerance=0.0)
    with pytest.raises(ValueError, match="max_cycles"):
        run_to_equilibrium(field, max_cycles=0)
