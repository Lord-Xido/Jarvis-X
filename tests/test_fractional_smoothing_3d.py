import math

import pytest

from jarvisx.fractional_smoothing_3d import (
    FractionalHierarchyConfig,
    Grid3D,
    classical_gradient_energy,
    equilibrium_residual,
    hierarchical_fractional_smooth,
    prolong_double,
    restrict_half,
    round_trip_error,
    run_to_equilibrium,
    spectral_fractional_step,
)


def test_constant_field_is_a_fractional_equilibrium():
    field = Grid3D.constant((4, 4, 4), 3.25)
    result = spectral_fractional_step(field, alpha=0.6, tau=1.0)

    assert result.values == pytest.approx(field.values, abs=1.0e-12)
    assert equilibrium_residual(result, alpha=0.6) == pytest.approx(0.0, abs=1.0e-12)


def test_fractional_step_preserves_mass_and_reduces_variance():
    field = Grid3D.impulse((4, 4, 4), (1, 1, 1), amplitude=8.0)
    result = spectral_fractional_step(field, alpha=0.7, tau=0.2)

    assert result.mass == pytest.approx(field.mass, abs=1.0e-10)
    assert result.variance < field.variance
    assert max(result.values) < max(field.values)
    assert min(result.values) > -1.0e-12


def test_coarse_expansion_contraction_is_identity():
    coarse = Grid3D(
        (2, 2, 2),
        tuple(float(index) for index in range(8)),
    )

    expanded = prolong_double(coarse)
    reconstructed = restrict_half(expanded)

    assert reconstructed.values == pytest.approx(coarse.values, abs=1.0e-12)
    assert round_trip_error(coarse) == pytest.approx(0.0, abs=1.0e-12)


def test_hierarchy_emits_mechanistic_3d_instruction_trace():
    field = Grid3D.impulse((4, 4, 4), (0, 0, 0), amplitude=4.0)
    config = FractionalHierarchyConfig(
        alphas=(1.0, 0.5),
        taus=(0.05, 0.20),
        coarse_blends=(0.30,),
    )

    result = hierarchical_fractional_smooth(field, config)
    opcodes = tuple(instruction.opcode for instruction in result.instructions)

    assert len(result.traces) == 2
    assert result.traces[0].shape == (4, 4, 4)
    assert result.traces[1].shape == (2, 2, 2)
    assert "RESTRICT_2X2X2" in opcodes
    assert opcodes.count("FRACTIONAL_HEAT_3D") == 2
    assert "PROLONG_2X2X2" in opcodes
    assert "FUSE_COARSE_FINE" in opcodes
    assert opcodes[-1] == "VERIFY_MASS_AND_UPDATE"


def test_hierarchy_preserves_mass_and_smooths_gradient():
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
        taus=(0.08, 0.20),
        coarse_blends=(0.25,),
    )

    result = hierarchical_fractional_smooth(field, config)

    assert result.mass_after == pytest.approx(result.mass_before, abs=1.0e-9)
    assert abs(result.mass_drift) < 1.0e-9
    assert classical_gradient_energy(result.field) < classical_gradient_energy(field)
    assert result.update_rms > 0.0


def test_zero_mean_omega_preserves_total_mass():
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


def test_repeated_smoothing_moves_toward_equilibrium():
    field = Grid3D.impulse((4, 4, 4), (1, 2, 3), amplitude=1.0)
    config = FractionalHierarchyConfig(
        alphas=(1.0, 0.65),
        taus=(0.20, 0.35),
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
