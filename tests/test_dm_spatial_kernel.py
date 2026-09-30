import math

import pytest

from jarvisx.dm_spatial_kernel import (
    OME6400SpatialKernel,
    SpatialInstruction,
    SpatialInputs,
    SpatialKernelConfig,
    SpatialOpcode,
    SpatialState,
    assemble_program,
    decode_program,
    default_program,
)


def test_default_program_is_real_fixed_width_bytecode() -> None:
    program = default_program()
    payload = assemble_program(program)

    assert len(program) == 8
    assert len(payload) == 64
    assert decode_program(payload) == program
    assert program[-1].opcode is SpatialOpcode.HALT


def test_geometry_matches_three_axis_equation() -> None:
    state = SpatialState(psi=0.5, theta=0.25, xi=0.75, omega=1.0)
    kernel = OME6400SpatialKernel(state=state)

    axes, singularity = kernel.geometry()

    assert axes[0] == pytest.approx(kernel.config.phi * state.xi)
    assert axes[1] == pytest.approx(state.psi * state.theta)
    assert axes[2] == pytest.approx(kernel.config.recursion_base**state.omega)
    assert singularity == pytest.approx(state.psi * kernel.config.phi / axes[2])


def test_cycle_is_deterministic_and_bounded() -> None:
    inputs = SpatialInputs(
        observation=0.75,
        intent=0.4,
        prediction=0.2,
        refinement=0.1,
        grad_theta=-0.05,
        grad_h=0.025,
    )
    first = OME6400SpatialKernel()
    second = OME6400SpatialKernel()

    a = first.run(6, inputs)
    b = second.run(6, inputs)

    assert a == b
    assert first.state == second.state
    assert len(a.digest) == 64
    for value in (first.state.psi, first.state.theta, first.state.xi, first.state.omega):
        assert abs(value) <= first.config.projection_limit
        assert math.isfinite(value)


def test_virtual_rom_capacity_does_not_allocate_6_4_gb() -> None:
    kernel = OME6400SpatialKernel()

    assert kernel.virtual_rom_bytes == 6_400_000_000
    assert kernel.physical_rom_bytes == 64


def test_error_drives_pulse_telemetry() -> None:
    low = OME6400SpatialKernel(state=SpatialState(psi=0.0, theta=1.0))
    high = OME6400SpatialKernel(state=SpatialState(psi=0.0, theta=1.0))

    low_frame = low.execute_cycle(SpatialInputs(observation=0.0))
    high_frame = high.execute_cycle(SpatialInputs(observation=1.0))

    assert high_frame.pulse > low_frame.pulse


def test_invalid_inputs_are_rejected_without_mutation() -> None:
    kernel = OME6400SpatialKernel()
    before = kernel.state

    with pytest.raises(ValueError, match="finite"):
        kernel.execute_cycle(SpatialInputs(observation=float("nan")))

    assert kernel.state == before


def test_program_without_halt_is_rejected() -> None:
    with pytest.raises(ValueError, match="HALT"):
        OME6400SpatialKernel(program=(SpatialInstruction(SpatialOpcode.ENCODE),))


def test_cycle_budget_is_enforced() -> None:
    kernel = OME6400SpatialKernel(SpatialKernelConfig(max_cycles=2))
    kernel.run(2)

    with pytest.raises(RuntimeError, match="maximum cycle budget"):
        kernel.execute_cycle()


def test_bad_opcode_and_bad_bytecode_width_are_rejected() -> None:
    with pytest.raises(ValueError, match="multiple of 8"):
        decode_program(b"abc")
    with pytest.raises(ValueError, match="unsupported spatial opcode"):
        SpatialInstruction.from_bytes(bytes([0x01, 0, 0, 0, 0, 0, 0, 0]))
