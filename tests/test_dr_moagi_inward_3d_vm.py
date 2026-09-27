import math

import pytest

from jarvisx.dr_moagi_inward_3d_vm import (
    CandidateVector3D,
    Inward3DVMConfig,
    InwardSelfOptimizing3DVM,
    SpatialInstruction3D,
    SpatialOp,
    demo_bus,
)


def _config(**kwargs):
    base = dict(
        max_refine_steps=12,
        shadow_cycles=1,
        max_shadow_candidates=3,
        meta_interval=1,
        min_score_improvement=0.0,
    )
    base.update(kwargs)
    return Inward3DVMConfig(**base)


def test_spatial_instruction_round_trip_preserves_3d_address():
    instruction = SpatialInstruction3D(
        SpatialOp.PHI3D_ITER,
        flags=0xA5,
        x=0x123,
        y=0x456,
        z=0x789,
        arg=0xABC,
    )
    assert SpatialInstruction3D.unpack(instruction.pack()) == instruction
    assert instruction.pointer == (0x123, 0x456, 0x789)


def test_spatial_instruction_rejects_out_of_range_coordinates():
    with pytest.raises(ValueError):
        SpatialInstruction3D(SpatialOp.INGEST, z=4096)


def test_program_moves_through_input_latent_verify_and_output_regions():
    vm = InwardSelfOptimizing3DVM(_config())
    z_path = [instruction.z for instruction in vm.program]

    assert z_path[0] == 0
    assert any(300 <= z <= 699 for z in z_path)
    assert any(800 <= z <= 899 for z in z_path)
    assert z_path[-2] == 980
    assert len(vm.program_binary()) == 8 * len(vm.program)


def test_frame_executes_64_channel_bus_and_emits_bounded_fixed_point_telemetry():
    vm = InwardSelfOptimizing3DVM(_config())
    vm.load_bus(demo_bus())

    report = vm.run_frame()

    assert report.cycle == 1
    assert math.isfinite(report.mse)
    assert math.isfinite(report.fixed_point_residual)
    assert 1 <= report.refine_steps <= vm.engine.config.max_refine_steps
    assert 0.0 < report.spectral_bound < 1.0
    assert 0.0 <= report.convergence_radius <= 1.0
    assert len(report.error_field) <= vm.config.max_error_voxels
    assert vm.trace[-1].opcode == "HALT"


def test_shadow_meta_search_preserves_authoritative_state_and_routes_back_to_origin():
    vm = InwardSelfOptimizing3DVM(_config())
    vm.load_bus(demo_bus())
    vm.run_frame()

    report = vm.turn_inward()

    assert report.authoritative_state_unchanged is True
    assert report.evaluated_candidates == 4
    assert report.claim_status.endswith("external_sota_unverified")
    assert report.best.metrics.spectral_bound < 1.0
    assert vm.trace[-1].opcode in {"PROMOTE", "ROLLBACK"}
    assert vm.trace[-1].route_to == (0, 0, 0)


def test_autonomic_run_executes_state_and_meta_loops():
    vm = InwardSelfOptimizing3DVM(_config(meta_interval=1))
    vm.load_bus(demo_bus())

    report = vm.run_autonomic(2)

    assert len(report.frames) == 2
    assert len(report.meta) == 2
    assert report.trace_steps > len(vm.program) * 2
    assert report.frames[-1].cycle == 2


def test_candidate_vector_is_a_bounded_3d_policy_coordinate():
    assert CandidateVector3D(-1, 0, 1).manhattan == 2
    with pytest.raises(ValueError):
        CandidateVector3D(2, 0, 0)
