import math

from jarvisx.em_sig_3d_runtime import (
    EMSig3DConfig,
    build_source_program,
    logic_volume_stats,
    programmed_interference_field,
    run_3d_runtime,
    sparse_sensor_readout,
    spatial_latent_6bit,
    threshold_logic_volume,
)
from jarvisx.em_sig_emulation import EMSigConfig, create_world, qpsk_map


def test_qpsk_symbols_program_spatial_sources() -> None:
    symbols = qpsk_map((0, 0, 0, 1, 1, 1, 1, 0))
    program = build_source_program(
        symbols,
        source_count=4,
        radius=1.8,
        z_offset=0.7,
    )

    assert len(program) == 4
    assert all(math.isclose(row.amplitude, 1.0, abs_tol=1.0e-12) for row in program)
    assert len({round(row.phase_radians, 12) for row in program}) == 4
    assert {row.z for row in program} == {-0.7, 0.7}


def test_programmed_field_is_data_dependent_and_three_dimensional() -> None:
    em = EMSigConfig(
        materialized_extent=6,
        bit_count=4096,
        field_sources=8,
        seed=7,
    )
    config = EMSig3DConfig(em=em)

    bits_a = create_world(em.bit_count, 7)
    bits_b = create_world(em.bit_count, 8)
    program_a = build_source_program(
        qpsk_map(bits_a),
        source_count=em.field_sources,
        radius=config.source_radius,
        z_offset=config.source_z_offset,
    )
    program_b = build_source_program(
        qpsk_map(bits_b),
        source_count=em.field_sources,
        radius=config.source_radius,
        z_offset=config.source_z_offset,
    )

    field_a, peak_a = programmed_interference_field(config, program_a)
    field_b, peak_b = programmed_interference_field(config, program_b)

    assert len(field_a) == 6**3
    assert len(field_b) == 6**3
    assert peak_a > 0.0
    assert peak_b > 0.0
    assert field_a != field_b
    assert min(field_a) >= 0.0
    assert max(field_a) <= 1.0


def test_ternary_volume_and_sparse_sensor_geometry_are_accounted() -> None:
    extent = 6
    field = tuple(index / (extent**3 - 1) for index in range(extent**3))
    readout, logic = threshold_logic_volume(
        field,
        extent=extent,
        deadband=0.20,
    )
    stats = logic_volume_stats(logic)
    sensors = sparse_sensor_readout(
        readout,
        logic,
        extent=extent,
        stride=2,
    )

    assert stats["voxels"] == extent**3
    assert stats["negative"] + stats["deadband"] + stats["positive"] == extent**3
    assert set(logic) == {-1, 0, 1}
    assert len(sensors) == 3**3
    assert all(0 <= row.x < extent for row in sensors)
    assert all(0 <= row.y < extent for row in sensors)
    assert all(0 <= row.z < extent for row in sensors)
    assert 0 <= spatial_latent_6bit(logic, sensors) <= 63


def test_end_to_end_3d_runtime_reaches_bounded_fixed_points() -> None:
    em = EMSigConfig(
        materialized_extent=6,
        bit_count=4096,
        field_sources=8,
        correction_fraction=1.0,
        max_correction_cycles=3,
        seed=13,
    )
    report = run_3d_runtime(
        EMSig3DConfig(
            em=em,
            logic_deadband=0.20,
            sensor_stride=2,
        )
    )

    assert report["schema_version"] == "jarvisx.em-sig-3d-runtime.v1"
    assert report["provenance"] == "simulated"

    layers = report["layers"]
    assert layers["L1_source_program_3d"]["source_count"] == 8
    assert layers["L2_field_volume_3d"]["voxels"] == 6**3
    assert "not a Maxwell" in layers["L2_field_volume_3d"]["boundary"]

    stats = layers["L3_logic_volume_3d"]["stats"]
    assert stats["negative"] + stats["deadband"] + stats["positive"] == 6**3

    assert layers["L4_sparse_sensors"]["count"] == 3**3
    assert layers["L5_spatial_latent"]["fixed_point"] is True
    assert layers["L5_spatial_latent"]["invertible"] is False

    decoder = layers["L6_residual_decoder"]
    assert decoder["final_errors"] == 0
    assert decoder["fixed_point"] is True
    assert decoder["source_recovered"] is True

    verification = report["verification"]
    assert verification == {
        "latent_fixed_point": True,
        "decoder_fixed_point": True,
        "source_recovered": True,
        "logic_volume_accounted": True,
    }

    boundary = report["claim_boundary"]
    assert "physical RF transmission" in boundary["not_verified"]
    assert "measured electromagnetic logic" in boundary["not_verified"]


def test_runtime_is_seed_reproducible() -> None:
    em = EMSigConfig(
        materialized_extent=6,
        bit_count=4096,
        field_sources=8,
        correction_fraction=1.0,
        seed=21,
    )
    config = EMSig3DConfig(em=em, sensor_stride=3)

    assert run_3d_runtime(config) == run_3d_runtime(config)
