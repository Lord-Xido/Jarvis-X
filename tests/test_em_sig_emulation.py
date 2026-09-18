import math

from jarvisx.em_sig_emulation import (
    EMSigConfig,
    compression_design_estimate,
    create_world,
    modeled_rate_arithmetic,
    qpsk_map,
    run_emulation,
)


def test_qpsk_mapping_has_four_unit_energy_states() -> None:
    bits = (0, 0, 0, 1, 1, 1, 1, 0)
    symbols = qpsk_map(bits)

    assert len(set(symbols)) == 4
    assert all(math.isclose(abs(symbol), 1.0, rel_tol=0.0, abs_tol=1.0e-12) for symbol in symbols)


def test_120_gsas_rate_arithmetic_is_dimensionally_explicit() -> None:
    config = EMSigConfig(bit_count=4096, materialized_extent=6)
    timing = modeled_rate_arithmetic(config)

    assert timing["modeled_payload_rate_bps"] == 240.0e9
    assert math.isclose(timing["one_billion_bits_seconds"], 1.0e9 / 240.0e9)
    assert math.isclose(timing["one_trillion_bits_seconds"], 1.0e12 / 240.0e9)
    assert math.isclose(timing["modeled_1000_hop_propagation_seconds"], 140.0e-12)
    assert timing["one_trillion_bits_seconds"] > 1.0


def test_em_sig_trace_converges_without_overclaiming_physics() -> None:
    config = EMSigConfig(
        materialized_extent=6,
        bit_count=4096,
        correction_fraction=1.0,
        max_correction_cycles=3,
        seed=7,
    )
    report = run_emulation(config)

    assert report["schema_version"] == "jarvisx.em-sig-emulation.v1"
    assert report["provenance"] == "simulated"

    layers = report["layers"]
    l3 = layers["L3_spatial_field"]
    l4 = layers["L4_latent"]
    l5 = layers["L5_residual_correction"]

    assert l3["configured_array_elements"] == 1024
    assert l3["materialized_voxels"] == 6**3
    assert "not Poynting flux" in l3["boundary"]

    assert 0 <= l4["target_state_6bit"] <= 63
    assert 0 <= l4["final_state_6bit"] <= 63
    assert l4["latent_fixed_point"] is True

    trace = l5["trace"]
    errors = [row["errors"] for row in trace]
    assert errors == sorted(errors, reverse=True)
    assert l5["final_errors"] == 0
    assert l5["fixed_point"] is True
    assert l5["empirical_single_error_resolution"] == 1.0 / config.bit_count
    assert "does not verify a physical BER" in l5["boundary"]

    boundary = report["claim_boundary"]
    assert "1e-12 physical BER" in boundary["not_verified_by_this_run"]
    assert "120 GSa/s hardware" in boundary["not_verified_by_this_run"]


def test_world_is_seed_reproducible() -> None:
    assert create_world(128, 5) == create_world(128, 5)
    assert create_world(128, 5) != create_world(128, 6)


def test_sparse_compression_accounts_for_coordinates() -> None:
    estimate = compression_design_estimate(EMSigConfig())

    assert estimate["coordinate_bits_per_axis"] == 10
    assert estimate["indexed_bits_per_node"] == 36
    assert estimate["indexed_payload_bytes_minimum"] > estimate["implicit_coordinate_payload_bytes_optimistic"]
    assert estimate["indexed_compression_ratio"] < estimate["implicit_coordinate_compression_ratio_optimistic"]
