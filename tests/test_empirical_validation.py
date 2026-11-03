import json

import pytest

from jarvisx.empirical_validation import main, run_validation, write_report


def test_empirical_validation_passes_and_is_json_native(tmp_path) -> None:
    report = run_validation(repetitions=4, octree_max_depth=3)
    output = tmp_path / "evidence" / "report.json"

    write_report(report, output)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert report.passed
    assert payload["schema_version"] == "jarvisx.empirical-validation.v1"
    assert payload["summary"] == {
        "checks_passed": 6,
        "checks_total": 6,
        "passed": True,
    }
    assert {check["name"] for check in payload["checks"]} == {
        "vm_deterministic_replay",
        "omega_ledger_tamper_detection",
        "sparse_field_transactionality",
        "fractal_octree_closed_form",
        "fractional_smoothing_invariants",
        "dm3d_rom_transactional_closure",
    }


def test_repeatable_checks_keep_identical_deterministic_digests() -> None:
    first = run_validation(repetitions=3, octree_max_depth=2)
    second = run_validation(repetitions=3, octree_max_depth=2)

    first_by_name = {check.name: check for check in first.checks}
    second_by_name = {check.name: check for check in second.checks}

    vm_keys = (
        "program_digest_sha256",
        "final_state_digest_sha256",
        "trace_digest_sha256",
    )
    for key in vm_keys:
        assert first_by_name["vm_deterministic_replay"].metrics[key] == second_by_name[
            "vm_deterministic_replay"
        ].metrics[key]

    sparse_keys = ("state_digest_sha256", "journal_digest_sha256")
    for key in sparse_keys:
        assert first_by_name["sparse_field_transactionality"].metrics[key] == second_by_name[
            "sparse_field_transactionality"
        ].metrics[key]


def test_cli_writes_report_and_returns_success(tmp_path) -> None:
    output = tmp_path / "empirical.json"

    exit_code = main(
        (
            "--output",
            str(output),
            "--repetitions",
            "3",
            "--octree-max-depth",
            "2",
        )
    )

    assert exit_code == 0
    assert json.loads(output.read_text(encoding="utf-8"))["summary"]["passed"] is True


def test_validation_rejects_non_empirical_parameters() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        run_validation(repetitions=1)
    with pytest.raises(ValueError, match="non-negative"):
        run_validation(octree_max_depth=-1)


def test_dm3d_rom_receipt_distinguishes_virtual_extent_from_resident_state() -> None:
    from jarvisx.empirical_validation import dm3d_rom_conformance_check

    result = dm3d_rom_conformance_check()
    assert result.passed
    assert result.metrics["virtual_byte_positions"] == 10**27
    assert result.metrics["resident_tiles"] == 1
    assert result.metrics["full_byte_reconstruction"] is True
    assert result.metrics["self_loop_rejected"] is True
    assert result.metrics["incumbent_state_preserved"] is True
    assert result.metrics["source_tile_bytes"] == 64**3
    assert result.metrics["quantized_latent_bytes"] == 8**3
    assert result.metrics["signed_residual_entries"] == 64**3
