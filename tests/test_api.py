import pytest
from fastapi import HTTPException

from jarvisx.api import CodecRoundTripRequest, RunRequest, codec_roundtrip, health, run_code


def test_health_reports_bounded_capability_boundary() -> None:
    payload = health()

    assert payload["status"] == "ok"
    capabilities = payload["capabilities"]
    assert isinstance(capabilities, dict)
    assert capabilities["deterministic_vm"] is True
    assert capabilities["dr_moagi_codec_3d_reference"] is True
    assert capabilities["virtual_depth_is_physical_throughput"] is False


def test_run_endpoint_executes_fresh_transactional_vm() -> None:
    result = run_code(
        RunRequest(
            source="\n".join(
                (
                    "SET Ψ 10",
                    "SET Φ 20",
                    "ADD A Ψ Φ",
                    "HALT",
                )
            )
        )
    )

    registers = result["registers"]
    assert isinstance(registers, dict)
    assert registers["A"] == 30
    assert result["cycles"] == 4
    assert result["ledger_entries"] == 4
    assert result["ledger_valid"] is True


def test_run_endpoint_rejects_empty_source() -> None:
    with pytest.raises(HTTPException) as caught:
        run_code(RunRequest(source="   "))

    assert caught.value.status_code == 400


def test_codec_endpoint_performs_bounded_round_trip() -> None:
    result = codec_roundtrip(
        CodecRoundTripRequest(
            shape=(2, 2, 2),
            values=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            quant_step=0.25,
            virtual_depth=1_000_000,
        )
    )

    assert result["committed"] is True
    assert result["virtual_depth"] == 1_000_000
    assert result["measured_microsteps_executed"] == 1
    reconstructed = result["reconstructed_values"]
    assert isinstance(reconstructed, list)
    assert len(reconstructed) == 8
