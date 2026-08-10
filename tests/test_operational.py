import pytest

from jarvisx.operational import capability_manifest, execute_source


def test_execute_source_returns_verified_receipt() -> None:
    receipt = execute_source(
        "\n".join(
            (
                "SET Ψ 7",
                "SET Φ 5",
                "ADD A Ψ Φ",
                "SUB B Ψ Φ",
                "HALT",
            )
        )
    )

    assert receipt.registers["A"] == 12
    assert receipt.registers["B"] == 2
    assert receipt.cycles == 5
    assert receipt.ledger_entries == 5
    assert receipt.trace_entries == 5
    assert receipt.ledger_valid


def test_execute_source_rejects_empty_and_invalid_cycle_budget() -> None:
    with pytest.raises(ValueError, match="empty"):
        execute_source("   ")
    with pytest.raises(ValueError, match="max_cycles"):
        execute_source("HALT", max_cycles=0)


def test_capability_manifest_preserves_authority_boundary() -> None:
    manifest = capability_manifest()

    assert manifest["schema"] == "jarvisx.operational.v1"
    authority = manifest["authority"]
    assert isinstance(authority, dict)
    assert authority["vm_core"] == "authoritative"
    assert authority["visualization"] == "non-authoritative interface"
    invariants = manifest["invariants"]
    assert isinstance(invariants, dict)
    assert invariants["virtual_depth_is_physical_throughput"] is False
