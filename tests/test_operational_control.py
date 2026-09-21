from __future__ import annotations

from jarvisx.operational_control import (
    CANONICAL_MODULES,
    canonical_module_status,
    collect_operational_receipt,
    discover_entrypoints,
)


def test_operational_registry_exposes_all_installed_jarvisx_entrypoints():
    entrypoints = discover_entrypoints()
    names = {entry.name for entry in entrypoints}

    assert "jarvisx" in names
    assert "jarvisx-system" in names
    assert "jarvisx-dr-moagi-os" in names
    assert all(entry.present for entry in entrypoints)


def test_canonical_operational_modules_are_present():
    modules = canonical_module_status()

    assert tuple(item.module for item in modules) == CANONICAL_MODULES
    assert all(item.present for item in modules)


def test_unified_operational_verification_executes_bounded_authority_paths():
    receipt = collect_operational_receipt(run_smoke=True)

    assert receipt.ok
    assert receipt.core_transaction is not None
    assert receipt.core_transaction["committed"] is True
    assert receipt.core_transaction["runtime_verified"] is True
    assert receipt.core_transaction["observed_omega"] == 18

    assert receipt.sparse_3d_cycle is not None
    assert receipt.sparse_3d_cycle["committed"] is True
    assert receipt.sparse_3d_cycle["journal_valid"] is True
    assert receipt.sparse_3d_cycle["active_cells_after"] > 0
    assert receipt.sparse_3d_cycle["transport_bytes"] > 0
