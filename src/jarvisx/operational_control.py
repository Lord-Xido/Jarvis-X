"""Unified operational control plane for Jarvis-X.

This module does not replace the individual runtimes. It provides one bounded,
machine-readable system surface that:

* discovers every installed Jarvis-X console entry point;
* verifies that each entry point resolves to an installed module;
* checks canonical authority modules are present;
* executes a deterministic SystemRuntime transaction through CTR-style commit;
* executes one bounded sparse Dr Moagi 3D OS cycle; and
* emits a single operational receipt suitable for CI and deployment gates.

The control plane deliberately avoids arbitrary host-command execution and does
not grant new capabilities to any subsystem.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Sequence

ENTRYPOINT_PREFIX = "jarvisx"
SCHEMA_VERSION = 1

CANONICAL_MODULES = (
    "jarvisx.core",
    "jarvisx.system_runtime",
    "jarvisx.dr_moagi_os",
    "jarvisx.cognitive_field_geometry",
    "jarvisx.permeation",
    "jarvisx.verification",
    "jarvisx.nexus3d_cli",
)


@dataclass(frozen=True)
class EntryPointStatus:
    name: str
    value: str
    module: str
    present: bool


@dataclass(frozen=True)
class ModuleStatus:
    module: str
    present: bool


@dataclass(frozen=True)
class OperationalReceipt:
    schema_version: int
    python_version: str
    platform: str
    entrypoints: tuple[EntryPointStatus, ...]
    canonical_modules: tuple[ModuleStatus, ...]
    core_transaction: dict[str, Any] | None
    sparse_3d_cycle: dict[str, Any] | None
    ok: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def discover_entrypoints() -> tuple[EntryPointStatus, ...]:
    """Return every installed Jarvis-X console entry point in deterministic order."""
    discovered: list[EntryPointStatus] = []
    for entry in metadata.entry_points(group="console_scripts"):
        if not entry.name.startswith(ENTRYPOINT_PREFIX):
            continue
        module = entry.value.partition(":")[0].strip()
        discovered.append(
            EntryPointStatus(
                name=entry.name,
                value=entry.value,
                module=module,
                present=bool(module) and importlib.util.find_spec(module) is not None,
            )
        )
    return tuple(sorted(discovered, key=lambda item: item.name))


def canonical_module_status() -> tuple[ModuleStatus, ...]:
    return tuple(
        ModuleStatus(module=module, present=importlib.util.find_spec(module) is not None)
        for module in CANONICAL_MODULES
    )


def _core_transaction_probe() -> dict[str, Any]:
    from .assembler import Assembler
    from .parser import Parser
    from .system_runtime import CAP_VM_EXECUTE, ExecutionRequest, SystemRuntime

    source = "SET Ψ 7\nSET Φ 11\nADD Ω Ψ Φ\nHALT"
    program = tuple(Assembler().assemble(Parser().parse(source)))
    runtime = SystemRuntime()
    receipt = runtime.execute(
        ExecutionRequest(
            request_id="operational-control-core-smoke-v1",
            program=program,
            granted_capabilities=frozenset({CAP_VM_EXECUTE}),
        )
    )
    state = receipt.state_dict()
    verified = runtime.verify()
    expected_value = 18
    ok = bool(receipt.committed and verified and state.get("Ω") == expected_value)
    return {
        "ok": ok,
        "status": receipt.status.value,
        "committed": receipt.committed,
        "cycles": receipt.cycles,
        "expected_omega": expected_value,
        "observed_omega": state.get("Ω"),
        "state_hash": receipt.state_hash,
        "vm_ledger_head": receipt.vm_ledger_head,
        "audit_head": receipt.audit_head,
        "runtime_verified": verified,
    }


def _sparse_3d_probe() -> dict[str, Any]:
    from .dr_moagi_os import DrMoagiOSConfig, DrMoagiOSKernel, demo_field

    config = DrMoagiOSConfig(
        side=16,
        max_active_cells=4_096,
        deep_distiller_max_latent_cells=2_048,
        fixed_point_passes=1,
        state_dir=None,
    )
    kernel = DrMoagiOSKernel(config)
    kernel.boot(restore=False)
    kernel.load(demo_field(config.side))
    report = kernel.step()
    status = kernel.status()
    journal_valid = kernel.journal.verify()
    ok = bool(
        report.committed
        and journal_valid
        and status.get("journal_valid") is True
        and report.active_cells_after > 0
        and report.transport_bytes > 0
    )
    return {
        "ok": ok,
        "cycle": report.cycle,
        "committed": report.committed,
        "active_cells_before": report.active_cells_before,
        "active_cells_after": report.active_cells_after,
        "latent_cells": report.latent_cells,
        "transport_bytes": report.transport_bytes,
        "transport_hash": report.transport_hash,
        "theta_hash": report.theta_hash,
        "state_hash": report.state_hash,
        "journal_hash": report.journal_hash,
        "journal_valid": journal_valid,
        "rejection_reason": report.rejection_reason,
    }


def _guarded_probe(probe: Any) -> dict[str, Any]:
    try:
        return probe()
    except Exception as exc:
        return {
            "ok": False,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }


def collect_operational_receipt(*, run_smoke: bool) -> OperationalReceipt:
    entrypoints = discover_entrypoints()
    modules = canonical_module_status()
    structural_ok = bool(entrypoints) and all(item.present for item in entrypoints)
    structural_ok = structural_ok and all(item.present for item in modules)

    core_transaction = _guarded_probe(_core_transaction_probe) if run_smoke else None
    sparse_3d_cycle = _guarded_probe(_sparse_3d_probe) if run_smoke else None

    smoke_ok = True
    if run_smoke:
        smoke_ok = bool(core_transaction and core_transaction.get("ok"))
        smoke_ok = smoke_ok and bool(sparse_3d_cycle and sparse_3d_cycle.get("ok"))

    return OperationalReceipt(
        schema_version=SCHEMA_VERSION,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        entrypoints=entrypoints,
        canonical_modules=modules,
        core_transaction=core_transaction,
        sparse_3d_cycle=sparse_3d_cycle,
        ok=bool(structural_ok and smoke_ok),
    )


def _write_receipt(
    receipt: OperationalReceipt,
    *,
    output: Path | None,
    pretty: bool,
) -> None:
    encoded = json.dumps(
        receipt.as_dict(),
        indent=2 if pretty else None,
        sort_keys=True,
    )
    print(encoded)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jarvisx-system",
        description="Unified bounded operational control plane for Jarvis-X",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status", help="verify installed system structure")
    status.add_argument("--pretty", action="store_true")
    status.add_argument("--output", type=Path)

    verify = sub.add_parser(
        "verify",
        help="run structural checks plus canonical VM and sparse 3D smoke transactions",
    )
    verify.add_argument("--pretty", action="store_true")
    verify.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = collect_operational_receipt(run_smoke=args.command == "verify")
    _write_receipt(
        receipt,
        output=args.output,
        pretty=bool(args.pretty),
    )
    return 0 if receipt.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
