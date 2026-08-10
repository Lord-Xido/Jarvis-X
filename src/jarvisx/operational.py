"""Dependency-light operational facade for canonical Jarvis-X execution paths."""

from __future__ import annotations

from dataclasses import dataclass

from .assembler import Assembler
from .core import CodexVM
from .parser import Parser


@dataclass(frozen=True)
class VMExecutionReceipt:
    """Observable result of one isolated deterministic VM execution."""

    registers: dict[str, int]
    cycles: int
    ledger_entries: int
    ledger_valid: bool
    trace_entries: int

    def to_dict(self) -> dict[str, object]:
        return {
            "registers": dict(self.registers),
            "cycles": self.cycles,
            "ledger_entries": self.ledger_entries,
            "ledger_valid": self.ledger_valid,
            "trace_entries": self.trace_entries,
        }


def execute_source(source: str, *, max_cycles: int = 10_000) -> VMExecutionReceipt:
    """Parse, assemble and execute source in one fresh transactional CodexVM."""

    if not isinstance(source, str):
        raise TypeError("source must be a string")
    if not source.strip():
        raise ValueError("source cannot be empty")
    if max_cycles < 1:
        raise ValueError("max_cycles must be positive")

    ast = Parser().parse(source)
    bytecode = Assembler().assemble(ast)
    vm = CodexVM(max_cycles=max_cycles)
    vm.load(bytecode)
    registers = vm.run()
    return VMExecutionReceipt(
        registers=registers,
        cycles=vm.cycles,
        ledger_entries=len(vm.ledger.chain),
        ledger_valid=vm.ledger.verify(),
        trace_entries=len(vm.tracer.log),
    )


def capability_manifest() -> dict[str, object]:
    """Return the implemented-versus-bounded operational surface exposed by this package."""

    return {
        "schema": "jarvisx.operational.v1",
        "system": "Jarvis-X",
        "authority": {
            "vm_core": "authoritative",
            "omega_journal": "authoritative provenance",
            "dr_moagi_codec_3d": "alpha bounded reference",
            "visualization": "non-authoritative interface",
        },
        "invariants": {
            "deterministic_vm": True,
            "transactional_vm": True,
            "bounded_vm_cycles": True,
            "dr_moagi_codec_3d_reference": True,
            "codec_integrity_digest": True,
            "codec_anchor_preservation": True,
            "codec_commit_or_rollback": True,
            "virtual_depth_is_physical_throughput": False,
        },
    }
