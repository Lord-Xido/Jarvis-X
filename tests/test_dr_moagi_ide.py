from __future__ import annotations

from pathlib import Path

import pytest

from jarvisx.dr_moagi_ide import ANNRegistry, ProjectStore, execute_program, refactor_program


def test_execute_program_runs_real_codex_vm() -> None:
    result = execute_program("SET A 21\nSET B 21\nADD C A B\nHALT", max_cycles=16)
    assert result["registers"]["C"] == 42
    assert result["cycles"] == 4
    assert len(result["bytecode"]) == 4
    assert result["trace"][-1]["opcode"] == 0x0A


def test_execute_program_rejects_unknown_opcode() -> None:
    with pytest.raises(ValueError, match="unsupported opcode"):
        execute_program("SHELL rm -rf /\nHALT")


def test_project_store_round_trip(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "ide.sqlite3")
    created = store.save(name="demo.jx", source="SET A 1\nHALT")
    loaded = store.get(created["id"])
    assert loaded["name"] == "demo.jx"
    assert loaded["source"].endswith("HALT")
    assert store.list()[0]["id"] == created["id"]
    assert store.delete(created["id"])


def test_refactorer_is_bounded_and_emits_receipt() -> None:
    result = refactor_program(
        "SET A 2\nSET B 3\nADD C A B\nHALT\nSET D 9",
        seed=41,
        max_cycles=100,
        max_mutations=4,
    )
    assert result["total_cycles_used"] <= 100
    assert result["mutations_applied"] <= 4
    assert result["journaled"] in {True, False}
    assert "HALT" in result["output_program"]


def test_ann_registry_evaluates_and_optimizes() -> None:
    registry = ANNRegistry(max_sessions=2)
    session = registry.create(side=3, seed=41)
    values = [0.1] * session["summary"]["nodes"]
    before = registry.evaluate(session["session_id"], values)
    optimized = registry.optimize(session["session_id"], values, max_epochs=2)
    assert before["metrics"]["loss"]["total"] >= 0.0
    assert optimized["report"]["attempted_epochs"] <= 2
    assert optimized["epoch"] >= 0
