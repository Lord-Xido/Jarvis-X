"""Operational services for the browser-facing Dr Moagi ANN IDE.

This module deliberately exposes the Jarvis-X bounded VM, deterministic code
refactorer, inward-4D ANN and persistent project store. It does not expose an
arbitrary operating-system shell or unrestricted Python execution.
"""

from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from .assembler import Assembler, REG_MAP
from .code_editor_automata import (
    ConfidenceThresholdPolicy,
    CycleImprovementPolicy,
    DeterministicCodeEditingAutomata,
    MemorySafetyPolicy,
    RefactoringParameter,
    TransformType,
    TransformWhitelistPolicy,
)
from .core import CodexVM
from .inward4d_ann import Inward4DANN, Inward4DConfig
from .parser import Parser

MAX_SOURCE_BYTES = 256 * 1024
MAX_TRACE_ROWS = 2_000
MAX_ANN_SESSIONS = 16


def jsonable(value: Any) -> Any:
    """Convert dataclasses/tuples recursively into JSON-compatible values."""

    if is_dataclass(value) and not isinstance(value, type):
        return {key: jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    return value


def _normalized_source(source: str) -> str:
    if not isinstance(source, str):
        raise TypeError("source must be text")
    if len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise ValueError(f"source exceeds {MAX_SOURCE_BYTES} bytes")

    lines: list[str] = []
    for raw in source.splitlines():
        code = raw.split("#", 1)[0].strip()
        if code:
            lines.append(code)
    if not lines:
        raise ValueError("program is empty")
    if lines[-1].split()[0].upper() != "HALT":
        lines.append("HALT")
    return "\n".join(lines)


def _validate_ast(ast: list[list[str]]) -> None:
    specs = {"SET": 3, "ADD": 4, "SUB": 4, "HALT": 1}
    for line_number, node in enumerate(ast, start=1):
        if not node:
            continue
        opcode = node[0].upper()
        if opcode not in specs:
            raise ValueError(f"line {line_number}: unsupported opcode {node[0]!r}")
        if len(node) != specs[opcode]:
            raise ValueError(
                f"line {line_number}: {opcode} expects {specs[opcode] - 1} operand(s)"
            )
        node[0] = opcode
        if opcode == "SET":
            if node[1] not in REG_MAP:
                raise ValueError(f"line {line_number}: unknown register {node[1]!r}")
            try:
                immediate = int(node[2], 10)
            except ValueError as exc:
                raise ValueError(f"line {line_number}: SET immediate must be an integer") from exc
            if not 0 <= immediate <= (1 << 48) - 1:
                raise ValueError(f"line {line_number}: SET immediate is outside unsigned 48-bit range")
        elif opcode in {"ADD", "SUB"}:
            for register in node[1:]:
                if register not in REG_MAP:
                    raise ValueError(
                        f"line {line_number}: unknown register {register!r}"
                    )


def execute_program(
    source: str,
    *,
    max_cycles: int = 10_000,
    enable_reflex: bool = False,
) -> dict[str, Any]:
    """Parse, assemble and transactionally execute one bounded Jarvis-X program."""

    if isinstance(max_cycles, bool) or not isinstance(max_cycles, int) or not 1 <= max_cycles <= 100_000:
        raise ValueError("max_cycles must be an integer in [1, 100000]")

    normalized = _normalized_source(source)
    ast = Parser().parse(normalized)
    _validate_ast(ast)
    bytecode = Assembler().assemble(ast)
    if not bytecode:
        raise ValueError("assembler produced no bytecode")

    vm = CodexVM(enable_reflex=enable_reflex, max_cycles=max_cycles)
    vm.load(bytecode)
    registers = vm.run()
    trace = [
        {"opcode": opcode, "registers": snapshot}
        for opcode, snapshot in vm.tracer.log[-MAX_TRACE_ROWS:]
    ]
    return {
        "normalized_source": normalized,
        "bytecode": [f"0x{word:016X}" for word in bytecode],
        "cycles": vm.cycles,
        "registers": registers,
        "trace": trace,
        "trace_truncated": len(vm.tracer.log) > MAX_TRACE_ROWS,
    }


def refactor_program(
    source: str,
    *,
    seed: int = 41,
    max_cycles: int = 1_000,
    max_mutations: int = 10,
) -> dict[str, Any]:
    """Run the repository's conservative deterministic refactoring automaton."""

    normalized = _normalized_source(source)
    automata = DeterministicCodeEditingAutomata(cycle_limit=min(max_cycles, 10_000))
    automata.add_policy(ConfidenceThresholdPolicy(0.70))
    automata.add_policy(CycleImprovementPolicy())
    automata.add_policy(MemorySafetyPolicy(max_increase_percent=0.0))
    automata.add_policy(
        TransformWhitelistPolicy(
            {
                TransformType.DEAD_CODE_ELIMINATION,
                TransformType.CONST_PROPAGATION,
            }
        )
    )
    params = RefactoringParameter(
        seed=seed,
        max_depth=3,
        max_cycles=max_cycles,
        max_mutations=max_mutations,
        cost_model="combined",
        allow_unsafe=False,
        allow_heuristic=False,
    )
    return automata.refactor(normalized, params).to_dict()


class ProjectStore:
    """Small SQLite-backed project store with explicit size bounds."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    source TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _validate(name: str, source: str) -> None:
        cleaned = name.strip()
        if not cleaned or len(cleaned) > 120:
            raise ValueError("project name must contain 1-120 characters")
        if len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
            raise ValueError(f"source exceeds {MAX_SOURCE_BYTES} bytes")

    def save(self, *, name: str, source: str, project_id: str | None = None) -> dict[str, Any]:
        self._validate(name, source)
        now = time.time()
        identifier = project_id or uuid.uuid4().hex
        with self._lock, self._connect() as connection:
            existing = connection.execute(
                "SELECT created_at FROM projects WHERE id = ?", (identifier,)
            ).fetchone()
            created_at = float(existing["created_at"]) if existing else now
            connection.execute(
                """
                INSERT INTO projects(id, name, source, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    source = excluded.source,
                    updated_at = excluded.updated_at
                """,
                (identifier, name.strip(), source, created_at, now),
            )
        return self.get(identifier)

    def get(self, project_id: str) -> dict[str, Any]:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT id, name, source, created_at, updated_at FROM projects WHERE id = ?",
                (project_id,),
            ).fetchone()
        if row is None:
            raise KeyError(project_id)
        return dict(row)

    def list(self, limit: int = 100) -> list[dict[str, Any]]:
        bounded = max(1, min(int(limit), 500))
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, name, created_at, updated_at, length(source) AS source_chars
                FROM projects ORDER BY updated_at DESC LIMIT ?
                """,
                (bounded,),
            ).fetchall()
        return [dict(row) for row in rows]

    def delete(self, project_id: str) -> bool:
        with self._lock, self._connect() as connection:
            cursor = connection.execute("DELETE FROM projects WHERE id = ?", (project_id,))
            return cursor.rowcount > 0


class EventJournal:
    """In-memory bounded event stream used by HTTP and WebSocket telemetry."""

    def __init__(self, max_events: int = 500) -> None:
        self._events: deque[dict[str, Any]] = deque(maxlen=max_events)
        self._sequence = 0
        self._lock = threading.RLock()

    def emit(self, event_type: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:
            self._sequence += 1
            event = {
                "sequence": self._sequence,
                "timestamp": time.time(),
                "type": event_type,
                "payload": payload or {},
            }
            self._events.append(event)
            return dict(event)

    def since(self, sequence: int = 0) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(event) for event in self._events if event["sequence"] > sequence]

    @property
    def sequence(self) -> int:
        with self._lock:
            return self._sequence


class ANNRegistry:
    """Bounded registry of mutable inward-4D ANN sessions."""

    def __init__(self, max_sessions: int = MAX_ANN_SESSIONS) -> None:
        self.max_sessions = max_sessions
        self._sessions: dict[str, Inward4DANN] = {}
        self._created: dict[str, float] = {}
        self._lock = threading.RLock()

    def create(
        self,
        *,
        side: int = 6,
        fold_factor: float = 1.0,
        learning_rate: float = 0.005,
        prune_threshold: float = 0.15,
        seed: int = 41,
    ) -> dict[str, Any]:
        config = Inward4DConfig(
            side=side,
            fold_factor=fold_factor,
            learning_rate=learning_rate,
            prune_threshold=prune_threshold,
            seed=seed,
        )
        model = Inward4DANN(config)
        identifier = uuid.uuid4().hex
        with self._lock:
            if len(self._sessions) >= self.max_sessions:
                oldest = min(self._created, key=self._created.get)
                self._sessions.pop(oldest, None)
                self._created.pop(oldest, None)
            self._sessions[identifier] = model
            self._created[identifier] = time.time()
        return {
            "session_id": identifier,
            "epoch": model.epoch,
            "summary": model.arithmetic_summary(),
        }

    def _get(self, session_id: str) -> Inward4DANN:
        with self._lock:
            model = self._sessions.get(session_id)
        if model is None:
            raise KeyError(session_id)
        return model

    def status(self, session_id: str) -> dict[str, Any]:
        model = self._get(session_id)
        return {
            "session_id": session_id,
            "epoch": model.epoch,
            "summary": model.arithmetic_summary(),
            "active_synapses": model.active_synapse_count,
            "active_wrap_synapses": model.active_wrap_synapse_count,
        }

    def evaluate(self, session_id: str, values: list[float]) -> dict[str, Any]:
        model = self._get(session_id)
        metrics = model.evaluate(values)
        forward = model.forward(values)
        return {
            "session_id": session_id,
            "epoch": model.epoch,
            "metrics": jsonable(metrics),
            "latent_sample": list(forward.latent[:32]),
            "reconstruction_sample": list(forward.reconstruction[:32]),
        }

    def optimize(
        self,
        session_id: str,
        values: list[float],
        *,
        max_epochs: int = 20,
        tolerance: float | None = None,
    ) -> dict[str, Any]:
        model = self._get(session_id)
        report = model.optimize(values, max_epochs=max_epochs, tolerance=tolerance)
        return {
            "session_id": session_id,
            "epoch": model.epoch,
            "report": jsonable(report),
            "summary": model.arithmetic_summary(),
        }

    def delete(self, session_id: str) -> bool:
        with self._lock:
            existed = session_id in self._sessions
            self._sessions.pop(session_id, None)
            self._created.pop(session_id, None)
            return existed
