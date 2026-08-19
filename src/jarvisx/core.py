from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from os import PathLike

from .control_plane import OmegaEvidenceChain, StateEnvelope
from .debugger import Debugger
from .decoder import Decoder
from .ethics import LambdaShield
from .executor import Executor
from .ledger import OmegaLedger
from .ledger_store import PersistentLedger
from .memory import Memory
from .reflex import ReflexEngine
from .registers import Registers
from .sandbox import Sandbox
from .tracer import Tracer


@dataclass(frozen=True)
class _VMCheckpoint:
    registers: dict[str, int]
    memory: bytes
    cycles: int
    running: bool
    ledger_length: int
    trace_length: int
    control_length: int


class CodexVM:
    """Deterministic transactional Jarvis-X bytecode virtual machine.

    Each instruction executes against a complete authoritative-state checkpoint.
    A failed execution, validation, journal write, trace write, or unified
    control-plane receipt restores the checkpoint and stops the VM. Reflex
    stabilization is opt-in and occurs before the canonical post-instruction
    snapshot is journalled, traced and admitted into the common evidence plane.
    """

    def __init__(
        self,
        *,
        enable_reflex: bool = False,
        ledger_path: str | PathLike[str] | None = None,
        max_cycles: int = 10_000,
    ) -> None:
        if max_cycles < 1:
            raise ValueError("max_cycles must be positive")

        self.regs = Registers()
        self.mem = Memory()
        self.decoder = Decoder()
        self.executor = Executor(self.regs)
        self.ledger = PersistentLedger(ledger_path) if ledger_path else OmegaLedger()
        self.ethics = LambdaShield()
        self.reflex = ReflexEngine()
        self.enable_reflex = bool(enable_reflex)
        self.sandbox = Sandbox(max_cycles=max_cycles)
        self.debugger = Debugger(self)
        self.tracer = Tracer()
        self.control_plane = OmegaEvidenceChain()
        self.program: list[int] = []
        self.cycles = 0
        self.running = False

    def _checkpoint(self) -> _VMCheckpoint:
        return _VMCheckpoint(
            registers=self.regs.snapshot(),
            memory=self.mem.snapshot(),
            cycles=self.cycles,
            running=self.running,
            ledger_length=self.ledger.checkpoint(),
            trace_length=self.tracer.checkpoint(),
            control_length=self.control_plane.checkpoint(),
        )

    def _restore(self, checkpoint: _VMCheckpoint) -> None:
        self.regs.restore(checkpoint.registers)
        self.mem.restore(checkpoint.memory)
        self.cycles = checkpoint.cycles
        self.running = checkpoint.running
        self.ledger.restore(checkpoint.ledger_length)
        self.tracer.restore(checkpoint.trace_length)
        self.control_plane.restore(checkpoint.control_length)

    def _control_payload(self) -> dict[str, object]:
        memory = self.mem.snapshot()
        return {
            "registers": self.regs.snapshot(),
            "memory_sha256": hashlib.sha256(memory).hexdigest(),
            "memory_bytes": len(memory),
            "cycles": self.cycles,
            "running": self.running,
        }

    def _control_envelope(self, *, authoritative: bool) -> StateEnvelope:
        payload = self._control_payload()
        registers = payload["registers"]
        memory_bytes = payload["memory_bytes"]
        assert isinstance(registers, dict)
        assert isinstance(memory_bytes, int)
        return StateEnvelope.from_payload(
            state_type="jarvisx.vm-state",
            state_version=1,
            dimensions=(len(registers), memory_bytes),
            payload=payload,
            authoritative=authoritative,
        )

    def load(self, bytecode: Iterable[int]) -> None:
        program = list(bytecode)
        if not program:
            raise ValueError("bytecode program cannot be empty")
        for index, word in enumerate(program):
            if not isinstance(word, int) or isinstance(word, bool):
                raise TypeError(f"bytecode word {index} is not an integer")
            if not 0 <= word <= 0xFFFFFFFFFFFFFFFF:
                raise ValueError(f"bytecode word {index} is outside unsigned 64-bit range")

        self.program = program
        self.regs["IP"] = 0
        self.cycles = 0
        self.running = True

    def step(self) -> bool:
        if not self.running:
            raise RuntimeError("VM is not running; load a program first")

        ip = self.regs["IP"]
        if not 0 <= ip < len(self.program):
            self.running = False
            raise RuntimeError(f"instruction pointer {ip} is outside the loaded program")

        instruction = self.decoder.decode(self.program[ip])
        if not self.ethics.allow(instruction):
            self.running = False
            raise RuntimeError("Lambda policy blocked instruction")

        checkpoint = self._checkpoint()
        before = self._control_envelope(authoritative=True)
        try:
            next_cycles = self.cycles + 1
            self.sandbox.enforce(next_cycles)

            should_continue = bool(self.executor.execute(instruction))
            if self.enable_reflex:
                self.reflex.stabilize(self.regs)

            self.regs["IP"] = ip + 1
            self.cycles = next_cycles
            self.running = should_continue

            snapshot = self.regs.snapshot()
            self.ledger.log(snapshot, instruction.opcode)
            self.tracer.record(instruction, snapshot)

            candidate = self._control_envelope(authoritative=False)
            after = self._control_envelope(authoritative=True)
            self.control_plane.append(
                subsystem="codex-vm",
                operation=f"opcode:{int(instruction.opcode)}",
                decision="commit",
                before=before,
                candidate=candidate,
                after=after,
                metrics={"cycles": self.cycles},
            )
        except Exception:
            try:
                self._restore(checkpoint)
            except Exception as rollback_error:
                self.running = False
                raise RuntimeError("VM transaction rollback failed") from rollback_error
            self.running = False
            raise

        return should_continue

    def run(self) -> dict[str, int]:
        if not self.running:
            raise RuntimeError("VM is not running; load a program first")
        while self.running:
            self.step()
        return self.regs.snapshot()
