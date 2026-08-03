from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from os import PathLike

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


class CodexVM:
    """Deterministic transactional Jarvis-X bytecode virtual machine.

    Each instruction executes against a complete authoritative-state checkpoint.
    A failed execution, validation, journal write, or trace write restores the
    checkpoint and stops the VM. Reflex stabilization is opt-in and occurs
    before the canonical post-instruction snapshot is journalled and traced.
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
        )

    def _restore(self, checkpoint: _VMCheckpoint) -> None:
        self.regs.restore(checkpoint.registers)
        self.mem.restore(checkpoint.memory)
        self.cycles = checkpoint.cycles
        self.running = checkpoint.running
        self.ledger.restore(checkpoint.ledger_length)
        self.tracer.restore(checkpoint.trace_length)

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
