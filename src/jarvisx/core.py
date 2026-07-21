"""Deterministic Jarvis-X bytecode virtual machine."""

from pathlib import Path
from typing import Union

from .debugger import Debugger
from .decoder import Decoder
from .ethics import LambdaShield
from .executor import Executor
from .ledger_store import PersistentLedger
from .memory import Memory
from .reflex import ReflexEngine
from .registers import Registers
from .sandbox import Sandbox
from .tracer import Tracer


class CodexVM:
    """Execute bytecode without hidden register mutation.

    Reflex stabilization remains available, but it is opt-in. Applying it after
    every instruction by default changes architectural operands between `SET`
    and `ADD`, violating the ISA's deterministic arithmetic semantics.
    """

    def __init__(
        self,
        reflex_enabled: bool = False,
        ledger_path: Union[str, Path] = "omega_ledger.json",
    ):
        self.regs = Registers()
        self.mem = Memory()
        self.decoder = Decoder()
        self.executor = Executor(self.regs)
        self.ledger = PersistentLedger(ledger_path)
        self.ethics = LambdaShield()
        self.reflex = ReflexEngine()
        self.reflex_enabled = bool(reflex_enabled)
        self.sandbox = Sandbox()
        self.debugger = Debugger(self)
        self.tracer = Tracer()
        self.program = []
        self.cycles = 0
        self.running = True

    def load(self, bytecode):
        self.program = bytecode
        self.regs["IP"] = 0
        self.running = True

    def step(self):
        ip = self.regs["IP"]
        if ip < 0 or ip >= len(self.program):
            raise RuntimeError("instruction pointer is outside the loaded program")
        instr = self.decoder.decode(self.program[ip])

        if not self.ethics.allow(instr):
            raise RuntimeError("Ethics blocked instruction")

        cont = self.executor.execute(instr)
        if self.reflex_enabled:
            self.reflex.stabilize(self.regs)

        snapshot = self.regs.snapshot()
        self.ledger.log(snapshot, instr.opcode)
        self.tracer.record(instr, snapshot)

        self.regs["IP"] += 1
        self.cycles += 1
        self.sandbox.enforce(self.cycles)

        if not cont:
            self.running = False

    def run(self):
        while self.running:
            self.step()
