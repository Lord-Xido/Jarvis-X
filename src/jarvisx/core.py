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


class CodexVM:
    """Deterministic Jarvis-X VM with optional reflex and persistence layers."""

    def __init__(self, *, ledger_path=None, reflex_enabled=False, max_cycles=10000):
        self.regs = Registers()
        self.mem = Memory()
        self.decoder = Decoder()
        self.executor = Executor(self.regs)
        self.ledger = PersistentLedger(ledger_path) if ledger_path else OmegaLedger()
        self.ethics = LambdaShield()
        self.reflex = ReflexEngine(enabled=reflex_enabled)
        self.sandbox = Sandbox(max_cycles=max_cycles)
        self.debugger = Debugger(self)
        self.tracer = Tracer()
        self.program = []
        self.cycles = 0
        self.running = False

    def load(self, bytecode):
        if not bytecode:
            raise ValueError("program cannot be empty")
        self.program = list(bytecode)
        self.regs["IP"] = 0
        self.cycles = 0
        self.running = True

    def step(self):
        if not self.running:
            return False
        ip = self.regs["IP"]
        if not 0 <= ip < len(self.program):
            self.running = False
            raise RuntimeError("instruction pointer left program bounds")
        if self.debugger.check():
            return False

        instr = self.decoder.decode(self.program[ip])
        if not self.ethics.allow(instr):
            raise RuntimeError("Ethics blocked instruction")

        cont = self.executor.execute(instr)
        reflex_delta = self.reflex.stabilize(self.regs)
        self.regs["IP"] += 1
        self.cycles += 1
        self.sandbox.enforce(self.cycles)

        snapshot = self.regs.snapshot()
        self.ledger.log(snapshot, instr.opcode)
        self.tracer.record(instr, snapshot)

        if reflex_delta:
            self.tracer.log[-1] = (
                instr.opcode,
                {**snapshot, "REFLEX_DELTA": reflex_delta},
            )

        if not cont:
            self.running = False
        return cont

    def run(self):
        while self.running:
            if self.step() is False and self.debugger.check():
                break
        return self.regs.snapshot()
