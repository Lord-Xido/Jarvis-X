from .registers import Registers
from .memory import Memory
from .decoder import Decoder
from .executor import Executor
from .ledger_store import PersistentLedger
from .ethics import LambdaShield
from .reflex import ReflexEngine
from .sandbox import Sandbox
from .debugger import Debugger
from .tracer import Tracer

class CodexVM:
    def __init__(self):
        self.regs = Registers()
        self.mem = Memory()
        self.decoder = Decoder()
        self.executor = Executor(self.regs)
        self.ledger = OmegaLedger()
        self.ethics = LambdaShield()
        self.reflex = ReflexEngine()
        self.sandbox = Sandbox()
        self.debugger = Debugger(self)
        self.tracer = Tracer()
        self.program = []
        self.cycles = 0
        self.running = True

    def load(self, bytecode):
        self.program = bytecode
        self.regs["IP"] = 0

    def step(self):
        ip = self.regs["IP"]
        instr = self.decoder.decode(self.program[ip])

        if not self.ethics.allow(instr):
            raise RuntimeError("Ethics blocked instruction")

        cont = self.executor.execute(instr)
        self.ledger.log(self.regs.snapshot(), instr.opcode)
        self.tracer.record(instr, self.regs.snapshot())
        self.reflex.stabilize(self.regs)

        self.regs["IP"] += 1
        self.cycles += 1
        self.sandbox.enforce(self.cycles)

        if not cont:
            self.running = False

    def run(self):
        while self.running:
            self.step()
