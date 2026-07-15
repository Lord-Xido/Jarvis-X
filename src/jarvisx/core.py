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
from .electronic import ElectronicSubstrate


class CodexVM:
    def __init__(self, electronics=None):
        self.regs = Registers()
        self.mem = Memory()
        self.decoder = Decoder()
        self.executor = Executor(self.regs)
        self.ledger = PersistentLedger()
        self.ethics = LambdaShield()
        self.reflex = ReflexEngine()
        self.sandbox = Sandbox()
        self.debugger = Debugger(self)
        self.tracer = Tracer()
        self.electronics = electronics or ElectronicSubstrate()
        self.last_electronic_trace = None
        self.program = []
        self.cycles = 0
        self.running = True

    def load(self, bytecode):
        self.program = list(bytecode)
        self.regs["IP"] = 0
        self.cycles = 0
        self.running = True
        self.last_electronic_trace = None
        self.electronics.reset()

    def step(self):
        ip = self.regs["IP"]
        if ip < 0 or ip >= len(self.program):
            raise RuntimeError("Instruction pointer outside loaded program")

        word = self.program[ip]
        instr = self.decoder.decode(word)

        if not self.ethics.allow(instr):
            raise RuntimeError("Ethics blocked instruction")

        before = self.regs.snapshot()
        cont = self.executor.execute(instr)
        after = self.regs.snapshot()
        electronic_trace = self.electronics.tick(instr, before, after, word)

        if self.electronics.config.enforce_limits and not electronic_trace.lambda_accept:
            for name, value in before.items():
                self.regs[name] = value
            raise RuntimeError("Electronic Λ-gate rejected instruction commit")

        self.last_electronic_trace = electronic_trace
        self.ledger.log(after, instr.opcode)
        self.tracer.record(instr, after)
        self.reflex.stabilize(self.regs)

        self.regs["IP"] += 1
        self.cycles += 1
        self.sandbox.enforce(self.cycles)

        if not cont:
            self.running = False

    def run(self):
        while self.running:
            self.step()
