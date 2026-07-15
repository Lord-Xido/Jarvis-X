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
from .cognitive import CognitiveKernel, CognitiveVMBridge


class CodexVM:
    def __init__(self):
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
        self.cognitive = CognitiveKernel()
        self.cognitive_bridge = CognitiveVMBridge(self.regs, self.cognitive)
        self.program = []
        self.cycles = 0
        self.running = True

    def load(self, bytecode):
        self.program = bytecode
        self.regs["IP"] = 0
        self.running = True

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

    def cognitive_cycle(self, values):
        """Execute one atomic hierarchical intelligence transaction.

        Kernel state, journal state, and VM registers are restored together if
        the register projection fails after the kernel has accepted a candidate.
        """
        state_before = self.cognitive.state
        journal_length = len(self.cognitive.journal)
        registers_before = self.regs.snapshot()

        try:
            return self.cognitive_bridge.cycle(values)
        except Exception:
            self.cognitive.state = state_before
            del self.cognitive.journal[journal_length:]
            for name, value in registers_before.items():
                self.regs[name] = value
            raise
