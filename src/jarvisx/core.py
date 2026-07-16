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
from .geometric_rvis import GeometricFeedbackRuntime


class CodexVM:
    def __init__(self, reflex_enabled=False):
        self.regs = Registers()
        self.mem = Memory()
        self.decoder = Decoder()
        self.executor = Executor(self.regs)
        self.ledger = PersistentLedger()
        self.ethics = LambdaShield()
        self.reflex = ReflexEngine()
        self.reflex_enabled = bool(reflex_enabled)
        self.sandbox = Sandbox()
        self.debugger = Debugger(self)
        self.tracer = Tracer()
        self.cognitive = CognitiveKernel()
        self.cognitive_bridge = CognitiveVMBridge(self.regs, self.cognitive)
        self.geometric = GeometricFeedbackRuntime()
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
        if self.reflex_enabled:
            self.reflex.stabilize(self.regs)

        self.regs["IP"] += 1
        self.cycles += 1
        self.sandbox.enforce(self.cycles)

        if not cont:
            self.running = False

    def run(self):
        while self.running:
            self.step()

    def apply_reflex(self):
        """Apply one explicit reflex-stabilisation step to the register bank."""
        self.reflex.stabilize(self.regs)

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

    def geometric_feedback(self, values, cycles=None):
        """Run the inward 3D geometric feedback loop atomically.

        Every committed decoded output becomes the next cycle's input. The
        final accepted geometric state is projected into the existing Greek
        register bank. Exceptions restore geometric state, journal, and
        registers together.
        """
        state_before = self.geometric.state
        journal_length = len(self.geometric.journal)
        registers_before = self.regs.snapshot()

        try:
            results = self.geometric.run_feedback(values, cycles)
            if not results:
                return results
            final = results[-1]
            if not final.committed:
                self.regs["Λ"] = 0
                return results

            root = final.hierarchy[-1].values[0]
            self.regs["Ξ"] = sum(final.encoded)
            self.regs["Ψ"] = root
            self.regs["Φ"] = sum(final.evolved)
            self.regs["Λ"] = 1
            self.regs["Ω"] = sum(final.omega_after)
            self.regs["𝒮"] = int(final.metrics["best_reconstruction_l1"])
            self.regs["Π"] = sum(final.output)
            return results
        except Exception:
            self.geometric.state = state_before
            del self.geometric.journal[journal_length:]
            for name, value in registers_before.items():
                self.regs[name] = value
            raise
