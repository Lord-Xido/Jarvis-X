from .debugger import Debugger
from .decoder import Decoder
from .electronic import ElectronicSubstrate
from .ethics import LambdaShield
from .executor import Executor
from .ledger_store import PersistentLedger
from .memory import Memory
from .reflex import ReflexEngine
from .registers import Registers
from .sandbox import Sandbox
from .tracer import Tracer


class CodexVM:
    def __init__(
        self,
        electronics=None,
        ledger=None,
        memory=None,
        sandbox=None,
        reflex_enabled=False,
    ):
        self.regs = Registers()
        self.mem = memory or Memory()
        self.decoder = Decoder()
        self.executor = Executor(self.regs, self.mem)
        self.ledger = ledger or PersistentLedger(path=None)
        self.ethics = LambdaShield()
        self.reflex = ReflexEngine()
        self.reflex_enabled = bool(reflex_enabled)
        self.sandbox = sandbox or Sandbox()
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
        self.tracer.log.clear()

    def _restore_transaction(self, checkpoint):
        self.regs.restore(checkpoint["registers"])
        self.mem.restore(checkpoint["memory"])
        self.electronics.restore(checkpoint["electronics"])
        self.tracer.restore(checkpoint["tracer"])
        self.cycles = checkpoint["cycles"]
        self.running = checkpoint["running"]
        self.last_electronic_trace = checkpoint["last_electronic_trace"]
        try:
            self.ledger.restore(checkpoint["ledger"])
        except Exception:
            pass

    def step(self):
        ip = self.regs["IP"]
        if ip < 0 or ip >= len(self.program):
            raise RuntimeError("Instruction pointer outside loaded program")

        word = self.program[ip]
        instr = self.decoder.decode(word)
        if not self.ethics.allow(instr):
            raise RuntimeError("Ethics blocked instruction")

        checkpoint = {
            "registers": self.regs.snapshot(),
            "memory": self.mem.snapshot(),
            "electronics": self.electronics.checkpoint(),
            "tracer": self.tracer.checkpoint(),
            "ledger": self.ledger.checkpoint(),
            "cycles": self.cycles,
            "running": self.running,
            "last_electronic_trace": self.last_electronic_trace,
        }

        try:
            result = self.executor.execute(instr, current_ip=ip)
            if self.reflex_enabled:
                self.reflex.stabilize(self.regs)
            next_ip = ip + 1 if result.next_ip is None else int(result.next_ip)
            if result.continue_running and not 0 <= next_ip < len(self.program):
                raise RuntimeError("Branch target outside loaded program")
            self.regs["IP"] = next_ip

            next_cycle = self.cycles + 1
            self.sandbox.enforce(next_cycle)
            after = self.regs.snapshot()
            electronic_trace = self.electronics.tick(
                instr, checkpoint["registers"], after, word
            )
            if (
                self.electronics.config.enforce_limits
                and not electronic_trace.lambda_accept
            ):
                raise RuntimeError("Electronic Λ-gate rejected instruction commit")

            self.ledger.log(
                after,
                instr.opcode,
                metadata={
                    "cycle": next_cycle,
                    "electronic_checksum": electronic_trace.register_checksum,
                    "electronic_lambda_accept": electronic_trace.lambda_accept,
                    "telemetry_source": electronic_trace.source,
                },
            )
            self.tracer.record(instr, after)
            self.last_electronic_trace = electronic_trace
            self.cycles = next_cycle
            self.running = result.continue_running
            return electronic_trace
        except Exception:
            self._restore_transaction(checkpoint)
            raise

    def run(self, max_steps=None):
        steps = 0
        while self.running:
            if max_steps is not None and steps >= int(max_steps):
                raise RuntimeError("Run step limit exceeded")
            self.step()
            steps += 1
        return self.state_snapshot()

    def state_snapshot(self):
        return {
            "registers": self.regs.snapshot(),
            "cycles": self.cycles,
            "running": self.running,
            "ledger_entries": len(self.ledger.chain),
            "ledger_valid": self.ledger.verify(),
            "electronics": self.electronics.snapshot(),
        }
