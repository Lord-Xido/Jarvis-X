from typing import Any, Optional

from .debugger import Debugger
from .decoder import Decoder
from .ethics import LambdaShield
from .executor import Executor
from .geometric_memory import (
    PermeationResult,
    VisualMemoryANN,
    Volume3D,
    make_demo_volume,
)
from .ledger_store import PersistentLedger
from .memory import Memory
from .reflex import ReflexEngine
from .registers import Registers
from .sandbox import Sandbox
from .tracer import Tracer


class CodexVM:
    def __init__(self, ledger_path: Optional[str] = "omega_ledger.json"):
        self.regs = Registers()
        self.mem = Memory()
        self.decoder = Decoder()
        self.executor = Executor(self.regs)
        self.ledger = PersistentLedger(path=ledger_path)
        self.ethics = LambdaShield()
        self.reflex = ReflexEngine()
        self.sandbox = Sandbox()
        self.debugger = Debugger(self)
        self.tracer = Tracer()
        self.geometric = VisualMemoryANN()
        self._mm3d = None
        self.program = []
        self.cycles = 0
        self.running = True

    def reset_execution(self) -> None:
        """Reset request-local execution state without clearing registers."""

        self.regs["IP"] = 0
        self.cycles = 0
        self.running = True
        self.tracer = Tracer()

    def load(self, bytecode) -> None:
        self.program = list(bytecode)
        self.reset_execution()

    def step(self) -> None:
        ip = self.regs["IP"]
        if not 0 <= ip < len(self.program):
            raise RuntimeError("instruction pointer outside loaded program")

        instr = self.decoder.decode(self.program[ip])
        if not self.ethics.allow(instr):
            raise RuntimeError("policy blocked instruction")

        cont = self.executor.execute(instr)
        snapshot = self.regs.snapshot()
        self.ledger.log(snapshot, instr.opcode)
        self.tracer.record(instr, snapshot)
        self.reflex.stabilize(self.regs)

        self.regs["IP"] += 1
        self.cycles += 1
        self.sandbox.enforce(self.cycles)

        if not cont:
            self.running = False

    def run(self) -> None:
        while self.running:
            self.step()

    def permeate_volume(
        self,
        observed: Volume3D,
        target: Optional[Volume3D] = None,
        auto_optimize: bool = True,
    ) -> PermeationResult:
        action = "V3D.PERMEATE"
        if not self.ethics.allow_action(action):
            raise RuntimeError("policy blocked geometric permeation")

        result = self.geometric.permeate(
            observed,
            target=target,
            auto_optimize=auto_optimize,
        )
        summary = result.summary()
        self.ledger.log(summary, action)
        self.tracer.record_event(action, summary)
        return result

    def run_visual_memory(
        self,
        size: int = 12,
        auto_optimize: bool = True,
    ) -> PermeationResult:
        return self.permeate_volume(
            make_demo_volume(size),
            auto_optimize=auto_optimize,
        )

    def run_mm3d_cycle(self, psi_input: Any, config=None):
        """Execute the bounded MM3D Ω⁴ cycle under VM policy and journaling."""

        action = "MM3D.CYCLE"
        if not self.ethics.allow_action(action):
            raise RuntimeError("policy blocked MM3D cycle")

        from .mm3d_omega4 import MM3DEngine

        if self._mm3d is None or (config is not None and self._mm3d.config != config):
            if self._mm3d is not None:
                self._mm3d.close()
            self._mm3d = MM3DEngine(config)

        result = self._mm3d.cycle(psi_input)
        summary = result.summary()
        self.ledger.log(summary, action)
        self.tracer.record_event(action, summary)
        return result

    def close(self) -> None:
        if self._mm3d is not None:
            self._mm3d.close()
            self._mm3d = None
