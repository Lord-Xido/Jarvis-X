"""Jarvis-X unified deterministic virtual machine."""

from .abstraction3d import AbstractionANNCore3D
from .ann30d_safe import SafeANNProcessor30D
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
    def __init__(
        self,
        ledger_path=None,
        max_cycles=10000,
        max_program_words=10000,
        max_active_cells=100000,
        reflex_enabled=False,
    ):
        self.regs = Registers()
        self.mem = Memory()
        self.decoder = Decoder()
        self.ann30d = SafeANNProcessor30D(max_active_cells=max_active_cells)
        self.abstraction3d = AbstractionANNCore3D(
            max_active_nodes=max_active_cells
        )
        self.executor = Executor(
            self.regs,
            self.ann30d,
            self.abstraction3d,
        )
        self.ledger = PersistentLedger(ledger_path)
        self.ethics = LambdaShield()
        self.reflex = ReflexEngine(enabled=reflex_enabled)
        self.sandbox = Sandbox(max_cycles, max_program_words, max_active_cells)
        self.debugger = Debugger(self)
        self.tracer = Tracer(max_entries=max_cycles)
        self.program = []
        self.cycles = 0
        self.running = False

    def load(
        self,
        bytecode,
        ann_input=None,
        ann_target=0.0,
        reset_registers=True,
    ):
        program = list(bytecode)
        self.sandbox.validate_program(program)
        self.program = program
        if reset_registers:
            self.regs.reset()
        self.regs["IP"] = 0
        self.cycles = 0
        self.running = True
        self.tracer.clear()
        self.executor.set_ann_context(ann_input, ann_target)
        self.ann30d.reset_run_state()
        self.abstraction3d.reset_run_state()

    def _ann_metadata(self):
        snapshot = self.ann30d.snapshot()
        if snapshot.coordinate is None:
            return {}
        data = dict(vars(snapshot))
        data["state_hash"] = self.ann30d.state_hash()
        return data

    def _abstraction_metadata(self):
        snapshot = self.abstraction3d.snapshot()
        if snapshot.cycles == 0:
            return {}
        data = dict(vars(snapshot))
        data["state_hash"] = self.abstraction3d.state_hash()
        return data

    def step(self):
        if not self.running:
            return False
        ip = self.regs["IP"]
        if ip < 0 or ip >= len(self.program):
            self.running = False
            raise RuntimeError(f"instruction pointer outside program: {ip}")
        instr = self.decoder.decode(self.program[ip])
        if not self.ethics.allow(instr):
            self.running = False
            raise RuntimeError(f"LambdaShield blocked opcode 0x{instr.opcode:02X}")

        try:
            cont = self.executor.execute(instr)
            self.reflex.stabilize(self.regs)
            self.regs["IP"] = ip + 1
            self.cycles += 1
            active_cells = (
                self.ann30d.field.active_cells
                + self.abstraction3d.lattice.active_nodes
            )
            self.sandbox.enforce(self.cycles, active_cells)

            ann_metadata = self._ann_metadata()
            abstraction_metadata = self._abstraction_metadata()
            metadata = {}
            if ann_metadata:
                metadata["ann30d"] = ann_metadata
            if abstraction_metadata:
                metadata["abstraction3d"] = abstraction_metadata
            self.ledger.log(
                self.regs.snapshot(),
                instr.opcode,
                metadata=metadata,
            )
            self.tracer.record(
                instr,
                self.regs.snapshot(),
                ann=metadata or None,
            )
        except Exception:
            self.running = False
            raise

        if not cont:
            self.running = False
        return self.running

    def run(self):
        while self.running:
            self.step()
        return self.snapshot()

    def snapshot(self):
        ann = dict(vars(self.ann30d.snapshot()))
        ann["state_hash"] = self.ann30d.state_hash()
        abstraction = dict(vars(self.abstraction3d.snapshot()))
        abstraction["state_hash"] = self.abstraction3d.state_hash()
        return {
            "registers": self.regs.snapshot(),
            "cycles": self.cycles,
            "running": self.running,
            "ledger_entries": len(self.ledger.chain),
            "ledger_valid": self.ledger.verify(),
            "ann30d": ann,
            "abstraction3d": abstraction,
        }
