"""Unified scalar, 30D, and sparse 3D abstraction execution unit."""

from .abstraction3d import AbstractionANNCore3D, Instruction3D, Opcode3D
from .ann30d import Instruction30D, Opcode30D
from .ann30d_safe import SafeANNProcessor30D
from .opcodes import OPCODES
from .registers import REG_NAMES

ANN_OPCODE_MAP = {
    OPCODES["LOAD30"]: Opcode30D.LOAD,
    OPCODES["ENCODE30"]: Opcode30D.ENCODE30,
    OPCODES["PLACE30"]: Opcode30D.PLACE30,
    OPCODES["FIELD30"]: Opcode30D.FIELD30,
    OPCODES["PREDICT30"]: Opcode30D.PREDICT30,
    OPCODES["COMPARE30"]: Opcode30D.COMPARE,
    OPCODES["UPDATE_MEMORY30"]: Opcode30D.UPDATE_MEMORY,
    OPCODES["PROJECT30"]: Opcode30D.PROJECT,
    OPCODES["DECODE30"]: Opcode30D.DECODE30,
    OPCODES["HALT30"]: Opcode30D.HALT,
}

ABSTRACTION_OPCODE_MAP = {
    OPCODES["LOAD3D"]: Opcode3D.LOAD,
    OPCODES["ABSTRACT3D"]: Opcode3D.ABSTRACT,
    OPCODES["ROUTE3D"]: Opcode3D.ROUTE,
    OPCODES["ATTEND3D"]: Opcode3D.ATTEND,
    OPCODES["PREDICT3D"]: Opcode3D.PREDICT,
    OPCODES["COMPARE3D"]: Opcode3D.COMPARE,
    OPCODES["LEARN3D"]: Opcode3D.LEARN,
    OPCODES["PROJECT3D"]: Opcode3D.PROJECT,
    OPCODES["DECODE3D"]: Opcode3D.DECODE,
    OPCODES["HALT3D"]: Opcode3D.HALT,
}


class Executor:
    def __init__(self, registers, ann30d=None, abstraction3d=None):
        self.regs = registers
        self.ann30d = ann30d or SafeANNProcessor30D()
        self.abstraction3d = abstraction3d or AbstractionANNCore3D()
        self.ann_input = None
        self.ann_target = 0.0

    def set_ann_context(self, input_vector=None, target=0.0):
        self.ann_input = input_vector
        self.ann_target = float(target)

    @staticmethod
    def _register_name(index):
        if index < 0 or index >= len(REG_NAMES):
            raise RuntimeError(f"register index outside range: {index}")
        return REG_NAMES[index]

    def _sync_ann_registers(self):
        snapshot = self.ann30d.snapshot()
        scale = 1000000
        self.regs["A"] = int(round(snapshot.prediction * scale))
        self.regs["B"] = int(round(snapshot.residual * scale))
        self.regs["Ω"] = int(round(snapshot.memory * scale))
        self.regs["C"] = snapshot.active_cells
        self.regs["D"] = snapshot.cycles

    def _sync_abstraction_registers(self):
        snapshot = self.abstraction3d.snapshot()
        scale = 1000000
        self.regs["A"] = int(round(snapshot.prediction * scale))
        self.regs["B"] = int(round(snapshot.residual * scale))
        self.regs["Ω"] = int(round(snapshot.memory_norm * scale))
        self.regs["C"] = snapshot.active_nodes
        self.regs["D"] = snapshot.cycles
        self.regs["FLAGS"] = int(round(snapshot.loss * scale))

    def execute(self, instr):
        if instr.opcode == OPCODES["SET"]:
            self.regs[self._register_name(instr.dst)] = instr.imm

        elif instr.opcode == OPCODES["ADD"]:
            a = self.regs[self._register_name(instr.src1)]
            b = self.regs[self._register_name(instr.src2)]
            self.regs[self._register_name(instr.dst)] = a + b

        elif instr.opcode == OPCODES["SUB"]:
            a = self.regs[self._register_name(instr.src1)]
            b = self.regs[self._register_name(instr.src2)]
            self.regs[self._register_name(instr.dst)] = a - b

        elif instr.opcode in ANN_OPCODE_MAP:
            should_continue = self.ann30d.execute(
                Instruction30D(ANN_OPCODE_MAP[instr.opcode]),
                input_vector=self.ann_input,
                target=self.ann_target,
            )
            self._sync_ann_registers()
            if not should_continue:
                return False

        elif instr.opcode in ABSTRACTION_OPCODE_MAP:
            should_continue = self.abstraction3d.execute(
                Instruction3D(ABSTRACTION_OPCODE_MAP[instr.opcode]),
                input_vector=self.ann_input,
                target=self.ann_target,
            )
            self._sync_abstraction_registers()
            if not should_continue:
                return False

        elif instr.opcode == OPCODES["HALT"]:
            return False

        else:
            raise RuntimeError(f"unsupported opcode: 0x{instr.opcode:02X}")

        return True
