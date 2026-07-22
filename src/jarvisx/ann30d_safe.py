"""Operational safety wrapper for the sparse 30D ANN processor."""

import hashlib
import json
import math

from .ann30d import Instruction30D, Opcode30D, VirtualANNProcessor30D


class SafeANNProcessor30D(VirtualANNProcessor30D):
    def __init__(
        self,
        *args,
        max_active_cells=100000,
        max_input_length=4096,
        max_program_length=256,
        max_abs_input=1000000.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.max_active_cells = int(max_active_cells)
        self.max_input_length = int(max_input_length)
        self.max_program_length = int(max_program_length)
        self.max_abs_input = float(max_abs_input)
        if min(self.max_active_cells, self.max_input_length, self.max_program_length) <= 0:
            raise ValueError("30D safety limits must be positive")

    def reset_run_state(self):
        self.registers = {
            "INPUT": (),
            "RAW_LATENT": None,
            "LATENT": None,
            "COORD": None,
            "PREDICTION": 0.0,
            "TARGET": 0.0,
            "ERROR": 0.0,
            "OUTPUT": (),
        }
        self.cycles = 0
        self.halted = False

    def _validate_source(self, values):
        if values is None or not values:
            raise ValueError("input vector must not be empty")
        if len(values) > self.max_input_length:
            raise ValueError("input vector exceeds configured length limit")
        source = tuple(float(value) for value in values)
        if not all(math.isfinite(value) for value in source):
            raise ValueError("input vector must contain only finite values")
        if any(abs(value) > self.max_abs_input for value in source):
            raise ValueError("input value exceeds configured magnitude limit")
        return source

    def execute(self, instruction, input_vector=None, target=None):
        registers_before = dict(self.registers)
        cycles_before = self.cycles
        halted_before = self.halted
        coordinate = self.registers.get("COORD")
        coordinate = tuple(coordinate) if coordinate is not None else None
        old_cell = self.field.peek(coordinate).clone() if coordinate is not None and self.field.peek(coordinate) else None
        old_existed = old_cell is not None

        try:
            if instruction.opcode == Opcode30D.LOAD:
                source = instruction.operand if instruction.operand is not None else input_vector
                input_vector = self._validate_source(source)
                if target is not None and not math.isfinite(float(target)):
                    raise ValueError("target must be finite")

            result = super().execute(instruction, input_vector=input_vector, target=target)
            if self.field.active_cells > self.max_active_cells:
                raise MemoryError("30D active-cell quota exceeded")

            current = self.registers.get("COORD")
            if current is not None and instruction.opcode in {
                Opcode30D.PLACE30,
                Opcode30D.FIELD30,
                Opcode30D.UPDATE_MEMORY,
                Opcode30D.PROJECT,
            }:
                super().project(tuple(current))
            return result
        except Exception:
            self.registers = registers_before
            self.cycles = cycles_before
            self.halted = halted_before
            if coordinate is not None:
                if old_existed:
                    self.field._cells[coordinate] = old_cell
                else:
                    self.field._cells.pop(coordinate, None)
            raise

    def run(self, input_vector, target=0.0, program=None):
        self.reset_run_state()
        selected = tuple(program) if program is not None else self.default_program()
        if not selected:
            raise ValueError("program must not be empty")
        if len(selected) > self.max_program_length:
            raise ValueError("program exceeds configured instruction limit")
        if selected[-1].opcode != Opcode30D.HALT:
            raise ValueError("program must terminate with HALT")
        for instruction in selected:
            if not self.execute(instruction, input_vector=input_vector, target=target):
                break
        return self.snapshot()

    def state_hash(self):
        cells = []
        for coordinate in sorted(self.field._cells):
            cell = self.field._cells[coordinate]
            cells.append({
                "coordinate": coordinate,
                "activation": cell.activation,
                "electric": cell.electric,
                "magnetic": cell.magnetic,
                "memory": cell.memory,
                "prediction": cell.prediction,
                "residual": cell.residual,
                "visits": cell.visits,
            })
        payload = json.dumps(cells, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
