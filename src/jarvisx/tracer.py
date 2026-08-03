from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class Tracer:
    def __init__(self) -> None:
        self.log: list[tuple[int, dict[str, Any]]] = []

    def checkpoint(self) -> int:
        return len(self.log)

    def restore(self, checkpoint: int) -> None:
        if (
            not isinstance(checkpoint, int)
            or isinstance(checkpoint, bool)
            or checkpoint < 0
            or checkpoint > len(self.log)
        ):
            raise ValueError("trace checkpoint is invalid")
        del self.log[checkpoint:]

    def reset(self) -> None:
        self.log.clear()

    def record(self, instr: Any, regs: Mapping[str, Any]) -> None:
        self.log.append((int(instr.opcode), dict(regs)))
