from collections.abc import Iterable, Sequence

REG_MAP: dict[str, int]
OPCODES: dict[str, int]

def encode(
    opcode: int,
    dst: int = ...,
    src1: int = ...,
    src2: int = ...,
    imm: int = ...,
) -> int: ...

class Assembler:
    def assemble(self, ast: Iterable[Sequence[str]]) -> list[int]: ...
