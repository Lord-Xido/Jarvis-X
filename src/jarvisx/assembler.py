from __future__ import annotations

from collections.abc import Iterable, Sequence

REG_MAP = {
    "Ξ": 0,
    "Ψ": 1,
    "Φ": 2,
    "Λ": 3,
    "Ω": 4,
    "Θ": 5,
    "𝒮": 6,
    "Π": 7,
    "A": 8,
    "B": 9,
    "C": 10,
    "D": 11,
    "IP": 12,
    "SP": 13,
    "FLAGS": 14,
    "TMP": 15,
}

OPCODES = {
    "SET": 0x01,
    "ADD": 0x03,
    "SUB": 0x04,
    "HALT": 0x0A,
}


def encode(opcode: int, dst: int = 0, src1: int = 0, src2: int = 0, imm: int = 0) -> int:
    return (opcode << 56) | (dst << 40) | (src1 << 32) | (src2 << 24) | (imm << 8)


class Assembler:
    def assemble(self, ast: Iterable[Sequence[str]]) -> list[int]:
        bytecode: list[int] = []
        for node in ast:
            op = node[0]
            if op == "SET":
                bytecode.append(encode(OPCODES["SET"], REG_MAP[node[1]], 0, 0, int(node[2])))
            elif op == "ADD":
                bytecode.append(
                    encode(
                        OPCODES["ADD"],
                        REG_MAP[node[1]],
                        REG_MAP[node[2]],
                        REG_MAP[node[3]],
                        0,
                    )
                )
            elif op == "SUB":
                bytecode.append(
                    encode(
                        OPCODES["SUB"],
                        REG_MAP[node[1]],
                        REG_MAP[node[2]],
                        REG_MAP[node[3]],
                        0,
                    )
                )
            elif op == "HALT":
                bytecode.append(encode(OPCODES["HALT"]))
        return bytecode
