"""Strict assembler for the unified scalar, 30D, and 3D Jarvis-X ISA."""

from .opcodes import OPCODES

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

NO_OPERAND_OPS = {
    "HALT",
    "LOAD30",
    "ENCODE30",
    "PLACE30",
    "FIELD30",
    "PREDICT30",
    "COMPARE30",
    "UPDATE_MEMORY30",
    "PROJECT30",
    "DECODE30",
    "HALT30",
    "LOAD3D",
    "ABSTRACT3D",
    "ROUTE3D",
    "ATTEND3D",
    "PREDICT3D",
    "COMPARE3D",
    "LEARN3D",
    "PROJECT3D",
    "DECODE3D",
    "HALT3D",
}


def encode(opcode, dst=0, src1=0, src2=0, imm=0):
    if not 0 <= opcode <= 0xFF:
        raise ValueError("opcode must fit in 8 bits")
    if any(not 0 <= value <= 0xFF for value in (dst, src1, src2)):
        raise ValueError("register index must fit in 8 bits")
    if not -32768 <= imm <= 65535:
        raise ValueError("immediate must fit in 16 bits")
    encoded_imm = imm & 0xFFFF
    return (
        (opcode << 56)
        | (dst << 40)
        | (src1 << 32)
        | (src2 << 24)
        | (encoded_imm << 8)
    )


class Assembler:
    @staticmethod
    def _register(name):
        try:
            return REG_MAP[name]
        except KeyError as exc:
            raise ValueError(f"unknown register: {name}") from exc

    def assemble(self, ast):
        bytecode = []
        for line_number, node in enumerate(ast, start=1):
            if not node:
                continue
            op = node[0].upper()
            if op not in OPCODES:
                raise ValueError(f"unknown opcode on line {line_number}: {op}")
            try:
                if op == "SET":
                    if len(node) != 3:
                        raise ValueError("SET requires: SET <register> <integer>")
                    bytecode.append(
                        encode(OPCODES[op], self._register(node[1]), imm=int(node[2]))
                    )
                elif op in {"ADD", "SUB"}:
                    if len(node) != 4:
                        raise ValueError(f"{op} requires: {op} <dst> <src1> <src2>")
                    bytecode.append(
                        encode(
                            OPCODES[op],
                            self._register(node[1]),
                            self._register(node[2]),
                            self._register(node[3]),
                        )
                    )
                elif op in NO_OPERAND_OPS:
                    if len(node) != 1:
                        raise ValueError(f"{op} does not accept operands")
                    bytecode.append(encode(OPCODES[op]))
                else:
                    raise ValueError(f"unsupported opcode: {op}")
            except (ValueError, IndexError) as exc:
                raise ValueError(f"assembly error on line {line_number}: {exc}") from exc
        if not bytecode:
            raise ValueError("program must contain at least one instruction")
        return bytecode
