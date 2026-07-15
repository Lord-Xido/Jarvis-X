REG_MAP = {
    "Ξ": 0, "Ψ": 1, "Φ": 2, "Λ": 3, "Ω": 4,
    "Θ": 5, "𝒮": 6, "Π": 7,
    "A": 8, "B": 9, "C": 10, "D": 11,
    "IP": 12, "SP": 13, "FLAGS": 14, "TMP": 15,
}

OPCODES = {
    "SET": 0x01,
    "MOV": 0x02,
    "ADD": 0x03,
    "SUB": 0x04,
    "LOAD": 0x05,
    "STORE": 0x06,
    "CMP": 0x07,
    "JMP": 0x08,
    "JZ": 0x09,
    "HALT": 0x0A,
    "JNZ": 0x0B,
    "MUL": 0x0C,
    "XOR": 0x0D,
    "AND": 0x0E,
    "OR": 0x0F,
}


def encode(opcode, dst=0, src1=0, src2=0, imm=0):
    return (
        ((int(opcode) & 0xFF) << 56)
        | ((int(dst) & 0xFF) << 40)
        | ((int(src1) & 0xFF) << 32)
        | ((int(src2) & 0xFF) << 24)
        | ((int(imm) & 0xFFFF) << 8)
    )


class Assembler:
    def _register(self, name):
        try:
            return REG_MAP[name]
        except KeyError as exc:
            raise ValueError("Unknown register: {}".format(name)) from exc

    def _value(self, token, labels):
        if token in labels:
            return labels[token]
        try:
            return int(token, 0)
        except ValueError as exc:
            raise ValueError("Unknown label or integer: {}".format(token)) from exc

    def _expect(self, node, count):
        if len(node) != count:
            raise ValueError(
                "{} expects {} operand(s), received {}".format(
                    node[0], count - 1, len(node) - 1
                )
            )

    def assemble(self, ast):
        labels = {}
        instructions = []
        pc = 0

        for node in ast:
            if len(node) == 1 and node[0].endswith(":"):
                label = node[0][:-1]
                if not label:
                    raise ValueError("Empty label")
                if label in labels:
                    raise ValueError("Duplicate label: {}".format(label))
                labels[label] = pc
                continue
            instructions.append(node)
            pc += 1

        bytecode = []
        for node in instructions:
            op = node[0].upper()
            if op not in OPCODES:
                raise ValueError("Unknown opcode: {}".format(op))

            if op == "SET":
                self._expect(node, 3)
                bytecode.append(
                    encode(OPCODES[op], dst=self._register(node[1]), imm=self._value(node[2], labels))
                )
            elif op == "MOV":
                self._expect(node, 3)
                bytecode.append(
                    encode(OPCODES[op], dst=self._register(node[1]), src1=self._register(node[2]))
                )
            elif op in {"ADD", "SUB", "MUL", "XOR", "AND", "OR"}:
                self._expect(node, 4)
                bytecode.append(
                    encode(
                        OPCODES[op],
                        dst=self._register(node[1]),
                        src1=self._register(node[2]),
                        src2=self._register(node[3]),
                    )
                )
            elif op == "LOAD":
                self._expect(node, 3)
                bytecode.append(
                    encode(OPCODES[op], dst=self._register(node[1]), imm=self._value(node[2], labels))
                )
            elif op == "STORE":
                self._expect(node, 3)
                bytecode.append(
                    encode(OPCODES[op], src1=self._register(node[1]), imm=self._value(node[2], labels))
                )
            elif op == "CMP":
                self._expect(node, 3)
                bytecode.append(
                    encode(OPCODES[op], src1=self._register(node[1]), src2=self._register(node[2]))
                )
            elif op in {"JMP", "JZ", "JNZ"}:
                self._expect(node, 2)
                bytecode.append(encode(OPCODES[op], imm=self._value(node[1], labels)))
            elif op == "HALT":
                self._expect(node, 1)
                bytecode.append(encode(OPCODES[op]))

        return bytecode
