from .instruction import Instruction


class Decoder:
    def decode(self, word):
        if not isinstance(word, int) or word < 0 or word >= 1 << 64:
            raise ValueError("instruction word must be an unsigned 64-bit integer")
        opcode = (word >> 56) & 0xFF
        dst = (word >> 40) & 0xFF
        src1 = (word >> 32) & 0xFF
        src2 = (word >> 24) & 0xFF
        imm = (word >> 8) & 0xFFFF
        if imm & 0x8000:
            imm -= 0x10000
        return Instruction(opcode, dst, src1, src2, imm)
