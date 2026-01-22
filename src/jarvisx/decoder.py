from .instruction import Instruction

class Decoder:
    def decode(self, word):
        opcode = (word >> 56) & 0xFF
        dst    = (word >> 40) & 0xFF
        src1   = (word >> 32) & 0xFF
        src2   = (word >> 24) & 0xFF
        imm    = (word >> 8) & 0xFFFF
        return Instruction(opcode, dst, src1, src2, imm)
