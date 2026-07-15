from .instruction import Instruction


class Decoder:
    def decode(self, word):
        word = int(word) & 0xFFFFFFFFFFFFFFFF
        opcode = (word >> 56) & 0xFF
        dst = (word >> 40) & 0xFF
        src1 = (word >> 32) & 0xFF
        src2 = (word >> 24) & 0xFF
        imm_raw = (word >> 8) & 0xFFFF
        imm = imm_raw - 0x10000 if imm_raw & 0x8000 else imm_raw
        return Instruction(opcode, dst, src1, src2, imm)
