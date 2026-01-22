class Instruction:
    def __init__(self, opcode, dst=0, src1=0, src2=0, imm=0):
        self.opcode = opcode
        self.dst = dst
        self.src1 = src1
        self.src2 = src2
        self.imm = imm
