from jarvisx.parser import Parser
from jarvisx.assembler import Assembler

def test_assembler():
    ast = Parser().parse("SET Ψ 7\nHALT")
    bc = Assembler().assemble(ast)
    assert len(bc) == 2
