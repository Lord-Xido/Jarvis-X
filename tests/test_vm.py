from jarvisx.core import CodexVM
from jarvisx.parser import Parser
from jarvisx.assembler import Assembler

def test_add():
    code = "SET Ψ 10\nSET Φ 20\nADD A Ψ Φ\nHALT"
    ast = Parser().parse(code)
    bc = Assembler().assemble(ast)
    vm = CodexVM()
    vm.load(bc)
    vm.run()
    assert vm.regs["A"] == 30
