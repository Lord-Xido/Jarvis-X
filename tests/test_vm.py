from jarvisx.assembler import Assembler
from jarvisx.core import CodexVM
from jarvisx.parser import Parser


def test_add_preserves_exact_assembly_semantics():
    code = "SET Ψ 10\nSET Φ 20\nADD A Ψ Φ\nHALT"
    ast = Parser().parse(code)
    bytecode = Assembler().assemble(ast)
    vm = CodexVM()
    vm.load(bytecode)
    vm.run()
    assert vm.regs["A"] == 30


def test_reflex_stabilisation_is_explicit():
    vm = CodexVM()
    vm.regs["Ψ"] = 10
    vm.regs["Φ"] = 20

    vm.apply_reflex()

    assert vm.regs["Φ"] == 19
