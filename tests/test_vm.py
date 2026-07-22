from jarvisx.assembler import Assembler
from jarvisx.core import CodexVM
from jarvisx.parser import Parser


def assemble(source):
    return Assembler().assemble(Parser().parse(source))


def test_add():
    vm = CodexVM()
    vm.load(assemble("SET Ψ 10\nSET Φ 20\nADD A Ψ Φ\nHALT"))
    vm.run()
    assert vm.regs["A"] == 30
    assert vm.ledger.verify()


def test_loop_memory_and_control_flow():
    code = """
    SET A 3
    SET B 1
    SET C 0
    loop:
    ADD C C A
    SUB A A B
    CMP A Ξ
    JNZ loop
    STORE C 0
    LOAD D 0
    HALT
    """
    vm = CodexVM()
    vm.load(assemble(code))
    vm.run()
    assert vm.regs["D"] == 6
    assert vm.mem.load_int(0) == 6
    assert vm.cycles == 18
    assert len(vm.ledger.chain) == vm.cycles
