from jarvisx.assembler import Assembler
from jarvisx.core import CodexVM
from jarvisx.parser import Parser


def assemble(source):
    return Assembler().assemble(Parser().parse(source))


def test_add_is_deterministic_without_reflex_side_effects():
    vm = CodexVM()
    vm.load(assemble("SET Ψ 10\nSET Φ 20\nADD A Ψ Φ\nHALT"))
    state = vm.run()

    assert state["A"] == 30
    assert state["Φ"] == 20
    assert vm.ledger.verify()


def test_reflex_loop_is_explicitly_opt_in():
    vm = CodexVM(reflex_enabled=True)
    vm.load(assemble("SET Ψ 10\nSET Φ 20\nHALT"))
    state = vm.run()

    assert state["Φ"] == 18
    assert any("REFLEX_DELTA" in snapshot for _, snapshot in vm.tracer.log)


def test_loading_a_new_program_resets_execution_state():
    vm = CodexVM()
    vm.load(assemble("SET A 7\nHALT"))
    vm.run()
    assert vm.cycles == 2

    vm.load(assemble("SET A 9\nHALT"))
    vm.run()
    assert vm.cycles == 2
    assert vm.regs["A"] == 9
