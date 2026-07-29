import pytest

from jarvisx.assembler import Assembler
from jarvisx.core import CodexVM
from jarvisx.parser import Parser


def assemble(source: str) -> list[int]:
    return Assembler().assemble(Parser().parse(source))


def test_add() -> None:
    vm = CodexVM()
    vm.load(assemble("SET Ψ 10\nSET Φ 20\nADD A Ψ Φ\nHALT"))

    final_state = vm.run()

    assert final_state["A"] == 30
    assert vm.cycles == 4
    assert vm.ledger.verify()
    assert len(vm.ledger.chain) == 4


def test_reflex_stabilization_is_explicitly_opt_in() -> None:
    program = assemble("SET Ψ 10\nSET Φ 20\nHALT")

    authoritative = CodexVM()
    authoritative.load(program)
    authoritative.run()

    adaptive = CodexVM(enable_reflex=True)
    adaptive.load(program)
    adaptive.run()

    assert authoritative.regs["Φ"] == 20
    assert adaptive.regs["Φ"] != authoritative.regs["Φ"]


def test_load_rejects_empty_program() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        CodexVM().load([])


def test_load_rejects_non_64_bit_words() -> None:
    vm = CodexVM()
    with pytest.raises(ValueError, match="unsigned 64-bit"):
        vm.load([1 << 64])
    with pytest.raises(TypeError, match="not an integer"):
        vm.load([True])


def test_step_rejects_out_of_bounds_instruction_pointer() -> None:
    vm = CodexVM()
    vm.load(assemble("HALT"))
    vm.regs["IP"] = 9

    with pytest.raises(RuntimeError, match="outside the loaded program"):
        vm.step()
