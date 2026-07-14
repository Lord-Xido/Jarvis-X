import pytest

from jarvisx.assembler import Assembler
from jarvisx.core import CodexVM
from jarvisx.parser import Parser


def assemble(source):
    return Assembler().assemble(Parser().parse(source))


def test_scalar_vm_executes_and_can_be_reused():
    vm = CodexVM()
    vm.load(assemble("SET Ψ 10\nSET Φ 20\nADD A Ψ Φ\nHALT"))
    first = vm.run()
    assert first["registers"]["A"] == 30
    assert first["ledger_valid"] is True

    vm.load(assemble("SET A 7\nHALT"))
    second = vm.run()
    assert second["registers"]["A"] == 7
    assert second["cycles"] == 2


def test_unified_30d_opcodes_run_through_codex_vm():
    source = """
    LOAD30
    ENCODE30
    PLACE30
    FIELD30
    PREDICT30
    COMPARE30
    UPDATE_MEMORY30
    PROJECT30
    DECODE30
    HALT30
    """
    vm = CodexVM()
    vm.load(assemble(source), ann_input=[0.8, -0.3, 0.5, 1.0], ann_target=0.8)
    result = vm.run()
    ann = result["ann30d"]
    assert ann["halted"] is True
    assert ann["active_cells"] == 1
    assert ann["cycles"] == 10
    assert len(ann["output"]) == 4
    assert result["registers"]["C"] == 1
    assert result["ledger_valid"] is True
    assert len(vm.tracer.log) == 10


def test_unknown_opcode_and_bad_order_are_rejected():
    with pytest.raises(ValueError, match="unknown opcode"):
        assemble("PERMEATE30\nHALT")

    vm = CodexVM()
    vm.load(assemble("FIELD30\nHALT30"), ann_input=[1.0])
    with pytest.raises(RuntimeError):
        vm.run()


def test_instruction_pointer_out_of_range_is_rejected():
    vm = CodexVM()
    vm.load(assemble("SET A 1"))
    with pytest.raises(RuntimeError, match="instruction pointer"):
        vm.run()
