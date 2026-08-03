import pytest

from jarvisx.assembler import Assembler, encode
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


def test_reflex_effect_is_in_authoritative_receipt() -> None:
    vm = CodexVM(enable_reflex=True)
    vm.load(assemble("SET Ψ 10\nSET Φ 20\nHALT"))

    final_state = vm.run()

    assert vm.ledger.chain[-1]["state"] == final_state
    assert vm.tracer.log[-1][1] == final_state


def test_sandbox_rejection_preserves_last_committed_state() -> None:
    vm = CodexVM(max_cycles=1)
    vm.load(assemble("SET A 1\nSET A 2\nHALT"))

    assert vm.step()

    with pytest.raises(RuntimeError, match="Sandbox limit exceeded"):
        vm.step()

    assert vm.regs["A"] == 1
    assert vm.regs["IP"] == 1
    assert vm.cycles == 1
    assert len(vm.ledger.chain) == 1
    assert len(vm.tracer.log) == 1
    assert vm.running is False


def test_post_execution_receipt_failure_rolls_back_authoritative_state(monkeypatch) -> None:
    vm = CodexVM()
    vm.load(assemble("SET A 7\nHALT"))
    before_registers = vm.regs.snapshot()
    before_memory = vm.mem.snapshot()

    def fail_log(state: object, opcode: int) -> None:
        raise OSError("receipt unavailable")

    monkeypatch.setattr(vm.ledger, "log", fail_log)

    with pytest.raises(OSError, match="receipt unavailable"):
        vm.step()

    assert vm.regs.snapshot() == before_registers
    assert vm.mem.snapshot() == before_memory
    assert vm.cycles == 0
    assert len(vm.ledger.chain) == 0
    assert len(vm.tracer.log) == 0
    assert vm.running is False


def test_unknown_opcode_fails_closed_without_receipt() -> None:
    vm = CodexVM()
    vm.load([encode(0xFF)])
    before_registers = vm.regs.snapshot()

    with pytest.raises(RuntimeError, match="Unsupported opcode 0xFF"):
        vm.step()

    assert vm.regs.snapshot() == before_registers
    assert vm.cycles == 0
    assert not vm.ledger.chain
    assert not vm.tracer.log
    assert vm.running is False


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
