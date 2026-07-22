import pytest

from jarvisx.assembler import Assembler
from jarvisx.core import CodexVM
from jarvisx.electronic import ElectronicConfig, ElectronicSubstrate
from jarvisx.parser import Parser


def test_rejected_instruction_restores_all_transactional_state():
    substrate = ElectronicSubstrate(
        ElectronicConfig(clock_hz=100_000_000_000.0, enforce_limits=True)
    )
    vm = CodexVM(electronics=substrate)
    vm.load(Assembler().assemble(Parser().parse("SET A 1\nHALT")))

    with pytest.raises(RuntimeError, match="Electronic Λ-gate"):
        vm.step()

    assert vm.regs["A"] == 0
    assert vm.regs["IP"] == 0
    assert vm.cycles == 0
    assert vm.running is True
    assert vm.electronics.cycle == 0
    assert vm.electronics.trace == []
    assert vm.ledger.chain == []
    assert vm.tracer.log == []
