from jarvisx.api import app
from jarvisx.assembler import Assembler
from jarvisx.core import CodexVM
from jarvisx.parser import Parser


def assemble(source):
    return Assembler().assemble(Parser().parse(source))


def test_vm_can_execute_multiple_loaded_programs():
    vm = CodexVM(ledger_path=None)
    vm.load(assemble("SET A 5\nHALT"))
    vm.run()
    assert vm.regs["A"] == 5
    assert vm.cycles == 2

    vm.load(assemble("SET B 7\nHALT"))
    vm.run()
    assert vm.regs["B"] == 7
    assert vm.cycles == 2


def test_geometric_permeation_is_vm_authoritative_and_audited():
    vm = CodexVM(ledger_path=None)
    result = vm.run_visual_memory(size=8)

    assert result.reconstruction.shape == (8, 8, 8)
    assert vm.ledger.chain[-1]["payload"].endswith("|V3D.PERMEATE")
    assert vm.tracer.log[-1][0] == "V3D.PERMEATE"


def test_policy_gate_can_block_geometric_permeation():
    vm = CodexVM(ledger_path=None)
    vm.ethics.block_action("V3D.PERMEATE")

    try:
        vm.run_visual_memory(size=8)
    except RuntimeError as exc:
        assert "policy blocked" in str(exc)
    else:
        raise AssertionError("blocked geometric action was executed")


def test_fastapi_surface_matches_declared_runtime_stack():
    paths = {route.path for route in app.routes}
    assert "/health" in paths
    assert "/run" in paths
    assert "/visual-memory" in paths
