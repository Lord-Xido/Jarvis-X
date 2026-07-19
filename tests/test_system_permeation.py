import numpy as np

from jarvisx.api import app
from jarvisx.assembler import Assembler
from jarvisx.core import CodexVM
from jarvisx.mm3d_omega4 import MM3DConfig
from jarvisx.parser import Parser


def assemble(source):
    return Assembler().assemble(Parser().parse(source))


def tiny_mm3d_config():
    return MM3DConfig(
        xi_size=8,
        manifold_dim=256,
        latent_size=4,
        codebook_size=128,
        codebook_dim=8,
        projection_rank=8,
        metric_rank=4,
        exploration_depth=2,
        render_image_size=8,
        render_video_frames=2,
        render_audio_samples=128,
        max_reference_bytes=32 * 1024 * 1024,
        seed=12345,
    )


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


def test_mm3d_cycle_is_vm_authoritative_and_audited():
    config = tiny_mm3d_config()
    psi = np.linspace(-1.0, 1.0, config.manifold_dim, dtype=np.float32)
    vm = CodexVM(ledger_path=None)

    result = vm.run_mm3d_cycle(psi, config=config)

    assert result.latent_code.shape == (4, 4, 4)
    assert vm.ledger.chain[-1]["payload"].endswith("|MM3D.CYCLE")
    assert vm.tracer.log[-1][0] == "MM3D.CYCLE"
    vm.close()


def test_policy_gate_can_block_mm3d_cycle():
    config = tiny_mm3d_config()
    vm = CodexVM(ledger_path=None)
    vm.ethics.block_action("MM3D.CYCLE")

    try:
        vm.run_mm3d_cycle(np.ones(config.manifold_dim), config=config)
    except RuntimeError as exc:
        assert "policy blocked" in str(exc)
    else:
        raise AssertionError("blocked MM3D action was executed")


def test_fastapi_surface_matches_declared_runtime_stack():
    paths = {route.path for route in app.routes}
    assert "/health" in paths
    assert "/run" in paths
    assert "/visual-memory" in paths
    assert "/mm3d-cycle" in paths
