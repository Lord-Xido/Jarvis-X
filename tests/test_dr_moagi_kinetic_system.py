from __future__ import annotations

from jarvisx.dr_moagi_kinetic_system import CanonicalKineticDrMoagiSystem
from jarvisx.dr_moagi_meta_optimizer import MetaSearchConfig
from jarvisx.dr_moagi_os import DrMoagiOSConfig, DrMoagiOSKernel, demo_field


def test_kinetic_meta_cycle_preserves_world_state_and_emits_receipt(tmp_path):
    config = DrMoagiOSConfig(
        side=16,
        max_active_cells=512,
        fixed_point_passes=1,
        state_dir=tmp_path,
    )
    kernel = DrMoagiOSKernel(config)
    kernel.boot(restore=False)
    kernel.load(demo_field(16))
    kernel.step()

    before = kernel.status()
    before_hash = before["state_hash"]
    before_cycle = kernel.cycle
    before_iteration = before["distiller"]["iteration"]

    system = CanonicalKineticDrMoagiSystem(
        kernel,
        search=MetaSearchConfig(
            max_candidates=3,
            probe_cycles=1,
            confirm_cycles=1,
            max_eval_cells=64,
            survivors=1,
            min_relative_improvement=0.0,
        ),
    )
    report = system.turn_inward()
    after = system.status()
    receipt = system.last_kinetic_receipt

    assert receipt is not None
    assert receipt.stages[:6] == (
        "snapshot",
        "observe",
        "encode",
        "propose",
        "shadow",
        "verify",
    )
    assert receipt.stages[-2:] == ("journal", "reenter")
    assert after["state_hash"] == before_hash
    assert after["cycle"] == before_cycle
    assert after["distiller"]["iteration"] == before_iteration
    assert after["canonical_kinetic_meta"]["epoch"] == 1
    assert after["canonical_kinetic_meta"]["journal_valid"] is True
    assert after["canonical_kinetic_meta"]["receipt_head"] == receipt.receipt_hash
    assert system.kinetic_journal.verify()

    if report.promoted:
        assert receipt.decision == "commit"
        assert system.kernel.config.state_dir == tmp_path
    else:
        assert receipt.decision == "rollback"
        assert receipt.parent_state_hash == receipt.resulting_state_hash
