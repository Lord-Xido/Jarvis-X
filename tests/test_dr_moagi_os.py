from __future__ import annotations

import pytest

from jarvisx.dr_moagi_bitplane import SparseBitPlane3D, fold_and_attenuate
from jarvisx.dr_moagi_os import DrMoagiOSConfig, DrMoagiOSKernel, OSLifecycle, demo_field


def test_uint64_bitplane_packs_arbitrary_z_depth_without_dense_allocation():
    plane = SparseBitPlane3D.from_scalar_field(
        {
            (1, 2, 0): 1.0,
            (1, 2, 64): 0.75,
            (1, 2, 129): 0.5,
            (4, 4, 4): 0.1,
        },
        side=130,
        activation_threshold=0.5,
    )

    assert plane.words_per_column == 3
    assert plane.active_bits == 3
    assert plane.packed_words == 3
    assert plane.logical_words == 130 * 130 * 3
    words = plane.as_word_map()
    assert words[(1, 2, 0)] == 1
    assert words[(1, 2, 1)] == 1
    assert words[(1, 2, 2)] == 2


def test_bitplane_hamming_velocity_and_entropy_are_measured():
    before = SparseBitPlane3D.from_scalar_field(
        {(0, 0, 0): 1.0}, side=4, activation_threshold=0.5
    )
    after = SparseBitPlane3D.from_scalar_field(
        {(0, 0, 1): 1.0}, side=4, activation_threshold=0.5
    )

    metrics = after.metrics(before)

    assert metrics.phase_velocity == pytest.approx(2 / 64)
    assert 0.0 < metrics.entropy < 1.0
    assert metrics.kinetic_energy > 0.0


def test_inward_fold_contracts_toward_centroid_and_attenuates():
    folded = fold_and_attenuate(
        {(8, 4, 4): 1.0},
        side=9,
        contraction=0.5,
        attenuation=0.5,
    )

    assert list(folded) == [(6, 4, 4)]
    assert 0.0 < folded[(6, 4, 4)] < 1.0


def test_kernel_runs_full_transaction_and_persists_checkpoint(tmp_path):
    config = DrMoagiOSConfig(
        side=16,
        max_active_cells=4_096,
        fixed_point_passes=1,
        state_dir=tmp_path,
    )
    kernel = DrMoagiOSKernel(config)
    kernel.boot(restore=False)
    kernel.load(demo_field(16))

    report = kernel.step()

    assert report.committed
    assert report.active_cells_after > 0
    assert report.logical_words == 16 * 16
    assert len(report.journal_hash) == 64
    assert kernel.journal.verify()
    assert kernel.status()["journal_valid"] is True
    checkpoint = tmp_path / "checkpoint.json"
    assert checkpoint.exists()

    kernel.shutdown()
    restored = DrMoagiOSKernel(config)
    status = restored.boot(restore=True)
    assert status["loaded"] is True
    assert status["cycle"] == 1
    assert status["state_hash"] == report.state_hash


def test_kernel_rejects_lossy_cycle_transactionally_and_halts(tmp_path):
    config = DrMoagiOSConfig(
        side=8,
        max_active_cells=256,
        contraction=0.0,
        attenuation=0.0,
        block_size=2,
        quantization=0.01,
        max_reconstruction_mse=0.0,
        fixed_point_passes=0,
        state_dir=tmp_path,
    )
    kernel = DrMoagiOSKernel(config)
    kernel.boot(restore=False)
    initial = {(2, 2, 2): 1.0, (3, 2, 2): 0.1}
    kernel.load(initial)

    report = kernel.step()

    assert not report.committed
    assert report.rejection_reason == "validator rejected candidate"
    assert kernel.snapshot()["state_hash"] == DrMoagiOSKernel._state_hash(initial)
    assert kernel.lifecycle is OSLifecycle.HALTED
    assert kernel.journal.verify()


def test_os_api_exposes_control_plane_routes():
    from jarvisx.dr_moagi_os_api import app

    paths = {route.path for route in app.routes}
    assert "/" in paths
    assert "/healthz" in paths
    assert "/v1/os/boot" in paths
    assert "/v1/os/load" in paths
    assert "/v1/os/step" in paths
    assert "/v1/os/run" in paths
    assert "/v1/os/autorun/start" in paths
    assert "/v1/os/snapshot" in paths
    assert "/v1/os/bitplane" in paths
    assert "/metrics" in paths


def test_os_dashboard_is_live_threejs_control_plane_without_fake_physical_metrics():
    from jarvisx.dr_moagi_os_ui import DR_MOAGI_OS_HTML

    assert "three.min.js" in DR_MOAGI_OS_HTML
    assert "OrbitControls" in DR_MOAGI_OS_HTML
    assert "/v1/os/status" in DR_MOAGI_OS_HTML
    assert "/v1/os/step" in DR_MOAGI_OS_HTML
    assert "/v1/os/run" in DR_MOAGI_OS_HTML
    assert "/v1/os/snapshot" in DR_MOAGI_OS_HTML
    assert "Measured Runtime Telemetry" in DR_MOAGI_OS_HTML
    assert "render-only" in DR_MOAGI_OS_HTML
    assert "Reality Gap γ = 0.0000" not in DR_MOAGI_OS_HTML
    assert "128.4 / 512 GB" not in DR_MOAGI_OS_HTML
    assert "J/m³" not in DR_MOAGI_OS_HTML
    assert "3,850 tok/s" not in DR_MOAGI_OS_HTML
