from __future__ import annotations

import pytest

from jarvisx.dr_moagi_bitplane import SparseBitPlane3D, fold_and_attenuate
from jarvisx.dr_moagi_os import DrMoagiOSConfig, DrMoagiOSKernel, OSLifecycle, demo_field
from jarvisx.dr_moagi_os_store import SparseStateCodec3D


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


def test_exact_morton_transport_round_trips_sparse_float_state():
    codec = SparseStateCodec3D()
    field = {
        (0, 0, 0): 0.1,
        (63, 2, 19): -0.75,
        (511, 511, 511): 1.0,
        (257, 129, 33): 0.3333333333333333,
    }

    packet = codec.encode(field, side=512)
    decoded = codec.decode(packet)

    assert decoded == field
    assert packet.active_cells == len(field)
    assert packet.encoded_bytes > 0
    assert len(packet.checksum_sha256) == 64


def test_kernel_runs_full_transaction_and_persists_checkpoint(tmp_path):
    config = DrMoagiOSConfig(
        side=16,
        max_active_cells=4_096,
        deep_distiller_max_latent_cells=2_048,
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
    assert report.distiller_iteration == 1
    assert report.transport_bytes > 0
    assert len(report.transport_hash) == 64
    assert len(report.theta_hash) == 64
    assert len(report.journal_hash) == 64
    assert kernel.journal.verify()
    status = kernel.status()
    assert status["journal_valid"] is True
    assert status["distiller"]["iteration"] == 1
    assert status["transport"]["format"] == "DMOS2"
    checkpoint = tmp_path / "checkpoint.json"
    assert checkpoint.exists()

    kernel.shutdown()
    restored = DrMoagiOSKernel(config)
    restored_status = restored.boot(restore=True)
    assert restored_status["loaded"] is True
    assert restored_status["cycle"] == 1
    assert restored_status["state_hash"] == report.state_hash
    assert restored_status["distiller"]["iteration"] == 1
    assert restored_status["distiller"]["theta_hash"] == report.theta_hash


def test_kernel_export_import_is_exact_end_to_end():
    config = DrMoagiOSConfig(
        side=32,
        max_active_cells=2_048,
        deep_distiller_max_latent_cells=1_024,
        state_dir=None,
    )
    source = demo_field(32)
    first = DrMoagiOSKernel(config)
    first.boot(restore=False)
    first.load(source)
    packet = first.export_state_packet()
    first_hash = first.status()["state_hash"]

    second = DrMoagiOSKernel(config)
    second.boot(restore=False)
    second.import_state_packet(packet.payload, expected_checksum=packet.checksum_sha256)

    assert second.status()["state_hash"] == first_hash
    assert second.snapshot()["total_active_cells"] == len(source)


def test_late_transport_failure_rolls_back_state_and_adaptation(monkeypatch):
    config = DrMoagiOSConfig(
        side=16,
        max_active_cells=4_096,
        deep_distiller_max_latent_cells=2_048,
        fixed_point_passes=0,
        state_dir=None,
    )
    kernel = DrMoagiOSKernel(config)
    kernel.boot(restore=False)
    kernel.load(demo_field(16))
    before = kernel.status()
    before_hash = before["state_hash"]
    before_theta_hash = before["distiller"]["theta_hash"]
    before_iteration = before["distiller"]["iteration"]

    def fail_transport(_field):
        raise ValueError("injected transport failure")

    monkeypatch.setattr(kernel, "_verified_transport", fail_transport)
    report = kernel.step()
    after = kernel.status()

    assert not report.committed
    assert "transport verification failed" in (report.rejection_reason or "")
    assert after["state_hash"] == before_hash
    assert after["distiller"]["theta_hash"] == before_theta_hash
    assert after["distiller"]["iteration"] == before_iteration
    assert kernel.lifecycle is OSLifecycle.HALTED


def test_kernel_rejects_lossy_cycle_transactionally_and_halts(tmp_path):
    config = DrMoagiOSConfig(
        side=8,
        max_active_cells=256,
        deep_distiller_max_latent_cells=128,
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
    before_theta = kernel.status()["distiller"]["theta_hash"]

    report = kernel.step()

    assert not report.committed
    assert report.rejection_reason == "validator rejected candidate"
    assert kernel.snapshot()["state_hash"] == DrMoagiOSKernel._state_hash(initial)
    assert kernel.status()["distiller"]["theta_hash"] == before_theta
    assert kernel.lifecycle is OSLifecycle.HALTED
    assert kernel.journal.verify()


def test_kernel_capabilities_describe_bounded_end_to_end_runtime():
    kernel = DrMoagiOSKernel(DrMoagiOSConfig(side=8, state_dir=None))
    capabilities = kernel.capabilities()

    assert capabilities["adaptive_core"] == "DM-DD transactional residual auto-iteration"
    assert capabilities["transport"] == "DMOS2 exact Morton-delta float64 packet"
    assert capabilities["arbitrary_host_commands"] is False
    assert capabilities["self_rewriting_code"] is False
    assert capabilities["dense_logical_allocation"] is False


def test_os_api_exposes_control_plane_and_meta_routes():
    from jarvisx.dr_moagi_os_api import app

    paths = {route.path for route in app.routes}
    assert "/" in paths
    assert "/healthz" in paths
    assert "/v1/os/capabilities" in paths
    assert "/v1/os/boot" in paths
    assert "/v1/os/load" in paths
    assert "/v1/os/step" in paths
    assert "/v1/os/run" in paths
    assert "/v1/os/autorun/start" in paths
    assert "/v1/os/snapshot" in paths
    assert "/v1/os/bitplane" in paths
    assert "/v1/os/export" in paths
    assert "/v1/os/import" in paths
    assert "/v1/os/meta/status" in paths
    assert "/v1/os/meta/lattice" in paths
    assert "/v1/os/meta/optimize" in paths
    assert "/metrics" in paths


def test_os_dashboard_is_live_threejs_self_loop_control_plane_without_fake_metrics():
    from jarvisx.dr_moagi_os_ui import DR_MOAGI_OS_HTML

    assert "three.min.js" in DR_MOAGI_OS_HTML
    assert "OrbitControls" in DR_MOAGI_OS_HTML
    assert "/v1/os/status" in DR_MOAGI_OS_HTML
    assert "/v1/os/step" in DR_MOAGI_OS_HTML
    assert "/v1/os/run" in DR_MOAGI_OS_HTML
    assert "/v1/os/snapshot" in DR_MOAGI_OS_HTML
    assert "/v1/os/meta/lattice" in DR_MOAGI_OS_HTML
    assert "/v1/os/meta/optimize" in DR_MOAGI_OS_HTML
    assert "Measured Runtime Telemetry" in DR_MOAGI_OS_HTML
    assert "Inward Meta-Optimizer" in DR_MOAGI_OS_HTML
    assert "SPARSE STATE METRIC g = I + C / tr(C)" in DR_MOAGI_OS_HTML
    assert "PROVISIONAL ≠ AUTHORITATIVE" in DR_MOAGI_OS_HTML
    assert "external_sota_verified" in DR_MOAGI_OS_HTML
    assert "matrixWorld.elements" not in DR_MOAGI_OS_HTML
    assert "Reality Gap γ = 0.0000" not in DR_MOAGI_OS_HTML
    assert "Fixed Point (ΔΨ): 0.00000" not in DR_MOAGI_OS_HTML
    assert "128.4 / 512 GB" not in DR_MOAGI_OS_HTML
    assert "J/m³" not in DR_MOAGI_OS_HTML
    assert "3,850 tok/s" not in DR_MOAGI_OS_HTML
