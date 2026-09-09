from __future__ import annotations

from jarvisx.sparse_dense_bridge import (
    DenseTile,
    DenseTileResult,
    SparseDenseBridgeConfig,
    SparseDenseTileBridge,
    SparseDenseTransactionalRuntime,
)
from jarvisx.dr_moagi_multiparallel import MultiparallelConfig


class OffsetBackend:
    def __init__(self, offset: float) -> None:
        self.offset = offset
        self.depths: list[int] = []

    def process(self, tile: DenseTile) -> DenseTileResult:
        self.depths.append(tile.depth)
        values = list(tile.values)
        for coordinate in tile.active_coordinates:
            local_x = coordinate[0] - tile.origin[0]
            local_y = coordinate[1] - tile.origin[1]
            for channel in range(tile.channels):
                index = (((channel * tile.depth + tile.centre_depth) * tile.side + local_y) * tile.side) + local_x
                values[index] += self.offset
        return DenseTileResult(tuple(values), metrics=(("offset", self.offset),))


def _runtime() -> SparseDenseTransactionalRuntime:
    runtime = SparseDenseTransactionalRuntime(
        MultiparallelConfig(
            side=32,
            depth=2,
            max_active_loops=32,
            max_channels=4,
            expand_halo=False,
            global_coupling=0.0,
            max_reconstruction_mse=1.0,
        )
    )
    runtime.load({(1, 1): (1.0, 0.0), (2, 3): (0.5, -0.5)})
    return runtime


def test_sparse_dense_bridge_commits_backend_candidate_without_conflating_depth_axes():
    runtime = _runtime()
    backend = OffsetBackend(0.1)
    bridge = SparseDenseTileBridge(
        SparseDenseBridgeConfig(
            tile_side=4,
            tile_depth=5,
            max_tiles=4,
            max_candidate_mse=0.1,
            output_blend=1.0,
            convergence_tol=1.0e-8,
        )
    )

    result = bridge.run(runtime, backend)
    surface = runtime.snapshot_surface()

    assert result.receipt.committed
    assert result.tile_count == 1
    assert backend.depths == [5]
    assert runtime.config.depth == 2
    assert surface[(1, 1)] == (1.1, 0.1)
    assert surface[(2, 3)] == (0.6, -0.4)
    assert result.state_delta_rms > 0.0
    assert result.receipt.after_state_hash == result.receipt.candidate_state_hash


def test_sparse_dense_bridge_rolls_back_on_epistemic_gate_failure():
    runtime = _runtime()
    before = runtime.snapshot_surface()
    bridge = SparseDenseTileBridge(
        SparseDenseBridgeConfig(tile_side=4, tile_depth=3, max_candidate_mse=1.0)
    )

    result = bridge.run(
        runtime,
        OffsetBackend(0.1),
        epistemic_validator=lambda _candidate: False,
    )

    assert not result.receipt.committed
    assert result.receipt.verification.epistemic == "FAIL"
    assert runtime.snapshot_surface() == before
    assert result.receipt.after_state_hash == result.receipt.parent_state_hash


def test_sparse_dense_bridge_rolls_back_on_distortion_budget():
    runtime = _runtime()
    before = runtime.snapshot_surface()
    bridge = SparseDenseTileBridge(
        SparseDenseBridgeConfig(
            tile_side=4,
            tile_depth=3,
            max_candidate_mse=1.0e-6,
            output_blend=1.0,
        )
    )

    result = bridge.run(runtime, OffsetBackend(1.0))

    assert not result.receipt.committed
    assert result.receipt.verification.numerical == "FAIL"
    assert runtime.snapshot_surface() == before


def test_run_until_converged_stops_on_fixed_point():
    runtime = _runtime()
    bridge = SparseDenseTileBridge(
        SparseDenseBridgeConfig(
            tile_side=4,
            tile_depth=3,
            max_candidate_mse=0.0,
            output_blend=1.0,
            convergence_tol=0.0,
        )
    )

    runs = bridge.run_until_converged(runtime, OffsetBackend(0.0), max_rounds=5)

    assert len(runs) == 1
    assert runs[0].receipt.committed
    assert runs[0].converged
