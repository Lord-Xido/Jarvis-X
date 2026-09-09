#!/usr/bin/env python3
"""Sparse 1000x1000 -> bounded dense Torch tile operational bridge."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jarvisx.dr_moagi_multiparallel import MultiparallelConfig  # noqa: E402
from jarvisx.sparse_dense_bridge import (  # noqa: E402
    DenseTile,
    DenseTileResult,
    SparseDenseBridgeConfig,
    SparseDenseTileBridge,
    SparseDenseTransactionalRuntime,
)

from engine import VolumetricMultimodalInterpreter  # noqa: E402


class TorchVolumetricTileBackend:
    """Inference adapter from a bridge DenseTile into the merged Torch engine."""

    def __init__(
        self,
        model: VolumetricMultimodalInterpreter,
        *,
        recursive_depth: int = 2,
        convergence_tol: float = 0.0,
    ) -> None:
        if recursive_depth < 0:
            raise ValueError("recursive_depth must be non-negative")
        if not math.isfinite(convergence_tol) or convergence_tol < 0.0:
            raise ValueError("convergence_tol must be finite and non-negative")
        self.model = model
        self.recursive_depth = recursive_depth
        self.convergence_tol = convergence_tol

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def process(self, tile: DenseTile) -> DenseTileResult:
        if tile.channels != 16:
            raise ValueError("Torch volumetric outer shell requires exactly 16 channels")
        if tile.side != self.model.base_res or tile.depth != self.model.base_res:
            raise ValueError("dense tile side/depth must equal the Torch model base_res")
        tensor = torch.tensor(tile.values, dtype=torch.float32, device=self.device).view(
            1,
            tile.channels,
            tile.depth,
            tile.side,
            tile.side,
        )
        self.model.eval()
        with torch.no_grad():
            output, _mid, _inner, _core = self.model(
                tensor,
                recursive_depth=self.recursive_depth,
                commit_state=False,
                convergence_tol=self.convergence_tol,
            )
        delta_rms = float((output - tensor).pow(2).mean().sqrt().cpu())
        recursive_steps = int(self.model.last_metrics.get("recursive_steps", 0))
        values = tuple(float(value) for value in output.detach().cpu().reshape(-1).tolist())
        return DenseTileResult(
            values=values,
            metrics=(
                ("dense_delta_rms", delta_rms),
                ("recursive_steps", recursive_steps),
            ),
        )


def build_demo_runtime() -> SparseDenseTransactionalRuntime:
    runtime = SparseDenseTransactionalRuntime(
        MultiparallelConfig(
            side=1000,
            depth=4,
            max_active_loops=128,
            max_channels=16,
            expand_halo=False,
            global_coupling=0.0,
            max_reconstruction_mse=1.0,
        )
    )
    vector_a = tuple(0.05 * (index + 1) for index in range(16))
    vector_b = tuple(-0.025 * (index + 1) for index in range(16))
    runtime.load({(8, 8): vector_a, (9, 8): vector_b, (8, 9): vector_b, (9, 9): vector_a})
    return runtime


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runtime = build_demo_runtime()
    model = VolumetricMultimodalInterpreter(base_res=16, device=device).to(device)
    backend = TorchVolumetricTileBackend(model, recursive_depth=2, convergence_tol=1.0e-6)
    bridge = SparseDenseTileBridge(
        SparseDenseBridgeConfig(
            tile_side=16,
            tile_depth=16,
            max_tiles=4,
            max_candidate_mse=0.25,
            output_blend=0.01,
            convergence_tol=1.0e-5,
        )
    )
    runs = bridge.run_until_converged(runtime, backend, max_rounds=3)
    final = runs[-1]
    print(
        {
            "device": str(device),
            "rounds": len(runs),
            "committed": final.receipt.committed,
            "converged": final.converged,
            "candidate_mse": final.candidate_mse,
            "state_delta_rms": final.state_delta_rms,
            "receipt_hash": final.receipt.receipt_hash,
            "active_loops": runtime.active_loop_count,
            "logical_loops": runtime.logical_loop_count,
        }
    )
    return 0 if final.receipt.committed else 2


if __name__ == "__main__":
    raise SystemExit(main())
