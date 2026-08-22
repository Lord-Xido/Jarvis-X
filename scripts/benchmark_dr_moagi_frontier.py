"""Deterministic empirical benchmark for the Dr Moagi frontier runtime."""

from __future__ import annotations

import json

from jarvisx.dr_moagi_autoexec import AutoExecPolicy
from jarvisx.dr_moagi_frontier import (
    DrMoagiFrontierRuntime,
    FrontierConfig,
    SolverConfig,
    SparseAndersonSolver3D,
)
from jarvisx.dr_moagi_os import demo_field


def main() -> int:
    solver = SparseAndersonSolver3D(
        SolverConfig(tolerance=1.0e-8, max_iterations=30, depth=4, damping=1.0)
    )

    def affine(field: dict[tuple[int, int, int], float]) -> dict[tuple[int, int, int], float]:
        value = float(field.get((0, 0, 0), 0.0))
        return {(0, 0, 0): 0.5 * value + 0.5}

    initial = {(0, 0, 0): 0.0}
    plain = solver.solve_plain(affine, initial)
    accelerated = solver.solve(affine, initial)

    runtime = DrMoagiFrontierRuntime(
        FrontierConfig(
            side=16,
            max_active_cells=4_096,
            policy=AutoExecPolicy(block_size=2, quantization=0.01, prune_epsilon=0.0),
            max_iterations=12,
        )
    )
    runtime.load(demo_field(16))
    report = runtime.step()

    payload = {
        "affine_fixed_point": {
            "plain_iterations": plain.iterations,
            "accelerated_iterations": accelerated.iterations,
            "iteration_speedup": plain.iterations / max(1, accelerated.iterations),
            "plain_residual": plain.residual,
            "accelerated_residual": accelerated.residual,
        },
        "frontier_cycle": report.as_dict(),
        "claim_status": runtime.status()["claim_status"],
    }
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))

    if not accelerated.converged:
        raise SystemExit("accelerated solver failed to converge")
    if accelerated.iterations >= plain.iterations:
        raise SystemExit("accelerated solver did not beat the plain affine baseline")
    if not report.committed:
        raise SystemExit("frontier cycle did not commit")
    if report.selected_objective > report.plain_objective + 1.0e-15:
        raise SystemExit("frontier selection regressed the internal baseline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
