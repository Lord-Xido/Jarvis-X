from __future__ import annotations

from jarvisx.dm_vomegaxi_fixed_point import DMvOmegaXiFixedPointConfig
from jarvisx.reality_reckoning import (
    RealityConstrainedDMvOmegaXiEngine,
    SparseRealityAnchor,
)


def main() -> None:
    world = {
        (2, 2, 2): 0.25,
        (3, 2, 2): 0.25,
    }
    initial = {
        (2, 2, 2): 0.75,
        (3, 2, 2): 0.75,
    }

    anchor = SparseRealityAnchor(
        world,
        tolerance=1.0e-4,
        correction_gain=0.5,
    )
    engine = RealityConstrainedDMvOmegaXiEngine(
        anchor,
        DMvOmegaXiFixedPointConfig(
            fixed_point_tolerance=1.0e-4,
            max_iterations=64,
        ),
    )
    engine.load(initial)

    for report in engine.run_until_fixed_point():
        print(
            f"iter={report.iteration:02d} "
            f"internal={report.fixed_point_residual:.6g} "
            f"external={report.external_residual_after:.6g} "
            f"converged={report.converged}"
        )

    print(engine.status())


if __name__ == "__main__":
    main()
