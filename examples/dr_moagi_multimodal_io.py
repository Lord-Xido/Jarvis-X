"""Minimal Dr Moagi multimodal 3D I/O reference example."""

from jarvisx.dr_moagi_multimodal_io import (
    DrMoagiMultimodalConfig,
    DrMoagiMultimodalRuntime,
    IdentityMediumAdapter,
    MediumChannel,
)


def main() -> None:
    runtime = DrMoagiMultimodalRuntime(
        [
            MediumChannel("vision", IdentityMediumAdapter()),
            MediumChannel("audio", IdentityMediumAdapter()),
            MediumChannel("network", IdentityMediumAdapter(), input_weight=0.5),
        ],
        config=DrMoagiMultimodalConfig(
            side=16,
            vector_width=4,
            dt=0.1,
            max_active_cells=1024,
        ),
    )
    runtime.load({})

    result = runtime.step(
        {
            "vision": {(2, 2, 2): (0.8, 0.1, 0.0, 0.0)},
            "audio": {(2, 2, 2): (0.2, 0.7, 0.0, 0.0)},
            "network": {(3, 2, 2): (0.0, 0.0, 0.9, 0.1)},
        }
    )

    print("cycle:", result.metrics.cycle)
    print("committed:", result.metrics.committed)
    print("active cells:", result.metrics.active_cells_after)
    print("loss:", result.metrics.loss)


if __name__ == "__main__":
    main()
