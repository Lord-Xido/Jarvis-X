"""Run the bounded DM-vOmegaXi+ SK-3D bytecode reference."""

from __future__ import annotations

import json
from dataclasses import asdict

from jarvisx.dm_spatial_kernel import OME6400SpatialKernel, SpatialInputs


def main() -> None:
    kernel = OME6400SpatialKernel()
    frame = kernel.run(
        12,
        SpatialInputs(
            observation=0.72,
            intent=0.45,
            prediction=0.12,
            refinement=0.08,
            grad_theta=-0.02,
            grad_h=0.01,
        ),
    )
    receipt = {
        "engine": "OME-6400 / DM-vOmegaXi+ SK-3D",
        "virtual_rom_bytes": kernel.virtual_rom_bytes,
        "physical_rom_bytes": kernel.physical_rom_bytes,
        "state": asdict(kernel.state),
        "frame": asdict(frame),
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
