"""Demonstrate sparse writes, adaptive reblocking, and ROM restoration."""

from pathlib import Path

from jarvisx.virtual3d import OperationalMode, Virtual3DComputer, VolumeGeometry


def main() -> None:
    geometry = VolumeGeometry(
        extent=(6400, 6400, 6400),
        block_shape=(1024, 1024, 1024),
        cell_bytes=1_000_000_000,
    )
    computer = Virtual3DComputer(
        geometry=geometry,
        mode=OperationalMode.ADAPTIVE,
    )

    computer.write((10, 20, 30), b"architectural-state")
    computer.write((6399, 6399, 6399), b"boundary-state")

    before = computer.statistics()
    report = computer.optimize_layout(
        [
            (256, 256, 256),
            (512, 512, 512),
            (1024, 1024, 1024),
        ]
    )

    destination = Path("virtual3d-demo.jxrom")
    fingerprint = computer.save_rom(str(destination))
    restored = Virtual3DComputer.load_rom(str(destination))

    assert restored.read((10, 20, 30)) == b"architectural-state"
    assert restored.read((6399, 6399, 6399)) == b"boundary-state"

    print("Logical capacity (bytes):", before.logical_capacity_bytes)
    print("Allocated cells:", before.allocated_cells)
    print("Selected block shape:", report.selected.block_shape)
    print("ROM fingerprint:", fingerprint)
    print("ROM path:", destination.resolve())


if __name__ == "__main__":
    main()
