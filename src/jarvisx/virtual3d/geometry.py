"""Geometry and address translation for the sparse 3D virtual computer."""

from dataclasses import dataclass
from math import ceil
from typing import Tuple

Coordinate3D = Tuple[int, int, int]


@dataclass(frozen=True)
class BlockAddress:
    """Resolved block index and local offset for one logical coordinate."""

    block: Coordinate3D
    offset: Coordinate3D


@dataclass(frozen=True)
class VolumeGeometry:
    """Immutable geometry contract for a sparse logical 3D volume.

    ``extent`` and ``block_shape`` are measured in logical cells. ``cell_bytes``
    defines the maximum payload represented by one cell. Setting ``cell_bytes``
    to one gives byte-addressable semantics. Setting it to one billion models
    the 6400^3 one-gigabyte-cell design without allocating that capacity.
    """

    extent: Coordinate3D = (6400, 6400, 6400)
    block_shape: Coordinate3D = (1024, 1024, 1024)
    cell_bytes: int = 1

    def __post_init__(self) -> None:
        if any(value <= 0 for value in self.extent):
            raise ValueError("extent dimensions must be positive")
        if any(value <= 0 for value in self.block_shape):
            raise ValueError("block dimensions must be positive")
        if self.cell_bytes <= 0:
            raise ValueError("cell_bytes must be positive")

    @property
    def logical_cells(self) -> int:
        x, y, z = self.extent
        return x * y * z

    @property
    def logical_capacity_bytes(self) -> int:
        return self.logical_cells * self.cell_bytes

    @property
    def block_grid_shape(self) -> Coordinate3D:
        return tuple(
            int(ceil(axis / block))
            for axis, block in zip(self.extent, self.block_shape)
        )  # type: ignore[return-value]

    @property
    def maximum_block_count(self) -> int:
        x, y, z = self.block_grid_shape
        return x * y * z

    def validate_coordinate(self, coordinate: Coordinate3D) -> None:
        if len(coordinate) != 3:
            raise ValueError("coordinate must contain exactly three values")
        for axis, limit in zip(coordinate, self.extent):
            if axis < 0 or axis >= limit:
                raise IndexError(
                    "coordinate {} lies outside extent {}".format(
                        coordinate, self.extent
                    )
                )

    def map_coordinate(self, coordinate: Coordinate3D) -> BlockAddress:
        self.validate_coordinate(coordinate)
        block = tuple(
            axis // block_axis
            for axis, block_axis in zip(coordinate, self.block_shape)
        )
        offset = tuple(
            axis % block_axis
            for axis, block_axis in zip(coordinate, self.block_shape)
        )
        return BlockAddress(block=block, offset=offset)  # type: ignore[arg-type]

    def global_coordinate(
        self, block: Coordinate3D, offset: Coordinate3D
    ) -> Coordinate3D:
        coordinate = tuple(
            block_axis * block_size + local_axis
            for block_axis, block_size, local_axis in zip(
                block, self.block_shape, offset
            )
        )
        self.validate_coordinate(coordinate)  # type: ignore[arg-type]
        return coordinate  # type: ignore[return-value]

    def effective_block_shape(self, block: Coordinate3D) -> Coordinate3D:
        grid = self.block_grid_shape
        if any(axis < 0 or axis >= limit for axis, limit in zip(block, grid)):
            raise IndexError("block {} lies outside grid {}".format(block, grid))
        shape = []
        for block_axis, block_size, extent_axis in zip(
            block, self.block_shape, self.extent
        ):
            start = block_axis * block_size
            shape.append(min(block_size, extent_axis - start))
        return tuple(shape)  # type: ignore[return-value]
