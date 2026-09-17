"""Exact structural model for the canonical 1000 GB Cloud ROM profile.

The module is deliberately arithmetic-first.  It does not allocate the logical
1 TB volume.  Instead it exposes exact conversions, address mappings and
hardware/tensor decomposition for the canonical decimal 1000 GB state:

    10_000 x 10_000 x 10_000 voxels x 1 byte = 10**12 bytes.

This keeps logical geometry separate from physical resident memory, in line
with ADR-019.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from math import ceil
from typing import Any


DECIMAL_GB_BYTES = 10**9
DECIMAL_TB_BYTES = 10**12
GIB_BYTES = 2**30
KIB_BYTES = 2**10


@dataclass(frozen=True)
class HardwareBreakdown:
    logical_bytes: int
    logical_bits: int
    minimum_address_bits: int
    cache_line_bytes: int
    cache_lines: int
    page_bytes: int
    pages: int


@dataclass(frozen=True)
class TensorBreakdown:
    byte_voxel_edge: int
    byte_voxels: int
    bit_voxel_edge: int
    bit_voxels: int
    fp32_values: int
    fp16_values: int
    int8_values: int


@dataclass(frozen=True)
class MultimediaBreakdown:
    raw_4k_rgb_frame_bytes: int
    raw_4k_rgb_frames: int
    raw_4k_rgb_seconds_at_24fps: float
    cd_stereo_bytes_per_second: int
    cd_stereo_seconds: float
    cd_stereo_days: float


@dataclass(frozen=True)
class TileCover:
    tile_edge: int
    tile_bytes: int
    tiles_per_axis: int
    covering_tiles: int
    full_tile_equivalent: float
    has_partial_boundary_tiles: bool


class ThousandGBGeometry:
    """Exact byte-addressable geometry for 1000 decimal GB.

    The canonical volume is one byte per voxel over a 10,000^3 lattice.  All
    methods are metadata/arithmetic operations; no dense trillion-byte array is
    created.
    """

    axis_edge = 10_000
    logical_voxels = axis_edge**3
    logical_bytes = logical_voxels
    logical_bits = logical_bytes * 8

    def __init__(self) -> None:
        if self.logical_bytes != DECIMAL_TB_BYTES:
            raise RuntimeError("canonical geometry must equal exactly 10^12 bytes")

    @property
    def decimal_gb(self) -> int:
        return self.logical_bytes // DECIMAL_GB_BYTES

    @property
    def decimal_tb(self) -> int:
        return self.logical_bytes // DECIMAL_TB_BYTES

    @property
    def gibibytes(self) -> float:
        return self.logical_bytes / GIB_BYTES

    @staticmethod
    def thousand_gib_bytes() -> int:
        """Return the exact byte count of 1000 GiB, distinct from 1000 GB."""
        return 1000 * GIB_BYTES

    @staticmethod
    def thousand_gib_bits() -> int:
        return ThousandGBGeometry.thousand_gib_bytes() * 8

    @staticmethod
    def minimum_address_bits(byte_count: int) -> int:
        """Minimum unsigned address width needed for byte offsets [0, n-1]."""
        if byte_count <= 0:
            raise ValueError("byte_count must be positive")
        return (byte_count - 1).bit_length()

    def hardware_breakdown(
        self,
        *,
        cache_line_bytes: int = 64,
        page_bytes: int = 4096,
    ) -> HardwareBreakdown:
        if cache_line_bytes <= 0 or page_bytes <= 0:
            raise ValueError("cache line and page sizes must be positive")
        if self.logical_bytes % cache_line_bytes:
            raise ValueError("cache line size must exactly divide the canonical volume")
        if self.logical_bytes % page_bytes:
            raise ValueError("page size must exactly divide the canonical volume")
        return HardwareBreakdown(
            logical_bytes=self.logical_bytes,
            logical_bits=self.logical_bits,
            minimum_address_bits=self.minimum_address_bits(self.logical_bytes),
            cache_line_bytes=cache_line_bytes,
            cache_lines=self.logical_bytes // cache_line_bytes,
            page_bytes=page_bytes,
            pages=self.logical_bytes // page_bytes,
        )

    def tensor_breakdown(self) -> TensorBreakdown:
        return TensorBreakdown(
            byte_voxel_edge=10_000,
            byte_voxels=10_000**3,
            bit_voxel_edge=20_000,
            bit_voxels=20_000**3,
            fp32_values=self.logical_bytes // 4,
            fp16_values=self.logical_bytes // 2,
            int8_values=self.logical_bytes,
        )

    def offset(self, x: int, y: int, z: int) -> int:
        """Map (x, y, z) to a row-major byte offset without materialization."""
        n = self.axis_edge
        if not (0 <= x < n and 0 <= y < n and 0 <= z < n):
            raise IndexError("coordinate outside canonical 10000^3 volume")
        return x + n * (y + n * z)

    def coordinate(self, offset: int) -> tuple[int, int, int]:
        """Invert :meth:`offset` exactly."""
        if not 0 <= offset < self.logical_bytes:
            raise IndexError("offset outside canonical 1000 GB volume")
        n = self.axis_edge
        z, rem = divmod(offset, n * n)
        y, x = divmod(rem, n)
        return x, y, z

    def tile_cover(self, tile_edge: int) -> TileCover:
        """Describe cubic tiles needed to cover the logical volume.

        `covering_tiles` includes boundary tiles when the tile edge does not
        exactly divide 10,000. `full_tile_equivalent` is the logical byte count
        divided by one full tile's byte count and may therefore be fractional.
        """
        if tile_edge <= 0:
            raise ValueError("tile_edge must be positive")
        tile_bytes = tile_edge**3
        tiles_per_axis = ceil(self.axis_edge / tile_edge)
        return TileCover(
            tile_edge=tile_edge,
            tile_bytes=tile_bytes,
            tiles_per_axis=tiles_per_axis,
            covering_tiles=tiles_per_axis**3,
            full_tile_equivalent=self.logical_bytes / tile_bytes,
            has_partial_boundary_tiles=(self.axis_edge % tile_edge) != 0,
        )

    def multimedia_breakdown(self) -> MultimediaBreakdown:
        """Raw-data equivalents under explicit uncompressed assumptions."""
        raw_4k_rgb_frame_bytes = 3840 * 2160 * 3
        raw_4k_rgb_frames = self.logical_bytes // raw_4k_rgb_frame_bytes
        cd_stereo_bytes_per_second = 44_100 * 2 * 2
        cd_stereo_seconds = self.logical_bytes / cd_stereo_bytes_per_second
        return MultimediaBreakdown(
            raw_4k_rgb_frame_bytes=raw_4k_rgb_frame_bytes,
            raw_4k_rgb_frames=raw_4k_rgb_frames,
            raw_4k_rgb_seconds_at_24fps=raw_4k_rgb_frames / 24.0,
            cd_stereo_bytes_per_second=cd_stereo_bytes_per_second,
            cd_stereo_seconds=cd_stereo_seconds,
            cd_stereo_days=cd_stereo_seconds / 86_400.0,
        )

    def report(self) -> dict[str, Any]:
        """Return a machine-readable canonical structural report."""
        return {
            "standard": {
                "decimal_gb": self.decimal_gb,
                "decimal_tb": self.decimal_tb,
                "bytes": self.logical_bytes,
                "bits": self.logical_bits,
                "gibibytes_equivalent": self.gibibytes,
                "1000_gib_bytes": self.thousand_gib_bytes(),
                "1000_gib_bits": self.thousand_gib_bits(),
            },
            "geometry": {
                "axis_edge": self.axis_edge,
                "voxel_shape": [self.axis_edge] * 3,
                "bytes_per_voxel": 1,
                "logical_voxels": self.logical_voxels,
            },
            "hardware": asdict(self.hardware_breakdown()),
            "tensor": asdict(self.tensor_breakdown()),
            "tile_100": asdict(self.tile_cover(100)),
            "tile_64": asdict(self.tile_cover(64)),
            "multimedia": asdict(self.multimedia_breakdown()),
            "claims": {
                "dense_allocation_required": False,
                "geometry_implies_compression": False,
                "geometry_implies_throughput": False,
            },
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print the exact Dr Moagi 1000 GB byte-wise structural geometry."
    )
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args(argv)
    print(json.dumps(ThousandGBGeometry().report(), indent=args.indent, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
