"""Sparse 3D virtual computer, adaptive layout optimizer, and ROM format."""

import hashlib
import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Tuple

from .codec import EncodedBlock, SparseBlock, ZlibSparseBlockCodec
from .geometry import Coordinate3D, VolumeGeometry


class OperationalMode(str, Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    ADAPTIVE = "adaptive"


@dataclass(frozen=True)
class VolumeStatistics:
    logical_capacity_bytes: int
    maximum_blocks: int
    allocated_blocks: int
    allocated_cells: int
    physical_payload_bytes: int
    encoded_bytes: int
    reads: int
    writes: int


@dataclass(frozen=True)
class OptimizationCandidate:
    block_shape: Coordinate3D
    allocated_blocks: int
    encoded_bytes: int
    layout_bytes: int
    objective: float


@dataclass(frozen=True)
class OptimizationReport:
    previous_block_shape: Coordinate3D
    selected: OptimizationCandidate
    candidates: Tuple[OptimizationCandidate, ...]


class Virtual3DComputer:
    """Sparse logical 3D address space with verified block persistence.

    The implementation is sparse at two levels: blocks are allocated lazily,
    and cells inside a block are allocated lazily. It therefore never creates a
    dense ``B^3`` array merely because the logical block shape is large.
    """

    schema_version = "jarvisx.virtual3d.rom.v1"

    def __init__(
        self,
        geometry: Optional[VolumeGeometry] = None,
        mode: OperationalMode = OperationalMode.ONLINE,
    ) -> None:
        self.geometry = geometry or VolumeGeometry()
        self.mode = OperationalMode(mode)
        self.blocks: Dict[Coordinate3D, SparseBlock] = {}
        self.encoded_blocks: Dict[Coordinate3D, EncodedBlock] = {}
        self.reads = 0
        self.writes = 0

    def _allocate_block(self, index: Coordinate3D) -> SparseBlock:
        block = SparseBlock(
            index=index,
            shape=self.geometry.effective_block_shape(index),
            cell_bytes=self.geometry.cell_bytes,
        )
        self.blocks[index] = block
        return block

    def _materialize_block(self, index: Coordinate3D) -> Optional[SparseBlock]:
        block = self.blocks.get(index)
        if block is not None:
            return block
        encoded = self.encoded_blocks.get(index)
        if encoded is None:
            return None
        block = ZlibSparseBlockCodec.decode(encoded)
        self.blocks[index] = block
        return block

    def read(self, coordinate: Coordinate3D) -> bytes:
        address = self.geometry.map_coordinate(coordinate)
        self.reads += 1
        block = self._materialize_block(address.block)
        if block is None:
            return b""
        return block.read(address.offset)

    def write(self, coordinate: Coordinate3D, value: bytes) -> None:
        address = self.geometry.map_coordinate(coordinate)
        self.writes += 1
        block = self._materialize_block(address.block)
        if block is None:
            if not value:
                return
            block = self._allocate_block(address.block)
        block.write(address.offset, value)
        self.encoded_blocks.pop(address.block, None)
        if block.is_empty:
            self.blocks.pop(address.block, None)

    def iter_allocated_cells(self) -> Iterator[Tuple[Coordinate3D, bytes]]:
        indexes = sorted(set(self.blocks) | set(self.encoded_blocks))
        for index in indexes:
            block = self._materialize_block(index)
            if block is None:
                continue
            for offset, value in block.iter_cells():
                yield self.geometry.global_coordinate(index, offset), value

    def compact_block(self, index: Coordinate3D) -> Optional[EncodedBlock]:
        block = self._materialize_block(index)
        if block is None:
            return None
        if block.is_empty:
            self.blocks.pop(index, None)
            self.encoded_blocks.pop(index, None)
            return None
        encoded = ZlibSparseBlockCodec.optimize(block)
        self.encoded_blocks[index] = encoded
        if self.mode in (OperationalMode.OFFLINE, OperationalMode.ADAPTIVE):
            self.blocks.pop(index, None)
        return encoded

    def compact_all(self) -> Tuple[EncodedBlock, ...]:
        indexes = sorted(set(self.blocks) | set(self.encoded_blocks))
        encoded = []
        for index in indexes:
            result = self.compact_block(index)
            if result is not None:
                encoded.append(result)
        return tuple(encoded)

    def reblocked(self, block_shape: Coordinate3D) -> "Virtual3DComputer":
        geometry = VolumeGeometry(
            extent=self.geometry.extent,
            block_shape=block_shape,
            cell_bytes=self.geometry.cell_bytes,
        )
        rebuilt = Virtual3DComputer(geometry=geometry, mode=self.mode)
        for coordinate, value in self.iter_allocated_cells():
            rebuilt.write(coordinate, value)
        rebuilt.reads = self.reads
        rebuilt.writes = self.writes
        return rebuilt

    def _candidate_score(
        self, block_shape: Coordinate3D, layout_entry_bytes: int
    ) -> Tuple[OptimizationCandidate, "Virtual3DComputer"]:
        candidate = self.reblocked(block_shape)
        encoded = candidate.compact_all()
        encoded_bytes = sum(item.compressed_bytes for item in encoded)
        layout_bytes = len(encoded) * layout_entry_bytes
        score = float(encoded_bytes + layout_bytes)
        return (
            OptimizationCandidate(
                block_shape=block_shape,
                allocated_blocks=len(encoded),
                encoded_bytes=encoded_bytes,
                layout_bytes=layout_bytes,
                objective=score,
            ),
            candidate,
        )

    def optimize_layout(
        self,
        candidate_block_shapes: Iterable[Coordinate3D],
        layout_entry_bytes: int = 64,
    ) -> OptimizationReport:
        if layout_entry_bytes < 0:
            raise ValueError("layout_entry_bytes must be non-negative")
        shapes = [self.geometry.block_shape]
        for shape in candidate_block_shapes:
            if shape not in shapes:
                shapes.append(shape)
        evaluations = [
            self._candidate_score(shape, layout_entry_bytes) for shape in shapes
        ]
        selected_score, selected_volume = min(
            evaluations,
            key=lambda item: (
                item[0].objective,
                item[0].allocated_blocks,
                item[0].block_shape,
            ),
        )
        previous = self.geometry.block_shape
        self.geometry = selected_volume.geometry
        self.blocks = selected_volume.blocks
        self.encoded_blocks = selected_volume.encoded_blocks
        return OptimizationReport(
            previous_block_shape=previous,
            selected=selected_score,
            candidates=tuple(item[0] for item in evaluations),
        )

    def statistics(self) -> VolumeStatistics:
        materialized = list(self.blocks.values())
        encoded_only = set(self.encoded_blocks) - set(self.blocks)
        allocated_cells = sum(block.allocated_cells for block in materialized)
        physical_payload = sum(
            block.physical_payload_bytes for block in materialized
        )
        for index in encoded_only:
            block = ZlibSparseBlockCodec.decode(self.encoded_blocks[index])
            allocated_cells += block.allocated_cells
            physical_payload += block.physical_payload_bytes
        return VolumeStatistics(
            logical_capacity_bytes=self.geometry.logical_capacity_bytes,
            maximum_blocks=self.geometry.maximum_block_count,
            allocated_blocks=len(set(self.blocks) | set(self.encoded_blocks)),
            allocated_cells=allocated_cells,
            physical_payload_bytes=physical_payload,
            encoded_bytes=sum(
                item.compressed_bytes for item in self.encoded_blocks.values()
            ),
            reads=self.reads,
            writes=self.writes,
        )

    def _rom_document(self) -> Dict[str, object]:
        encoded = self.compact_all()
        return {
            "blocks": [
                item.to_document()
                for item in sorted(encoded, key=lambda block: block.index)
            ],
            "geometry": {
                "block_shape": list(self.geometry.block_shape),
                "cell_bytes": self.geometry.cell_bytes,
                "extent": list(self.geometry.extent),
            },
            "mode": self.mode.value,
            "reads": self.reads,
            "schema": self.schema_version,
            "writes": self.writes,
        }

    def to_rom_bytes(self) -> bytes:
        document = self._rom_document()
        canonical = json.dumps(
            document, sort_keys=True, separators=(",", ":")
        )
        envelope = {
            "fingerprint": hashlib.sha256(
                canonical.encode("utf-8")
            ).hexdigest(),
            "state": document,
        }
        return json.dumps(
            envelope, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")

    @classmethod
    def from_rom_bytes(cls, payload: bytes) -> "Virtual3DComputer":
        envelope = json.loads(payload.decode("utf-8"))
        document = envelope["state"]
        canonical = json.dumps(
            document, sort_keys=True, separators=(",", ":")
        )
        fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if fingerprint != envelope["fingerprint"]:
            raise ValueError("ROM fingerprint mismatch")
        if document["schema"] != cls.schema_version:
            raise ValueError(
                "unsupported ROM schema: {}".format(document["schema"])
            )
        geometry_document = document["geometry"]
        geometry = VolumeGeometry(
            extent=tuple(geometry_document["extent"]),
            block_shape=tuple(geometry_document["block_shape"]),
            cell_bytes=int(geometry_document["cell_bytes"]),
        )
        computer = cls(
            geometry=geometry,
            mode=OperationalMode(document["mode"]),
        )
        computer.reads = int(document["reads"])
        computer.writes = int(document["writes"])
        for item in document["blocks"]:
            encoded = EncodedBlock.from_document(item)
            computer.encoded_blocks[encoded.index] = encoded
        return computer

    def save_rom(self, path: str) -> str:
        payload = self.to_rom_bytes()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_bytes(payload)
        os.replace(str(temporary), str(destination))
        envelope = json.loads(payload.decode("utf-8"))
        return str(envelope["fingerprint"])

    @classmethod
    def load_rom(cls, path: str) -> "Virtual3DComputer":
        return cls.from_rom_bytes(Path(path).read_bytes())
