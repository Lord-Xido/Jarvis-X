from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Iterator

from .geometry import Address3D, GeometricProgramLayout
from .isa import Instruction


@dataclass(slots=True)
class VoxelPage:
    """A sparse 3D ROM voxel containing bytecode and immutable metadata."""

    address: Address3D
    instructions: tuple[Instruction, ...] = ()
    parameters: bytes = b""
    neighbors: tuple[Address3D, ...] = ()
    lambda_mask: int = 0xFFFF
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def digest(self) -> str:
        h = hashlib.sha256()
        h.update(str(self.address).encode())
        for instruction in self.instructions:
            h.update(instruction.encode())
        h.update(self.parameters)
        h.update(json.dumps(self.metadata, sort_keys=True, default=str).encode())
        return h.hexdigest()


class Sparse3DROM:
    """Sparse virtual ROM. Unallocated coordinates consume no physical page."""

    def __init__(self, layout: GeometricProgramLayout | None = None) -> None:
        self.layout = layout or GeometricProgramLayout()
        self._pages: dict[Address3D, VoxelPage] = {}
        self._instruction_count = 0

    def map_page(self, page: VoxelPage, *, replace: bool = False) -> None:
        if page.address in self._pages and not replace:
            raise KeyError(f"ROM page already mapped at {page.address}")
        self._pages[page.address] = page

    def get_page(self, address: Address3D) -> VoxelPage | None:
        return self._pages.get(address)

    def fetch_instruction(self, address: Address3D, slot: int = 0) -> Instruction:
        page = self._pages.get(address)
        if page is None:
            raise KeyError(f"unmapped ROM address {address}")
        try:
            return page.instructions[slot]
        except IndexError as exc:
            raise KeyError(f"instruction slot {slot} missing at {address}") from exc

    def load_program(self, instructions: list[Instruction]) -> Address3D:
        self._pages.clear()
        self._instruction_count = len(instructions)
        for index, instruction in enumerate(instructions):
            address = self.layout.address_for_index(index)
            self.map_page(VoxelPage(address=address, instructions=(instruction,)))
        return self.layout.address_for_index(0)

    def instruction_address(self, index: int) -> Address3D:
        if not 0 <= index < self._instruction_count:
            raise IndexError(index)
        return self.layout.address_for_index(index)

    def __len__(self) -> int:
        return len(self._pages)

    def __iter__(self) -> Iterator[VoxelPage]:
        for address in sorted(self._pages):
            yield self._pages[address]

    @property
    def manifest_digest(self) -> str:
        h = hashlib.sha256()
        for page in self:
            h.update(page.digest.encode())
        return h.hexdigest()

    def stats(self) -> dict[str, int | str]:
        return {
            "mapped_pages": len(self._pages),
            "instructions": self._instruction_count,
            "virtual_extent": "sparse/unbounded",
            "manifest_sha256": self.manifest_digest,
        }
