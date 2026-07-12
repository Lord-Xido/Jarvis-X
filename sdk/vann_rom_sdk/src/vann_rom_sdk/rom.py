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
    sealed_digest: str | None = None

    @property
    def digest(self) -> str:
        h = hashlib.sha256()
        h.update(json.dumps({
            "address": {
                "x": self.address.x,
                "y": self.address.y,
                "z": self.address.z,
                "level": self.address.level,
                "space": self.address.space,
            },
            "neighbors": [
                {
                    "x": neighbor.x,
                    "y": neighbor.y,
                    "z": neighbor.z,
                    "level": neighbor.level,
                    "space": neighbor.space,
                }
                for neighbor in self.neighbors
            ],
            "lambda_mask": self.lambda_mask,
            "metadata": self.metadata,
        }, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8"))
        for instruction in self.instructions:
            h.update(instruction.encode())
        h.update(self.parameters)
        return h.hexdigest()

    def seal(self) -> None:
        self.sealed_digest = self.digest

    def verify_seal(self) -> None:
        if self.sealed_digest is None:
            raise RuntimeError(f"ROM page at {self.address} is not sealed")
        if self.digest != self.sealed_digest:
            raise RuntimeError(f"ROM page integrity failure at {self.address}")


class Sparse3DROM:
    """Sparse virtual ROM with explicit sealing and integrity verification."""

    def __init__(self, layout: GeometricProgramLayout | None = None) -> None:
        self.layout = layout or GeometricProgramLayout()
        self._pages: dict[Address3D, VoxelPage] = {}
        self._instruction_count = 0
        self._sealed = False
        self._sealed_manifest: str | None = None

    @property
    def sealed(self) -> bool:
        return self._sealed

    def map_page(self, page: VoxelPage, *, replace: bool = False) -> None:
        if self._sealed:
            raise RuntimeError("ROM is sealed; remapping requires loading a new image")
        if page.address in self._pages and not replace:
            raise KeyError(f"ROM page already mapped at {page.address}")
        self._pages[page.address] = page

    def get_page(self, address: Address3D) -> VoxelPage | None:
        return self._pages.get(address)

    def fetch_instruction(self, address: Address3D, slot: int = 0) -> Instruction:
        if not self._sealed:
            raise RuntimeError("cannot execute an unsealed ROM image")
        page = self._pages.get(address)
        if page is None:
            raise KeyError(f"unmapped ROM address {address}")
        page.verify_seal()
        try:
            return page.instructions[slot]
        except IndexError as exc:
            raise KeyError(f"instruction slot {slot} missing at {address}") from exc

    def load_program(self, instructions: list[Instruction]) -> Address3D:
        if not instructions:
            raise ValueError("program cannot be empty")
        self._sealed = False
        self._sealed_manifest = None
        self._pages.clear()
        self._instruction_count = len(instructions)
        for index, instruction in enumerate(instructions):
            address = self.layout.address_for_index(index)
            self.map_page(VoxelPage(
                address=address,
                instructions=(instruction,),
                metadata={"image_version": 1, "instruction_index": index},
            ))
        self.seal()
        return self.layout.address_for_index(0)

    def seal(self) -> None:
        for page in self._pages.values():
            page.seal()
        self._sealed = True
        self._sealed_manifest = self.manifest_digest

    def verify_manifest(self) -> None:
        if not self._sealed or self._sealed_manifest is None:
            raise RuntimeError("ROM image is not sealed")
        for page in self._pages.values():
            page.verify_seal()
        if self.manifest_digest != self._sealed_manifest:
            raise RuntimeError("ROM manifest integrity failure")

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
        h.update(b"VANN-ROM-IMAGE-v1")
        for page in self:
            digest = page.sealed_digest if page.sealed_digest is not None else page.digest
            h.update(digest.encode("ascii"))
        return h.hexdigest()

    def stats(self) -> dict[str, int | str | bool]:
        return {
            "mapped_pages": len(self._pages),
            "instructions": self._instruction_count,
            "virtual_extent": "sparse/unbounded",
            "sealed": self._sealed,
            "manifest_sha256": self.manifest_digest,
        }
