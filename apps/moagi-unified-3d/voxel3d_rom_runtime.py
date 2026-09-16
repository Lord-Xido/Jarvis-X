#!/usr/bin/env python3
"""Reference parser/runtime bridge for Jarvis X VOXEL3D ROM images.

This module intentionally separates *observed binary structure* from *assigned
instruction semantics*. It validates and exposes the ROM image, but does not
pretend undocumented VM opcodes are executable host instructions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence
import argparse
import json
import re
import struct

MAGIC = b"\x7fVOXEL3D"
KNOWN_ANCHORS: tuple[str, ...] = (
    "GLSL_SVO_ENGINE",
    "SVO_COMPUTE_SHADER_KERNEL_VECTOR",
    "RAYMARCHER_INIT",
    "AUTO_EXEC_LOOP_0",
    "ENGINE_STATE_RUN",
    "VOXEL_COUNT_1MB3",
    "STACK_TRACE_ZERO",
    "VOXEL_ENGINE_EOF",
)

PHASES: tuple[str, ...] = (
    "ROM_BYTE_IMAGE",
    "DECODE",
    "SPATIAL_VM",
    "GLSL_SVO_ENGINE",
    "SVO_COMPUTE_ENCODER",
    "LATENT_CORE_Z_T",
    "DECODER",
    "RECONSTRUCTION_X_HAT_T",
    "RESIDUAL_CONTRAST",
    "CTR_VERIFY_CORRECT",
    "SCHEDULER_PI_T",
    "ENGINE_STATE_RUN",
    "RAYMARCHER_INIT",
    "FRAME_OUTPUT",
)

RECURSIVE_INVARIANT: tuple[str, ...] = (
    "GENERATE",
    "CONTRAST",
    "RECKON",
    "VERIFY",
    "CORRECT",
    "RECUR",
)


class RomFormatError(ValueError):
    """Raised when a byte sequence is not a valid VOXEL3D image."""


@dataclass(frozen=True)
class Anchor:
    name: str
    offset: int


@dataclass(frozen=True)
class PrintableString:
    offset: int
    text: str


@dataclass(frozen=True)
class RawWord:
    offset: int
    value: int

    @property
    def hex(self) -> str:
        return f"0x{self.value:08X}"


@dataclass(frozen=True)
class RuntimeTrace:
    cycle: int
    phases: tuple[str, ...]
    recursive_invariant: tuple[str, ...]
    verified: bool
    correction_committed: bool


class Voxel3DRomImage:
    """Immutable view over a VOXEL3D ROM byte image."""

    def __init__(self, data: bytes):
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("data must be bytes-like")
        raw = bytes(data)
        if not raw.startswith(MAGIC):
            prefix = raw[: len(MAGIC)].hex().upper()
            raise RomFormatError(
                f"invalid VOXEL3D magic: expected {MAGIC.hex().upper()}, got {prefix}"
            )
        self._data = raw

    @classmethod
    def from_hex(cls, text: str) -> "Voxel3DRomImage":
        cleaned = re.sub(r"[^0-9A-Fa-f]", "", text)
        if len(cleaned) % 2:
            raise RomFormatError("hex stream has an odd number of nibbles")
        try:
            return cls(bytes.fromhex(cleaned))
        except ValueError as exc:
            raise RomFormatError(str(exc)) from exc

    @classmethod
    def from_file(cls, path: str | Path) -> "Voxel3DRomImage":
        p = Path(path)
        if p.suffix.lower() in {".hex", ".txt"}:
            return cls.from_hex(p.read_text(encoding="utf-8"))
        return cls(p.read_bytes())

    @property
    def data(self) -> bytes:
        return self._data

    @property
    def size(self) -> int:
        return len(self._data)

    def anchors(self) -> tuple[Anchor, ...]:
        found: list[Anchor] = []
        for name in KNOWN_ANCHORS:
            start = 0
            needle = name.encode("ascii")
            while True:
                offset = self._data.find(needle, start)
                if offset < 0:
                    break
                found.append(Anchor(name=name, offset=offset))
                start = offset + 1
        return tuple(sorted(found, key=lambda item: item.offset))

    def printable_strings(self, minimum_length: int = 6) -> tuple[PrintableString, ...]:
        if minimum_length < 1:
            raise ValueError("minimum_length must be >= 1")
        pattern = re.compile(rb"[\x20-\x7E]{%d,}" % minimum_length)
        return tuple(
            PrintableString(offset=m.start(), text=m.group().decode("ascii"))
            for m in pattern.finditer(self._data)
        )

    def find_anchor(self, name: str) -> Anchor | None:
        needle = name.encode("ascii")
        offset = self._data.find(needle)
        return None if offset < 0 else Anchor(name=name, offset=offset)

    def raw_words(
        self,
        start: int = 0,
        end: int | None = None,
        *,
        word_size: int = 4,
        byteorder: str = "little",
    ) -> Iterator[RawWord]:
        if word_size not in (1, 2, 4, 8):
            raise ValueError("word_size must be one of 1, 2, 4, 8")
        if byteorder not in ("little", "big"):
            raise ValueError("byteorder must be 'little' or 'big'")
        stop = self.size if end is None else min(end, self.size)
        if start < 0 or stop < start:
            raise ValueError("invalid byte range")
        for offset in range(start, stop - word_size + 1, word_size):
            value = int.from_bytes(self._data[offset : offset + word_size], byteorder)
            yield RawWord(offset=offset, value=value)

    def words_after_anchor(
        self,
        name: str,
        *,
        count: int = 32,
        skip_nul_padding: bool = True,
    ) -> tuple[RawWord, ...]:
        anchor = self.find_anchor(name)
        if anchor is None:
            return ()
        start = anchor.offset + len(name)
        if skip_nul_padding:
            while start < self.size and self._data[start] == 0:
                start += 1
        words = self.raw_words(start=start, word_size=4, byteorder="little")
        result: list[RawWord] = []
        for word in words:
            result.append(word)
            if len(result) >= count:
                break
        return tuple(result)

    def telemetry(self, opcode_preview_count: int = 24) -> dict:
        anchors = self.anchors()
        candidate_words = self.words_after_anchor(
            "AUTO_EXEC_LOOP_0", count=opcode_preview_count
        )
        return {
            "format": "VOXEL3D",
            "magic": MAGIC.hex().upper(),
            "size_bytes": self.size,
            "anchors": [asdict(item) for item in anchors],
            "ascii_strings": [asdict(item) for item in self.printable_strings()],
            "auto_exec_raw_words": [
                {"offset": word.offset, "value": word.value, "hex": word.hex}
                for word in candidate_words
            ],
            "opcode_semantics": "UNSPECIFIED",
            "native_executable": False,
        }


class JarvisXVoxelRuntime:
    """Reference orchestration shell for the VOXEL3D -> unified-3D boundary."""

    def __init__(self, rom: Voxel3DRomImage):
        self.rom = rom
        self.cycle = 0

    def step(self, *, residual_verified: bool) -> RuntimeTrace:
        """Advance one reference orchestration cycle.

        `residual_verified` represents the result of the external CTR/verification
        layer. Persistent correction is committed only after verification.
        """
        self.cycle += 1
        return RuntimeTrace(
            cycle=self.cycle,
            phases=PHASES,
            recursive_invariant=RECURSIVE_INVARIANT,
            verified=bool(residual_verified),
            correction_committed=bool(residual_verified),
        )

    def describe(self) -> dict:
        return {
            "rom": self.rom.telemetry(),
            "state_model": ["X_t", "Z_t", "Omega_t", "X_hat_t", "R_t", "Pi_t"],
            "phases": list(PHASES),
            "recursive_invariant": list(RECURSIVE_INVARIANT),
            "cycle": self.cycle,
        }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect a Jarvis X VOXEL3D ROM image")
    parser.add_argument("image", help="Path to binary ROM image or .hex/.txt hex stream")
    parser.add_argument(
        "--step",
        action="store_true",
        help="emit one reference runtime cycle in addition to ROM telemetry",
    )
    parser.add_argument(
        "--verified",
        action="store_true",
        help="mark the reference cycle as externally CTR-verified",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    rom = Voxel3DRomImage.from_file(args.image)
    runtime = JarvisXVoxelRuntime(rom)
    payload = runtime.describe()
    if args.step:
        payload["trace"] = asdict(runtime.step(residual_verified=args.verified))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
