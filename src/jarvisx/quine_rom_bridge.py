"""Bounded executable bridge for the React Artifact (9) quine-ROM opcode profile.

This is a profile adapter, not a replacement for the canonical Jarvis-X VM.
Virtual geometry: 2**30 BIT coordinates along each of three axes (2**90
logical bit locations). Only payload chunks, 8**3 latents and XOR residuals
are materialized. Exact recovery depends on retaining the residual.

Q_ENC / Q_DEC use the established jarvisx.dm3d_rom 64**3 -> 8**3 codec.
Q_FOLD describes sparse Morton-localized chunk addressing. Q_OPT selects a
smaller lossless transport (zlib levels) without altering authoritative data.
Q_JMP_SELF is a *bounded*, verified return to the first instruction; it does
not self-modify code, prove universal convergence, or run forever.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from jarvisx.dm3d_rom import (
    LATENT_BYTES,
    TILE_BYTES,
    TILE_EDGE,
    decode_tile,
    encode_tile,
)

AXIS_BITS = 1 << 30
VIRTUAL_BITS = AXIS_BITS**3
MAX_INPUT_BYTES = 32 * 1024 * 1024
MAX_CYCLES = 32
FORMAT = "JX-QROM-BRIDGE-1"
MAGIC = b"JXQRB1\x00\x00"
HEADER = struct.Struct("<8sQ")

Q_NOP = 0x00
Q_OPT = 0x0E
Q_DEC = 0xD1
Q_JMP_SELF = 0xCC
Q_ENC = 0xE1
Q_FOLD = 0xF0
ISA_NAMES = {
    Q_NOP: "Q_NOP", Q_OPT: "Q_OPT", Q_DEC: "Q_DEC",
    Q_JMP_SELF: "Q_JMP_SELF", Q_ENC: "Q_ENC", Q_FOLD: "Q_FOLD",
}
DEFAULT_PROGRAM = (Q_ENC, Q_FOLD, Q_DEC, Q_OPT, Q_JMP_SELF, Q_NOP)


def morton_decode_90(address: int) -> tuple[int, int, int]:
    """Bijection from a 90-bit linear bit index into three 30-bit axes."""
    if not 0 <= address < VIRTUAL_BITS:
        raise ValueError("bit address outside 2**90-bit virtual cube")
    coords = [0, 0, 0]
    for i in range(30):
        for axis in range(3):
            coords[axis] |= ((address >> (3 * i + axis)) & 1) << i
    return tuple(coords)  # type: ignore[return-value]


def morton_encode_90(x: int, y: int, z: int) -> int:
    if any(not 0 <= v < AXIS_BITS for v in (x, y, z)):
        raise ValueError("coordinate outside 2**30-bit virtual axis")
    out = 0
    for i in range(30):
        out |= ((x >> i) & 1) << (3 * i)
        out |= ((y >> i) & 1) << (3 * i + 1)
        out |= ((z >> i) & 1) << (3 * i + 2)
    return out


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _decompress_bounded(blob: bytes, expected_size: int) -> bytes:
    """Reject corrupt or expansion-bomb data before materialization."""
    if expected_size < 0 or expected_size > TILE_BYTES:
        raise ValueError("invalid decompressed block length")
    try:
        stream = zlib.decompressobj()
        output = stream.decompress(blob, expected_size + 1)
        if stream.unconsumed_tail or not stream.eof or stream.unused_data:
            raise ValueError("invalid compressed payload")
        output += stream.flush()
    except zlib.error as exc:
        raise ValueError("corrupt compressed payload") from exc
    if len(output) != expected_size:
        raise ValueError("compressed payload length mismatch")
    return output


@dataclass
class Frame:
    index: int
    real_bytes: int
    latent_z: bytes
    residual_z: bytes
    source_hash: str
    level: int

    def decode(self) -> bytes:
        latent = _decompress_bounded(self.latent_z, LATENT_BYTES)
        residual = _decompress_bounded(self.residual_z, TILE_BYTES)
        prediction = decode_tile(bytearray(latent))
        reconstructed = bytes(p ^ r for p, r in zip(prediction, residual))
        raw = reconstructed[:self.real_bytes]
        if len(raw) != self.real_bytes or sha256(raw) != self.source_hash:
            raise ValueError("frame integrity violation")
        return raw

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "real_bytes": self.real_bytes,
            "latent_z": base64.b64encode(self.latent_z).decode("ascii"),
            "residual_z": base64.b64encode(self.residual_z).decode("ascii"),
            "source_hash": self.source_hash,
            "level": self.level,
        }

    @classmethod
    def from_dict(cls, x: dict[str, Any]) -> "Frame":
        index, length, level = int(x["index"]), int(x["real_bytes"]), int(x["level"])
        if index < 0 or not 0 < length <= TILE_BYTES or not 1 <= level <= 9:
            raise ValueError("invalid frame metadata")
        return cls(
            index, length,
            base64.b64decode(x["latent_z"], validate=True),
            base64.b64decode(x["residual_z"], validate=True),
            str(x["source_hash"]), level,
        )


def _encode_frame(index: int, raw: bytes, step: int = 8, level: int = 6) -> Frame:
    if not raw or len(raw) > TILE_BYTES:
        raise ValueError("invalid source chunk")
    padded = bytearray(raw)
    padded.extend(b"\x00" * (TILE_BYTES - len(padded)))
    latent = encode_tile(padded, quant_step=step)
    prediction = decode_tile(latent)
    residual = bytes(a ^ b for a, b in zip(padded, prediction))
    return Frame(index, len(raw), zlib.compress(latent, level),
                 zlib.compress(residual, level), sha256(raw), level)


def _frames_to_bytes(frames: Sequence[Frame], total_length: int, digest: str) -> bytes:
    if total_length == 0 and frames:
        raise ValueError("nonempty frames for empty source")
    if len(frames) != (total_length + TILE_BYTES - 1) // TILE_BYTES:
        raise ValueError("frame count mismatch")
    out = bytearray()
    for i, frame in enumerate(frames):
        if frame.index != i:
            raise ValueError("noncontiguous frame order")
        expected = min(TILE_BYTES, total_length - len(out))
        if frame.real_bytes != expected:
            raise ValueError("frame length mismatch")
        out.extend(frame.decode())
    if len(out) != total_length or sha256(out) != digest:
        raise ValueError("payload SHA-256 verification failed")
    return bytes(out)


@dataclass
class QuineROMBridge:
    """Deterministic, bounded opcode executor: source -> frames -> same source."""

    source: bytes
    frames: list[Frame] = field(default_factory=list)
    folded: list[dict[str, Any]] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)
    reconstructed: bytes | None = None
    cycles: int = 0

    def __post_init__(self) -> None:
        if len(self.source) > MAX_INPUT_BYTES:
            raise ValueError("input exceeds bounded profile maximum")
        self.source_sha256 = sha256(self.source)

    def run(self, *, program: Sequence[int] = DEFAULT_PROGRAM, cycles: int = 2) -> dict[str, Any]:
        if not 1 <= cycles <= MAX_CYCLES:
            raise ValueError("cycles out of allowed bounded range")
        if not program or any(op not in ISA_NAMES for op in program):
            raise ValueError("unknown opcode or empty program")
        if program.count(Q_JMP_SELF) != 1 or tuple(program[-2:]) != (Q_JMP_SELF, Q_NOP):
            raise ValueError("program must close with Q_JMP_SELF, Q_NOP")
        if tuple(program[:3]) != (Q_ENC, Q_FOLD, Q_DEC):
            raise ValueError("program must begin Q_ENC, Q_FOLD, Q_DEC")
        self.trace.clear()
        self.cycles = 0
        for cycle in range(cycles):
            for pc, op in enumerate(program):
                if op == Q_ENC:
                    self.frames = [
                        _encode_frame(i // TILE_BYTES, self.source[i:i + TILE_BYTES])
                        for i in range(0, len(self.source), TILE_BYTES)
                    ]
                elif op == Q_FOLD:
                    if self.frames is None:
                        raise RuntimeError("Q_FOLD before Q_ENC")
                    self.folded = [
                        {
                            "index": frame.index,
                            "bit_start": frame.index * TILE_BYTES * 8,
                            "morton_xyz": morton_decode_90(frame.index * TILE_BYTES * 8),
                            "latent_shape": [8, 8, 8],
                            "source_shape": [64, 64, 64],
                        }
                        for frame in self.frames
                    ]
                elif op == Q_DEC:
                    self.reconstructed = _frames_to_bytes(
                        self.frames, len(self.source), self.source_sha256
                    )
                elif op == Q_OPT:
                    optimized = []
                    for frame in self.frames:
                        latent = _decompress_bounded(frame.latent_z, LATENT_BYTES)
                        resid = _decompress_bounded(frame.residual_z, TILE_BYTES)
                        choices = [
                            Frame(frame.index, frame.real_bytes, zlib.compress(latent, level),
                                  zlib.compress(resid, level), frame.source_hash, level)
                            for level in (1, 6, 9)
                        ]
                        candidate = min(choices, key=lambda f: len(f.latent_z) + len(f.residual_z))
                        if candidate.decode() != frame.decode():
                            raise ValueError("optimization violated byte-exact gate")
                        optimized.append(candidate)
                    self.frames = optimized
                elif op == Q_JMP_SELF:
                    if self.reconstructed != self.source:
                        raise RuntimeError("Q_JMP_SELF requires verified fixed-point identity")
                elif op == Q_NOP:
                    pass
                self.trace.append({"cycle": cycle, "pc": pc, "opcode": ISA_NAMES[op]})
            self.cycles += 1
        verified = _frames_to_bytes(self.frames, len(self.source), self.source_sha256)
        if verified != self.source:
            raise RuntimeError("final verification failed")
        return {
            "profile": FORMAT,
            "source_sha256": self.source_sha256,
            "roundtrip_sha256": sha256(verified),
            "bit_exact": True,
            "virtual_axis_bits": AXIS_BITS,
            "virtual_bit_addresses": str(VIRTUAL_BITS),
            "materialized_chunks": len(self.frames),
            "stored_compressed_bytes": sum(len(f.latent_z) + len(f.residual_z) for f in self.frames),
            "cycles_completed": self.cycles,
            "instructions_executed": len(self.trace),
            "fixed_point_scope": "exact round-trip of supplied finite payload; not learned convergence",
        }

    def export(self, path: Path) -> None:
        if self.reconstructed != self.source:
            raise RuntimeError("verify before export")
        manifest = {
            "format": FORMAT, "source_sha256": self.source_sha256,
            "source_length": len(self.source), "axis_bits": AXIS_BITS,
            "frames": [f.to_dict() for f in self.frames],
        }
        raw = json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode("utf-8")
        path.write_bytes(HEADER.pack(MAGIC, len(raw)) + raw)


def restore(path: Path) -> bytes:
    payload = path.read_bytes()
    if len(payload) < HEADER.size or len(payload) > MAX_INPUT_BYTES * 2 + 1024 * 1024:
        raise ValueError("archive size out of bounds")
    magic, json_length = HEADER.unpack_from(payload)
    if magic != MAGIC or json_length != len(payload) - HEADER.size:
        raise ValueError("invalid QROM archive header")
    doc = json.loads(payload[HEADER.size:])
    if doc["format"] != FORMAT or doc["axis_bits"] != AXIS_BITS:
        raise ValueError("incompatible QROM profile")
    length = doc["source_length"]
    if not isinstance(length, int) or not 0 <= length <= MAX_INPUT_BYTES:
        raise ValueError("invalid declared source length")
    frames = [Frame.from_dict(x) for x in doc["frames"]]
    return _frames_to_bytes(frames, length, str(doc["source_sha256"]))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Jarvis-X bounded quine ROM bridge")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("integrate", help="encode a finite source, verify, save archive")
    build.add_argument("source", type=Path)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--cycles", type=int, default=2)
    verify = sub.add_parser("restore", help="restore and verify byte-exact source")
    verify.add_argument("archive", type=Path)
    verify.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "integrate":
        bridge = QuineROMBridge(args.source.read_bytes())
        report = bridge.run(cycles=args.cycles)
        bridge.export(args.output)
        print(json.dumps(report, indent=2))
    else:
        result = restore(args.archive)
        args.output.write_bytes(result)
        print(json.dumps({"restored_bytes": len(result), "sha256": sha256(result),
                          "bit_exact": True}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
