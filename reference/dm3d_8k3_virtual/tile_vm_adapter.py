#!/usr/bin/env python3
"""Bounded JX8KT1 -> typed float32 tile tensor reference bytecode adapter.

This is a separate experimental profile sharing the legacy DM3D *instruction word
layout*, not its opcode table. It never invokes or mutates production DM3D VM state.
Independent raw reference comparison is required for all tiles before execution.
"""
from __future__ import annotations

import argparse
from array import array
from dataclasses import dataclass
from enum import IntEnum
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import struct
import sys
from typing import Sequence

from verify_packet import MAGIC, RAW_TILE_BYTES, decode, morton3, verify

WORLD_SIDE = 8192
TILE_SIDE = 16
TILE_VALUES = 4096
MAX_TILES = 1024
MAX_PACKET_BYTES = 2 * 1024 * 1024
MAX_PROGRAM_WORDS = 16


class TileOp(IntEnum):
    LOAD_TILE = 0x70
    DECODE_2BIT = 0x71
    MAP_FLOAT32 = 0x72
    REPACK_2BIT = 0x73
    VERIFY_CTR = 0x74
    EMIT_PROVISIONAL = 0x75
    HALT = 0xFF


DEFAULT_PROGRAM = (
    TileOp.LOAD_TILE,
    TileOp.DECODE_2BIT,
    TileOp.MAP_FLOAT32,
    TileOp.REPACK_2BIT,
    TileOp.VERIFY_CTR,
    TileOp.EMIT_PROVISIONAL,
    TileOp.HALT,
)


def pack_word(op: TileOp, dst: int = 0, a: int = 0, b: int = 0, imm: int = 0) -> int:
    if not all(0 <= v <= 255 for v in (dst, a, b)) or not 0 <= imm < (1 << 32):
        raise ValueError("invalid VM instruction fields")
    return (int(op) << 56) | (dst << 48) | (a << 40) | (b << 32) | imm


def parse_program(data: bytes) -> tuple[TileOp, ...]:
    if not data or len(data) % 8 or len(data) // 8 > MAX_PROGRAM_WORDS:
        raise ValueError("bytecode must contain 1..16 complete 64-bit words")
    words = struct.unpack(">" + "Q" * (len(data) // 8), data)
    ops = []
    for word in words:
        if word & ((1 << 56) - 1):
            raise ValueError("v1 instruction requires zero operands and reserved fields")
        try:
            ops.append(TileOp(word >> 56))
        except ValueError as exc:
            raise ValueError("unknown/unsupported JX8K VM opcode") from exc
    if tuple(ops) != DEFAULT_PROGRAM:
        raise ValueError("v1 VM enforces exact bounded verify-before-emit program")
    return tuple(ops)


def default_bytecode() -> bytes:
    return b"".join(struct.pack(">Q", pack_word(op)) for op in DEFAULT_PROGRAM)


@dataclass(frozen=True)
class TileAddress:
    tile_x: int
    tile_y: int
    tile_z: int

    @property
    def global_origin(self) -> tuple[int, int, int]:
        return (self.tile_x * TILE_SIDE, self.tile_y * TILE_SIDE, self.tile_z * TILE_SIDE)

    @property
    def morton39(self) -> int:
        return morton3(*self.global_origin)


@dataclass(frozen=True)
class TileRecord:
    address: TileAddress
    encoded: bytes
    reference: bytes


def parse_records(packet: bytes, raw: bytes, observations: Sequence[dict]) -> list[TileRecord]:
    if len(packet) > MAX_PACKET_BYTES or len(observations) > MAX_TILES:
        raise ValueError("physical budget exceeded")
    if not packet.startswith(MAGIC):
        raise ValueError("wrong packet profile")
    if len(raw) != len(observations) * RAW_TILE_BYTES:
        raise ValueError("raw tile reference length mismatch")
    cursor = len(MAGIC)
    records = []
    seen = set()
    for idx, obs in enumerate(observations):
        if cursor + 8 > len(packet):
            raise ValueError("truncated tile metadata")
        x, y, z, size = struct.unpack_from("<HHHH", packet, cursor)
        cursor += 8
        if size < 1 or cursor + size > len(packet):
            raise ValueError("invalid packet payload length")
        address = TileAddress(x, y, z)
        if not all(0 <= c < WORLD_SIDE // TILE_SIDE for c in (x, y, z)):
            raise ValueError("out-of-range tile coordinate")
        if address in seen:
            raise ValueError("duplicate tile coordinate")
        seen.add(address)
        if (x, y, z) != (obs["x"], obs["y"], obs["z"]):
            raise ValueError("metadata/address mismatch")
        content = packet[cursor:cursor + size]
        cursor += size
        records.append(TileRecord(address, content, raw[idx*RAW_TILE_BYTES:(idx+1)*RAW_TILE_BYTES]))
    if cursor != len(packet):
        raise ValueError("trailing binary records")
    return records


def unpack_tensor(raw: bytes) -> array:
    if len(raw) != RAW_TILE_BYTES:
        raise ValueError("invalid tile width")
    # Values 0,1,2,3 are exactly representable in IEEE-754 float32.
    return array("f", ((raw[i >> 2] >> (2*(i & 3))) & 3 for i in range(TILE_VALUES)))


def pack_tensor(tensor: array) -> bytes:
    if tensor.typecode != "f" or len(tensor) != TILE_VALUES:
        raise ValueError("tensor must contain exactly 4096 float32 elements")
    out = bytearray(RAW_TILE_BYTES)
    for i, v in enumerate(tensor):
        if not math.isfinite(v) or v not in (0.0, 1.0, 2.0, 3.0):
            raise ValueError("non-discrete tensor value")
        out[i >> 2] |= int(v) << (2 * (i & 3))
    return bytes(out)


def tensor_telemetry(tensor: array) -> dict:
    histogram = [0] * 4
    sx = sy = sz = count = 0
    for idx, value in enumerate(tensor):
        v = int(value)
        histogram[v] += 1
        if v:
            x = idx % TILE_SIDE
            y = (idx // TILE_SIDE) % TILE_SIDE
            z = idx // (TILE_SIDE*TILE_SIDE)
            sx += x
            sy += y
            sz += z
            count += 1
    return {
        "counts": histogram,
        "active_voxels": count,
        "local_centroid": [round(c / count, 6) for c in (sx, sy, sz)] if count else None,
    }


class BoundedTileVM:
    def __init__(self, program: bytes):
        self.program = parse_program(program)
        self.trace = []
        self.steps = 0

    def run(self, report: dict, packet: bytes, raw: bytes) -> dict:
        # Fail closed BEFORE touching any VM working state. This comparison is to
        # the supplied independent raw reference, not external-world correctness.
        if len(packet) > MAX_PACKET_BYTES or len(raw) > MAX_TILES*RAW_TILE_BYTES:
            raise ValueError("packet exceeds VM resource limits")
        receipt = verify(report, packet, raw)
        if receipt["status"] != "PASS" or report["tiles_verified"] > MAX_TILES:
            raise ValueError("unverified binary substrate")
        records = parse_records(packet, raw, report["observations"])
        if not records:
            raise ValueError("empty execution batch")
        self.trace = []
        self.steps = 0
        for rec in records:
            current = None
            tile_bytes = None
            tensor = None
            repacked = None
            evidence = None
            for op in self.program[:-1]:
                self.steps += 1
                if self.steps > MAX_TILES*(len(self.program)-1):
                    raise ValueError("execution budget exceeded")
                if op == TileOp.LOAD_TILE:
                    current = rec.address
                elif op == TileOp.DECODE_2BIT:
                    if current is None:
                        raise ValueError("LOAD_TILE required")
                    tile_bytes = decode(rec.encoded)
                elif op == TileOp.MAP_FLOAT32:
                    if tile_bytes is None:
                        raise ValueError("DECODE_2BIT required")
                    tensor = unpack_tensor(tile_bytes)
                elif op == TileOp.REPACK_2BIT:
                    if tensor is None:
                        raise ValueError("MAP_FLOAT32 required")
                    repacked = pack_tensor(tensor)
                elif op == TileOp.VERIFY_CTR:
                    if repacked is None or tile_bytes is None or repacked != tile_bytes or repacked != rec.reference:
                        raise ValueError("CTR tile verification failed")
                    evidence = {
                        "tile_coord": [current.tile_x, current.tile_y, current.tile_z],
                        "morton39": current.morton39,
                        "tensor_shape": [16, 16, 16],
                        "tensor_dtype": "float32",
                        "bit_exact": True,
                        **tensor_telemetry(tensor),
                    }
                elif op == TileOp.EMIT_PROVISIONAL:
                    if evidence is None:
                        raise ValueError("CTR verification required before emitting")
                    self.trace.append(evidence)
                else:
                    raise ValueError("unsupported operation")
        self.steps += 1  # HALT
        if len(self.trace) != len(records):
            raise ValueError("not all verified tile proposals were emitted")
        return {
            "schema": "JX8K-TILEVM-1",
            "status": "PASS",
            "authority": "provisional",
            "commit_eligible": False,
            "production_state_modified": False,
            "program_sha256": sha256(default_bytecode()).hexdigest(),
            "raw_sha256": receipt["raw_sha256"],
            "packet_sha256": receipt["packet_sha256"],
            "tiles_processed": len(self.trace),
            "physical_tensor_values": len(self.trace)*TILE_VALUES,
            "float32_working_set_bytes_per_tile": TILE_VALUES*4,
            "instruction_steps": self.steps,
            "typed_state": {"spatial":"39-bit Morton / 8192^3",
                            "geometry":"2-bit -> float32[16,16,16] -> 2-bit",
                            "verification":"exact against independent raw fixture",
                            "memory":"observation only"},
            "gates": {"bit_roundtrip":"pass", "geometry_profile":"pass",
                      "resource_bound":"pass", "integrity_against_reference":"pass",
                      "external_world":"not_evaluated", "learned_ann":"not_implemented",
                      "production_approval":"not_evaluated"},
            "observations": self.trace,
        }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", required=True)
    ap.add_argument("--encoded", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--receipt")
    ap.add_argument("--program")
    ap.add_argument("--program-out")
    args = ap.parse_args()
    program = Path(args.program).read_bytes() if args.program else default_bytecode()
    vm = BoundedTileVM(program)
    report = json.loads(Path(args.report).read_text())
    packet, raw = Path(args.encoded).read_bytes(), Path(args.raw).read_bytes()
    result = vm.run(report, packet, raw)
    out = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.receipt:
        target = Path(args.receipt)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(out)
        os.replace(temporary, target)
    if args.program_out:
        Path(args.program_out).write_bytes(default_bytecode())
    print(json.dumps({k:v for k,v in result.items() if k!="observations"},sort_keys=True,indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print("JX8K tile VM rejected input:", exc, file=sys.stderr)
        raise SystemExit(1) from exc
