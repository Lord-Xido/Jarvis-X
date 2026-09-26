#!/usr/bin/env python3
"""
Dr Moagi 3D 1GB x 1GB x 1GB Sparse Auto-Encoding/Decoding Bytecode ROM
Reference VM / executable ROM builder

Logical space
-------------
Each axis spans 1,000,000,000 byte coordinates:

    V = {0, ..., 999,999,999}^3

This is a VIRTUAL 10^27-byte address space.  The implementation never allocates
the dense cube.  It materializes only active 64^3 tiles.

Pipeline
--------
virtual byte field
  -> sparse active tile
  -> 3D block encoder
  -> quantized latent cube
  -> residual-guided latent refinement
  -> 3D decoder
  -> signed residual field
  -> correction
  -> verification
  -> recur

The bytecode ROM contains fixed-width instructions.  It is a compact machine
description of the virtual cube, not a dense 10^27-byte dump.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

MAGIC = b"DM3DROM1"
VERSION = 1

AXIS_BYTES = 1_000_000_000
TILE_EDGE = 64
TILE_BYTES = TILE_EDGE**3
LATENT_EDGE = 8
LATENT_BYTES = LATENT_EDGE**3
TILES_PER_AXIS = (AXIS_BYTES + TILE_EDGE - 1) // TILE_EDGE

HEADER_STRUCT = struct.Struct("<8sIQIIIQQ16s")
HEADER_SIZE = HEADER_STRUCT.size
INSTR_STRUCT = struct.Struct("<BBHIIIIIII")
INSTR_SIZE = INSTR_STRUCT.size

OP_NOP = 0x00
OP_FILL_TILE = 0x01
OP_ENCODE = 0x10
OP_REFINE = 0x11
OP_DECODE = 0x12
OP_RESIDUAL = 0x13
OP_CORRECT = 0x14
OP_VERIFY = 0x15
OP_DROP_TILE = 0x16
OP_INWARD_LOOP_K = 0x17
OP_JUMP = 0x20
OP_HALT = 0xFF

OP_NAMES = {
    OP_NOP: "NOP",
    OP_FILL_TILE: "FILL_TILE",
    OP_ENCODE: "ENCODE",
    OP_REFINE: "REFINE",
    OP_DECODE: "DECODE",
    OP_RESIDUAL: "RESIDUAL",
    OP_CORRECT: "CORRECT",
    OP_VERIFY: "VERIFY",
    OP_DROP_TILE: "DROP_TILE",
    OP_INWARD_LOOP_K: "INWARD_LOOP_K",
    OP_JUMP: "JUMP",
    OP_HALT: "HALT",
}


def clamp_u8(v: int) -> int:
    return 0 if v < 0 else 255 if v > 255 else int(v)


def tile_index(x: int, y: int, z: int, edge: int = TILE_EDGE) -> Tuple[int, int, int]:
    return x // edge, y // edge, z // edge


def tile_origin(tx: int, ty: int, tz: int, edge: int = TILE_EDGE) -> Tuple[int, int, int]:
    return tx * edge, ty * edge, tz * edge


def split_by_3(v: int) -> int:
    # Locality hint only: tile coordinates require more than the 21 useful bits here.
    v &= 0x1FFFFF
    v = (v | (v << 32)) & 0x1F00000000FFFF
    v = (v | (v << 16)) & 0x1F0000FF0000FF
    v = (v | (v << 8)) & 0x100F00F00F00F00F
    v = (v | (v << 4)) & 0x10C30C30C30C30C3
    v = (v | (v << 2)) & 0x1249249249249249
    return v


def morton3_local(x: int, y: int, z: int) -> int:
    return split_by_3(x) | (split_by_3(y) << 1) | (split_by_3(z) << 2)


def idx3(x: int, y: int, z: int, n: int) -> int:
    return (z * n + y) * n + x


def deterministic_tile(tx: int, ty: int, tz: int, pattern: int) -> bytearray:
    """Procedurally materialize one 64^3 byte tile."""
    out = bytearray(TILE_BYTES)
    seed = (
        (tx * 73856093)
        ^ (ty * 19349663)
        ^ (tz * 83492791)
        ^ (pattern * 2654435761)
    ) & 0xFFFFFFFF
    for z in range(TILE_EDGE):
        for y in range(TILE_EDGE):
            base = idx3(0, y, z, TILE_EDGE)
            for x in range(TILE_EDGE):
                if pattern == 0:
                    v = 0
                elif pattern == 1:
                    v = (3 * x + 5 * y + 7 * z + (x * y) // 8 + (z * z) // 16 + (seed & 31)) & 0xFF
                elif pattern == 2:
                    cx = x - TILE_EDGE // 2
                    cy = y - TILE_EDGE // 2
                    cz = z - TILE_EDGE // 2
                    r2 = cx * cx + cy * cy + cz * cz
                    v = (r2 // 7 + (seed & 63)) & 0xFF
                else:
                    h = (
                        seed
                        ^ (x * 0x45D9F3B)
                        ^ (y * 0x119DE1F3)
                        ^ (z * 0x3449)
                    ) & 0xFFFFFFFF
                    h ^= h >> 16
                    h = (h * 0x45D9F3B) & 0xFFFFFFFF
                    h ^= h >> 16
                    v = h & 0xFF
                out[base + x] = v
    return out


def encode_tile(src: bytearray, quant_step: int = 8) -> bytearray:
    """Encode one 64^3 tile into an 8^3 quantized block-average latent cube."""
    q = max(1, int(quant_step))
    block = TILE_EDGE // LATENT_EDGE
    latent = bytearray(LATENT_BYTES)

    for lz in range(LATENT_EDGE):
        for ly in range(LATENT_EDGE):
            for lx in range(LATENT_EDGE):
                total = 0
                count = 0
                z0, y0, x0 = lz * block, ly * block, lx * block
                for z in range(z0, z0 + block):
                    for y in range(y0, y0 + block):
                        base = idx3(x0, y, z, TILE_EDGE)
                        for x in range(block):
                            total += src[base + x]
                            count += 1
                avg = total / count
                quant = int(round(avg / q) * q)
                latent[idx3(lx, ly, lz, LATENT_EDGE)] = clamp_u8(quant)
    return latent


def decode_tile(latent: bytearray) -> bytearray:
    """Decode one 8^3 latent cube into a 64^3 nearest-block reconstruction."""
    block = TILE_EDGE // LATENT_EDGE
    out = bytearray(TILE_BYTES)
    for lz in range(LATENT_EDGE):
        for ly in range(LATENT_EDGE):
            for lx in range(LATENT_EDGE):
                value = latent[idx3(lx, ly, lz, LATENT_EDGE)]
                z0, y0, x0 = lz * block, ly * block, lx * block
                for z in range(z0, z0 + block):
                    for y in range(y0, y0 + block):
                        base = idx3(x0, y, z, TILE_EDGE)
                        out[base : base + block] = bytes([value]) * block
    return out


def residual_field(src: bytearray, recon: bytearray) -> List[int]:
    return [int(a) - int(b) for a, b in zip(src, recon)]


def mse_from_residual(resid: List[int]) -> float:
    if not resid:
        return 0.0
    return sum(r * r for r in resid) / len(resid)


def refine_latent(
    src: bytearray,
    latent: bytearray,
    iterations: int = 4,
    alpha_num: int = 1,
    alpha_den: int = 1,
) -> bytearray:
    """Pool reconstruction residuals inward and correct the latent representation."""
    block = TILE_EDGE // LATENT_EDGE
    z = bytearray(latent)
    iters = max(0, min(64, int(iterations)))
    den = max(1, int(alpha_den))
    num = max(0, min(den, int(alpha_num)))

    for _ in range(iters):
        recon = decode_tile(z)
        for lz in range(LATENT_EDGE):
            for ly in range(LATENT_EDGE):
                for lx in range(LATENT_EDGE):
                    total = 0
                    count = 0
                    z0, y0, x0 = lz * block, ly * block, lx * block
                    for zz in range(z0, z0 + block):
                        for yy in range(y0, y0 + block):
                            base = idx3(x0, yy, zz, TILE_EDGE)
                            for xx in range(block):
                                total += int(src[base + xx]) - int(recon[base + xx])
                                count += 1
                    correction = int(round((total / count) * num / den))
                    i = idx3(lx, ly, lz, LATENT_EDGE)
                    z[i] = clamp_u8(int(z[i]) + correction)
        if num == den:
            break
    return z


def inward_loop_k(
    src: bytearray,
    latent: bytearray,
    logical_iterations: int,
    physical_budget: int = 256,
) -> Tuple[bytearray, int, bool]:
    """Execute a bounded fused recurrence and fast-forward only at an exact fixed point.

    Each physical step applies the same deterministic one-step latent refinement.
    If a step leaves the latent unchanged, all later applications of that recurrence
    are identical, so the remaining logical iterations can be elided exactly.

    Returns the latent state, physical step count, and fixed-point flag.
    """
    logical = max(0, int(logical_iterations))
    if logical == 0:
        return bytearray(latent), 0, True

    budget = max(1, min(logical, int(physical_budget)))
    z = bytearray(latent)
    physical_steps = 0

    for _ in range(budget):
        next_z = refine_latent(src, z, iterations=1, alpha_num=1, alpha_den=1)
        physical_steps += 1
        if next_z == z:
            return z, physical_steps, True
        z = next_z

    return z, physical_steps, False


def correct_with_residual(recon: bytearray, resid: List[int]) -> bytearray:
    return bytearray(clamp_u8(int(v) + int(r)) for v, r in zip(recon, resid))


TileKey = Tuple[int, int, int]


@dataclass
class VMStats:
    instructions: int = 0
    verifies: int = 0
    verified_exact: int = 0
    logical_refine_iterations: int = 0
    logical_voxel_updates: int = 0
    physical_refine_steps: int = 0
    elided_fixed_point_steps: int = 0
    elided_logical_updates: int = 0


class SparseVolume:
    def __init__(self) -> None:
        self.tiles: Dict[TileKey, bytearray] = {}

    def put(self, key: TileKey, tile: bytearray) -> None:
        if len(tile) != TILE_BYTES:
            raise ValueError("tile has wrong byte length")
        self.tiles[key] = tile

    def get(self, key: TileKey) -> bytearray:
        if key not in self.tiles:
            raise KeyError(f"tile {key} is not materialized")
        return self.tiles[key]

    def drop(self, key: TileKey) -> None:
        self.tiles.pop(key, None)


class DM3DVM:
    def __init__(self, trace: bool = True) -> None:
        self.volume = SparseVolume()
        self.latent: Dict[TileKey, bytearray] = {}
        self.recon: Dict[TileKey, bytearray] = {}
        self.resid: Dict[TileKey, List[int]] = {}
        self.corrected: Dict[TileKey, bytearray] = {}
        self.pc = 0
        self.trace = trace
        self.stats = VMStats()

    @staticmethod
    def _check_tile_coord(key: TileKey) -> None:
        for value in key:
            if not (0 <= value < TILES_PER_AXIS):
                raise ValueError(f"tile coordinate {key} outside virtual 1GB^3 cube")

    def run(self, rom: bytes) -> VMStats:
        header = parse_header(rom)
        code_offset = int(header["code_offset"])
        count = int(header["instr_count"])
        code_end = code_offset + count * INSTR_SIZE
        if code_end > len(rom):
            raise ValueError("ROM code section truncated")

        self.pc = 0
        while self.pc < count:
            off = code_offset + self.pc * INSTR_SIZE
            fields = INSTR_STRUCT.unpack_from(rom, off)
            op, flags, _, x, y, z, p0, p1, p2, p3 = fields
            key = (x, y, z)
            self.stats.instructions += 1

            if self.trace:
                print(
                    f"[pc={self.pc:04d}] {OP_NAMES.get(op, hex(op)):<10} "
                    f"tile={key} p=({p0},{p1},{p2},{p3})"
                )

            advance = True

            if op == OP_NOP:
                pass
            elif op == OP_FILL_TILE:
                self._check_tile_coord(key)
                self.volume.put(key, deterministic_tile(x, y, z, p0))
            elif op == OP_ENCODE:
                self._check_tile_coord(key)
                self.latent[key] = encode_tile(self.volume.get(key), quant_step=max(1, p0 or 8))
            elif op == OP_REFINE:
                self._check_tile_coord(key)
                if key not in self.latent:
                    raise RuntimeError("REFINE before ENCODE")
                self.latent[key] = refine_latent(
                    self.volume.get(key),
                    self.latent[key],
                    iterations=p0 or 4,
                    alpha_num=p1 or 1,
                    alpha_den=p2 or 1,
                )
            elif op == OP_INWARD_LOOP_K:
                self._check_tile_coord(key)
                if key not in self.latent:
                    raise RuntimeError("INWARD_LOOP_K before ENCODE")

                logical_iterations = max(1, int(p0 or 1))
                physical_budget = max(1, int(p1 or 256))
                logical_lanes = int(p2 or TILE_BYTES)
                if not (1 <= logical_lanes <= TILE_BYTES):
                    raise RuntimeError(
                        f"INWARD_LOOP_K logical lane span must be in [1,{TILE_BYTES}]"
                    )

                next_latent, physical_steps, fixed = inward_loop_k(
                    self.volume.get(key),
                    self.latent[key],
                    logical_iterations=logical_iterations,
                    physical_budget=physical_budget,
                )
                self.latent[key] = next_latent
                covered_iterations = (
                    logical_iterations if fixed else min(logical_iterations, physical_steps)
                )
                self.stats.logical_refine_iterations += covered_iterations
                self.stats.logical_voxel_updates += covered_iterations * logical_lanes
                self.stats.physical_refine_steps += physical_steps

                if fixed and physical_steps < logical_iterations:
                    elided = logical_iterations - physical_steps
                    self.stats.elided_fixed_point_steps += elided
                    self.stats.elided_logical_updates += elided * logical_lanes

                if self.trace:
                    print(
                        "           inward-loop "
                        f"logical={logical_iterations:,} lanes={logical_lanes:,} "
                        f"physical={physical_steps:,} fixed={fixed}"
                    )

                if (flags & 0x01) and not fixed and physical_steps < logical_iterations:
                    raise RuntimeError(
                        "strict INWARD_LOOP_K budget ended before fixed point "
                        "or requested logical iteration count"
                    )
            elif op == OP_DECODE:
                self._check_tile_coord(key)
                if key not in self.latent:
                    raise RuntimeError("DECODE before ENCODE")
                self.recon[key] = decode_tile(self.latent[key])
            elif op == OP_RESIDUAL:
                self._check_tile_coord(key)
                if key not in self.recon:
                    raise RuntimeError("RESIDUAL before DECODE")
                self.resid[key] = residual_field(self.volume.get(key), self.recon[key])
                if self.trace:
                    print(f"           residual MSE={mse_from_residual(self.resid[key]):.6f}")
            elif op == OP_CORRECT:
                self._check_tile_coord(key)
                if key not in self.resid or key not in self.recon:
                    raise RuntimeError("CORRECT before RESIDUAL")
                self.corrected[key] = correct_with_residual(self.recon[key], self.resid[key])
            elif op == OP_VERIFY:
                self._check_tile_coord(key)
                self.stats.verifies += 1
                candidate = self.corrected.get(key, self.recon.get(key))
                if candidate is None:
                    raise RuntimeError("VERIFY before DECODE")
                source = self.volume.get(key)
                exact = candidate == source
                if exact:
                    self.stats.verified_exact += 1
                if self.trace:
                    digest = hashlib.sha256(candidate).hexdigest()[:16]
                    src_digest = hashlib.sha256(source).hexdigest()[:16]
                    print(
                        f"           verify exact={exact} sha256={digest} source={src_digest}"
                    )
                if (flags & 0x01) and not exact:
                    raise RuntimeError("strict verification failed")
            elif op == OP_DROP_TILE:
                self.volume.drop(key)
                self.latent.pop(key, None)
                self.recon.pop(key, None)
                self.resid.pop(key, None)
                self.corrected.pop(key, None)
            elif op == OP_JUMP:
                if p0 >= count:
                    raise RuntimeError("jump target outside program")
                self.pc = p0
                advance = False
            elif op == OP_HALT:
                break
            else:
                raise RuntimeError(f"unknown opcode 0x{op:02x}")

            if advance:
                self.pc += 1

        return self.stats


def pack_instruction(
    op: int,
    x: int = 0,
    y: int = 0,
    z: int = 0,
    p0: int = 0,
    p1: int = 0,
    p2: int = 0,
    p3: int = 0,
    flags: int = 0,
) -> bytes:
    values = [x, y, z, p0, p1, p2, p3]
    for value in values:
        if not (0 <= int(value) <= 0xFFFFFFFF):
            raise ValueError(f"instruction argument out of u32 range: {value}")
    return INSTR_STRUCT.pack(op, flags, 0, *map(int, values))


def build_rom(instructions: List[bytes], flags: int = 0) -> bytes:
    code = b"".join(instructions)
    header = HEADER_STRUCT.pack(
        MAGIC,
        VERSION,
        AXIS_BYTES,
        TILE_EDGE,
        LATENT_EDGE,
        len(instructions),
        HEADER_SIZE,
        flags,
        b"\x00" * 16,
    )
    return header + code


def parse_header(rom: bytes) -> Dict[str, object]:
    if len(rom) < HEADER_SIZE:
        raise ValueError("ROM too short")
    magic, version, axis, tile, latent, count, code_offset, flags, _ = HEADER_STRUCT.unpack_from(
        rom, 0
    )
    if magic != MAGIC:
        raise ValueError("bad ROM magic")
    return {
        "magic": magic.decode("ascii"),
        "version": version,
        "axis_bytes": axis,
        "tile_edge": tile,
        "latent_edge": latent,
        "instr_count": count,
        "code_offset": code_offset,
        "flags": flags,
        "rom_bytes": len(rom),
    }


def demo_program() -> List[bytes]:
    keys = [
        (0, 0, 0),
        (1_000_000, 2_000_000, 3_000_000),
        (TILES_PER_AXIS - 1, TILES_PER_AXIS - 2, TILES_PER_AXIS - 3),
    ]
    program: List[bytes] = []
    for i, (x, y, z) in enumerate(keys):
        program.extend(
            [
                pack_instruction(OP_FILL_TILE, x, y, z, p0=1 + (i % 2)),
                pack_instruction(OP_ENCODE, x, y, z, p0=8),
                pack_instruction(OP_REFINE, x, y, z, p0=4, p1=1, p2=1),
                pack_instruction(OP_DECODE, x, y, z),
                pack_instruction(OP_RESIDUAL, x, y, z),
                pack_instruction(OP_CORRECT, x, y, z),
                pack_instruction(OP_VERIFY, x, y, z, flags=0x01),
            ]
        )
    program.append(pack_instruction(OP_HALT))
    return program


def million_by_million_program() -> List[bytes]:
    """Build an exact 1,000,000-lane x 1,000,000-iteration logical workload.

    Four padded 64^3 tiles cover the lane axis. p2 records the number of
    valid logical lanes in each tile, so accounting excludes padding.
    """
    logical_lanes = 1_000_000
    logical_iterations = 1_000_000
    remaining = logical_lanes
    program: List[bytes] = []
    tile_id = 0

    while remaining > 0:
        lanes = min(TILE_BYTES, remaining)
        x, y, z = (tile_id, 0, 0)
        program.extend(
            [
                pack_instruction(OP_FILL_TILE, x, y, z, p0=1 + (tile_id % 2)),
                pack_instruction(OP_ENCODE, x, y, z, p0=8),
                pack_instruction(
                    OP_INWARD_LOOP_K,
                    x,
                    y,
                    z,
                    p0=logical_iterations,
                    p1=256,
                    p2=lanes,
                    flags=0x01,
                ),
                pack_instruction(OP_DECODE, x, y, z),
                pack_instruction(OP_RESIDUAL, x, y, z),
                pack_instruction(OP_CORRECT, x, y, z),
                pack_instruction(OP_VERIFY, x, y, z, flags=0x01),
            ]
        )
        remaining -= lanes
        tile_id += 1

    program.append(pack_instruction(OP_HALT))
    return program


def write_demo_rom(path: Path) -> bytes:
    rom = build_rom(demo_program())
    path.write_bytes(rom)
    return rom


def inspect_rom(path: Path) -> None:
    rom = path.read_bytes()
    print(json.dumps(parse_header(rom), indent=2))
    header = parse_header(rom)
    for pc in range(int(header["instr_count"])):
        off = int(header["code_offset"]) + pc * INSTR_SIZE
        op, flags, _, x, y, z, p0, p1, p2, p3 = INSTR_STRUCT.unpack_from(rom, off)
        print(
            f"{pc:04d}: {OP_NAMES.get(op, hex(op)):<10} flags=0x{flags:02x} "
            f"tile=({x},{y},{z}) p=({p0},{p1},{p2},{p3})"
        )


def print_geometry() -> None:
    logical_voxels = AXIS_BYTES**3
    levels = math.ceil(math.log2(AXIS_BYTES))
    print(f"axis bytes           : {AXIS_BYTES:,}")
    print(f"virtual byte voxels  : 10^{int(math.log10(logical_voxels))} = {logical_voxels}")
    print(f"tile edge            : {TILE_EDGE}")
    print(f"tile bytes           : {TILE_BYTES:,}")
    print(f"tiles / axis         : {TILES_PER_AXIS:,}")
    print(f"octree depth to root : {levels}")
    print(f"latent bytes / tile  : {LATENT_BYTES:,}")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dr Moagi 3D sparse AE/AD bytecode ROM reference VM")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="build the demonstration ROM image")
    p_build.add_argument("path", nargs="?", default="dr_moagi_3d_1gb3.rom")

    p_run = sub.add_parser("run", help="execute a ROM image")
    p_run.add_argument("path")
    p_run.add_argument("--quiet", action="store_true")

    p_demo = sub.add_parser("demo", help="build and execute a demo ROM")
    p_demo.add_argument("path", nargs="?", default="dr_moagi_3d_1gb3.rom")
    p_demo.add_argument("--quiet", action="store_true")

    p_million = sub.add_parser(
        "million-loop",
        help="execute the exact 1,000,000 x 1,000,000 logical inward-loop workload",
    )
    p_million.add_argument("--quiet", action="store_true")

    p_inspect = sub.add_parser("inspect", help="dump ROM header and bytecode")
    p_inspect.add_argument("path")

    sub.add_parser("geometry", help="print virtual geometry")
    args = parser.parse_args(argv)

    if args.cmd == "build":
        path = Path(args.path)
        rom = write_demo_rom(path)
        print(f"wrote {path} ({len(rom)} bytes)")
        print_geometry()
        return 0
    if args.cmd == "run":
        rom = Path(args.path).read_bytes()
        print(json.dumps(parse_header(rom), indent=2))
        print(DM3DVM(trace=not args.quiet).run(rom))
        return 0
    if args.cmd == "demo":
        path = Path(args.path)
        rom = write_demo_rom(path)
        print(f"wrote {path} ({len(rom)} bytes)")
        print_geometry()
        print(DM3DVM(trace=not args.quiet).run(rom))
        return 0
    if args.cmd == "million-loop":
        vm = DM3DVM(trace=not args.quiet)
        stats = vm.run(build_rom(million_by_million_program()))
        print(stats)
        return 0
    if args.cmd == "inspect":
        inspect_rom(Path(args.path))
        return 0
    if args.cmd == "geometry":
        print_geometry()
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
