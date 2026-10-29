#!/usr/bin/env python3
"""DM3D-ENV1: bounded public byte-envelope and in-memory transaction reference.

Not the proprietary DM-PCS protocol, encryption, or authoritative runtime.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import struct

MAGIC = b"DM3DENV1"
VERSION = 1
PROFILE = b"DM3D-CUBE-U8-V1"
LAYOUT = 1
BOUNDARY = 1
HEADER = struct.Struct(">8sHH16sBBHIIIQ32s")
MAX_SIDE = 64
MAX_PAYLOAD = MAX_SIDE**3
GATE_NAMES = ("type", "geometry", "resource", "reconstruction", "integrity", "policy")


@dataclass(frozen=True)
class Geometry:
    side: int
    profile_id: str = PROFILE.decode("ascii")
    layout: str = "x-fastest"
    boundary: str = "periodic"

    def check(self) -> None:
        if type(self.side) is not int or not (2 <= self.side <= MAX_SIDE):
            raise ValueError("invalid side")
        if self.side & (self.side - 1):
            raise ValueError("side must be a power of two")
        if (self.profile_id != PROFILE.decode("ascii")
                or self.layout != "x-fastest" or self.boundary != "periodic"):
            raise ValueError("unsupported geometry profile")

    @property
    def size(self) -> int:
        self.check()
        return self.side**3


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pack(data: bytes, geometry: Geometry) -> bytes:
    geometry.check()
    if type(data) is not bytes or len(data) != geometry.size:
        raise ValueError("payload must be exactly side^3 bytes")
    header = HEADER.pack(
        MAGIC, VERSION, 0, PROFILE.ljust(16, b"\0"), LAYOUT, BOUNDARY, 0,
        geometry.side, geometry.side, geometry.side,
        len(data), hashlib.sha256(data).digest(),
    )
    return header + data


def unpack(blob: bytes) -> tuple[Geometry, bytes]:
    if type(blob) is not bytes or not HEADER.size <= len(blob) <= HEADER.size + MAX_PAYLOAD:
        raise ValueError("invalid envelope size")
    (magic, version, flags, profile, layout, boundary, reserved,
     dx, dy, dz, length, expected) = HEADER.unpack_from(blob)
    if (magic != MAGIC or version != VERSION or flags != 0
            or profile != PROFILE.ljust(16, b"\0") or layout != LAYOUT
            or boundary != BOUNDARY or reserved != 0):
        raise ValueError("invalid envelope header")
    geometry = Geometry(dx)
    geometry.check()
    if dy != dx or dz != dx or length != geometry.size or len(blob) != HEADER.size + length:
        raise ValueError("geometry or payload length mismatch")
    data = blob[HEADER.size:]
    if hashlib.sha256(data).digest() != expected:
        raise ValueError("integrity failure: payload SHA-256 does not match")
    return geometry, data


def row_index(x: int, y: int, z: int, side: int) -> int:
    Geometry(side).check()
    if any(type(v) is not int or not 0 <= v < side for v in (x, y, z)):
        raise ValueError("coordinates out of bounds")
    return x + side*y + side*side*z


def morton_index(x: int, y: int, z: int, side: int) -> int:
    row_index(x, y, z, side)
    bits = side.bit_length() - 1
    return sum(
        (((x >> k) & 1) << (3*k)) |
        (((y >> k) & 1) << (3*k+1)) |
        (((z >> k) & 1) << (3*k+2))
        for k in range(bits)
    )


def to_morton(data: bytes, geometry: Geometry) -> bytes:
    geometry.check()
    if len(data) != geometry.size:
        raise ValueError("wrong length")
    side = geometry.side
    out = bytearray(len(data))
    for z in range(side):
        for y in range(side):
            for x in range(side):
                out[morton_index(x, y, z, side)] = data[x + side*y + side*side*z]
    return bytes(out)


def from_morton(data: bytes, geometry: Geometry) -> bytes:
    geometry.check()
    if len(data) != geometry.size:
        raise ValueError("wrong length")
    side = geometry.side
    out = bytearray(len(data))
    for z in range(side):
        for y in range(side):
            for x in range(side):
                out[x + side*y + side*side*z] = data[morton_index(x, y, z, side)]
    return bytes(out)


@dataclass(frozen=True)
class Proposal:
    epoch: int
    before_sha256: str
    after_sha256: str
    envelope: bytes
    gates: tuple[tuple[str, str], ...]

    def receipt(self) -> dict:
        return {
            "format": "DM3D-ENV1-reference-receipt",
            "epoch": self.epoch,
            "state_hash_before": self.before_sha256,
            "state_hash_candidate": self.after_sha256,
            "gates": dict(self.gates), "decision": "PENDING",
        }


class ReferenceTransaction:
    """One-domain in-memory reference; NOT a durable multi-domain transaction."""

    def __init__(self, initial: bytes, geometry: Geometry):
        self.geometry = geometry
        self.authoritative = unpack(pack(initial, geometry))[1]
        self.epoch = 0
        self.journal_head = "0" * 64

    def stage(self, candidate: bytes) -> Proposal:
        encoded = pack(candidate, self.geometry)
        verified_geometry, verified = unpack(encoded)
        if verified != candidate or verified_geometry != self.geometry:
            raise ValueError("candidate reconstruction mismatch")
        return Proposal(
            epoch=self.epoch,
            before_sha256=digest(self.authoritative),
            after_sha256=digest(candidate),
            envelope=encoded,
            gates=tuple((name, "PASS") for name in GATE_NAMES)
                  + (("external_evidence", "NOT_APPLICABLE"),),
        )

    def commit(self, proposal: Proposal, *, approve: bool = False) -> dict:
        if not approve:
            raise PermissionError("explicit approval is required")
        if (type(proposal) is not Proposal or proposal.epoch != self.epoch
                or proposal.before_sha256 != digest(self.authoritative)):
            raise ValueError("stale or invalid proposal")
        expected_gates = dict.fromkeys(GATE_NAMES, "PASS") | {"external_evidence": "NOT_APPLICABLE"}
        if dict(proposal.gates) != expected_gates:
            raise ValueError("unsupported verification gates")
        geometry, candidate = unpack(proposal.envelope)
        if geometry != self.geometry or digest(candidate) != proposal.after_sha256:
            raise ValueError("candidate verification failed")
        previous_head = self.journal_head
        next_epoch = self.epoch + 1
        record = (b"DM3D-ENV1-COMMIT\0" + bytes.fromhex(previous_head)
                  + next_epoch.to_bytes(8, "big")
                  + bytes.fromhex(proposal.before_sha256)
                  + bytes.fromhex(proposal.after_sha256))
        next_head = digest(record)
        self.authoritative = candidate
        self.epoch = next_epoch
        self.journal_head = next_head
        return {
            **proposal.receipt(), "decision": "COMMITTED", "epoch": next_epoch,
            "journal_parent": previous_head, "journal_head": next_head,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("pack")
    p.add_argument("input", type=Path)
    p.add_argument("side", type=int)
    p.add_argument("output", type=Path)
    p = sub.add_parser("unpack")
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p = sub.add_parser("inspect")
    p.add_argument("input", type=Path)
    args = parser.parse_args()
    if args.action == "pack":
        result = pack(args.input.read_bytes(), Geometry(args.side))
        args.output.write_bytes(result)
    else:
        geometry, raw = unpack(args.input.read_bytes())
        if args.action == "unpack":
            args.output.write_bytes(raw)
        else:
            print(json.dumps({
                "format": "DM3D-ENV1", "side": geometry.side,
                "payload_bytes": len(raw), "sha256": digest(raw),
            }, sort_keys=True))


if __name__ == "__main__":
    main()
