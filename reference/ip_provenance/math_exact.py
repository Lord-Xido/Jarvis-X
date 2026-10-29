"""Exact rational/bitwise mathematical witness for the *frozen* 3D engine.

This is a bounded proof for the 8^3 status representation and a 1000-step
rational contraction, not a proof that an arbitrary neural network, virtual
1e18 lattice, physical hardware or external transaction behaves perfectly.
"""
from __future__ import annotations

import hashlib
from fractions import Fraction
from itertools import product
import json
from math import prod
from typing import Any

from reference.canonical_shadow_tx import geometry_witness as gm
from reference.dm3d_env1 import envelope as env
from reference.ip_provenance.freeze import FROZEN_COMMIT, canonical_bytes, merkle

DOMAINS = ("ann", "geometry", "os", "vm")
VERTICES = ((1, 1, 1), (6, 6, 1), (6, 1, 6), (1, 6, 6))
STATES = ("ABSENT", "PREPARED", "COMMITTED", "ABORTED", "UNREACHABLE")
STEPS = 1000


def determinant(u: tuple[int, ...], v: tuple[int, ...],
                w: tuple[int, ...]) -> int:
    cross = (v[1] * w[2] - v[2] * w[1],
             v[2] * w[0] - v[0] * w[2],
             v[0] * w[1] - v[1] * w[0])
    return sum(u[i] * cross[i] for i in range(3))


def volume_exact() -> Fraction:
    a, b, c, d = VERTICES
    u = tuple(b[i] - a[i] for i in range(3))
    v = tuple(c[i] - a[i] for i in range(3))
    w = tuple(d[i] - a[i] for i in range(3))
    return Fraction(determinant(u, v, w), 6)


def status_coordinate_code(domain_i: int, state_i: int) -> int:
    if not 0 <= domain_i < 4 or not 0 <= state_i < 5:
        raise ValueError("invalid domain/status")
    return 8 * domain_i + state_i + 1


def encode_exact(indices: tuple[int, int, int, int]) -> bytes:
    """Independent integer implementation; imports canonical envelope only."""
    if len(indices) != 4 or any(type(k) is not int or not 0 <= k < 5 for k in indices):
        raise ValueError("invalid status quadruple")
    raw = bytearray(512)
    for i, (coord, state) in enumerate(zip(VERTICES, indices)):
        x, y, z = coord
        raw[x + 8*y + 64*z] = status_coordinate_code(i, state)
    return env.pack(bytes(raw), env.Geometry(8))


def fraction_pair(value: Fraction) -> dict:
    return {"numerator": str(value.numerator), "denominator": str(value.denominator)}


def exact_recurrence() -> dict[str, Any]:
    """Execute all 1000 updates in exact rationals, validating E(k)=E0/4^k."""
    initial = (Fraction(1), Fraction(1, 2), Fraction(0), Fraction(0))
    current = initial
    e0 = sum((1 - v)**2 for v in current)
    for k in range(STEPS + 1):
        expected = tuple(1 - Fraction(1, 2**k)*(1 - z0) for z0 in initial)
        energy = sum((1 - x)**2 for x in current)
        if current != expected or energy != e0 / (4**k):
            raise AssertionError(f"exact rational contraction failed at k={k}")
        if k < STEPS:
            current = tuple((v + 1)/2 for v in current)
    return {
        "steps_executed_exactly": STEPS,
        "recurrence": "z[k+1]=(z[k]+1)/2",
        "closed_form": "z[k]=1-2^-k*(1-z[0])",
        "energy_law": "E[k]=E[0]/4^k",
        "initial": [fraction_pair(v) for v in initial],
        "E0": fraction_pair(e0),
        "E1000": fraction_pair(e0 / (4**STEPS)),
        "rational_proof": "PASS",
        "commit_semantics": "NONE: geometric state only; no participant apply",
    }


def audit_3d() -> dict[str, Any]:
    if gm.SIDE != 8 or gm.DOMAINS != DOMAINS or gm.STATUS != STATES:
        raise AssertionError("frozen geometric profile unexpectedly changed")
    if tuple(gm.VERTICES[name] for name in DOMAINS) != VERTICES:
        raise AssertionError("frozen coordinate assignment unexpectedly changed")
    volume = volume_exact()
    if volume != Fraction(-125, 3):
        raise AssertionError("tetrahedron has unexpected signed volume")
    if len(set(VERTICES)) != 4 or volume == 0:
        raise AssertionError("invalid tetrahedral 3D embedding")
    if env.HEADER.size != 84 or env.Geometry(8).size != 512:
        raise AssertionError("frozen 3D envelope changed")
    # All 512 Morton indices must be a one-to-one permutation.
    indexes = {
        env.morton_index(x, y, z, 8)
        for x, y, z in product(range(8), repeat=3)
    }
    if indexes != set(range(512)):
        raise AssertionError("invalid three-axis Morton permutation")
    states = product(range(len(STATES)), repeat=len(DOMAINS))
    packet_hashes: list[bytes] = []
    number = 0
    for indices in states:
        observed = {name: STATES[index] for name, index in zip(DOMAINS, indices)}
        packet = encode_exact(indices)
        if (len(packet) != 596 or gm.encode_lattice(observed) != packet
                or gm.decode_lattice(packet) != observed):
            raise AssertionError("3D encode/decoding mismatch")
        shape, raw = env.unpack(packet)
        if (shape != env.Geometry(8)
                or env.from_morton(env.to_morton(raw, shape), shape) != raw
                or sum(bool(v) for v in raw) != 4):
            raise AssertionError("3D Morton or nonzero-coordinate mismatch")
        packet_hashes.append(hashlib.sha256(
            b"JARVISX-IPFREEZE/v1/3d-packet\x00"
            + bytes(indices) + hashlib.sha256(packet).digest()
        ).digest())
        number += 1
    if number != 5**4:
        raise AssertionError("not all 3D status combinations were tested")
    return {
        "schema": "JarvisX/IPFreeze/3DMathExact1",
        "source_commit": FROZEN_COMMIT,
        "geometry": {"side": 8, "voxels": 512, "coordinates": {
            name: list(VERTICES[i]) for i, name in enumerate(DOMAINS)},
            "signed_volume_exact": fraction_pair(volume),
            "row_major": "x+8*y+64*z",
            "morton_permutation": "512/512 unique, exact inverse",
            "permutation_indexing": "3-bit interleave of x,y,z"},
        "codec": {
            "schema": "DM3DENV1/U8",
            "header_bytes": 84, "data_bytes": 512, "packet_bytes": 596,
            "status_states_per_domain": 5,
            "distinct_domain_codes": 20,
            "status_configurations_exhaustively_verified": number,
            "bit_identical_reconstruction": True,
            "status_atlas_merkle_sha256": merkle(packet_hashes),
        },
        "exact_contraction": exact_recurrence(),
        "verification_scope": (
            "Integer-coordinate / lossless status codec + exact rational "
            "contraction on frozen 3D witness; not a proof of AI performance, "
            "learned compression, legal ownership or distributed ACID"
        ),
        "external_evidence": "NOT_APPLICABLE",
        "production_deployment": "BLOCKED",
    }


def main() -> None:
    report = audit_3d()
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
