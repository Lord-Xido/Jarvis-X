"""Bounded 3D geometric witness of the real ADR-016 *shadow* transaction.

Four independently owned participant states are sampled read-only and placed
at the vertices of an exact regular tetrahedron in an 8x8x8 U8 lattice.
A DM3D-ENV1 envelope preserves the four states byte-for-byte, with a checked
Morton permutation and SHA-256 reconstruction. A contractive map is explicitly
a mathematical *visualization* of the observed progress, not an executor,
neural optimizer, transaction commit, or cross-process atomic snapshot.

No live backend, SQLite transaction, or participant is mutated.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from reference.canonical_shadow_tx.bridge import DOMAINS, _state_hash, encoded, sha
from reference.canonical_shadow_tx.durable import _frames
from reference.canonical_shadow_tx.multiprocess import Coordinator
from reference.dm3d_env1.envelope import (
    Geometry, from_morton, pack, row_index, to_morton, unpack,
)

SIDE = 8
GEOMETRY = Geometry(SIDE)
VERTICES = {
    "ann": (1, 1, 1),
    "geometry": (6, 6, 1),
    "os": (6, 1, 6),
    "vm": (1, 6, 6),
}
STATUS = ("ABSENT", "PREPARED", "COMMITTED", "ABORTED", "UNREACHABLE")
STATUS_CODE = {name: n for n, name in enumerate(STATUS)}
COLOR = {
    "ABSENT": "#95a1b8",
    "PREPARED": "#f4be56",
    "COMMITTED": "#20bbae",
    "ABORTED": "#e77c7c",
    "UNREACHABLE": "#ba78d5",
}
SAMPLE_ITERATIONS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000)


def signed_tetra_volume() -> float:
    """Scalar triple product / 6: exact nondegenerate 3D embedding."""
    a, b, c, d = (VERTICES[name] for name in DOMAINS)
    u = tuple(b[i] - a[i] for i in range(3))
    v = tuple(c[i] - a[i] for i in range(3))
    w = tuple(d[i] - a[i] for i in range(3))
    cross = (v[1]*w[2]-v[2]*w[1], v[2]*w[0]-v[0]*w[2],
             v[0]*w[1]-v[1]*w[0])
    return sum(u[i]*cross[i] for i in range(3)) / 6


def _validate_statuses(states: dict[str, str]) -> None:
    if type(states) is not dict or tuple(states) != DOMAINS:
        raise ValueError("exactly four ordered spatial domains are required")
    if any(type(v) is not str or v not in STATUS_CODE for v in states.values()):
        raise ValueError("invalid participant status")


def encode_lattice(states: dict[str, str]) -> bytes:
    """Injective 3D mapping; each domain/status has a distinct U8 code."""
    _validate_statuses(states)
    data = bytearray(GEOMETRY.size)
    for index, domain in enumerate(DOMAINS):
        x, y, z = VERTICES[domain]
        data[row_index(x, y, z, SIDE)] = index*8 + STATUS_CODE[states[domain]] + 1
    packet = pack(bytes(data), GEOMETRY)
    shape, raw = unpack(packet)
    if shape != GEOMETRY or raw != bytes(data) or (
        from_morton(to_morton(raw, shape), shape) != raw
    ):
        raise ValueError("3D packet or Morton reconstruction failed")
    if decode_lattice(packet) != states:
        raise ValueError("3D state reconstruction failed")
    return packet


def decode_lattice(packet: bytes) -> dict[str, str]:
    shape, raw = unpack(packet)
    if shape != GEOMETRY:
        raise ValueError("unexpected geometry profile")
    data = bytearray(raw)
    states: dict[str, str] = {}
    for index, domain in enumerate(DOMAINS):
        value_index = row_index(*VERTICES[domain], SIDE)
        value = data[value_index]
        offset = value - (index*8 + 1)
        if not 0 <= offset < len(STATUS):
            raise ValueError("unknown or conflicting vertex state")
        states[domain] = STATUS[offset]
        data[value_index] = 0
    if any(data):
        raise ValueError("unmapped 3D voxels are not permitted")
    return states


def relaxation(states: dict[str, str], *, iterations: int = 1000) -> dict[str, Any]:
    """z(k)=1-2^-k(1-z0); E(k)=4^-k E(0), no participant changes.

    This closed-form trajectory is analytically equivalent to k recurrence
    steps and avoids falsely benchmarking unexecuted ANN iterations.
    """
    _validate_statuses(states)
    if type(iterations) is not int or not 0 <= iterations <= 1000:
        raise ValueError("only 0..1000 bounded recurrence steps are supported")
    initial = tuple(1.0 if states[name] == "COMMITTED" else
                    0.5 if states[name] == "PREPARED" else 0.0
                    for name in DOMAINS)
    initial_energy = sum((1-v)**2 for v in initial)
    samples = []
    for k in sorted(set(n for n in SAMPLE_ITERATIONS if n <= iterations) | {iterations}):
        factor = 0.5**k
        z = tuple(1 - factor*(1-v) for v in initial)
        energy = initial_energy*(factor**2)
        if not math.isfinite(energy) or any(not math.isfinite(t) for t in z):
            raise ArithmeticError("nonfinite 3D contraction")
        if energy > initial_energy*4**(-k) + 1e-300:
            raise AssertionError("contraction bound violated")
        samples.append({"k": k, "z": list(z), "squared_error": energy})
    return {
        "equation": "z[k+1]=(z[k]+1)/2; E[k+1]=E[k]/4",
        "interpretation": "closed-form display trajectory only; no state mutation",
        "iterations": iterations,
        "initial": list(initial), "initial_squared_error": initial_energy,
        "samples": samples,
        "logical_commit_fixed_point": all(v == "COMMITTED" for v in states.values()),
    }


@dataclass(frozen=True)
class Witness:
    txid: str
    stage: str
    base_epoch: int
    fence: int
    target_sha256: str
    participant_statuses: dict[str, str]
    verifiable: bool
    complete: bool
    gate_results: dict[str, str]
    packet: bytes
    contraction: dict[str, Any]
    warning: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "CanonicalShadowTx/GeometricWitness3D1",
            "txid": self.txid, "stage": self.stage,
            "base_epoch": self.base_epoch, "fence": self.fence,
            "candidate_sha256": self.target_sha256,
            "lattice": {"side": SIDE, "profile": GEOMETRY.profile_id,
                        "layout": GEOMETRY.layout, "boundary": GEOMETRY.boundary,
                        "vertices": {name: list(VERTICES[name]) for name in DOMAINS},
                        "tetrahedron_signed_volume": signed_tetra_volume(),
                        "envelope_sha256": sha(self.packet),
                        "voxel_sha256": sha(unpack(self.packet)[1]),
                        "envelope_size_bytes": len(self.packet),
                        "morton_inverse": "PASS"},
            "statuses": self.participant_statuses,
            "gates": self.gate_results,
            "verified": self.verifiable,
            "complete": self.complete,
            "relaxation": self.contraction,
            "external_evidence": "NOT_APPLICABLE",
            "authority": "shadow-only/read-only geometric observation",
            "warning": self.warning,
        }


def witness(coordinator: Coordinator, txid: str, *,
            iterations: int = 1000) -> Witness:
    """Bind a 3D snapshot to coordinator evidence, with fail-closed rechecks.

    No atomic read across separate processes can be inferred. Detect a
    coordinator change before returning; concurrent participant writes can
    still interleave before final COMPLETE and cannot be certified.
    """
    from reference.canonical_shadow_tx.multiprocess import _id
    txid = _id(txid)
    record = coordinator._record(txid)
    if record is None:
        raise ValueError("unknown transaction")
    epoch, fence, manifest, candidate, stage, receipt = record
    frames = _frames(bytes(manifest))
    if _state_hash(frames) != candidate:
        raise ValueError("coordinator candidate hash mismatch")
    expected = dict(zip(DOMAINS, frames))
    statuses: dict[str, str] = {}
    per_domain: dict[str, str] = {}
    for domain in DOMAINS:
        path = coordinator.root / (domain + ".sqlite3")
        if not path.is_file():
            statuses[domain] = "ABSENT"
            per_domain[domain] = "MISSING"
            continue
        try:
            with coordinator._participant(domain) as participant:
                actual = participant.status(txid)
                version, actual_fence, current = participant.state()
                integrity = participant.verify()
        except (OSError, RuntimeError, ValueError):
            statuses[domain] = "UNREACHABLE"
            per_domain[domain] = "UNREACHABLE"
            continue
        status = actual if actual in STATUS_CODE else "ABSENT" if actual is None else "UNREACHABLE"
        statuses[domain] = status
        if status == "COMMITTED":
            valid = (integrity and version == epoch + 1 and
                     actual_fence == fence and current == expected[domain])
        elif status in ("PREPARED", "ABORTED"):
            valid = integrity and version == epoch and actual_fence == fence
        elif status == "ABSENT":
            valid = integrity and version == epoch
        else:
            valid = False
        per_domain[domain] = "PASS" if valid else "FAIL"

    stage_match = {
        "PREPARING": {"ABSENT", "PREPARED"},
        "PREPARED": {"PREPARED"},
        "COMMIT_DECIDED": {"PREPARED", "COMMITTED"},
        "COMPLETE": {"COMMITTED"},
        "ABORTED": {"ABSENT", "ABORTED"},
    }
    stage_gate = (stage in stage_match and
                  set(statuses.values()) <= stage_match[stage])
    consistent_coordinator = (record == coordinator._record(txid))
    coordinator_epoch = coordinator._meta()[0]
    epoch_gate = (coordinator_epoch == (epoch+1 if stage == "COMPLETE" else epoch))
    all_valid = all(x == "PASS" for x in per_domain.values())
    complete = stage == "COMPLETE" and all(x == "COMMITTED" for x in statuses.values())
    if complete:
        try:
            complete = coordinator.coordinated_read(txid) == frames
        except (RuntimeError, OSError, ValueError):
            complete = False
    all_verified = all_valid and stage_gate and consistent_coordinator and epoch_gate
    # Incomplete states can be structurally valid but never be certified as
    # a completed commit; an unavailable/missing participant is fail-closed.
    gates = {
        "canonical_manifest": "PASS",
        "participant_profiles": "PASS" if all_valid else "FAIL",
        "stage_consistency": "PASS" if stage_gate else "FAIL",
        "coordinator_epoch": "PASS" if epoch_gate else "FAIL",
        "coordinator_stable": "PASS" if consistent_coordinator else "FAIL",
        "commit_complete": "PASS" if complete else "BLOCKED",
        "distributed_atomic_visibility": "NOT_PROVEN",
        "external_evidence": "NOT_APPLICABLE",
    }
    for domain in DOMAINS:
        gates["domain_" + domain] = per_domain[domain]
    if stage == "COMPLETE" and not complete:
        all_verified = False
    packet = encode_lattice(statuses)
    return Witness(txid, stage, epoch, fence, candidate, statuses, all_verified,
                   complete, gates, packet, relaxation(statuses, iterations=iterations),
                   "3D coordinate/relaxation is an observability projection only; "
                   "no independent transaction commit or performance gain; "
                   "direct reads across participants are not atomic")


def svg_view(w: Witness) -> str:
    """A deterministic 3D isometric projection, not an independent state gate."""
    def project(position: tuple[int, int, int]) -> tuple[float, float]:
        x, y, z = position
        return 320 + 32*(x-z), 258 + 17*(x+z) - 39*y
    points = {domain: project(VERTICES[domain]) for domain in DOMAINS}
    elements = ['<svg xmlns="http://www.w3.org/2000/svg" width="680" height="510" '
                'viewBox="0 0 680 510" role="img" '
                'aria-label="3D tetrahedral shadow transaction witness">',
                '<rect width="680" height="510" rx="22" fill="#0d172a"/>',
                '<text x="24" y="38" font-size="18" fill="#e2edf9">'
                'Shadow transaction geometric witness</text>']
    for i, name in enumerate(DOMAINS):
        for other in DOMAINS[i+1:]:
            x1, y1 = points[name]
            x2, y2 = points[other]
            elements.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" '
                            f'x2="{x2:.1f}" y2="{y2:.1f}" '
                            'stroke="#587089" stroke-width="2" opacity=".75"/>')
    for domain in DOMAINS:
        x, y = points[domain]
        state = w.participant_statuses[domain]
        elements.extend([
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="17" '
            f'fill="{COLOR[state]}" stroke="#e6f3fa" stroke-width="2"/>',
            f'<text x="{x:.1f}" y="{y-25:.1f}" font-size="14" '
            f'fill="#e6f3fa" text-anchor="middle">{domain.upper()}</text>',
            f'<text x="{x:.1f}" y="{y+35:.1f}" font-size="11" '
            f'fill="#d3dde6" text-anchor="middle">{state}</text>',
        ])
    elements.append('<text x="22" y="458" font-size="13" fill="#d3dde6">'
                    + escape(f'stage={w.stage} | complete={w.complete} | '
                             f'verified={w.verifiable}') + '</text>')
    elements.append('<text x="22" y="480" font-size="11" fill="#92a5bc">'
                    'One coordinate per participant; shaded lines are geometric links, '
                    'not network bandwidth</text></svg>')
    return "\n".join(elements) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("txid")
    parser.add_argument("--prefix", type=Path, default=Path("geometric-witness-3d"))
    parser.add_argument("--ipc", action="store_true",
                        help="read participants through authenticated Unix RPC")
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()
    if args.ipc:
        from reference.canonical_shadow_tx.ipc import IPCCoordinator
        authority = IPCCoordinator
    else:
        authority = Coordinator
    with authority(args.root) as coordinator:
        result = witness(coordinator, args.txid, iterations=args.iterations)
    args.prefix.parent.mkdir(parents=True, exist_ok=True)
    args.prefix.with_suffix(".json").write_bytes(encoded(result.as_dict()) + b"\n")
    args.prefix.with_suffix(".env").write_bytes(result.packet)
    args.prefix.with_suffix(".svg").write_text(svg_view(result), encoding="utf-8")
    print(json.dumps({"stage": result.stage, "verified": result.verifiable,
                      "complete": result.complete,
                      "json": str(args.prefix.with_suffix(".json")),
                      "env": str(args.prefix.with_suffix(".env")),
                      "svg": str(args.prefix.with_suffix(".svg"))}, sort_keys=True))


if __name__ == "__main__":
    main()
