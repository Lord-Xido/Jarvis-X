#!/usr/bin/env python3
"""DM-vOmegaXi+ Quine v2.0: bounded, deterministic, 3D evolutionary source quine.

A research simulator, not a distributed replicator or a trained ANN.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

TEMPLATE = "#!/usr/bin/env python3\n\"\"\"DM-vOmegaXi+ Quine v2.0: bounded, deterministic, 3D evolutionary source quine.\n\nA research simulator, not a distributed replicator or a trained ANN.\n\"\"\"\nfrom __future__ import annotations\n\nimport argparse\nimport hashlib\nimport json\nimport random\nimport sys\nimport time\nfrom pathlib import Path\n\nTEMPLATE = @@TEMPLATE@@\nGENOME = @@GENOME@@\n\nSIDE = 8\nCOARSE = SIDE // 2\nMAX_STEPS = 1000\nMAX_LOGICAL_STREAMS = 1_000_000\n\n\ndef source_of(genome: list[int]) -> str:\n    \"\"\"Recreate the full executable source without reading the source file.\"\"\"\n    if len(genome) != 3 or any(type(w) is not int or w < 0 or w > 16 for w in genome):\n        raise ValueError(\"genome must contain three integers in [0, 16]\")\n    template_key = \"TEMPLATE = \" + \"@@TEMPLATE@@\"\n    genome_key = \"GENOME = \" + \"@@GENOME@@\"\n    a = TEMPLATE.replace(genome_key, \"GENOME = \" + json.dumps(genome, separators=(\",\", \":\")), 1)\n    return a.replace(template_key, \"TEMPLATE = \" + json.dumps(TEMPLATE, ensure_ascii=True), 1)\n\n\ndef digest(data: str) -> str:\n    return hashlib.sha256(data.encode(\"utf-8\")).hexdigest()\n\n\ndef at(x: int, y: int, z: int, dim: int) -> int:\n    return (z * dim + y) * dim + x\n\n\ndef make_volume() -> tuple[float, ...]:\n    # A fixed, modest synthetic 3D signal, not natural or clinical imaging.\n    return tuple(float(12 + 2*x + 3*y + 5*z + (x*y + 2*z) % 5)\n                 for z in range(SIDE) for y in range(SIDE) for x in range(SIDE))\n\n\ndef encode(volume: tuple[float, ...]) -> tuple[float, ...]:\n    result = []\n    for bz in range(COARSE):\n        for by in range(COARSE):\n            for bx in range(COARSE):\n                total = sum(volume[at(2*bx+dx, 2*by+dy, 2*bz+dz, SIDE)]\n                            for dz in (0, 1) for dy in (0, 1) for dx in (0, 1))\n                result.append(total / 8.0)\n    return tuple(result)\n\n\nVOLUME = make_volume()\nLATENT = encode(VOLUME)\n\n\ndef gradient(bx: int, by: int, bz: int, axis: int) -> float:\n    c = [bx, by, bz]\n    k = c[axis]\n    lo, hi = max(0, k-1), min(COARSE-1, k+1)\n    c[axis] = lo\n    left = LATENT[at(c[0], c[1], c[2], COARSE)]\n    c[axis] = hi\n    right = LATENT[at(c[0], c[1], c[2], COARSE)]\n    return (right - left) / (2.0 * (hi - lo))\n\n\ndef mse(genome: list[int]) -> float:\n    total = 0.0\n    for z in range(SIDE):\n        for y in range(SIDE):\n            for x in range(SIDE):\n                bx, by, bz = x // 2, y // 2, z // 2\n                base = LATENT[at(bx, by, bz, COARSE)]\n                estimate = base\n                for axis, offset in enumerate((x % 2 - 0.5, y % 2 - 0.5, z % 2 - 0.5)):\n                    estimate += (genome[axis] / 16.0) * gradient(bx, by, bz, axis) * offset\n                error = VOLUME[at(x, y, z, SIDE)] - estimate\n                total += error * error\n    return total / len(VOLUME)\n\n\ndef evolve(start: list[int], steps: int, seed: int):\n    rng = random.Random(seed)\n    current = list(start)\n    best = mse(current)\n    records = []\n    accepted = 0\n    for step in range(1, steps + 1):\n        candidate = list(current)\n        axis = rng.randrange(3)\n        candidate[axis] = max(0, min(16, candidate[axis] + rng.choice((-1, 1))))\n        trial = mse(candidate)\n        improved = trial < best - 1e-12\n        parent_sha = digest(source_of(current))\n        candidate_sha = digest(source_of(candidate))\n        if improved:\n            current, best = candidate, trial\n            accepted += 1\n        records.append({\n            \"step\": step, \"axis\": axis, \"accepted\": improved,\n            \"parent_sha256\": parent_sha, \"candidate_sha256\": candidate_sha,\n            \"committed_sha256\": digest(source_of(current)),\n            \"loss_mse\": best, \"genome\": current[:]\n        })\n    return current, best, accepted, records\n\n\ndef write_new(path: str, data: str) -> None:\n    # Exclusive creation prevents accidental replacement of the running program.\n    dest = Path(path)\n    if dest.resolve() == Path(__file__).resolve():\n        raise ValueError(\"cannot overwrite the active source\")\n    with dest.open(\"x\", encoding=\"utf-8\", newline=\"\\n\") as handle:\n        handle.write(data)\n\n\ndef selftest() -> None:\n    assert Path(__file__).read_bytes() == source_of(GENOME).encode(\"utf-8\"), \"quine byte mismatch\"\n    assert len(VOLUME) == 512 and len(LATENT) == 64\n    assert abs(mse([0, 0, 0]) - mse([0, 0, 0])) < 1e-15\n    descendant, best, accepted, records = evolve(list(GENOME), 100, 707)\n    assert best <= mse(GENOME) + 1e-12\n    assert all(records[i][\"loss_mse\"] <= records[i-1][\"loss_mse\"] + 1e-12\n               for i in range(1, len(records)))\n    assert digest(source_of(descendant)) == records[-1][\"committed_sha256\"]\n    assert accepted <= 100\n    print(\"PASS: exact source quine, 3D codec, bounded mutation, monotone accept/rollback, SHA-256 trace\")\n\n\ndef main(argv: list[str] | None = None) -> int:\n    parser = argparse.ArgumentParser(description=__doc__)\n    parser.add_argument(\"--quine\", action=\"store_true\", help=\"emit exact own source bytes\")\n    parser.add_argument(\"--selftest\", action=\"store_true\")\n    parser.add_argument(\"--steps\", type=int, default=32)\n    parser.add_argument(\"--seed\", type=int, default=707)\n    parser.add_argument(\"--streams\", type=int, default=1_000_000,\n                        help=\"virtual logical destinations; no physical broadcast\")\n    parser.add_argument(\"--emit-child\", help=\"exclusively create a descendant .py source file\")\n    parser.add_argument(\"--trace-out\", help=\"exclusively create a deterministic JSONL trace file\")\n    parser.add_argument(\"--bench\", action=\"store_true\", help=\"report local wall-clock timing only\")\n    args = parser.parse_args(argv)\n    if args.quine:\n        sys.stdout.buffer.write(source_of(GENOME).encode(\"utf-8\"))\n        return 0\n    if args.selftest:\n        selftest()\n        return 0\n    if not 0 <= args.steps <= MAX_STEPS:\n        parser.error(\"steps must be in [0, 1000]\")\n    if not 0 <= args.streams <= MAX_LOGICAL_STREAMS:\n        parser.error(\"streams must be in [0, 1000000]\")\n    start_loss = mse(GENOME)\n    start_ns = time.perf_counter_ns()\n    child, final_loss, accepted, trace = evolve(list(GENOME), args.steps, args.seed)\n    elapsed_ns = time.perf_counter_ns() - start_ns\n    child_source = source_of(child)\n    if args.emit_child:\n        write_new(args.emit_child, child_source)\n    if args.trace_out:\n        write_new(args.trace_out, \"\".join(json.dumps(row, sort_keys=True) + \"\\n\" for row in trace))\n    summary = {\n        \"engine\": \"DM-vOmegaXi+ bounded quine v2.0\",\n        \"geometry\": [SIDE, SIDE, SIDE],\n        \"latent\": [COARSE, COARSE, COARSE],\n        \"steps\": args.steps,\n        \"accepted\": accepted,\n        \"initial_genome\": GENOME,\n        \"final_genome\": child,\n        \"initial_mse\": start_loss,\n        \"final_mse\": final_loss,\n        \"source_sha256\": digest(source_of(GENOME)),\n        \"descendant_sha256\": digest(child_source),\n        \"logical_streams\": args.streams,\n        \"copies_materialized\": int(bool(args.emit_child)),\n        \"speedup_1000x_proven\": False,\n    }\n    if args.bench:\n        summary[\"local_elapsed_ns\"] = elapsed_ns\n        summary[\"note\"] = \"Timing is not a speedup measurement or stable benchmark\"\n    print(json.dumps(summary, indent=2, sort_keys=True))\n    return 0\n\n\nif __name__ == \"__main__\":\n    raise SystemExit(main())\n"
GENOME = [0,0,0]

SIDE = 8
COARSE = SIDE // 2
MAX_STEPS = 1000
MAX_LOGICAL_STREAMS = 1_000_000


def source_of(genome: list[int]) -> str:
    """Recreate the full executable source without reading the source file."""
    if len(genome) != 3 or any(type(w) is not int or w < 0 or w > 16 for w in genome):
        raise ValueError("genome must contain three integers in [0, 16]")
    template_key = "TEMPLATE = " + "@@TEMPLATE@@"
    genome_key = "GENOME = " + "@@GENOME@@"
    a = TEMPLATE.replace(genome_key, "GENOME = " + json.dumps(genome, separators=(",", ":")), 1)
    return a.replace(template_key, "TEMPLATE = " + json.dumps(TEMPLATE, ensure_ascii=True), 1)


def digest(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def at(x: int, y: int, z: int, dim: int) -> int:
    return (z * dim + y) * dim + x


def make_volume() -> tuple[float, ...]:
    # A fixed, modest synthetic 3D signal, not natural or clinical imaging.
    return tuple(float(12 + 2*x + 3*y + 5*z + (x*y + 2*z) % 5)
                 for z in range(SIDE) for y in range(SIDE) for x in range(SIDE))


def encode(volume: tuple[float, ...]) -> tuple[float, ...]:
    result = []
    for bz in range(COARSE):
        for by in range(COARSE):
            for bx in range(COARSE):
                total = sum(volume[at(2*bx+dx, 2*by+dy, 2*bz+dz, SIDE)]
                            for dz in (0, 1) for dy in (0, 1) for dx in (0, 1))
                result.append(total / 8.0)
    return tuple(result)


VOLUME = make_volume()
LATENT = encode(VOLUME)


def gradient(bx: int, by: int, bz: int, axis: int) -> float:
    c = [bx, by, bz]
    k = c[axis]
    lo, hi = max(0, k-1), min(COARSE-1, k+1)
    c[axis] = lo
    left = LATENT[at(c[0], c[1], c[2], COARSE)]
    c[axis] = hi
    right = LATENT[at(c[0], c[1], c[2], COARSE)]
    return (right - left) / (2.0 * (hi - lo))


def mse(genome: list[int]) -> float:
    total = 0.0
    for z in range(SIDE):
        for y in range(SIDE):
            for x in range(SIDE):
                bx, by, bz = x // 2, y // 2, z // 2
                base = LATENT[at(bx, by, bz, COARSE)]
                estimate = base
                for axis, offset in enumerate((x % 2 - 0.5, y % 2 - 0.5, z % 2 - 0.5)):
                    estimate += (genome[axis] / 16.0) * gradient(bx, by, bz, axis) * offset
                error = VOLUME[at(x, y, z, SIDE)] - estimate
                total += error * error
    return total / len(VOLUME)


def evolve(start: list[int], steps: int, seed: int):
    rng = random.Random(seed)
    current = list(start)
    best = mse(current)
    records = []
    accepted = 0
    for step in range(1, steps + 1):
        candidate = list(current)
        axis = rng.randrange(3)
        candidate[axis] = max(0, min(16, candidate[axis] + rng.choice((-1, 1))))
        trial = mse(candidate)
        improved = trial < best - 1e-12
        parent_sha = digest(source_of(current))
        candidate_sha = digest(source_of(candidate))
        if improved:
            current, best = candidate, trial
            accepted += 1
        records.append({
            "step": step, "axis": axis, "accepted": improved,
            "parent_sha256": parent_sha, "candidate_sha256": candidate_sha,
            "committed_sha256": digest(source_of(current)),
            "loss_mse": best, "genome": current[:]
        })
    return current, best, accepted, records


def write_new(path: str, data: str) -> None:
    # Exclusive creation prevents accidental replacement of the running program.
    dest = Path(path)
    if dest.resolve() == Path(__file__).resolve():
        raise ValueError("cannot overwrite the active source")
    with dest.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(data)


def selftest() -> None:
    assert Path(__file__).read_bytes() == source_of(GENOME).encode("utf-8"), "quine byte mismatch"
    assert len(VOLUME) == 512 and len(LATENT) == 64
    assert abs(mse([0, 0, 0]) - mse([0, 0, 0])) < 1e-15
    descendant, best, accepted, records = evolve(list(GENOME), 100, 707)
    assert best <= mse(GENOME) + 1e-12
    assert all(records[i]["loss_mse"] <= records[i-1]["loss_mse"] + 1e-12
               for i in range(1, len(records)))
    assert digest(source_of(descendant)) == records[-1]["committed_sha256"]
    assert accepted <= 100
    print("PASS: exact source quine, 3D codec, bounded mutation, monotone accept/rollback, SHA-256 trace")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quine", action="store_true", help="emit exact own source bytes")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=707)
    parser.add_argument("--streams", type=int, default=1_000_000,
                        help="virtual logical destinations; no physical broadcast")
    parser.add_argument("--emit-child", help="exclusively create a descendant .py source file")
    parser.add_argument("--trace-out", help="exclusively create a deterministic JSONL trace file")
    parser.add_argument("--bench", action="store_true", help="report local wall-clock timing only")
    args = parser.parse_args(argv)
    if args.quine:
        sys.stdout.buffer.write(source_of(GENOME).encode("utf-8"))
        return 0
    if args.selftest:
        selftest()
        return 0
    if not 0 <= args.steps <= MAX_STEPS:
        parser.error("steps must be in [0, 1000]")
    if not 0 <= args.streams <= MAX_LOGICAL_STREAMS:
        parser.error("streams must be in [0, 1000000]")
    start_loss = mse(GENOME)
    start_ns = time.perf_counter_ns()
    child, final_loss, accepted, trace = evolve(list(GENOME), args.steps, args.seed)
    elapsed_ns = time.perf_counter_ns() - start_ns
    child_source = source_of(child)
    if args.emit_child:
        write_new(args.emit_child, child_source)
    if args.trace_out:
        write_new(args.trace_out, "".join(json.dumps(row, sort_keys=True) + "\n" for row in trace))
    summary = {
        "engine": "DM-vOmegaXi+ bounded quine v2.0",
        "geometry": [SIDE, SIDE, SIDE],
        "latent": [COARSE, COARSE, COARSE],
        "steps": args.steps,
        "accepted": accepted,
        "initial_genome": GENOME,
        "final_genome": child,
        "initial_mse": start_loss,
        "final_mse": final_loss,
        "source_sha256": digest(source_of(GENOME)),
        "descendant_sha256": digest(child_source),
        "logical_streams": args.streams,
        "copies_materialized": int(bool(args.emit_child)),
        "speedup_1000x_proven": False,
    }
    if args.bench:
        summary["local_elapsed_ns"] = elapsed_ns
        summary["note"] = "Timing is not a speedup measurement or stable benchmark"
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
