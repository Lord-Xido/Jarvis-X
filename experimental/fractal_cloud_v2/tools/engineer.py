#!/usr/bin/env python3
"""JARVIS-X Omega3 cloud-runner engineering agent.

Automated reproducibility, numerical equivalence, rollback/recovery and
bounded CPU-worker tuning. It never rewrites source or relaxes acceptance gates.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

EXPECTED_FINAL_HASH = '098794e9b04115c7'


def require(ok, why):
    if not ok:
        raise RuntimeError(why)


def call(binary, out, depth=2, steps=24, workers=4, extra=(), expected=0, events='none'):
    cmd = [str(binary), '--depth', str(depth), '--steps', str(steps),
           '--workers', str(workers), '--events', events, '--out', str(out),
           '--no-frames', *extra]
    start = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=90, check=False)
    elapsed = time.perf_counter() - start
    require(p.returncode == expected,
            f'command failed exit={p.returncode} expected={expected}: {cmd}\nSTDOUT={p.stdout[-1200:]}\nSTDERR={p.stderr[-1200:]}')
    return {'exit': p.returncode, 'seconds': round(elapsed, 6),
            'cmd': cmd, 'stdout_tail': p.stdout[-500:], 'stderr_tail': p.stderr[-500:]}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    require(2 <= args.depth <= 4, 'allowed depth 2..4')
    require(1 <= args.steps <= 48, 'allowed steps 1..48')
    binary = args.binary.resolve()
    golden = args.golden.resolve()
    require(binary.is_file() and golden.is_file(), 'binary and golden trace required')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    checks = []
    timings = {}

    def stage(label, *, depth=2, steps=24, workers=4, extra=(), expected=0, events='none'):
        p = out / label
        if p.exists():
            shutil.rmtree(p)
        r = call(binary, p, depth, steps, workers, extra, expected, events)
        timings[label] = r['seconds']
        return p

    ref = stage('golden_1w', workers=1, events='sample')
    require((ref / 'trace.csv').read_bytes() == golden.read_bytes(), 'v1 golden trace mismatch')
    require(list(csv.DictReader((ref / 'trace.csv').open()))[-1]['hash'] == EXPECTED_FINAL_HASH,
            'reference hash mismatch')
    checks.append('v1 canonical trace byte-match, 24 steps, 400 cells, 672 edges')

    for workers in (2, 4, 8):
        p = stage(f'golden_{workers}w', workers=workers)
        require((p / 'trace.csv').read_bytes() == golden.read_bytes(), f'{workers}w trace mismatch')
        require((p / 'checkpoint.bin').read_bytes() == (ref / 'checkpoint.bin').read_bytes(),
                f'{workers}w checkpoint mismatch')
    checks.append('1/2/4/8 CPU workers: identical full trace and checkpoints')

    partial = stage('partial_1step', steps=1, workers=1)
    resumed = stage('resume_8w', steps=24, workers=8,
                    extra=('--resume', str(partial / 'checkpoint.bin')))
    require((resumed / 'checkpoint.bin').read_bytes() == (ref / 'checkpoint.bin').read_bytes(),
            'checkpoint continuation changed final state')
    full = list(csv.DictReader((ref / 'trace.csv').open()))
    continuation = list(csv.DictReader((resumed / 'trace.csv').open()))
    require(continuation[0]['status'] == 'RESUME' and continuation[1:] == full[2:],
            'resume trace mismatch')
    checks.append('cross-worker checkpoint restart with identical final state')

    fault = stage('forced_rollback', steps=10, workers=4,
                  extra=('--inject-fault-step', '2'), expected=2)
    require((fault / 'checkpoint.bin').read_bytes() == (partial / 'checkpoint.bin').read_bytes(),
            'rollback checkpoint not identical to last committed step')
    checks.append('fault-injection rollback preserves previous committed checkpoint')

    bad = bytearray((partial / 'checkpoint.bin').read_bytes())
    bad[-1] ^= 1
    tampered = out / 'tampered.bin'
    tampered.write_bytes(bad)
    stage('tamper_rejected', workers=2,
          extra=('--resume', str(tampered)), expected=1)
    checks.append('modified checkpoint rejected before execution')

    d3a = stage('depth3_1w', depth=3, steps=24, workers=1)
    d3b = stage('depth3_4w', depth=3, steps=24, workers=4)
    require((d3a / 'trace.csv').read_bytes() == (d3b / 'trace.csv').read_bytes(),
            'depth3 trace differs by worker count')
    rows = list(csv.DictReader((d3b / 'trace.csv').open()))
    require(rows[-1]['nodes'] == '8000' and rows[-1]['edges'] == '14976',
            'depth3 geometric counts mismatch')
    checks.append('depth3 8000-cell graph: 14976 edges and cross-worker trace equivalence')

    if args.depth == 4:
        d4a = stage('depth4_1w', depth=4, steps=args.steps, workers=1)
        d4b = stage('depth4_4w', depth=4, steps=args.steps, workers=4)
        require((d4a / 'trace.csv').read_bytes() == (d4b / 'trace.csv').read_bytes(),
                'depth4 trace differs by worker count')
        row = list(csv.DictReader((d4b / 'trace.csv').open()))[-1]
        require(row['nodes'] == '160000' and row['edges'] == '311808',
                'depth4 geometric counts mismatch')
        checks.append('depth4 160000-cell graph: 311808 edges, cross-worker deterministic replay')

    report = {
        'schema': 'jx3d.autoengineering.report.v1',
        'status': 'PASS',
        'runtime': 'local CPU-threaded sharding on GitHub-hosted runner, not multi-node',
        'tests': checks,
        'reference_trace_sha256': sha(golden),
        'reference_checkpoint_sha256': sha(ref / 'checkpoint.bin'),
        'depth4_tested': args.depth == 4,
        'wall_seconds_by_stage': timings,
        'timing_caveat': 'Single-run wall times; not a statistically validated speedup',
        'advisory_fastest_reference_worker_count': min((1, 2, 4, 8), key=lambda w: timings[f'golden_{w}w']),
        'permissions': 'no code mutation; read-only test execution; CI artifacts only',
    }
    (out / 'engineering_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--golden', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--depth', type=int, default=4)
    p.add_argument('--steps', type=int, default=24)
    run(p.parse_args())


if __name__ == '__main__':
    main()
