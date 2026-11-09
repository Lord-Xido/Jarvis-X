#!/usr/bin/env python3
"""JARVIS-X Ω³ epistemic safety gate v2.1.

Independent process supervisor; never changes the locked C++ Q16 kernel.
TRUE = evidence verified; FALSE = contradiction/known failed arithmetic;
UNKNOWN = insufficient evidence, e.g. timeout or missing output.
Neither a second verifier on the same host nor an FNV checksum establishes
hardware fault independence or cryptographic authenticity.
"""
import argparse
import csv
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import struct
import subprocess
import time

FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
MASK64 = (1 << 64) - 1
HEADER = '<IIQQQ'
GOLDEN_HASH = '098794e9b04115c7'


class Verdict(str, Enum):
    TRUE = 'TRUE'
    FALSE = 'FALSE'
    UNKNOWN = 'UNKNOWN'


def digest(bytes_value):
    return hashlib.sha256(bytes_value).hexdigest()


def fnv_u(h, value, count):
    for k in range(count):
        h = ((h ^ ((value >> (8 * k)) & 255)) * FNV_PRIME) & MASK64
    return h


def fractal_coords(depth):
    alphabet = [(x, y, z) for z in range(3) for y in range(3)
                for x in range(3) if int(x == 1) + int(y == 1) + int(z == 1) <= 1]
    cells = [(0, 0, 0)]
    for _ in range(depth):
        cells = [(3 * x + dx, 3 * y + dy, 3 * z + dz)
                 for x, y, z in cells for dx, dy, dz in alphabet]
    return cells


def verify_evidence(root, depth, steps, golden=None):
    """Return (verdict, explanation, checkpoint SHA-256) without trusting run.json."""
    root = Path(root)
    try:
        report = json.loads((root / 'run.json').read_text())
        with (root / 'trace.csv').open(newline='') as inp:
            trace = list(csv.DictReader(inp))
        check = (root / 'checkpoint.bin').read_bytes()
    except (OSError, ValueError, KeyError) as exc:
        return Verdict.UNKNOWN, f'missing or incomplete execution evidence: {exc}', None

    if report.get('status') == 'ROLLBACK' or (trace and trace[-1].get('status') == 'ROLLBACK'):
        return Verdict.FALSE, 'runtime rejected its candidate and rolled back', digest(check)
    if report.get('status') != 'COMMITTED':
        return Verdict.UNKNOWN, 'runtime did not report a committed completion', digest(check)
    n = 20 ** depth
    edges = 2 * (20 ** depth - 8 ** depth)
    if (report.get('depth') != depth or report.get('nodes') != n or
            report.get('edges') != edges or report.get('steps_committed') != steps or
            report.get('steps_requested') != steps or
            report.get('actual_remote_workers') is not False):
        return Verdict.FALSE, 'runtime report contradicts requested geometry or execution', digest(check)
    if len(check) != 40 + 2 * n or check[:8] != b'JX3DCHK2':
        return Verdict.FALSE, 'checkpoint header or length mismatch', digest(check)
    d, t, cn, state_hash, parent_hash = struct.unpack_from(HEADER, check, 8)
    if (d, t, cn) != (depth, steps, n):
        return Verdict.FALSE, 'checkpoint geometry/step mismatch', digest(check)
    if not trace or len(trace) != steps + 1:
        return Verdict.FALSE, 'incomplete/noncanonical execution trace', digest(check)
    if trace[0].get('status') != 'INIT':
        return Verdict.FALSE, 'trace does not begin at INIT', digest(check)
    for expected_step, row in enumerate(trace):
        if (int(row['step']) != expected_step or int(row['nodes']) != n or
                int(row['edges']) != edges or
                row['status'] != ('INIT' if expected_step == 0 else 'COMMIT')):
            return Verdict.FALSE, 'trace sequence or geometry inconsistent', digest(check)
    if trace[-1].get('hash') != format(state_hash, '016x') or report.get('final_hash') != format(state_hash, '016x'):
        return Verdict.FALSE, 'trace/report/checkpoint hash mismatch', digest(check)

    state = struct.unpack_from(f'<{n}H', check, 40)
    if (abs(min(state) / 65535 - float(trace[-1]['min'])) > 0.00000002 or
            abs(max(state) / 65535 - float(trace[-1]['max'])) > 0.00000002):
        return Verdict.FALSE, 'trace bounds contradict committed state', digest(check)
    h = fnv_u(FNV_OFFSET, parent_hash, 8)
    h = fnv_u(h, steps, 4)
    for (x, y, z), q in zip(fractal_coords(depth), state):
        h = fnv_u(h, x, 4)
        h = fnv_u(h, y, 4)
        h = fnv_u(h, z, 4)
        h = fnv_u(h, q, 4)
    if h != state_hash:
        return Verdict.FALSE, 'independent Python recomputation rejected checkpoint digest', digest(check)
    if golden is not None:
        try:
            golden_data = Path(golden).read_bytes()
            trace_data = (root / 'trace.csv').read_bytes()
        except OSError as exc:
            return Verdict.UNKNOWN, f'golden comparison not available: {exc}', digest(check)
        if trace_data != golden_data:
            return Verdict.FALSE, 'canonical golden trace mismatch', digest(check)
        if depth == 2 and steps == 24 and format(state_hash, '016x') != GOLDEN_HASH:
            return Verdict.FALSE, 'known depth-2 baseline hash mismatch', digest(check)
    return Verdict.TRUE, 'independent final-checkpoint reconstruction and trace consistency verified', digest(check)


def execute_once(binary, attempt, depth, steps, workers, timeout, fault_step=None):
    cmd = [str(binary), '--depth', str(depth), '--steps', str(steps), '--workers',
           str(workers), '--events', 'none', '--no-frames', '--out', str(attempt)]
    if fault_step is not None:
        cmd += ['--inject-fault-step', str(fault_step)]
    started = time.monotonic()
    child = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, start_new_session=True)
    try:
        stdout, stderr = child.communicate(timeout=timeout)
        return child.returncode, round(time.monotonic() - started, 6), stdout[-1500:], stderr[-1500:]
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.communicate()
        return None, round(time.monotonic() - started, 6), '', 'watchdog deadline exceeded'


def atomic_json(path, payload):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, path)


def supervise(binary, output, depth, steps, workers, attempts, timeout, golden=None, fault_step=None):
    binary = Path(binary).resolve()
    output = Path(output).resolve()
    if not binary.is_file():
        raise ValueError('runtime binary not found')
    if not (1 <= depth <= 4 and 0 <= steps <= 10000 and 1 <= workers <= 64 and
            1 <= attempts <= 5 and 0 < timeout <= 600):
        raise ValueError('invalid bounded supervisor configuration')
    if output.exists() and any(output.iterdir()):
        raise ValueError('output must be empty; prevent stale evidence reuse')
    output.mkdir(parents=True, exist_ok=True)
    log = []
    final = Verdict.UNKNOWN
    explanation = 'insufficient evidence'
    verified = None
    for index in range(1, attempts + 1):
        directory = output / f'attempt_{index:02d}'
        directory.mkdir()
        exit_code, elapsed, stdout, stderr = execute_once(binary, directory, depth, steps, workers, timeout, fault_step)
        checkpoint_hash = None
        if exit_code is None:
            verdict, explanation = Verdict.UNKNOWN, 'execution watchdog timeout'
        elif exit_code == 2:
            verdict, explanation = Verdict.FALSE, 'runtime explicitly rejected a transition'
        elif exit_code != 0:
            verdict, explanation = Verdict.UNKNOWN, f'runtime exited {exit_code} without validated result'
        else:
            try:
                verdict, explanation, checkpoint_hash = verify_evidence(directory, depth, steps, golden)
            except (ValueError, TypeError, OverflowError, KeyError, struct.error) as exc:
                verdict, explanation = Verdict.FALSE, f'evidence contradiction: {exc}'
        log.append({'attempt': index, 'verdict': verdict.value, 'reason': explanation,
                    'exit_code': exit_code, 'elapsed_seconds': elapsed,
                    'stdout_tail': stdout, 'stderr_tail': stderr,
                    'checkpoint_sha256': checkpoint_hash})
        if verdict == Verdict.TRUE:
            # Publication is an explicit trust boundary; unverified attempts remain quarantined.
            temp = output / '.verified.tmp'
            shutil.copytree(directory, temp)
            os.replace(temp, output / 'verified')
            verified = str(output / 'verified')
            final = Verdict.TRUE
            break
        if verdict == Verdict.FALSE:
            final = Verdict.FALSE
            break
    status = 'COMMITTED' if final == Verdict.TRUE else ('ROLLBACK' if final == Verdict.FALSE else 'SAFE_HALT_UNKNOWN')
    result = {'schema': 'jx3d.epistemic.v1', 'verdict': final.value, 'status': status,
              'depth': depth, 'steps': steps, 'workers': workers,
              'max_attempts': attempts, 'timeout_seconds_per_attempt': timeout,
              'attempts': log, 'published_verified_directory': verified,
              'limitations': 'FNV-1a and duplicated same-host computation are not independent cryptographic or hardware proofs'}
    atomic_json(output / 'epistemic_decision.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--golden', type=Path)
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--steps', type=int, default=24)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--attempts', type=int, default=2)
    p.add_argument('--timeout-seconds', type=float, default=30)
    p.add_argument('--inject-fault-step', type=int)
    a = p.parse_args()
    result = supervise(a.binary, a.output, a.depth, a.steps, a.workers,
                       a.attempts, a.timeout_seconds, a.golden, a.inject_fault_step)
    print(json.dumps({'verdict': result['verdict'], 'status': result['status'],
                      'attempts_executed': len(result['attempts'])}))
    return {Verdict.TRUE.value: 0, Verdict.FALSE.value: 2, Verdict.UNKNOWN.value: 3}[result['verdict']]


if __name__ == '__main__':
    raise SystemExit(main())