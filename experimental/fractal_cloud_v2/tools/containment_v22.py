#!/usr/bin/env python3
"""JARVIS-X Ω³ v2.2 reference containment supervisor.

Proof scope: authentic deterministic replay for the documented JX3D genesis,
with a protected HMAC release boundary *if* its secret is inaccessible to the
untrusted producer and consumers reverify at use. This is NOT a TEE, production
sandbox, remote verifier or formal proof of permission isolation.
"""
import argparse
import csv
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import stat
import struct
import subprocess
import time
from enum import Enum

Q = 65535
FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
MASK = (1 << 64) - 1
COORD_OFFSETS = ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1))
SCHEMA = 'jx3d.containment.v2.2'
MAX_WORK = 6_000_000


class Verdict(str, Enum):
    TRUE = 'TRUE'
    FALSE = 'FALSE'
    UNKNOWN = 'UNKNOWN'


class UnavailableProof(Exception):
    pass


def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def fnv_add(h, val, width):
    for i in range(width):
        h = ((h ^ ((val >> (8 * i)) & 255)) * FNV_PRIME) & MASK
    return h


def coordinates(depth):
    alphabet = [(x, y, z) for z in range(3) for y in range(3) for x in range(3)
                if (x == 1) + (y == 1) + (z == 1) <= 1]
    nodes = [(0, 0, 0)]
    for _ in range(depth):
        nodes = [(3 * x + a, 3 * y + b, 3 * z + c)
                 for x, y, z in nodes for a, b, c in alphabet]
    return nodes


def state_digest(coords, q, parent, step):
    h = fnv_add(fnv_add(FNV_OFFSET, parent, 8), step, 4)
    for (x, y, z), v in zip(coords, q):
        for d in (x, y, z, v):
            h = fnv_add(h, d, 4)
    return h


def load_regular_file(path, max_bytes):
    path = Path(path)
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > max_bytes:
        raise ValueError(f'nonregular or oversized evidence: {path.name}')
    data = path.read_bytes()
    if len(data) > max_bytes:
        raise ValueError(f'evidence exceeds size limit: {path.name}')
    return data


def verify_replay(attempt, depth, steps, golden=None, deadline=None):
    """Recompute every transition, not merely the digest of submitted data.

    The model's trusted scope is specifically the fixed genesis/recurrence of
    the source-controlled JX3D v2 kernel with no --resume and no custom input.
    """
    attempt = Path(attempt)
    n = 20 ** depth
    if n * max(1, steps) > MAX_WORK:
        raise UnavailableProof('replay proof budget exceeded; refuse to certify')
    try:
        chk = load_regular_file(attempt / 'checkpoint.bin', 40 + 2 * n)
        trace_bytes = load_regular_file(attempt / 'trace.csv', (steps + 2) * 320)
        run_bytes = load_regular_file(attempt / 'run.json', 16384)
    except FileNotFoundError as exc:
        raise UnavailableProof(f'critical evidence missing: {exc.filename}') from exc
    if len(chk) != 40 + 2 * n or chk[:8] != b'JX3DCHK2':
        return Verdict.FALSE, 'checkpoint shape or header contradiction', {}
    d, t, cnt, submitted_hash, submitted_parent = struct.unpack_from('<IIQQQ', chk, 8)
    if (d, t, cnt) != (depth, steps, n):
        return Verdict.FALSE, 'checkpoint metadata contradicts selected workload', {}
    final_q = struct.unpack_from(f'<{n}H', chk, 40)
    report = json.loads(run_bytes)
    if (report.get('schema') != 'jx3d.run.v2' or report.get('depth') != depth or
        report.get('steps_requested') != steps or report.get('steps_committed') != steps or
        report.get('resumed_at_step') != 0 or report.get('status') != 'COMMITTED' or
        report.get('nodes') != n or report.get('edges') != 2*(n - 8 ** depth) or
        report.get('actual_remote_workers') is not False):
        return Verdict.FALSE, 'runtime metadata inconsistent or unsupported', {}
    lines = list(csv.DictReader(trace_bytes.decode('utf-8').splitlines()))
    if len(lines) != steps + 1:
        return Verdict.FALSE, 'trace step count mismatch', {}
    if golden is not None:
        try:
            trusted = load_regular_file(golden, (steps + 2)*320)
        except FileNotFoundError as exc:
            raise UnavailableProof('trusted golden reference missing') from exc
        if trace_bytes != trusted:
            return Verdict.FALSE, 'canonical golden trace mismatch', {}
    coords = coordinates(depth)
    index = {c:i for i,c in enumerate(coords)}
    neighbors = [[index[c2] for dx,dy,dz in COORD_OFFSETS
                  if (c2 := (x+dx,y+dy,z+dz)) in index]
                 for x,y,z in coords]
    q = [(((17*x + 29*y + 43*z)%1009)*Q+504)//1008 for x,y,z in coords]
    prev, parent = 0, 0
    for step in range(steps+1):
        if deadline is not None and time.monotonic() > deadline:
            raise UnavailableProof('independent replay deadline expired')
        digest = state_digest(coords,q,prev,step)
        row = lines[step]
        if (row['hash'] != f'{digest:016x}' or row['status'] != ('INIT' if step==0 else 'COMMIT')
            or int(row['step']) != step or int(row['nodes']) != n
            or int(row['edges']) != 2*(n - 8 ** depth)):
            return Verdict.FALSE, f'independent transition/trace discrepancy at step {step}', {}
        if abs(float(row['min']) - min(q)/Q) > 1e-8 or abs(float(row['max'])-max(q)/Q)>1e-8:
            return Verdict.FALSE, f'trace bounds discrepancy at step {step}', {}
        if step == steps:
            parent = prev
            prev = digest
            break
        updated = [((3 * len(nb) * q[i] + sum(q[j] for j in nb) + 2*len(nb))//(4*len(nb))
                    if nb else q[i]) for i,nb in enumerate(neighbors)]
        q = updated
        prev = digest
    if tuple(q) != final_q or prev != submitted_hash or parent != submitted_parent:
        return Verdict.FALSE, 'final checkpoint differs from independently replayed state', {}
    if report.get('final_hash') != f'{prev:016x}':
        return Verdict.FALSE, 'final report hash mismatch', {}
    return Verdict.TRUE, 'all transitions independently replayed from fixed genesis', {
        'checkpoint_sha256': sha256(chk), 'trace_sha256': sha256(trace_bytes),
        'run_report_sha256': sha256(run_bytes), 'final_digest': f'{prev:016x}',
        'depth':depth,'steps':steps,'nodes':n,'edges':2*(n-8**depth),
        'proof_method':'full-independent-deterministic-replay-v1',
    }


def read_secret(key_file):
    p = Path(key_file)
    s = p.lstat()
    if not stat.S_ISREG(s.st_mode) or (s.st_mode & 0o077):
        raise ValueError('signing key must be a regular, owner-only file (mode 0600)')
    data = p.read_bytes()
    if len(data) < 32:
        raise ValueError('signing key must contain at least 32 random bytes')
    return data


def sign_manifest(manifest, key):
    return hmac.new(key, canonical_json(manifest), hashlib.sha256).hexdigest()


def release_verify(release, key_file):
    """Consumer-side check; must be performed immediately before use."""
    try:
        root = Path(release)
        doc = json.loads(load_regular_file(root/'certificate.json', 16384))
        manifest = doc['manifest']
        if doc.get('algorithm') != 'HMAC-SHA256' or manifest.get('schema') != SCHEMA:
            return False, 'certificate schema or algorithm mismatch'
        if not hmac.compare_digest(sign_manifest(manifest, read_secret(key_file)), doc['mac']):
            return False, 'release certificate MAC invalid'
        if (set(manifest['artifact_sha256']) != {'checkpoint.bin','trace.csv','run.json'}
                or manifest.get('proof_method') != 'full-independent-deterministic-replay-v1'
                or not (1 <= int(manifest['depth']) <= 4 and 0 <= int(manifest['steps']) <= 10000)):
            return False, 'release certificate missing mandatory evidence or invalid replay proof'
        for name, expected in manifest['artifact_sha256'].items():
            if name not in ('checkpoint.bin','trace.csv','run.json'):
                return False, 'unapproved artifact path'
            maximum = 40+2*20**int(manifest['depth']) if name=='checkpoint.bin' else (int(manifest['steps'])+2)*320 if name=='trace.csv' else 16384
            if sha256(load_regular_file(root/name,maximum)) != expected:
                return False, f'published {name} changed after verification'
        return True, 'authenticated release and files match the certificate'
    except (KeyError, ValueError, TypeError, OSError, json.JSONDecodeError) as exc:
        return False, f'release not verified: {exc}'


def execute_once(binary, attempt, depth, steps, workers, timeout, injected=None):
    cmd = [str(binary), '--depth',str(depth),'--steps',str(steps),'--workers',str(workers),
           '--events','none','--no-frames','--out',str(attempt)]
    if injected is not None:
        cmd += ['--inject-fault-step',str(injected)]
    child = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, start_new_session=True)
    t0 = time.monotonic()
    try:
        stdout,stderr = child.communicate(timeout=timeout)
        return child.returncode,stdout[-600:],stderr[-600:],time.monotonic()-t0
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.communicate()
        return None,'','watchdog timeout',time.monotonic()-t0


def supervise(binary, output, key_file, depth=2, steps=24, workers=2,
              attempts=2, timeout=30, replay_timeout=60, golden=None, injected=None):
    """Reference harness; OS-level producer/verifier isolation is NOT provided."""
    binary, output, key_file = Path(binary).resolve(),Path(output).resolve(),Path(key_file).resolve()
    if not binary.is_file():
        raise ValueError('binary missing')
    if output == key_file or output in key_file.parents:
        raise ValueError('signing secret must be outside output tree')
    key = read_secret(key_file)
    if not (1<=depth<=4 and 0<=steps<=10000 and 1<=workers<=64 and 1<=attempts<=5 and
            0<timeout<=600 and 0<replay_timeout<=600):
        raise ValueError('invalid bounded configuration')
    if output.exists() and any(output.iterdir()):
        raise ValueError('stale output is prohibited')
    output.mkdir(parents=True,exist_ok=True)
    log=[];final=Verdict.UNKNOWN; reason='missing evidence'; published=None
    for number in range(1,attempts+1):
        attempt = output/f'attempt_{number:02d}'
        attempt.mkdir()
        status,stdout,stderr,elapsed=execute_once(binary,attempt,depth,steps,workers,timeout,injected)
        info={}; verdict=Verdict.UNKNOWN
        if status is None:
            reason='runtime timeout'
        elif status==2:
            verdict=Verdict.FALSE;reason='runtime explicitly rolled back'
        elif status!=0:
            reason=f'runtime exited {status} without trustworthy evidence'
        else:
            try:
                verdict,reason,info=verify_replay(attempt,depth,steps,golden,
                                                 deadline=time.monotonic()+replay_timeout)
            except UnavailableProof as exc:
                verdict=Verdict.UNKNOWN;reason=str(exc)
            except (OSError,ValueError,KeyError,TypeError,IndexError,OverflowError,struct.error,
                    UnicodeError,ZeroDivisionError) as exc:
                verdict=Verdict.FALSE;reason=f'malformed or contradictory evidence: {exc}'
        log.append({'attempt':number,'verdict':verdict.value,'reason':reason,
                    'runtime_exit':status,'runtime_seconds':round(elapsed,6),
                    'stdout_tail':stdout,'stderr_tail':stderr})
        if verdict==Verdict.TRUE:
            temp=output/'.release-staging'
            temp.mkdir(mode=0o700)
            hashes={}
            for name in ('checkpoint.bin','trace.csv','run.json'):
                shutil.copyfile(attempt/name,temp/name)
                hashes[name]=sha256((temp/name).read_bytes())
            # Recheck copy hashes against the *independently observed* evidence.
            for name, digest_key in (('checkpoint.bin','checkpoint_sha256'),
                                      ('trace.csv','trace_sha256'),('run.json','run_report_sha256')):
                if hashes[name] != info[digest_key]:
                    raise RuntimeError('race: evidence changed during publication')
            manifest={'schema':SCHEMA,'depth':depth,'steps':steps,'workers':workers,
                      'artifact_sha256':hashes,'final_digest':info['final_digest'],
                      'proof_method':info['proof_method'],
                      'issuer':'reference-hmac-governor',
                      'unique_release_id':secrets.token_hex(16)}
            cert={'manifest':manifest,'algorithm':'HMAC-SHA256','mac':sign_manifest(manifest,key)}
            (temp/'certificate.json').write_bytes(canonical_json(cert)+b'\n')
            os.replace(temp,output/'verified')
            passed,explanation=release_verify(output/'verified',key_file)
            if not passed:
                raise RuntimeError('post-publication check failed: '+explanation)
            final=Verdict.TRUE; published=str(output/'verified');break
        if verdict==Verdict.FALSE:
            final=Verdict.FALSE;break
    status={Verdict.TRUE:'COMMITTED',Verdict.FALSE:'ROLLBACK',Verdict.UNKNOWN:'SAFE_HALT_UNKNOWN'}[final]
    decision={'schema':SCHEMA,'verdict':final.value,'status':status,
              'published_verified_directory':published,'attempts':log,
              'limitations':'reference same-host harness; no OS isolation, attestation, fsync or TEE'}
    (output/'decision.json').write_bytes(canonical_json(decision)+b'\n')
    return decision


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--key-file',type=Path,required=True)
    parser.add_argument('--depth',type=int,default=2)
    parser.add_argument('--steps',type=int,default=24)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--attempts',type=int,default=2)
    parser.add_argument('--timeout-seconds',type=float,default=30)
    parser.add_argument('--replay-timeout-seconds',type=float,default=60)
    parser.add_argument('--golden',type=Path)
    parser.add_argument('--inject-fault-step',type=int)
    parser.add_argument('--verify-release',type=Path)
    args=parser.parse_args()
    if args.verify_release:
        ok,why=release_verify(args.verify_release,args.key_file)
        print(json.dumps({'valid':ok,'explanation':why}))
        return 0 if ok else 2
    result=supervise(args.binary,args.output,args.key_file,args.depth,args.steps,args.workers,
                     args.attempts,args.timeout_seconds,args.replay_timeout_seconds,
                     args.golden,args.inject_fault_step)
    print(json.dumps({'verdict':result['verdict'],'status':result['status']}))
    return {'TRUE':0,'FALSE':2,'UNKNOWN':3}[result['verdict']]

if __name__=='__main__':
    raise SystemExit(main())