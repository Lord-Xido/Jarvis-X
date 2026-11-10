#!/usr/bin/env python3
"""JARVIS-X Ω³ v2.3: provenance-bound release + fail-closed consuming gateway.

The producer does not possess signing authority. The *reference* publisher
snapshots untrusted evidence, invokes independent v2.2 mathematical replay,
and signs the validated snapshot with Ed25519. Consumers use pinned public
keys and verify immutable bytes read from file descriptors before returning
checkpoint data. OS isolation / storage ACLs must be supplied by deployment.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile
import time

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from containment_v22 import Verdict, UnavailableProof, verify_replay

SCHEMA = 'jx3d.release.v2.3'
POLICY_SCHEMA = 'jx3d.trust.v2.3'
EVIDENCE = ('checkpoint.bin', 'trace.csv', 'run.json')
HASH_REGEX = re.compile(r'^[0-9a-f]{64}$')
KEY_REGEX = re.compile(r'^[0-9a-f]{32}$')
HEX_SIGNATURE = re.compile(r'^[0-9a-f]{128}$')
ALLOWED_PROOF = 'full-independent-deterministic-replay-v1'
READ_CAPS = {'certificate.json': 16384, 'trace.csv': 3_300_000, 'run.json': 16384, 'checkpoint.bin': 400_040}

class VerificationError(Exception):
    pass

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('ascii')

def hex_sha(data):
    return hashlib.sha256(data).hexdigest()

def strict_json(data):
    def dedupe(pairs):
        result = {}
        for name, value in pairs:
            if name in result:
                raise VerificationError('duplicate JSON field: ' + name)
            result[name] = value
        return result
    try:
        return json.loads(data.decode('utf-8'), object_pairs_hook=dedupe,
                          parse_constant=lambda s: (_ for _ in ()).throw(VerificationError('non-finite JSON')))
    except (ValueError, UnicodeError) as e:
        raise VerificationError('invalid UTF-8 JSON: ' + str(e)) from e

def read_from_dirfd(fd, name, limit):
    if '/' in name or not name or name in ('.', '..'):
        raise VerificationError('unapproved artifact filename')
    flags = os.O_RDONLY | getattr(os, 'O_CLOEXEC', 0) | getattr(os, 'O_NOFOLLOW', 0)
    sub = os.open(name, flags, dir_fd=fd)
    try:
        meta = os.fstat(sub)
        if not stat.S_ISREG(meta.st_mode) or meta.st_size > limit or meta.st_size < 0:
            raise VerificationError('unsafe or oversized file: '+name)
        parts = []; size = 0
        while True:
            block = os.read(sub, min(65536,limit + 1 - size))
            if not block: break
            size += len(block)
            if size > limit: raise VerificationError('oversized streamed file: '+name)
            parts.append(block)
        if size != meta.st_size:
            raise VerificationError('artifact changed size during reading: '+name)
        return b''.join(parts)
    finally:
        os.close(sub)

def directory_fd(path):
    return os.open(path, os.O_RDONLY | getattr(os,'O_CLOEXEC',0) | getattr(os,'O_DIRECTORY',0) | getattr(os,'O_NOFOLLOW',0))

def read_snapshot(path, caps):
    fd = directory_fd(path)
    try:
        return {name: read_from_dirfd(fd, name,limit) for name,limit in caps.items()}
    finally:
        os.close(fd)

def load_signer(private_file):
    private_file = Path(private_file)
    parent = directory_fd(private_file.parent)
    try:
        stat_parent = os.fstat(parent)
        if stat_parent.st_mode & 0o022: raise VerificationError('signing key parent is group/other writable')
        fd = os.open(private_file.name, os.O_RDONLY | getattr(os,'O_NOFOLLOW',0), dir_fd=parent)
        try:
            s = os.fstat(fd)
            if not stat.S_ISREG(s.st_mode) or s.st_uid != os.geteuid() or (s.st_mode & 0o077) or s.st_size != 32:
                raise VerificationError('signing key must be 32-byte owner-only regular file')
            secret = os.read(fd,33)
            if len(secret) != 32: raise VerificationError('invalid signing-key length')
        finally: os.close(fd)
    finally: os.close(parent)
    return Ed25519PrivateKey.from_private_bytes(secret)

def signer_identity(private):
    public = private.public_key().public_bytes(Encoding.Raw,PublicFormat.Raw)
    return hex_sha(public)[:32], public

def validate_policy(policy):
    if not isinstance(policy, dict) or set(policy) != {'schema','trusted_keys','minimum_sequence'}:
        raise VerificationError('invalid trust policy shape')
    if policy['schema'] != POLICY_SCHEMA or type(policy['minimum_sequence']) is not int or policy['minimum_sequence'] < 0:
        raise VerificationError('trust policy schema or minimum sequence invalid')
    keys = policy['trusted_keys']
    if not isinstance(keys,dict) or not keys:
        raise VerificationError('no independent trust anchors')
    for kid, hexpub in keys.items():
        if not isinstance(kid,str) or not KEY_REGEX.fullmatch(kid) or not isinstance(hexpub,str) or not re.fullmatch(r'[0-9a-f]{64}',hexpub):
            raise VerificationError('bad public key entry')
        if hex_sha(bytes.fromhex(hexpub))[:32] != kid: raise VerificationError('public key identifier mismatch')
    return policy

def read_policy(policy_path):
    p=Path(policy_path)
    data=read_snapshot(p.parent,{p.name:64_000})[p.name]
    return validate_policy(strict_json(data))

def validated_manifest(manifest):
    required={'schema','key_id','sequence','release_id','depth','steps','nodes','final_digest',
              'proof_method','artifact_sha256','model'}
    if not isinstance(manifest,dict) or set(manifest)!=required:
        raise VerificationError('manifest has missing or unexpected fields')
    if manifest['schema']!=SCHEMA or manifest['proof_method']!=ALLOWED_PROOF or manifest['model']!='fixed-genesis-q16-six-neighbour':
        raise VerificationError('unsupported signed verification model')
    if not isinstance(manifest['key_id'],str) or not KEY_REGEX.fullmatch(manifest['key_id']):
        raise VerificationError('unrecognised key identifier syntax')
    if type(manifest['sequence']) is not int or manifest['sequence']<0 or not isinstance(manifest['release_id'],str) or not re.fullmatch(r'[0-9a-f]{32}',manifest['release_id']):
        raise VerificationError('invalid sequence or release identifier')
    depth=manifest['depth'];steps=manifest['steps']
    if type(depth) is not int or type(steps) is not int or not 1<=depth<=4 or not 0<=steps<=10000:
        raise VerificationError('invalid workload dimensions')
    if type(manifest['nodes']) is not int or manifest['nodes']!=20**depth:
        raise VerificationError('invalid number of cells')
    if not isinstance(manifest['final_digest'],str) or not re.fullmatch(r'[0-9a-f]{16}',manifest['final_digest']):
        raise VerificationError('invalid final digest')
    artifacts=manifest['artifact_sha256']
    if not isinstance(artifacts,dict) or set(artifacts)!=set(EVIDENCE) or not all(isinstance(v,str) and HASH_REGEX.fullmatch(v) for v in artifacts.values()):
        raise VerificationError('invalid or incomplete artifact digests')
    return manifest

def read_release(release_dir, trust_policy):
    """Return (immutable checkpoint bytes, authenticated manifest), not file paths.

    Enforce trust using a separately controlled pinned public-key policy.
    Captures exactly the bytes checked, eliminating post-check file-reopen TOCTOU
    for the returned in-memory value; this does not prevent other consumers
    from improperly reading the filesystem themselves.
    """
    policy=validate_policy(trust_policy)
    fd=directory_fd(release_dir)
    try:
        cert=strict_json(read_from_dirfd(fd,'certificate.json',READ_CAPS['certificate.json']))
        if not isinstance(cert,dict) or set(cert)!= {'algorithm','manifest','signature'} or cert['algorithm']!='Ed25519':
            raise VerificationError('incorrect certificate format')
        manifest=validated_manifest(cert['manifest'])
        kid=manifest['key_id']
        if kid not in policy['trusted_keys']:raise VerificationError('signer absent from independently pinned policy')
        if manifest['sequence']<policy['minimum_sequence']:raise VerificationError('release below independently maintained minimum sequence')
        sig=cert['signature']
        if not isinstance(sig,str) or not HEX_SIGNATURE.fullmatch(sig):raise VerificationError('invalid Ed25519 signature encoding')
        public=Ed25519PublicKey.from_public_bytes(bytes.fromhex(policy['trusted_keys'][kid]))
        try: public.verify(bytes.fromhex(sig), canonical(manifest))
        except InvalidSignature as e: raise VerificationError('release certificate signature invalid') from e
        checked={}
        for name in EVIDENCE:
            cap=READ_CAPS[name] if name != 'checkpoint.bin' else 40+2*20**manifest['depth']
            data=read_from_dirfd(fd,name,cap)
            if hex_sha(data)!=manifest['artifact_sha256'][name]:
                raise VerificationError('release artifact modified after signing: '+name)
            checked[name]=data
        chk=checked['checkpoint.bin']
        if len(chk)!=40+2*20**manifest['depth'] or chk[:8]!=b'JX3DCHK2':
            raise VerificationError('certified checkpoint geometry invalid')
        import struct
        d,t,n,h,_=struct.unpack_from('<IIQQQ',chk,8)
        if (d,t,n)!=(manifest['depth'],manifest['steps'],manifest['nodes']) or f'{h:016x}'!=manifest['final_digest']:
            raise VerificationError('signed manifest/checkpoint disagreement')
        return bytes(chk),dict(manifest)
    finally:
        os.close(fd)

def create_release(attempt_dir, releases_root, signer_file, sequence, depth, steps, golden=None, proof_timeout=120):
    """Trustworthy publication requires a protected governor OS identity.

    Note: the caller must ensure *only* the governor can write releases_root
    and the producer cannot read signer_file. This library cannot enforce
    deployment-wide policy from a shared process identity.
    """
    if type(sequence) is not int or sequence < 0 or type(depth) is not int or not 1<=depth<=4 or type(steps) is not int or not 0<=steps<=10000:
        raise VerificationError('invalid publisher workload configuration')
    private=load_signer(signer_file)
    key_id,_=signer_identity(private)
    releases_root=Path(releases_root)
    releases_root.mkdir(mode=0o700,parents=False,exist_ok=True)
    rfd=directory_fd(releases_root)
    try:
        sr=os.fstat(rfd)
        if sr.st_uid != os.geteuid() or sr.st_mode & 0o022 or sr.st_mode & 0o007:
            raise VerificationError('publication root must be governor-owned, not writable by group/others, and not accessible to others')
    finally:os.close(rfd)
    release_id=os.urandom(16).hex()
    stage=Path(tempfile.mkdtemp(prefix='.candidate-',dir=releases_root))
    try:
        # Snapshot from untrusted producer via no-follow, FD-based reads.
        snapshots=read_snapshot(attempt_dir,{
            'checkpoint.bin':40+2*20**depth,
            'trace.csv':(steps+2)*320,
            'run.json':16384,
        })
        for name,data in snapshots.items():
            with (stage/name).open('xb') as f:
                f.write(data);f.flush();os.fsync(f.fileno())
        # Independently validate the *staged snapshot* being signed, not
        # mutable original files which might change during replay.
        verdict,reason,proof=verify_replay(stage,depth,steps,golden,
                                           deadline=time.monotonic()+proof_timeout)
        if verdict!=Verdict.TRUE:
            raise VerificationError(f'replay refused {verdict.value}: {reason}')
        manifest={'schema':SCHEMA,'key_id':key_id,'sequence':sequence,'release_id':release_id,
                  'depth':depth,'steps':steps,'nodes':proof['nodes'],
                  'final_digest':proof['final_digest'],'proof_method':proof['proof_method'],
                  'model':'fixed-genesis-q16-six-neighbour',
                  'artifact_sha256':{name:hex_sha(data) for name,data in snapshots.items()}}
        validated_manifest(manifest)
        certificate={'algorithm':'Ed25519','manifest':manifest,
                     'signature':private.sign(canonical(manifest)).hex()}
        with (stage/'certificate.json').open('xb') as f:
            f.write(canonical(certificate)+b'\n');f.flush();os.fsync(f.fileno())
        # Optionally permit a separate read-only consumer group: a preprovisioned
        # setgid releases_root passes its group to this new stage directory.
        # No producer/group write permission is ever granted.
        for name in (*EVIDENCE, 'certificate.json'):
            os.chmod(stage/name, 0o640)
        os.chmod(stage, 0o750)
        # Flush staged directory entries before atomic rename.
        stagefd=directory_fd(stage)
        try:os.fsync(stagefd)
        finally:os.close(stagefd)
        dest=releases_root/release_id
        os.rename(stage,dest)
        parentfd=directory_fd(releases_root)
        try:os.fsync(parentfd)
        finally:os.close(parentfd)
        return dest,manifest
    finally:
        if stage.exists():shutil.rmtree(stage)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    issue=sub.add_parser('publish',help='verify untrusted snapshot and issue signed release')
    issue.add_argument('--attempt',type=Path,required=True)
    issue.add_argument('--releases-root',type=Path,required=True)
    issue.add_argument('--signer',type=Path,required=True)
    issue.add_argument('--sequence',type=int,required=True)
    issue.add_argument('--depth',type=int,required=True)
    issue.add_argument('--steps',type=int,required=True)
    issue.add_argument('--golden',type=Path)
    issue.add_argument('--proof-timeout',type=float,default=120)
    consume=sub.add_parser('consume',help='verify a release with pinned public policy and return bytes')
    consume.add_argument('--release',type=Path,required=True)
    consume.add_argument('--trust-policy',type=Path,required=True)
    consume.add_argument('--check-only',action='store_true')
    a=p.parse_args(argv)
    try:
        if a.command=='publish':
            path,manifest=create_release(a.attempt,a.releases_root,a.signer,a.sequence,a.depth,a.steps,a.golden,a.proof_timeout)
            print(json.dumps({'released':str(path),'key_id':manifest['key_id'],'sequence':manifest['sequence']}))
        else:
            checkpoint,manifest=read_release(a.release,read_policy(a.trust_policy))
            if a.check_only: print(json.dumps({'verified':True,'sequence':manifest['sequence'],'sha256':hex_sha(checkpoint)}))
            else: sys.stdout.buffer.write(checkpoint)
        return 0
    except (VerificationError,UnavailableProof,ValueError,OSError) as exc:
        print(json.dumps({'verified':False,'reason':str(exc)}),file=sys.stderr)
        return 3

if __name__=='__main__':
    raise SystemExit(main())
