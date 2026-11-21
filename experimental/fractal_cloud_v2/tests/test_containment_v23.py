#!/usr/bin/env python3
"""Adversarial regression for v2.3 separate publisher / consumer trust boundary.

Usage: python3 test_containment_v23.py ENGINE_BI N GOLDEN_CSV
"""
import csv
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding,PublicFormat,PrivateFormat,NoEncryption

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from containment_v23 import (VerificationError, create_release, read_release,
                             signer_identity, strict_json, read_policy, canonical,
                             validated_manifest, hex_sha, read_snapshot)
from containment_v22 import coordinates,state_digest

class Containment23(unittest.TestCase):
    binary=None
    golden=None

    def setUp(self):
        t=tempfile.TemporaryDirectory(prefix='jx3d-v23-test-')
        self.addCleanup(t.cleanup)
        self.root=Path(t.name)
        self.producer=self.root/'producer';self.producer.mkdir()
        self.secure=self.root/'secure';self.secure.mkdir(mode=0o700)
        self.public=self.root/'public';self.public.mkdir()
        self.releases=self.secure/'releases';self.releases.mkdir(mode=0o700)
        self.private=Ed25519PrivateKey.generate()
        self.key=self.secure/'signer.key'
        self.key.write_bytes(self.private.private_bytes(Encoding.Raw,PrivateFormat.Raw,NoEncryption()))
        self.key.chmod(0o600)
        self.key_id,self.public_key=signer_identity(self.private)
        self.policy={'schema':'jx3d.trust.v2.3','minimum_sequence':0,
                     'trusted_keys':{self.key_id:self.public_key.hex()}}
        self.policy_file=self.public/'trust.json'
        self.policy_file.write_bytes(canonical(self.policy))
        self._actual=None

    def run_actual(self):
        if self._actual is not None:return self._actual
        out=self.producer/'attempt';
        subprocess.run([str(self.binary),'--depth','2','--steps','24','--workers','2',
                        '--events','none','--no-frames','--out',str(out)],
                       check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,timeout=30)
        self._actual=out
        return out

    def release(self, **kw):
        kw.setdefault('attempt_dir',self.run_actual())
        kw.setdefault('releases_root',self.releases)
        kw.setdefault('signer_file',self.key)
        kw.setdefault('sequence',3)
        kw.setdefault('depth',2)
        kw.setdefault('steps',24)
        kw.setdefault('golden',self.golden)
        return create_release(**kw)[0]

    def consumer(self,path,policy=None):
        return read_release(path,self.policy if policy is None else policy)

    def test_01_genuine_signed_release_consumed(self):
        path=self.release()
        data,manifest=self.consumer(path)
        self.assertEqual(len(data),840)
        self.assertEqual(manifest['sequence'],3)
        self.assertEqual(hex_sha(data),manifest['artifact_sha256']['checkpoint.bin'])

    def test_02_valid_no_golden_independent_replay(self):
        path=self.release(golden=None)
        self.assertEqual(self.consumer(path)[1]['steps'],24)

    def test_03_forgery_recomputed_hash_rejected(self):
        orig=self.run_actual()
        b=bytearray((orig/'checkpoint.bin').read_bytes())
        d,t,n,h,parent=struct.unpack_from('<IIQQQ',b,8)
        q=list(struct.unpack_from(f'<{n}H',b,40));q[0]+=1
        struct.pack_into('<H',b,40,q[0]);newhash=state_digest(coordinates(d),q,parent,t)
        struct.pack_into('<Q',b,24,newhash)
        (orig/'checkpoint.bin').write_bytes(b)
        with (orig/'trace.csv').open(newline='') as f:
            reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
        rows[-1]['hash']=f'{newhash:016x}'
        rows[-1]['min']=f'{min(q)/65535:.8f}'
        rows[-1]['max']=f'{max(q)/65535:.8f}'
        rows[-1]['range']=f'{(max(q)-min(q))/65535:.8f}'
        with (orig/'trace.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)
        report=json.loads((orig/'run.json').read_text());report['final_hash']=f'{newhash:016x}'
        (orig/'run.json').write_text(json.dumps(report)+'\n')
        with self.assertRaises(VerificationError):self.release(golden=None)
        self.assertEqual(list(self.releases.iterdir()),[])

    def test_04_modified_artifact_rejected_at_consume(self):
        path=self.release()
        target=path/'checkpoint.bin';b=bytearray(target.read_bytes());b[-1]^=1;target.write_bytes(b)
        with self.assertRaisesRegex(VerificationError,'modified'):self.consumer(path)

    def test_05_modified_certificate_rejected(self):
        path=self.release();cert=path/'certificate.json';d=json.loads(cert.read_text());d['manifest']['steps']=100
        cert.write_bytes(canonical(d))
        with self.assertRaises(VerificationError):self.consumer(path)

    def test_06_wrong_public_key_not_accepted(self):
        path=self.release();other=Ed25519PrivateKey.generate()
        wrong_id,wrong_pub=signer_identity(other)
        with self.assertRaises(VerificationError):self.consumer(path,{'schema':'jx3d.trust.v2.3','minimum_sequence':0,'trusted_keys':{wrong_id:wrong_pub.hex()}})

    def test_07_replay_window_prevents_old_certificate(self):
        path=self.release(sequence=3)
        reject={**self.policy,'minimum_sequence':4}
        with self.assertRaisesRegex(VerificationError,'minimum sequence'):self.consumer(path,reject)

    def test_08_signer_revocation_rejects_previous_release(self):
        path=self.release()
        p=Ed25519PrivateKey.generate();kid,pub=signer_identity(p)
        revoke={**self.policy,'trusted_keys':{kid:pub.hex()}}
        with self.assertRaises(VerificationError):self.consumer(path,revoke)

    def test_09_symlinked_checkpoint_rejected(self):
        path=self.release();target=path/'checkpoint.bin';other=self.root/'checkpoint-copy'
        shutil.copyfile(target,other);target.unlink();target.symlink_to(other)
        with self.assertRaises(OSError):self.consumer(path)

    def test_10_signing_key_permissions_required(self):
        self.key.chmod(0o644)
        with self.assertRaises(VerificationError):self.release()

    def test_11_publication_directory_permissions_required(self):
        self.releases.chmod(0o777)
        with self.assertRaises(VerificationError):self.release()

    def test_12_consumer_receives_verified_bytes_not_mutable_file_path(self):
        path=self.release();data,manifest=self.consumer(path)
        (path/'checkpoint.bin').write_bytes(b'corrupt later')
        self.assertEqual(hex_sha(data),manifest['artifact_sha256']['checkpoint.bin'])
        with self.assertRaises(VerificationError):self.consumer(path)

    def test_13_unsigned_producer_checkpoint_cannot_be_consumed(self):
        attempt=self.run_actual()
        with self.assertRaises(OSError):self.consumer(attempt)

    def test_14_unauthorised_public_key_substitution_is_rejected(self):
        path=self.release()
        fake_private=Ed25519PrivateKey.generate();kid,pub=signer_identity(fake_private)
        cert=json.loads((path/'certificate.json').read_text());cert['manifest']['key_id']=kid
        cert['signature']=fake_private.sign(canonical(cert['manifest'])).hex()
        (path/'certificate.json').write_bytes(canonical(cert))
        with self.assertRaisesRegex(VerificationError,'absent'):self.consumer(path)

    def test_15_duplicate_json_fields_rejected(self):
        with self.assertRaisesRegex(VerificationError,'duplicate'):
            strict_json(b'{"algorithm":"Ed25519","algorithm":"HMAC"}')

    def test_16_bad_issuer_identity_in_trust_policy_rejected(self):
        malformed={**self.policy, 'trusted_keys':{'0'*32:self.public_key.hex()}}
        with self.assertRaises(VerificationError):self.consumer(self.release(),malformed)

    def test_17_decision_not_required_from_untrusted_producer(self):
        path=self.release()
        (self.run_actual()/'run.json').write_text('{}')
        self.assertTrue(self.consumer(path)[0].startswith(b'JX3DCHK2'))

    def test_18_command_line_consume_only_outputs_raw_verified_checkpoint(self):
        path=self.release()
        script=Path(__file__).resolve().parents[1]/'tools'/'containment_v23.py'
        r=subprocess.run([sys.executable,str(script),'consume','--release',str(path),
                          '--trust-policy',str(self.policy_file)],capture_output=True)
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertEqual(len(r.stdout),840)
        self.assertTrue(r.stdout.startswith(b'JX3DCHK2'))

    def test_19_command_line_failure_returns_nonzero_no_payload(self):
        path=self.release(); self.policy_file.write_text('{}')
        script=Path(__file__).resolve().parents[1]/'tools'/'containment_v23.py'
        r=subprocess.run([sys.executable,str(script),'consume','--release',str(path),
                          '--trust-policy',str(self.policy_file)],capture_output=True)
        self.assertNotEqual(r.returncode,0)
        self.assertEqual(r.stdout,b'')

    def test_20_group_readable_but_not_group_writable_publication(self):
        self.releases.chmod(0o2750)
        path=self.release()
        self.assertEqual(path.stat().st_mode & 0o777, 0o750)
        self.assertEqual((path/'checkpoint.bin').stat().st_mode & 0o777, 0o640)
        self.assertEqual((path/'certificate.json').stat().st_mode & 0o777, 0o640)
        self.assertTrue(self.consumer(path)[0].startswith(b'JX3DCHK2'))

    def test_21_symlinked_trust_policy_rejected(self):
        self.release()
        target=self.public/'copy.json'; target.write_bytes(self.policy_file.read_bytes())
        self.policy_file.unlink();self.policy_file.symlink_to(target)
        with self.assertRaises(OSError):read_policy(self.policy_file)

if __name__=='__main__':
    if len(sys.argv)!=3:
        raise SystemExit('usage: python test_containment_v23.py ENGINE_BINARY GOLDEN_CSV')
    Containment23.binary=Path(sys.argv[1]).resolve()
    Containment23.golden=Path(sys.argv[2]).resolve()
    sys.argv=[sys.argv[0]]
    unittest.main(verbosity=2)
