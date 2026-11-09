#!/usr/bin/env python3
"""Adversarial black-box regression for JX3D 2.2 reference containment.

Usage: python3 test_containment_v22.py /path/jx3d_cloud /path/golden.csv
"""
import csv
import json
import os
from pathlib import Path
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from containment_v22 import (Verdict, coordinates, fnv_add, state_digest,
                             supervise, verify_replay, release_verify, read_secret,
                             UnavailableProof, MAX_WORK)

class ContainmentTests(unittest.TestCase):
    binary = None
    golden = None

    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='jx3d-containment-')
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.key = self.root/'supervisor.key'
        self.key.write_bytes(os.urandom(32))
        self.key.chmod(0o600)

    def actual(self, name='real', depth=2, steps=24):
        target=self.root/name
        subprocess.run([str(self.binary),'--depth',str(depth),'--steps',str(steps),
                        '--workers','2','--events','none','--no-frames','--out',str(target)],
                       check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,timeout=40)
        return target

    def supervise_case(self,name='supervised',**kw):
        return supervise(self.binary,self.root/name,self.key,**kw)

    def forge_state_bundle(self, dir):
        chk_path=dir/'checkpoint.bin'
        buf=bytearray(chk_path.read_bytes())
        depth,step,n,_,parent=struct.unpack_from('<IIQQQ',buf,8)
        q=list(struct.unpack_from(f'<{n}H',buf,40))
        q[0] += 1
        struct.pack_into('<H',buf,40,q[0])
        hashed=state_digest(coordinates(depth),q,parent,step)
        struct.pack_into('<Q',buf,24,hashed)
        chk_path.write_bytes(buf)
        trace_path=dir/'trace.csv'
        with trace_path.open(newline='') as f:
            rows=list(csv.DictReader(f));fields=list(rows[0])
        rows[-1]['hash']=f'{hashed:016x}'
        rows[-1]['min']=f'{min(q)/65535:.8f}'
        rows[-1]['max']=f'{max(q)/65535:.8f}'
        rows[-1]['range']=f'{(max(q)-min(q))/65535:.8f}'
        with trace_path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n')
            w.writeheader();w.writerows(rows)
        report_path=dir/'run.json';report=json.loads(report_path.read_text())
        report['final_hash']=f'{hashed:016x}'
        report_path.write_text(json.dumps(report)+'\n')

    def fake_runtime(self,source):
        target=self.root/'fake.py'
        target.write_text('#!/usr/bin/env python3\nimport pathlib,shutil,sys\n'
                          'out=pathlib.Path(sys.argv[sys.argv.index("--out")+1])\n'
                          f'shutil.copytree(pathlib.Path({str(source)!r}),out,dirs_exist_ok=True)\n')
        target.chmod(0o755)
        return target

    def test_01_valid_golden_signed_release(self):
        result=self.supervise_case('good',golden=self.golden)
        self.assertEqual(result['verdict'],'TRUE')
        self.assertEqual(result['status'],'COMMITTED')
        self.assertTrue(release_verify(self.root/'good'/'verified',self.key)[0])

    def test_02_valid_without_golden_independent_replay(self):
        result=self.supervise_case('noncanonical',depth=2,steps=3,workers=1)
        self.assertEqual(result['verdict'],'TRUE')
        self.assertTrue(release_verify(self.root/'noncanonical'/'verified',self.key)[0])

    def test_03_self_consistent_forged_bundle_rejected_without_golden(self):
        forged=self.actual('forged')
        self.forge_state_bundle(forged)
        verdict,reason,_=verify_replay(forged,2,24)
        self.assertEqual(verdict,Verdict.FALSE,reason)

    def test_04_counterfeit_runtime_cannot_publish_forgery(self):
        forged=self.actual('forged')
        self.forge_state_bundle(forged)
        fake=self.fake_runtime(forged)
        result=supervise(fake,self.root/'counterfeit',self.key,attempts=1)
        self.assertEqual(result['status'],'ROLLBACK')
        self.assertFalse((self.root/'counterfeit'/'verified').exists())

    def test_05_watchdog_timeout_bounded_unknown(self):
        fake=self.root/'hang.py'
        fake.write_text('#!/usr/bin/env python3\nimport time\ntime.sleep(60)\n');fake.chmod(0o755)
        result=supervise(fake,self.root/'timeout',self.key,attempts=2,timeout=0.12)
        self.assertEqual(result['status'],'SAFE_HALT_UNKNOWN')
        self.assertEqual(len(result['attempts']),2)
        self.assertFalse((self.root/'timeout'/'verified').exists())

    def test_06_missing_evidence_unknown(self):
        fake=self.root/'empty.py';fake.write_text('#!/usr/bin/env python3\n');fake.chmod(0o755)
        result=supervise(fake,self.root/'missing',self.key,attempts=1)
        self.assertEqual(result['status'],'SAFE_HALT_UNKNOWN')

    def test_07_injected_arithmetic_error_is_false(self):
        result=self.supervise_case('arith_fault',injected=2)
        self.assertEqual(result['status'],'ROLLBACK')
        self.assertFalse((self.root/'arith_fault'/'verified').exists())

    def test_08_post_publication_mutation_detected(self):
        self.supervise_case('tamper',golden=self.golden)
        file=self.root/'tamper'/'verified'/'checkpoint.bin'
        changed=bytearray(file.read_bytes());changed[-1]^=1;file.write_bytes(changed)
        self.assertFalse(release_verify(self.root/'tamper'/'verified',self.key)[0])

    def test_09_signed_certificate_mutation_detected(self):
        self.supervise_case('cert',golden=self.golden)
        cert=self.root/'cert'/'verified'/'certificate.json'
        payload=json.loads(cert.read_text());payload['manifest']['workers']=99
        cert.write_text(json.dumps(payload))
        self.assertFalse(release_verify(self.root/'cert'/'verified',self.key)[0])

    def test_10_wrong_key_never_authenticates(self):
        self.supervise_case('wrong_key',golden=self.golden)
        other=self.root/'unrelated.key';other.write_bytes(os.urandom(32));other.chmod(0o600)
        self.assertFalse(release_verify(self.root/'wrong_key'/'verified',other)[0])

    def test_11_world_readable_secret_fails_closed(self):
        self.key.chmod(0o644)
        with self.assertRaises(ValueError):
            self.supervise_case('bad_permissions')

    def test_12_missing_golden_fails_unknown(self):
        result=self.supervise_case('nogolden',golden=self.root/'nonexistent.csv',attempts=1)
        self.assertEqual(result['status'],'SAFE_HALT_UNKNOWN')

    def test_13_oversized_proof_budget_unknown(self):
        with self.assertRaises(UnavailableProof):
            verify_replay(self.root/'missing',4,10000)

    def test_14_symlinked_evidence_rejected(self):
        orig=self.actual('origin')
        target=self.root/'symlinked';shutil.copytree(orig,target)
        (target/'checkpoint.bin').unlink()
        (target/'checkpoint.bin').symlink_to(orig/'checkpoint.bin')
        with self.assertRaises(ValueError):
            verify_replay(target,2,24)

    def test_15_preexisting_output_rejected(self):
        directory=self.root/'used';directory.mkdir();(directory/'file').write_text('data')
        with self.assertRaises(ValueError):
            supervise(self.binary,directory,self.key)

    def test_16_key_cannot_live_inside_publication_tree(self):
        output=self.root/'inside'
        output.mkdir()
        (output/'secret').write_bytes(os.urandom(32))
        (output/'secret').chmod(0o600)
        with self.assertRaises(ValueError):
            supervise(self.binary,output,output/'secret')

if __name__=='__main__':
    if len(sys.argv)!=3:
        raise SystemExit('Usage: test_containment_v22.py BINARY GOLDEN')
    ContainmentTests.binary=Path(sys.argv[1]).resolve()
    ContainmentTests.golden=Path(sys.argv[2]).resolve()
    sys.argv=[sys.argv[0]]
    unittest.main(verbosity=2)
