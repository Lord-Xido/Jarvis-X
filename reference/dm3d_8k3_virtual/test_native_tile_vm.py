"""Native C++ JX8KT1 consumer: atomic local sandbox and cross-runtime evidence tests."""
import hashlib
import json
import shutil
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

from native_receipt_link import link

ROOT = Path(__file__).resolve().parent


class NativeConsumerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which('g++') or shutil.which('clang++')
        if not compiler:
            raise RuntimeError('C++17 compiler required')
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.native = cls.root / 'native'
        cls.codec = cls.root / 'codec'
        options = ['-std=c++17', '-O2', '-Wall', '-Wextra', '-Wpedantic']
        subprocess.run([compiler, *options, str(ROOT/'native_tile_vm.cpp'), '-o', str(cls.native)], check=True)
        subprocess.run([compiler, *options, str(ROOT/'jarvis_x_8k_virtual.cpp'), '-o', str(cls.codec)], check=True)
        cls.report = cls.root / 'report.json'
        cls.raw = cls.root / 'raw.bin'
        cls.packet = cls.root / 'encoded.jx8k'
        subprocess.run([str(cls.codec), '--rounds', '3', '--budget', '64', '--candidates', '256', '--seed', '42',
                        '--report', str(cls.report), '--tiles', str(cls.raw), '--encoded', str(cls.packet)],
                       check=True, stdout=subprocess.DEVNULL)
        cls.metadata = json.loads(cls.report.read_text())
        cls.raw_data = cls.raw.read_bytes()
        cls.packet_data = cls.packet.read_bytes()

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def invoke(self, *additional):
        return subprocess.run([str(self.native), '--encoded', str(self.packet), '--raw', str(self.raw), *additional],
                              capture_output=True, text=True)

    def test_compile_selftest_and_192_tile_commit(self):
        self.assertEqual(subprocess.run([str(self.native), '--self-test'],capture_output=True).returncode,0)
        p = self.invoke()
        self.assertEqual(p.returncode,0,msg=p.stderr)
        receipt=json.loads(p.stdout)
        self.assertEqual(receipt['tiles_requested'],192)
        self.assertEqual(receipt['instruction_steps'],1153)
        self.assertTrue(receipt['sandbox_committed'])
        self.assertEqual(receipt['sandbox_records_after'],193)
        self.assertFalse(receipt['production_state_modified'])
        result=link(receipt, self.metadata, self.packet_data, self.raw_data)
        self.assertEqual(result['status'],'PASS')
        self.assertFalse(result['commit_eligible'])
        self.assertEqual(result['tiles_verified'],192)

    def test_rollback_after_staging_first_and_last(self):
        for idx in (0,16,191):
            with self.subTest(injected_rejection_at=idx):
                p=self.invoke('--force-reject-at',str(idx))
                self.assertEqual(p.returncode,2,msg=p.stderr)
                r=json.loads(p.stdout)
                self.assertTrue(r['rolled_back'])
                self.assertTrue(r['state_unchanged_after_rollback'])
                self.assertFalse(r['sandbox_committed'])
                self.assertFalse(r['production_state_modified'])
                self.assertEqual(r['tiles_staged'],idx+1)
                self.assertEqual(r['sandbox_records_after'],1)
                with self.assertRaises(ValueError):
                    link(r,self.metadata,self.packet_data,self.raw_data)

    def test_bad_binary_and_modified_program_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            tmp=Path(td)
            for name,b in [('magic',b'garbage!'+self.packet_data[8:]),
                           ('truncated',self.packet_data[:-1]),
                           ('suffix',self.packet_data+b'junk')]:
                path=tmp/(name+'.jx8k');path.write_bytes(b)
                proc=subprocess.run([str(self.native),'--encoded',str(path),'--raw',str(self.raw)],capture_output=True)
                self.assertNotEqual(proc.returncode,0)
            badprog=tmp/'evil.bc';badprog.write_bytes(b'\x75'+b'\0'*55)
            p=self.invoke('--program',str(badprog))
            self.assertNotEqual(p.returncode,0)
            self.assertNotIn('sandbox_committed',p.stdout)
            self.assertNotEqual(self.invoke('--force-reject-at','192').returncode,0)

    def test_integrity_crosscheck_detects_forged_metadata(self):
        native=json.loads(self.invoke().stdout)
        for bad in [dict(native,production_state_modified=True),
                    dict(native,instruction_steps=0),
                    dict(native,tiles_staged=191),
                    dict(native,observations=native['observations'][:-1])]:
            with self.assertRaises(ValueError):
                link(bad,self.metadata,self.packet_data,self.raw_data)
        clone=json.loads(json.dumps(native))
        clone['observations'][-1]['counts'][0]+=1
        with self.assertRaises(ValueError):
            link(clone,self.metadata,self.packet_data,self.raw_data)


if __name__=='__main__':
    unittest.main()
