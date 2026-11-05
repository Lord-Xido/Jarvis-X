#!/usr/bin/env python3
"""Offline, dependency-free fail-closed validation for the G3M bytecode plan."""
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

ROOT = Path(__file__).resolve().parents[1]

class PlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="g3m_isa_test_")
        cls.work = Path(cls.tmp.name)
        compiler = shutil.which('g++') or shutil.which('clang++') or shutil.which('c++')
        if compiler is None:
            raise unittest.SkipTest('C++17 compiler unavailable')
        cls.exe = cls.work / 'g3m_disasm'
        subprocess.run([compiler,'-std=c++17','-O2','-Wall','-Wextra','-pedantic',str(ROOT/'g3m_disasm.cpp'),'-o',str(cls.exe)],check=True)
        subprocess.run([sys.executable,str(ROOT/'emit_plan.py')],check=True,stdout=subprocess.DEVNULL)
        cls.reference = (ROOT/'g3m_pipeline.g3mb').read_bytes()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def check(self, data, should_pass):
        path=self.work / 'candidate.g3mb'
        path.write_bytes(data)
        proc=subprocess.run([str(self.exe),str(path)],capture_output=True,text=True)
        self.assertEqual(proc.returncode==0,should_pass,proc.stdout+'\n'+proc.stderr)

    def repaired(self, data):
        buf=bytearray(data)
        struct.pack_into('<I',buf,24,zlib.crc32(buf[32:])&0xffffffff)
        return bytes(buf)

    def test_valid(self):
        self.assertEqual(len(self.reference),272)
        self.check(self.reference,True)
    def test_truncated(self):
        self.check(self.reference[:-1],False)
    def test_wrong_crc(self):
        b=bytearray(self.reference);b[35]^=1
        self.check(bytes(b),False)
    def test_illegal_opcode_with_repaired_crc(self):
        b=bytearray(self.reference);b[32]=0xEE
        self.check(self.repaired(b),False)
    def test_wrong_order_with_repaired_crc(self):
        b=bytearray(self.reference);b[32+16]=0x03
        self.check(self.repaired(b),False)
    def test_invalid_point_count_with_repaired_crc(self):
        b=bytearray(self.reference);struct.pack_into('<I',b,32+8,1000001)
        self.check(self.repaired(b),False)
    def test_invalid_knn_with_repaired_crc(self):
        # Op at pc=4: k cannot be >= support (128).
        b=bytearray(self.reference);struct.pack_into('<I',b,32+4*16+8,128)
        self.check(self.repaired(b),False)
    def test_invalid_source_register_with_repaired_crc(self):
        b=bytearray(self.reference);struct.pack_into('<H',b,32+5*16+4,15)
        self.check(self.repaired(b),False)
    def test_bad_version(self):
        b=bytearray(self.reference);struct.pack_into('<H',b,8,2)
        self.check(bytes(b),False)

if __name__ == '__main__':
    unittest.main(verbosity=2)
