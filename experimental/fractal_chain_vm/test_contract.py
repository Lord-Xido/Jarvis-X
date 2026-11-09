#!/usr/bin/env python3
import csv
import pathlib
import struct
import subprocess
import sys
import tempfile

exe = sys.argv[1]
with tempfile.TemporaryDirectory() as temp:
    folder = pathlib.Path(temp)
    def run(name, *args):
        return subprocess.run([exe, '--prefix', str(folder/name), *args], capture_output=True, text=True)
    first = run('first', '--dim', '8', '--levels', '3', '--iterations', '8')
    second = run('second', '--dim', '8', '--levels', '3', '--iterations', '8')
    assert first.returncode == second.returncode == 0, (first.stderr, second.stderr)
    assert 'levels=3 cycles=31' in first.stdout, first.stdout
    a = (folder/'first.jx3v').read_bytes()
    b = (folder/'second.jx3v').read_bytes()
    assert a == b, 'nondeterministic output within host/compiler'
    assert a[:4] == b'JX3V'
    assert struct.unpack('<5I',a[4:24]) == (1,8,8,8,512)
    assert len(a) == 24+512*4
    bytecode = (folder/'first.jxbc').read_bytes()
    assert bytecode[:4] == b'JXBC'
    assert struct.unpack('<II',bytecode[4:12]) == (1,11)
    assert len(bytecode) == 12+11*12
    with (folder/'first.csv').open() as f:
        rows = list(csv.DictReader(f))
    with (folder/'second.csv').open() as f:
        again = list(csv.DictReader(f))
    assert len(rows) == len(again) == 31
    assert [int(x['level']) for x in rows if int(x['opcode'])==0x27] == [0,1,2]
    assert sum(int(x['opcode'])==0x31 for x in rows)==3
    assert [(x['input_hash'],x['output_hash']) for x in rows] == [(x['input_hash'],x['output_hash']) for x in again]
    assert run('bad', '--dim', '9').returncode != 0
    assert run('badlevels', '--levels', '0').returncode != 0
    print('PASS: deterministic replay, 31 traced dispatches, portable binary formats, bad-input rejection')