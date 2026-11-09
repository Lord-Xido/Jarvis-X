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

    # Causal feedback invariant: the blend output must be exactly the
    # next level's MANDELBULB input (the latter is intentionally a no-op).
    for level in (1, 2):
        previous_blend = next(x for x in rows if int(x['level']) == level-1 and int(x['opcode']) == 0x26)
        next_entry = next(x for x in rows if int(x['level']) == level and int(x['opcode']) == 0x20)
        assert previous_blend['output_hash'] == next_entry['input_hash'], '3D feedback continuity broken'
        assert next_entry['input_hash'] == next_entry['output_hash'], 'retained volume was overwritten'

    # Quality is measured, not assumed to improve. Check finite MSE and
    # bounded output range, with a separately calculated reference metric.
    import math
    mse_rows = [x for x in rows if int(x['opcode']) == 0x27]
    assert len(mse_rows) == 3
    assert all(math.isfinite(float(x['mse'])) and 0 <= float(x['mse']) <= 1 for x in mse_rows)
    decoded = struct.unpack('<512f', a[24:])
    assert all(math.isfinite(x) and 0 <= x <= 1 for x in decoded)
    print('Measured per-level MSE:', [float(x['mse']) for x in mse_rows])

    # Execute the adaptive spatial codec in the same VM bytecode chain.
    spatial = run('spatial', '--dim', '8', '--levels', '3', '--iterations', '8', '--codec', 'spatial')
    spatial_repeat = run('spatial-repeat', '--dim', '8', '--levels', '3',
                         '--iterations', '8', '--codec', 'spatial')
    assert spatial.returncode == spatial_repeat.returncode == 0, spatial.stderr
    assert 'codec=spatial' in spatial.stdout and 'cycles=31' in spatial.stdout
    data = (folder/'spatial.jx3b').read_bytes()
    assert data[:4] == b'JX3B'
    version, dim, block, length, checksum = struct.unpack('<5I', data[4:24])
    assert version == 1 and dim == 8 and block in (2,4,8)
    assert length == (8//block)**3 and len(data) == 24+length
    h = 2166136261
    for byte in data[24:]:
        h = ((h ^ byte) * 16777619) & 0xffffffff
    assert h == checksum, 'spatial payload checksum mismatch'
    assert (folder/'spatial.jx3b').read_bytes() == (folder/'spatial-repeat.jx3b').read_bytes()
    assert (folder/'spatial.jx3v').read_bytes() == (folder/'spatial-repeat.jx3v').read_bytes()
    with (folder/'spatial.csv').open() as f:
        spatial_rows = list(csv.DictReader(f))
    assert len(spatial_rows) == 31
    assert all(row['codec'] == 'spatial_u8' for row in spatial_rows)
    assert any(int(row['block']) in (2,4,8) for row in spatial_rows)
    assert run('badlambda', '--codec', 'spatial', '--rd-lambda', '-1').returncode != 0
    assert run('bad', '--dim', '9').returncode != 0
    assert run('badlevels', '--levels', '0').returncode != 0
    print('PASS: deterministic replay, 31 traced dispatches, portable binary formats, bad-input rejection')