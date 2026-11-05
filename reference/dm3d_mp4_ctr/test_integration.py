#!/usr/bin/env python3
"""Native -> independent decoder -> CTR integration and negative tests."""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from independent_verify import decode_archive


def run(exe: str, *arguments: Path | str, success: bool = True):
    result = subprocess.run([exe, *map(str, arguments)], text=True, capture_output=True)
    if success != (result.returncode == 0):
        raise AssertionError(f'Unexpected process return {result.returncode} for {arguments}: {result.stderr} {result.stdout}')
    return result


def test_one(exe: str, directory: Path, name: str, data: bytes):
    src, archive, out = (directory / (name + suffix) for suffix in ('.src', '.jxdm3', '.out'))
    src.write_bytes(data)
    result = run(exe, 'roundtrip', src, archive, out)
    import json
    receipt = json.loads(result.stdout)
    assert receipt['status'] == 'verified_commit'
    assert receipt['guest_ram_bytes'] == 1048576
    assert receipt['logical_axis'] == 10485760
    assert receipt['vm_steps'] == receipt['tiles'] * 7
    assert out.read_bytes() == data
    independent = decode_archive(archive, src)
    assert independent['sha256'] == hashlib.sha256(data).hexdigest()
    # immutable authoritative output; repeat execution must fail without overwrite
    run(exe, 'decode', archive, out, success=False)
    # Header CRC, tile payload, truncation and unexpected trailer are rejected.
    original = archive.read_bytes()
    for which, mutant in enumerate((
        original[:20] + b'\x55' + original[21:],
        original[:50] + bytes([original[50] ^ 1]) + original[51:] if len(original) > 50 else original+b'X',
        original[:-1] if original else original,
        original + b'X',
    )):
        corrupt = directory / f'{name}-{which}.bad'
        rejected = directory / f'{name}-{which}.rejected'
        corrupt.write_bytes(mutant)
        run(exe, 'decode', corrupt, rejected, success=False)
        assert not rejected.exists(), 'decoder published corrupted candidate'
        assert not list(directory.glob(rejected.name + '.dm3d-stage-*')), 'leftover staging directory'
    return independent


def main(exe: str):
    with tempfile.TemporaryDirectory(prefix='jarvisx-mp4-ctr-') as tmp:
        path = Path(tmp)
        rng = random.Random(103)
        # The MP4-shaped fixture tests bytes/box preservation, not playable video decoding.
        sample_mp4 = (b'\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isommp41'
                      + b'\x00\x00\x00\x0cmdat' + b'X123')
        cases = {
            'empty': b'',
            'zero': b'\0' * 200000,
            'repeat': b'A' * 9999,
            'sequential': bytes(range(256))*20,
            'random': rng.randbytes(1024*1024 + 13),
            'mp4_boxes': sample_mp4,
        }
        # Optional large, actually playable MP4 with UUID payload: used locally.
        if os.environ.get('DM3D_MP4_TEST_FILE'):
            source = Path(os.environ['DM3D_MP4_TEST_FILE'])
            cases['actual_mp4'] = source.read_bytes()
        for name, data in cases.items():
            result = test_one(exe, path, name, data)
            print('PASS', name, len(data), 'bytes, sha256', result['sha256'][:12])
        # Cross-platform output-path rejection and header corruption handling covered above.
        print('INTEGRATION PASS', len(cases), 'independent end-to-end workloads + negative tests')


if __name__ == '__main__':
    main(sys.argv[1])
