#!/usr/bin/env python3
"""Black-box 1000-iteration regression for the standalone C++ reference.

Requires the executable as argv[1]. Uses a synthetic deterministic fixture,
never a user payload or a bundled model, and checks each claimed invariant
without confusing lossy neural inference with exact byte preservation.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile


def invoke(*args, cwd):
    return subprocess.run([str(a) for a in args], cwd=cwd, check=True,
                          text=True, capture_output=True, timeout=120)


def main():
    if len(sys.argv) != 2:
        raise SystemExit('usage: test_recursive3d_1000.py PATH_TO_EXECUTABLE')
    exe = Path(sys.argv[1]).resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix='jarvisx-1000-') as scratch:
        root = Path(scratch)
        source = root / 'synthetic.bin'
        # 512 complete 2x2x2 patches, deterministic and independent of private data.
        payload = bytes(((j * 37 + (j // 8) * 19 + (j % 7) * 11) % 256)
                        for j in range(4096))
        source.write_bytes(payload)
        model = root / 'trained.jxnm'
        invoke(exe, 'selftest', cwd=root)
        invoke(exe, 'train', source, model, '8', '128', cwd=root)
        outputs = []
        for k in (1, 2):
            out = root / f'run{k}'
            invoke(exe, 'loop1000', source, model, out, '16', cwd=root)
            report = json.loads((out / 'verification_report.json').read_text())
            assert report['iterations_executed'] == 1000
            assert report['metrics_rows'] == 1001
            assert report['sampled_patches'] == 16
            assert report['training_during_feedback'] is False
            assert report['exact_full_source_file_binary_roundtrip'] is True
            assert (out / 'restored_source.bin').read_bytes() == payload
            assert report['morton_roundtrip_checks'] == 1001 * 16
            assert report['3d_lifting_roundtrip_checks'] == 1001 * 16
            assert math.isfinite(report['final_continuous_step_rmse'])
            assert math.isfinite(report['final_continuous_origin_rmse'])
            with (out / 'iteration_metrics.csv').open(newline='') as f:
                observations = list(csv.DictReader(f))
            assert len(observations) == 1001
            assert observations[0]['iteration'] == '0'
            assert observations[-1]['iteration'] == '1000'
            with (out / 'geometry_snapshots.csv').open(newline='') as f:
                snapshots = list(csv.DictReader(f))
            assert len(snapshots) == 16 * 12
            outputs.append(out)
        for name in ('iteration_metrics.csv', 'geometry_snapshots.csv',
                     'restored_source.bin', 'source.jx3d'):
            a = (outputs[0] / name).read_bytes()
            b = (outputs[1] / name).read_bytes()
            assert hashlib.sha256(a).digest() == hashlib.sha256(b).digest(), name
    print('PASS: 1000 transitions, geometric invariants, byte-exact archive, reproducible traces')


if __name__ == '__main__':
    main()
