"""Fixture-free native and hybrid round-trip tests; synthetic data only."""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path
import numpy as np
from hybrid import pack, read8
from verify8 import decode

P = Path(__file__).resolve().parent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--iterations', type=int, default=1_000_000)
    args = ap.parse_args()
    if args.iterations < 256:
        raise ValueError('iterations must be at least 256')
    rng = np.random.default_rng(710)
    frames = rng.integers(0, 256, (256, 4, 4, 4, 1), dtype=np.uint8)
    frames[:64] = 0
    for i in range(65, 128):
        frames[i] = np.roll(frames[i-1], 1, axis=2)
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        src = d/'input.raw'
        frames.tofile(src)
        for policy in (1, 2):
            out = d/f'{policy}.s3b'
            r = json.loads(subprocess.check_output([str(P/'core'), str(src), str(out), '4', '4', '4', '1', str(args.iterations), '256', '0', str(policy), '710']))
            np.testing.assert_array_equal(decode(out), frames)
            assert r['iterations'] == args.iterations and r['instructions'] == 12*args.iterations
            assert r['candidate_evaluations'] + r['skipped'] == args.iterations
            results[policy] = r
        assert (d/'1.s3b').read_bytes() == (d/'2.s3b').read_bytes()
        for k in ('hash_chain', 'weights', 'modes', 'record_bytes', 'baseline_bytes', 'accepted'):
            assert results[1][k] == results[2][k], k
        hybrid = {}
        for kind, a in [('image', frames[:1, 0].reshape(1,4,4,1)), ('video', frames[:8, 0]), ('audio', frames[:4, 0]), ('text', frames[:2, 0]), ('geometry', frames[128])]:
            out = d/f'{kind}.s3h'
            hybrid[kind] = pack(a, {'kind':kind}, out)
            np.testing.assert_array_equal(read8(out)[0], a)
        bad = d/'bad.s3b'
        bad.write_bytes((d/'1.s3b').read_bytes()[:-1])
        try:
            decode(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('truncation accepted')
    print(json.dumps({'native':results, 'hybrid':hybrid, 'truncated_packet':'rejected', 'status':'PASS'}, indent=2))

if __name__ == '__main__':
    main()
