"""Focused end-to-end regression for bounded Nano3D synthetic reference."""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verify_commit import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', required=True)
    parser.add_argument('--compiler', default='g++')
    args = parser.parse_args()
    engine = Path(args.engine).resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix='nano3d-ctest-') as base:
        root = Path(base)
        inv = subprocess.run([str(engine), '--no-emit', '--loops', '10000'], text=True, capture_output=True, check=True)
        assert 'trained_correct=32768/32768' in inv.stdout
        assert 'table_bytes=65536' in inv.stdout
        for bad in (['--files', '-1'], ['--files', '0'], ['--loops', '-3'], ['--files', '10001']):
            p = subprocess.run([str(engine), *bad, '--out', str(root / 'forbidden')], capture_output=True)
            assert p.returncode != 0, bad
        a = run(engine=engine, root=root/'published', release='r1', files=2, lines=40,
                compiler=args.compiler, inject_invalid=False)
        assert a['syntax_accepted_loc'] == 80
        release_dir = root/'published'/'releases'/'r1'
        for file_name, digest in a['source_sha256'].items():
            assert hashlib.sha256((release_dir/'source'/file_name).read_bytes()).hexdigest() == digest
        before = (root/'published'/'latest.json').read_bytes()
        try:
            run(engine=engine, root=root/'published', release='reject', files=2, lines=40,
                compiler=args.compiler, inject_invalid=True)
        except RuntimeError as exc:
            assert 'compiler rejected' in str(exc)
        else:
            raise AssertionError('invalid code was published')
        assert (root/'published'/'latest.json').read_bytes() == before
        assert not (root/'published'/'releases'/'reject').exists()
        b = run(engine=engine, root=root/'repeat', release='r2', files=2, lines=40,
                compiler=args.compiler, inject_invalid=False)
        assert a['source_sha256'] == b['source_sha256']
        latest = json.loads((root/'published'/'latest.json').read_text())
        assert latest['release'] == 'r1'
    print('PASS: addressing, synthetic training, CLI bounds, syntax, hashes, repeatability, rollback')


if __name__ == '__main__':
    main()
