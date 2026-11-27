#!/usr/bin/env python3
"""Transactional proof-of-mechanism harness for Jarvis X Nano3D.

Validation scope: independent g++/clang++ -fsyntax-only on each translation unit.
Does not prove semantics, linking, production security, or durable fsync.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


def run(*, engine: Path, root: Path, release: str, files: int, lines: int,
        compiler: str, inject_invalid: bool) -> dict:
    if not release or not all(ch.isalnum() or ch in '-_' for ch in release):
        raise ValueError('release must contain only letters, digits, - and _')
    if not (1 <= files <= 10000 and 1 <= lines <= 100000 and files * lines <= 1_000_000):
        raise ValueError('file and line counts out of bounds')
    engine = engine.resolve(strict=True)
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    releases = root / 'releases'
    releases.mkdir(exist_ok=True)
    published = releases / release
    if published.exists():
        raise FileExistsError(f'release already exists: {published}')

    start = time.perf_counter()
    # Temp stages and releases reside on the same filesystem so os.replace is atomic.
    stage = Path(tempfile.mkdtemp(prefix='.pending_', dir=root))
    try:
        output = stage / 'source'
        result = subprocess.run(
            [str(engine), '--files', str(files), '--lines', str(lines),
             '--loops', '100000', '--out', str(output)],
            capture_output=True, text=True, check=True,
        )
        sources = sorted(output.glob('generated_*.cpp'))
        if len(sources) != files:
            raise RuntimeError(f'file count mismatch: {len(sources)} vs {files}')
        if inject_invalid:
            # Preserve line count; force the independent C++ compiler gate to reject.
            content = sources[0].read_text(encoding='utf-8')
            first, rest = content.split('\n', 1)
            sources[0].write_text('int this is not valid;\n' + rest, encoding='utf-8')
        checked_lines = 0
        bytes_total = 0
        hashes = {}
        for source in sources:
            b = source.read_bytes()
            if b'\x00' in b:
                raise RuntimeError('unexpected NUL in C++ source')
            n = b.count(b'\n')
            if n != lines:
                raise RuntimeError(f'line count invalid in {source.name}: {n}')
            check = subprocess.run(
                [compiler, '-std=c++17', '-fsyntax-only', str(source)],
                text=True, capture_output=True,
            )
            if check.returncode != 0:
                raise RuntimeError(f'{source.name}: compiler rejected:\n{check.stderr[:1500]}')
            checked_lines += n
            bytes_total += len(b)
            hashes[source.name] = hashlib.sha256(b).hexdigest()
        if checked_lines != files * lines:
            raise RuntimeError('incorrect final LOC count')
        receipt = {
            'schema': 'jarvisx.nano3d.syntax-admission.v1',
            'release': release,
            'scope': 'deterministic synthetic code templates; g++/clang++ -fsyntax-only; NOT semantic verification',
            'files': len(sources),
            'syntax_accepted_loc': checked_lines,
            'source_bytes': bytes_total,
            'source_sha256': hashes,
            'compiler': compiler,
            'engine_output': result.stdout,
            'elapsed_seconds_to_receipt': time.perf_counter() - start,
        }
        (stage / 'receipt.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
        os.replace(stage, published)
        # Update the latest pointer only after immutable release publication.
        # A crash between these two atomic moves leaves previous latest intact.
        latest_data = {'release': release, 'receipt_path': str(published / 'receipt.json')}
        pointer_temp = root / f'.latest_{release}.tmp'
        pointer_temp.write_text(json.dumps(latest_data, sort_keys=True) + '\n')
        os.replace(pointer_temp, root / 'latest.json')
        return receipt
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--engine', default='./nano3d')
    p.add_argument('--root', default='transactional-output')
    p.add_argument('--release', required=True)
    p.add_argument('--files', type=int, default=8)
    p.add_argument('--lines', type=int, default=850)
    p.add_argument('--compiler', default='g++')
    p.add_argument('--inject-invalid', action='store_true')
    args = p.parse_args()
    try:
        result=run(engine=Path(args.engine),root=Path(args.root),release=args.release,
                   files=args.files,lines=args.lines,compiler=args.compiler,
                   inject_invalid=args.inject_invalid)
        print(json.dumps({k:result[k] for k in ('release','files','syntax_accepted_loc','source_bytes','elapsed_seconds_to_receipt')},indent=2))
        print('ACCEPTED: release published only after all translation units passed -fsyntax-only')
        return 0
    except Exception as e:
        print(f'REJECTED: {e}')
        return 1

if __name__=='__main__':
    raise SystemExit(main())