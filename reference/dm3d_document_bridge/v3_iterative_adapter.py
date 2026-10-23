#!/usr/bin/env python3
"""Public DM3D v3 bridge to a separately held, trusted iterative DOCX SDK.

Does not distribute or import the private runtime. No decoded program is
executed by this bridge. Running the private VM requires explicit approval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile

NS = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
BEGIN, END = '<<<DM3D-PYTHON-BEGIN>>>', '<<<DM3D-PYTHON-END>>>'
MAX_DOCX = 20 * 1024 * 1024
MAX_XML = 4 * 1024 * 1024
MAX_ITEMS = 600


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def inspect_docx(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size > MAX_DOCX:
        raise ValueError('DOCX missing or exceeds size limit')
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        names = [m.filename for m in members]
        if len(members) > MAX_ITEMS or len(names) != len(set(names)):
            raise ValueError('too many or duplicate DOCX entries')
        if any(m.file_size > MAX_DOCX or m.compress_size > MAX_DOCX for m in members):
            raise ValueError('oversized DOCX component')
        if sum(m.file_size for m in members) > 4 * MAX_DOCX:
            raise ValueError('DOCX inflated limit exceeded')
        if 'word/document.xml' not in names:
            raise ValueError('not a Word document')
        info = archive.getinfo('word/document.xml')
        if info.file_size > MAX_XML:
            raise ValueError('document XML too large')
        xml = archive.read(info)
    if b'<!DOCTYPE' in xml.upper() or b'<!ENTITY' in xml.upper():
        raise ValueError('XML entities forbidden')
    root = ET.fromstring(xml)
    paragraphs = []
    for paragraph in root.iter(NS + 'p'):
        parts = []
        for element in paragraph.iter():
            if element.tag == NS + 't' and element.text:
                parts.append(element.text)
            elif element.tag == NS + 'tab':
                parts.append('\t')
        paragraphs.append(''.join(parts))
    if paragraphs.count(BEGIN) != 1 or paragraphs.count(END) != 1:
        raise ValueError('DOCX source markers absent or ambiguous')
    start, stop = paragraphs.index(BEGIN), paragraphs.index(END)
    if stop <= start + 1:
        raise ValueError('DOCX source empty or incorrectly ordered')
    source = ('\n'.join(paragraphs[start+1:stop]) + '\n').encode('utf-8')
    return {'docx_sha256': digest(path.read_bytes()), 'source_sha256': digest(source),
            'source_bytes': len(source), 'source_lines': stop-start-1}


def geometry(package: Path) -> dict:
    """Read outer-level logical 3D tile coordinates; private decoder validates payload."""
    with zipfile.ZipFile(package) as archive:
        if 'manifest.json' not in archive.namelist():
            raise ValueError('DM3D outer manifest absent')
        info = archive.getinfo('manifest.json')
        if info.file_size > 4_000_000:
            raise ValueError('manifest too large')
        meta = json.loads(archive.read(info))
    blocks = meta.get('blocks')
    if not isinstance(blocks, list) or len(blocks) > 32_768:
        raise ValueError('invalid 3D block inventory')
    side = meta.get('grid_side')
    if type(side) is not int or side < 1 or side > 1024:
        raise ValueError('invalid 3D grid side')
    coords = []
    for index, block in enumerate(blocks):
        if block.get('index') != index:
            raise ValueError('noncontiguous 3D block order')
        xyz = block.get('xyz')
        if not isinstance(xyz, list) or len(xyz) != 3 or any(type(v) is not int or v < 0 for v in xyz):
            raise ValueError('invalid voxel coordinate')
        if any(v >= side for v in xyz):
            raise ValueError('voxel is outside the declared cube')
        Morton = sum(((v >> bit) & 1) << (3*bit+axis)
                     for bit in range(side.bit_length()) for axis, v in enumerate(xyz))
        if Morton != index:
            raise ValueError('Morton address mismatch')
        coords.append(xyz)
    if len(set(map(tuple, coords))) != len(coords):
        raise ValueError('duplicate voxel coordinate')
    return {'grid_side': meta.get('grid_side'), 'logical_blocks': len(coords),
            'block_xyz': coords[:64], 'truncated': len(coords) > 64}


def _invoke(runtime: Path, args: list[str], *, timeout: int = 120) -> dict:
    proc = subprocess.run([sys.executable, str(runtime), *args], cwd=runtime.parent,
                          stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True, timeout=timeout, check=False)
    if proc.returncode:
        raise RuntimeError(f'trusted runtime failed ({proc.returncode}): {proc.stderr[:800]}')
    if len(proc.stdout) > 2_000_000:
        raise ValueError('excessive runtime output')
    return json.loads(proc.stdout)


def run(docx: Path, runtime: Path, out: Path, runtime_sha256: str,
        *, rounds: int, layers: int, approved: bool) -> dict:
    if not approved:
        raise PermissionError('external runtime requires --approve-external-runtime')
    if not re.fullmatch('[a-fA-F0-9]{64}', runtime_sha256):
        raise ValueError('expected 64-digit pinned runtime SHA-256')
    if not runtime.is_file() or digest(runtime.read_bytes()) != runtime_sha256.lower():
        raise ValueError('trusted runtime digest mismatch')
    if not 1 <= rounds <= 32 or not 1 <= layers <= 4:
        raise ValueError('round or layer limit invalid')
    details = inspect_docx(docx)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError('output directory must be empty (prevents stale proof reuse)')
    out.mkdir(parents=True, exist_ok=True)
    report = _invoke(runtime, ['iterate-docx', str(docx.resolve()), '--out', str(out.resolve()),
                               '--rounds', str(rounds), '--layers', str(layers)])
    if report.get('engine') != 'DM3D-CLOSED-LOOP-2' or report.get('layers') != layers:
        raise ValueError('incompatible iterative runtime report')
    expected = ('original.py', 'best.py', 'best.dm3d', 'decoded_verified.py', 'iteration_report.json')
    if any(not (out / file).is_file() for file in expected):
        raise ValueError('incomplete runtime proof artifacts')
    original, best, reconstructed = ((out / p).read_bytes()
                                     for p in ('original.py', 'best.py', 'decoded_verified.py'))
    if digest(original) != details['source_sha256']:
        raise ValueError('DOCX source and runtime original mismatch')
    if best != reconstructed or digest(best) != report['final_sha256']:
        raise ValueError('runtime reconstructed source mismatch')
    with tempfile.TemporaryDirectory() as temp:
        target = Path(temp) / 'independent-decode.py'
        verification = _invoke(runtime, ['nested-decode', str((out/'best.dm3d').resolve()),
                                         str(target), '--layers', str(layers),
                                         '--sha256', report['final_sha256']])
        if target.read_bytes() != best or verification['sha256'] != report['final_sha256']:
            raise ValueError('independent decode verification failed')
    proof = {'verified': True, 'bridge': 'public-v3', **details,
             'accepted_iterations': report['accepted_iterations'],
             'termination': report['halt_reason'], 'final_sha256': report['final_sha256'],
             'geometry': geometry(out/'best.dm3d')}
    (out/'public_bridge_proof.json').write_text(json.dumps(proof, indent=2)+'\n', encoding='utf-8')
    return proof


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    check = sub.add_parser('inspect', help='inspect DOCX without executing code')
    check.add_argument('docx', type=Path)
    process = sub.add_parser('run', help='run independently held pinned local SDK')
    process.add_argument('docx', type=Path)
    process.add_argument('--runtime', type=Path, required=True)
    process.add_argument('--runtime-sha256', required=True)
    process.add_argument('--out', type=Path, required=True)
    process.add_argument('--rounds', type=int, default=8)
    process.add_argument('--layers', type=int, default=2)
    process.add_argument('--approve-external-runtime', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.command == 'inspect':
            result = inspect_docx(args.docx)
        else:
            result = run(args.docx, args.runtime, args.out, args.runtime_sha256,
                         rounds=args.rounds, layers=args.layers, approved=args.approve_external_runtime)
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, PermissionError, RuntimeError, FileNotFoundError,
            zipfile.BadZipFile, ET.ParseError, subprocess.TimeoutExpired, KeyError) as exc:
        print(f'ERROR {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
