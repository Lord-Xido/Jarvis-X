#!/usr/bin/env python3
"""Independently bind the sandbox native C++ receipt to raw JX8KT1 evidence.

This does not grant permission to mutate Jarvis-X production state.
"""
import argparse
import hashlib
import json
from pathlib import Path

from verify_packet import verify
from tile_vm_adapter import BoundedTileVM, default_bytecode


def link(native: dict, report: dict, packet: bytes, raw: bytes) -> dict:
    binary = verify(report, packet, raw)
    pyvm = BoundedTileVM(default_bytecode()).run(report, packet, raw)
    count = binary['verified_tiles']
    if (native.get('schema') != 'JX8K-NATIVE-TILEVM-1'
            or native.get('authority') != 'sandbox_only'
            or native.get('production_state_modified') is not False
            or native.get('commit_eligible') is not False
            or native.get('sandbox_committed') is not True
            or native.get('rolled_back') is not False
            or native.get('tiles_requested') != count
            or native.get('tiles_staged') != count
            or native.get('instruction_steps') != count * 6 + 1
            or native.get('packet_bytes') != len(packet)
            or native.get('sandbox_records_after') != count + 1
            or native.get('tensor_dtype') != 'float32'
            or native.get('voxels_per_tile') != 4096
            or native.get('fixed_point') != 'not_evaluated'
            or native.get('external_world') != 'not_evaluated'):
        raise ValueError('incompatible or incomplete native sandbox receipt')
    got = native.get('observations')
    expected = pyvm['observations']
    if not isinstance(got, list) or len(got) != count:
        raise ValueError('native observation count mismatch')
    for idx, (n, p) in enumerate(zip(got, expected)):
        r = report['observations'][idx]
        if (n.get('tile_coord') != p['tile_coord']
                or n.get('morton39') != p['morton39']
                or n.get('counts') != p['counts']
                or n.get('fnv1a_hex') != r['fnv1a_hex']):
            raise ValueError(f'cross-runtime mismatch at tile {idx}')
    return {
        'schema': 'JX8K-NATIVE-LINK-1',
        'status': 'PASS',
        'authority': 'provisional',
        'production_state_modified': False,
        'commit_eligible': False,
        'native_sandbox_commit_verified': True,
        'tiles_verified': count,
        'instruction_steps': native['instruction_steps'],
        'packet_sha256': binary['packet_sha256'],
        'raw_sha256': binary['raw_sha256'],
        'native_receipt_sha256': hashlib.sha256(json.dumps(native, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        'independent_comparison': 'exact per-tile address, Morton39, histograms, FNV and original 2-bit bytes',
        'production_authority_gate': 'not_evaluated',
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native-receipt', required=True)
    p.add_argument('--report', required=True)
    p.add_argument('--encoded', required=True)
    p.add_argument('--raw', required=True)
    p.add_argument('--out')
    a = p.parse_args()
    result = link(json.loads(Path(a.native_receipt).read_text()),
                  json.loads(Path(a.report).read_text()),
                  Path(a.encoded).read_bytes(), Path(a.raw).read_bytes())
    encoded = json.dumps(result, indent=2, sort_keys=True) + '\n'
    if a.out:
        Path(a.out).write_text(encoded)
    print(encoded, end='')


if __name__ == '__main__':
    main()
