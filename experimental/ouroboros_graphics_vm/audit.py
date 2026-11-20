#!/usr/bin/env python3
"""Independent, dependency-free output format and telemetry audit."""
import csv
import math
import pathlib
import sys

def audit(prefix: pathlib.Path, expected_frames: int, expected_cycles: int) -> None:
    pgm = prefix.with_name(prefix.name + '_framebuffer.pgm').read_bytes()
    header = b'P5\n128 128\n255\n'
    assert pgm.startswith(header), 'unexpected PGM header or dimensions'
    assert len(pgm) == len(header) + 16384, 'unexpected PGM framebuffer byte count'
    lines = prefix.with_name(prefix.name + '_neural_field.ply').read_text().splitlines()
    assert lines[0:2] == ['ply', 'format ascii 1.0'], 'invalid PLY magic'
    assert 'end_header' in lines, 'missing PLY header terminator'
    vertex_line = next(s for s in lines if s.startswith('element vertex '))
    edge_line = next(s for s in lines if s.startswith('element edge '))
    vertices = int(vertex_line.split()[-1])
    edges = int(edge_line.split()[-1])
    start = lines.index('end_header') + 1
    assert vertices > 0 and edges > 0
    assert len(lines) == start + vertices + edges, 'PLY record count mismatch'
    for row in lines[start:start + vertices]:
        values = row.split()
        assert len(values) == 6 and all(math.isfinite(float(x)) for x in values[:3])
    for row in lines[start + vertices:]:
        v1, v2 = map(int, row.split())
        assert 0 <= v1 < vertices and 0 <= v2 < vertices
    with prefix.with_name(prefix.name + '_telemetry.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == expected_frames, 'telemetry row count mismatch'
    previous = -1
    for i, row in enumerate(rows):
        assert int(row['display_frame']) == i
        cycles = int(row['vm_cycles'])
        assert previous <= cycles <= expected_cycles
        assert int(row['vm_frame']) == cycles // 48
        assert 0 <= int(row['active_neurons']) <= vertices
        assert 0 <= int(row['live_packets']) <= 900
        assert math.isfinite(float(row['mean_activation']))
        assert 0 <= float(row['mean_activation']) <= 1
        assert math.isfinite(float(row['frame_delta_mse']))
        previous = cycles
    assert int(rows[-1]['vm_cycles']) == expected_cycles, 'cycle mismatch'
    print(f'PASS PGM 128x128, PLY {vertices} vertices/{edges} edges, '
          f'{len(rows)} telemetry rows, {previous} cycles')

if __name__ == '__main__':
    if len(sys.argv) != 4:
        sys.exit('usage: audit.py OUTPUT_PREFIX EXPECTED_FRAMES EXPECTED_CYCLES')
    audit(pathlib.Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
