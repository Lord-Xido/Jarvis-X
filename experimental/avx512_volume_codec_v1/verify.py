#!/usr/bin/env python3
"""C++17 runtime/compile-failure tests, ISA inspection, independent trace audit."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import platform
import shlex
import shutil
import struct
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent


def audit(directory):
    summary = json.loads((directory / 'summary.json').read_text())
    with (directory / 'voxel_trace.csv').open() as source:
        voxels = list(csv.DictReader(source))
    with (directory / 'block_trace.csv').open() as source:
        blocks = list(csv.DictReader(source))
    values = [struct.unpack('<f', bytes.fromhex(row['input_bits'])[::-1])[0] for row in voxels]
    def raw(value):
        return struct.unpack('<I', struct.pack('<f', value))[0]
    expected = {}
    for row in blocks:
        x, y, z = (int(row[k]) for k in ('x', 'y', 'z'))
        total = 0.0
        for dz in range(2):
            for dy in range(2):
                for dx in range(2):
                    total += values[(2*z+dz)*4*34+(2*y+dy)*34+2*x+dx]
        mean_bits = raw(total * 0.125)
        assert int(row['mean_bits'], 16) == mean_bits
        expected[(x, y, z)] = mean_bits
    squared_sum = 0.0
    for i, row in enumerate(voxels):
        x, y, z = (int(row[k]) for k in ('x', 'y', 'z'))
        assert int(row['index']) == i == z*4*34+y*34+x
        reconstructed = expected[(x//2, y//2, z//2)]
        assert int(row['reconstruction_bits'], 16) == reconstructed
        value = struct.unpack('<f', reconstructed.to_bytes(4, 'little'))[0]
        residual = values[i] - value
        assert float(row['residual']) == residual
        assert float(row['squared_error']) == residual*residual
        squared_sum += residual*residual
    assert len(voxels) == 816 and len(blocks) == 102
    assert math.isclose(summary['mse'], squared_sum/816, rel_tol=1e-12, abs_tol=1e-12)
    assert summary['latent_payload_bytes'] == 408 and summary['input_payload_bytes'] == 3264
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sanitize', action='store_true')
    parser.add_argument('--output', default='build')
    args = parser.parse_args()
    build = ROOT / args.output
    build.mkdir(parents=True, exist_ok=True)
    compiler = shlex.split(os.environ.get('CXX', 'g++'))
    log = []
    def run(arguments, extra_env=None, expected_failure=False):
        command = [str(x) for x in arguments]
        environment = os.environ.copy()
        environment.update(extra_env or {})
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, env=environment)
        log.append('$ ' + shlex.join(command) + '\n' + result.stdout + result.stderr)
        if expected_failure:
            if result.returncode == 0:
                raise AssertionError('Invalid geometry unexpectedly compiled')
        elif result.returncode:
            raise RuntimeError('Command failed: ' + shlex.join(command) + '\n' + result.stderr)
        return result.stdout + result.stderr
    try:
        version = run(compiler + ['--version']).splitlines()[0]
        flags = ['-std=c++17', '-O2', '-Wall', '-Wextra', '-Wpedantic', '-Wconversion', '-Wshadow', '-Werror', '-ffp-contract=off']
        sources = ['scalar.cpp', 'avx512.cpp']
        for source, target in [('tests.cpp', 'codec_tests'), ('demo.cpp', 'codec_demo')]:
            run(compiler + flags + sources + [source, '-o', build / target])
        tests = run([build / 'codec_tests'])
        print(tests, end='', flush=True)
        disabled = build / 'scalar_only_tests'
        run(compiler + flags + ['-DJX_DISABLE_AVX512'] + sources + ['tests.cpp', '-o', disabled])
        run([disabled])
        invalid = [('<0,2,2,1>', 'positive'), ('<2,3,2,1>', 'even'), ('<2,2,2,0>', 'channels'),
                   ('<(std::size_t{1}<<63),4,2,1>', 'overflow')]
        for index, (template, message) in enumerate(invalid):
            path = build / ('invalid_' + str(index) + '.cpp')
            path.write_text('#include "codec.hpp"\nusing C=jx::volume::AVX512VolumeAutoencoderCodec' + template
                            + ';\nstatic_assert(sizeof(C)>0);\n')
            output = run(compiler + flags + ['-I', ROOT, '-c', path, '-o', build / 'invalid.o'], expected_failure=True)
            assert message in output
        receipt = build / 'receipt'
        run([build / 'codec_demo', receipt])
        summary = audit(receipt)
        object_file = build / 'avx512.o'
        run(compiler + flags + ['-c', 'avx512.cpp', '-o', object_file])
        isa_checked = False
        if shutil.which('objdump') and platform.machine().lower() in ('x86_64', 'amd64', 'i386', 'i686'):
            assembly = run(['objdump', '-d', '-M', 'intel', object_file])
            (build / 'avx512_disassembly.txt').write_text(assembly)
            assert 'zmm' in assembly and 'vfmadd' in assembly
            isa_checked = True
        if args.sanitize:
            sanitized_flags = ['-std=c++17', '-O1', '-g', '-Wall', '-Wextra', '-Wpedantic', '-Werror',
                               '-ffp-contract=off', '-fsanitize=address,undefined', '-fno-omit-frame-pointer']
            run(compiler + sanitized_flags + sources + ['tests.cpp', '-o', build / 'sanitized_tests'])
            run([build / 'sanitized_tests'], {'ASAN_OPTIONS': 'detect_leaks=0'})
        results = {'compiler': version, 'machine': platform.machine(), 'runtime_test_groups': 9,
                   'compile_rejection_cases': 4, 'scalar_only_build_passed': True,
                   'python_voxel_trace_audit_passed': True, 'avx512_isa_compilation_checked': isa_checked,
                   'native_avx512_execution_verified': summary['backend'] == 'avx512f',
                   'sanitizers_passed': args.sanitize, 'leak_detection_claim': False,
                   'performance_claim': False, 'demo': summary}
        (build / 'validation_summary.json').write_text(json.dumps(results, indent=2) + '\n')
        print(json.dumps(results, indent=2), flush=True)
    finally:
        (build / 'validation.log').write_text('\n'.join(log))


if __name__ == '__main__':
    main()
