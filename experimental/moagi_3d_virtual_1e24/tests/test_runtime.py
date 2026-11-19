#!/usr/bin/env python3
"""Independent Python reference for a bounded slice of the logical 10^24 address space."""
import pathlib
import subprocess
import sys
import tempfile

OUTER, INNER, LINES = 10**9, 10**6, 10**9
MASK = (1 << 64) - 1
PRIME = 1099511628211
INIT = 1469598103934665603


def mix(v):
    v = (v + 0x9E3779B97F4A7C15) & MASK
    v = ((v ^ (v >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    v = ((v ^ (v >> 27)) * 0x94D049BB133111EB) & MASK
    return v ^ (v >> 31)


def seed(x, y, z):
    s = mix(0x4D4F414749334431 ^ x)
    s = mix(s ^ y)
    return mix(s ^ z)


def constant(x, y, z):
    neighbours = (
        ((x - 1) % OUTER, y, z), ((x + 1) % OUTER, y, z),
        (x, (y - 1) % INNER, z), (x, (y + 1) % INNER, z),
        (x, y, (z - 1) % LINES), (x, y, (z + 1) % LINES),
    )
    s = seed(x, y, z)
    for xx, yy, zz in neighbours:
        s = mix(s ^ seed(xx, yy, zz))
    return s


def run(runtime, *args, ok=True):
    p = subprocess.run([str(runtime), *map(str, args)], text=True,
                       capture_output=True, check=False)
    if ok and p.returncode != 0:
        raise AssertionError(f"{args} returned {p.returncode}: {p.stderr}")
    if not ok and p.returncode == 0:
        raise AssertionError(f"invalid command accepted: {args}")
    return p.stdout.strip()


def main():
    runtime = pathlib.Path(sys.argv[1]).resolve()
    compiler = sys.argv[2]
    verification = run(runtime, '--verify')
    assert 'verification=PASS' in verification
    assert 'address_roundtrips=100006' in verification
    assert 'logical_line_instances=10^24' in verification

    addresses = ((0, 0, 0), (999999999, 999999, 999999999),
                 (123456789, 543210, 987654321), (0, 0, 999999999),
                 (999999999, 0, 0), (0, 999999, 0))
    for a in addresses:
        h = run(runtime, '--pack', *a)
        assert len(h) == 20
        assert run(runtime, '--unpack', h) == ' '.join(map(str, a))
        expected = f'  acc = (acc ^ {constant(*a)}ULL) * 1099511628211ULL;'.strip()
        assert run(runtime, '--line', *a) == expected
        assert run(runtime, '--line', *a) == run(runtime, '--line', *a)

    for a in (('0', '0', '-1'), ('0', '0', '+1'), ('not-a-number', '0', '0'),
              ('1000000000', '0', '0'), ('0', '1000000', '0'),
              ('0', '0', '1000000000'), ('18446744073709551616', '0', '0')):
        run(runtime, '--line', *a, ok=False)
    for invalid in ('z' * 20, '0' * 19, 'f' * 20):
        run(runtime, '--unpack', invalid, ok=False)

    x, y, n = 123456789, 543210, 1000
    with tempfile.TemporaryDirectory(prefix='moagi-3d-') as tmp:
        source = pathlib.Path(tmp) / 'example.cpp'
        binary = pathlib.Path(tmp) / 'example'
        summary = run(runtime, '--emit-demo', x, y, n, source)
        assert 'statements=1000' in summary
        lines = source.read_text(encoding='utf-8').splitlines()
        assert len(lines) == n + 6
        assert lines[4].strip() == f'acc = (acc ^ {constant(x, y, 0)}ULL) * {PRIME}ULL;'
        acc = INIT
        for z in range(n):
            acc = ((acc ^ constant(x, y, z)) * PRIME) & MASK
        assert f'expected_output={acc}' in summary
        subprocess.run([compiler, '-std=c++17', '-O2', '-Wall', '-Wextra',
                        '-Wpedantic', '-Werror', str(source), '-o', str(binary)],
                       check=True, timeout=90)
        result = subprocess.run([str(binary)], capture_output=True,
                                text=True, check=True, timeout=10)
        assert result.stdout.strip() == str(acc), (result.stdout, acc)
        assert acc == 451793898048098394, acc  # Gold reference from local prototype
    print('PASS: address, bounds, neighbour oracle, source generation, C++ compile, execution')


if __name__ == '__main__':
    main()
