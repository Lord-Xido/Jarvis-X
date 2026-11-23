# Jarvis-X Nano3D Predictive ANN (bounded research reference)

An isolated **C++17 proof of mechanism** for precomputing an ANN's predictions on a
32×32×32 synthetic context lattice, storing one 2-byte prediction per Morton
address, retrieving it during template-based C++ source emission, and publishing
only syntactically accepted translation units through a verification gate.

**No changes to the canonical Jarvis-X VM or authoritative state.**
This module is opt-in and non-authoritative.

## Numerical contract

- 32³ = 32,768 cells; 8 synthetic octant classes.
- Trained 3-input/8-output softmax linear classifier (deterministic seeded training).
- 15-bit, collision-free Morton indexing for this bounded domain.
- 32,768 × 2 bytes = 65,536 bytes (64 KiB) prediction table.
- Cache benchmark reports **amortized queries/s**, NOT dependent-lookup latency.
- Generated C++ is deliberately repetitive one-line template functions;
  it is not open-ended semantic code prediction.
- External compiler gate checks **-fsyntax-only**, NOT link correctness,
  behavior, unit-test coverage, security or real-world intelligence.
- Engine emits at most **1,000,000 LOC per invocation**; all counts are bounded.
- No measured million *verified* LOC/s claim. No nanosecond ANN inference claim.

## Reproduce on Linux with Python 3 and a C++17 compiler

```sh
cmake -S experimental/nano3d_predictive_ann -B build/nano3d -DCMAKE_BUILD_TYPE=Release
cmake --build build/nano3d --parallel 2
ctest --test-dir build/nano3d --output-on-failure

build/nano3d/jarvisx_nano3d --no-emit --loops 100000
python3 experimental/nano3d_predictive_ann/verify_commit.py \
    --engine build/nano3d/jarvisx_nano3d \
    --root /tmp/jx-nano3d-verified --release r1 --files 8 --lines 850
```

The verifier creates temporary candidate files, runs a C++ compiler on **every**
file, writes SHA-256 receipts, renames the completed release directory, then
updates a latest-pointer JSON file. Invalid syntax is rejected; the previous
pointer is preserved. This is single-writer visibility atomicity only, not
power-failure durability or hostile-code isolation. Use trusted paths and
compilers. Its test intentionally injects invalid C++ to verify rollback.

## Evidence

Local reference testing used GCC and Clang with strict warnings, and two
CTest checks passed (synthetic invariants and candidate rollback/replay).
Historical reference measurements, including roughly 0.54 ns of amortized
hot-cache lookup work and roughly 42.6k **syntax-accepted** LOC/s on one
run, are workload- and machine-dependent observations, not portable promises.
They exclude compilation to machine code, linking, behavior and production
performance. See [MODEL.md](MODEL.md).

For CI, see `.github/workflows/jx-nano3d-predictive-ann.yml`.
