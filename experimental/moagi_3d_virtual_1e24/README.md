# Moagi 3D Virtual C++ Runtime — 10²⁴ Logical Line Instances

**Experimental and bounded.** This strict C++17 prototype models a virtual address space of 10^9 outer indices × 10^6 inner indices × 10^9 C++ line indices = 10^24 logical line instances. It does not physically generate or execute 10^24 statements, compress arbitrary external data, or adapt code over 10^15 causal revisions.

## Architecture

Each (outer, inner, line) coordinate is validated, packed into 80 meaningful bits (30+20+30), and mapped through a fixed six-face-neighbour toroidal integer mixer. Its output is a deterministic 64-bit constant used in an emitted C++ statement: `acc = (acc ^ constant) * 1099511628211ULL;`. An arbitrary line can be regenerated on demand. Unique addresses need not produce unique source statements.

## Build and verification

```bash
cmake -S experimental/moagi_3d_virtual_1e24 -B build/moagi3d -DCMAKE_BUILD_TYPE=Release
cmake --build build/moagi3d --parallel
ctest --test-dir build/moagi3d --output-on-failure
./build/moagi3d/moagi_3d_virtual_code_runtime --verify
./build/moagi3d/moagi_3d_virtual_code_runtime --pack 999999999 999999 999999999
./build/moagi3d/moagi_3d_virtual_code_runtime --unpack ee6b27ffd08ffb9ac9ff
./build/moagi3d/moagi_3d_virtual_code_runtime --line 123456789 543210 987654321
./build/moagi3d/moagi_3d_virtual_code_runtime --emit-demo 123456789 543210 1000 demo.cpp
```

CTest checks 100,006 deterministic address round-trips, invalid-input rejection, an **independently implemented Python six-neighbour oracle**, and generation/compilation/execution of a 1,000-arithmetic-statement C++ program. Expected test output: `451793898048098394`. GCC and Clang passed locally before publication; GitHub CI independently determines remote status.

Generated source output is bounded to 1..1,000,000 arithmetic statements per request. Treat future user-modifiable/evolved generated source as untrusted code and sandbox before execution. Mixer is **not cryptographic**. The 80-bit packed address describes a logical key, not a 10-byte in-memory guarantee; exact lossless preservation of changing external data requires independent input/state logs, immutable rule versions and checkpoints.

See [TECHNICAL_NOTES.md](TECHNICAL_NOTES.md) for complete mathematical scope and measured local test details. Changes are isolated to this directory and its path-scoped CI workflow.
