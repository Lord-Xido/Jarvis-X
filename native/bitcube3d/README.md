# BitCube3D v2 — isolated native C++17 SDK, coding runtime, and offline IDE

An integration candidate for Jarvis-X (tracked by [issue #440](https://github.com/Lord-Xido/Jarvis-X/issues/440)). This subsystem is independent of existing bitcode, bytecode, and C++ modules; it is not wired into Jarvis-X's default executable or service paths.

## C++ ↔ Python ↔ Node.js golden interchange

The [cross-backend B3D1 contract](../../docs/B3D1_CROSS_BACKEND_CONTRACT.md)
introduces independent Python and Node.js format oracles and
C++-executed file exchange tests using exact shared SHA-256 golden vectors.
B3D1 remains uncompressed and unauthenticated, and is distinct from the
`DM3DROM1` tile codec.

## Build and validate

From the repository root:

```sh
cmake -S native/bitcube3d -B build/bitcube3d -DCMAKE_BUILD_TYPE=Release
cmake --build build/bitcube3d -j
ctest --test-dir build/bitcube3d --output-on-failure
(cd /tmp && /absolute/path/to/Jarvis-X/build/bitcube3d/bitcube3d-cpp run /absolute/path/to/Jarvis-X/native/bitcube3d/examples/3d_program.b3ds)
```

Alternatively, run the REPL with `build/bitcube3d/bitcube3d-cpp repl`. The standalone offline browser IDE is `native/bitcube3d/ide/index.html`; open it in a browser (no server needed). Browser code and C++ use the same B3D1 binary layout, but this PR's path-scoped CI validates the native runtime only.

## C++ SDK

```cpp
#include "bitcube3d/engine.hpp"
using namespace bitcube3d;
Engine cube;
cube.set(0, 0, 0, true);
cube.set(999999999, 999999999, 999999999, true);
Engine original = cube;
cube.forward(1000);
cube.inverse(1000);
if (!(cube == original)) return 1;
```

Link with the CMake target `bitcube3d_sdk`; for interpreter programs link with `bitcube3d_program`. Supported DSL commands include `RESET`, `SET`, `GET`, `CLEAR`, `FLIP`, `BOX`, `SPHERE`, `ASSERT`, `VERIFY`, `ENCODE`, `DECODE`, `SAVE`, `LOAD`, and `STATS`.

Generate a C++ DSL-wrapper application with:

```sh
build/bitcube3d/bitcube3d-cpp codegen native/bitcube3d/examples/3d_program.b3ds generated.cpp
g++ -std=c++17 -O2 -Inative/bitcube3d/include generated.cpp build/bitcube3d/libbitcube3d_program.a build/bitcube3d/libbitcube3d_sdk.a -o generated
```

## Mathematical and storage invariants

- Virtual address domain: coordinates `[0, 10^9-1]^3`, representing `10^27` logical bit positions, **never densely allocated**.
- Sparse brick: `16×16×16 = 4096` bits (`512` raw payload bytes) keyed by brick coordinates.
- Bit indexing within a brick is x-fastest: `i = x + 16*y + 256*z`.
- One forward round: 3 axes × strides `{1,2,4,8}`; each update is `high ^= low` over GF(2). The inverse reverses the pass order.
- Round-trip: `D^n(E^n(X)) = X` for valid states; decoding is exact and lossless.
- B3D1: `B3D1` magic, 32-bit LE cube side, 32-bit LE rounds, 64-bit LE active-brick count, then sorted 12-byte brick coordinates + 512-byte payload.
- Script execution caps active bricks (default 100,000), outstanding rounds (10,000), script lines (100,000), and per-statement geometry. The public `Engine` API does not enforce the script budget.

## Important limitations

This codec does not encrypt or compress data; it is not an ANN/LLM, learned optimizer, distributed runtime, or measured 10^27-bit processing capability. 3D geometric addresses do not imply processing parallelism. `codegen` produces C++ that invokes the DSL interpreter, not specialized fused C++ machine kernels. Scripts may read/write local files, and this engine is **not a sandbox** for untrusted scripts. Snapshots have no cryptographic authentication or checksum.
