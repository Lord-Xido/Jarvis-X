# Moagi 3D Virtual C++ Runtime — Mathematical and Operational Notes

## Scope and limits

This is a **strict C++17 virtual code generator**, not a machine executing 10^24 operations or a compressor for arbitrary external data. It uses a logical domain of 10^9 outer-iteration indices × 10^6 inner-iteration indices × 10^9 line indices = 10^24 logical line instances. The two iteration axes serve as independently addressable indices; there is **no stateful self-optimisation across these axes**.

## 3D addressing and 80-bit packing

- Coordinates: `0 <= x < 10^9`, `0 <= y < 10^6`, `0 <= z < 10^9`.
- Flattened mathematical rank: `rank = (x*10^6+y)*10^9+z`, requiring approximately 80 bits.
- Packed bit representation: `(x << 50) | (y << 30) | z`. Bit widths: 30+20+30=80, exactly 10 bytes. The implementation uses `uint16_t high` and `uint64_t low`, so no nonstandard 128-bit integer is required.
- This is the size of an **address**, not the size of the code it denotes.

## Stateless spatial generation

Let `h` be an immutable deterministic 64-bit integer mixer and `S(x,y,z)` a site seed. For face-adjacent sites on the 3D torus, the generated code literal is computed from the current site and all six neighbors:

`c(x,y,z) = fold_mix(S(x,y,z), S(x±1,y,z), S(x,y±1,z), S(x,y,z±1))`.

Toroidal boundaries wrap independently on all axes. The generated source line is:

`acc = (acc ^ c(x,y,z)) * 1099511628211ULL;`

Here `acc` is 64-bit unsigned; arithmetic is specified modulo `2^64`. A generated demo program combines an initial accumulator with `n` generated statements and prints its result. `mix64` is **not** cryptographic, and repeated line contents are possible: unique addresses do not imply unique source strings. This is genuine logical 3D neighbor dependence but **not** simulation of 3D physical hardware.

## Commands

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -Werror moagi_3d_virtual_code_runtime.cpp -o runtime
./runtime --verify
./runtime --pack 999999999 999999 999999999
./runtime --unpack ee6b27ffd08ffb9ac9ff
./runtime --line 999999999 999999 999999999
./runtime --emit-demo 123456789 543210 1000 demo.cpp
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic demo.cpp -o demo
./demo
```

## Observed verification (local container)

- GCC and Clang strict C++17 builds: PASS.
- 100,006 address round trips: PASS, including exact maximum valid coordinate.
- Three invalid-coordinate rejection checks: PASS. An invalid outer value returns exit code 2.
- Last 80-bit address: `ee6b27ffd08ffb9ac9ff`, decoding to `(999999999,999999,999999999)`.
- Emitted demo: exactly 1000 arithmetic statements + 6 structural lines = 1006 physical lines. File size: 59,513 bytes.
- Generated demo execution: `451793898048098394`.
- Independent Python reimplementation of six-neighbor seed and 64-bit arithmetic: same output, PASS.
- The generator does **not** compile all `10^15` logical program versions; the test compiles one 1000-statement demo. The compiler is also free to constant-fold the statements, so compiled execution time cannot prove one operation occurred for every source line.

## Information preservation versus virtual representation

The global generator source, exact version, seed rules and coordinate permit *reconstruction of defined synthetic code*. This does not imply a capability to losslessly compress arbitrary independently specified source code. A changed compiler, generator, seed, external input or nondeterministic value must be recorded for repeatable reconstruction; use immutable revisions, test vectors, content hashes (e.g. SHA-256), version metadata and backed-up logs. Hashes detect changes but do not recover missing source information.

For actual stateful evolution, let `t=x*10^6+y` and `theta_{t+1} = U(theta_t, outcomes_t)`. Independent random-access reconstruction then generally needs checkpoints + lossless intervening changes, a proven skip-ahead rule, or replay from the beginning. The prototype instead computes `line(x,y,z)` directly without evolving prior state.

## Complexity

- One packed address: 10 bytes excluding representation overhead.
- Generation of one line: constant space and constant arithmetic work (six-neighbor stencil).
- Generate `m` source lines: O(m) computation and O(1) additional memory when streamed.
- Verify `n` specified coordinates: O(n), not a proof of every address.
- Execute all logical line instances individually: Omega(10^24) elementary line visits unless algebraic structure or computation sharing eliminates the need to visit each site. Performance is **not** established by virtual addressability.

## Paths

- `moagi_3d_virtual_code_runtime.cpp`: compilable generator source.
- `moagi_3d_virtual_example.cpp`: generated and locally executed demo source.
- `MOAGI_3D_VIRTUAL_RUNTIME_TECHNICAL_NOTES.md`: this document.
