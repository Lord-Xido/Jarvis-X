# Jarvis X / DM3D V2 — Spectral Inward Fold

This is an **operational finite 3D C++17 numerical runtime**, an upgrade of `jarvis_x_dm3d.cpp`. It is **not** a trained cognitive neural network, an autonomous source-code self-rewriter, or a physical septillion-scale worker system. Its large address space is symbolic.

## Highlights

- `jarvis_x_dm3d_v2.cpp`: **888 lines**, C++17, requires Boost.Multiprecision header-only dependencies and C++ threads.
- Select `--solver auto`, `--solver fft`, or `--solver swarm`. Default is `auto`.
- Fast **closed-form spectral fixed point** on power-of-two periodic cubes of side 2–128; auto falls back to the verified swarm for even non-power-of-two sides.
- Retained arithmetic-indexed RAG, certified rational comparisons to π, binary/UTF-8 lossless 3D byte codec, 3D residual encoder, associative numerical memory, and save/resume.
- Each spectral solution is independently evaluated with the original spatial-domain objective before commitment.
- Spectral mode performs one direct optimisation and leaves subsequent requested iteration steps unchanged; it does **not** execute thousands of real optimisation iterations in that mode.

## Derivation

For `T` the synthetic target field, with periodic cube `N³`, the original objective is

`J(X) = [||X-T||² + mu * sum_(voxel,direction) (X_(r+e_d) - X_r)²] / (2*N³)`.

Its first-order optimality condition is

`(I - mu*Delta_3D) X* = T`.

The periodic Laplacian is diagonal in the 3D Fourier basis. If `k_j = 2π*j/N`, then

`Xhat*(j_x,j_y,j_z) = That(j_x,j_y,j_z) / [1+2*mu*(3-cos(kx)-cos(ky)-cos(kz))]`.

The denominator is `>=1` for `mu>=0`; hence the quadratic has a unique minimum, and the FFT solution exists and is well-conditioned with respect to the minimum eigenvalue.

The algorithm's asymptotic costs are approximately:

- Iterative: `O(K * N³ + K * workers)` for `K` steps, with several 3D stencil/memory passes per step.
- Spectral: `O(N³ log N)` in this radix-2 implementation (FFT + diagonal inversion + inverse FFT), plus one independent spatial objective verification `O(N³)`.

The spectral solver is **specialised to the stated linear, periodic, quadratic problem**; it is not a general nonlinear ANN optimiser. The π ledger still operates independently when the spectral path is chosen. Use `--solver swarm` for RAG-calibrated worker updates.

## Build

```bash
g++ -std=c++17 -O3 -pthread jarvis_x_dm3d_v2.cpp -o jarvis_x_dm3d_v2
./jarvis_x_dm3d_v2 --self-test
```

Requires Boost headers on the include path (`libboost-dev` on Ubuntu/Debian). Optional `CMakeLists.txt` included.

## Run

```bash
# Direct solver on 64^3
./jarvis_x_dm3d_v2 --side 64 --workers 48 --threads 4 \
  --steps 40 --solver auto --query 22/7 --top 4 \
  --trace trace.csv --ledger memory.csv --save-state state.dm3ds

# Iterative mode for exact apples-to-apples comparison
./jarvis_x_dm3d_v2 --side 64 --workers 48 --threads 4 \
  --steps 40 --solver swarm

# Non power-of-two automatically uses verified swarm
./jarvis_x_dm3d_v2 --side 30 --steps 5 --solver auto

# Lossless byte transformation
./jarvis_x_dm3d_v2 --encode-file input.png encoded.dm3d
./jarvis_x_dm3d_v2 --decode-file encoded.dm3d recovered.png
```

## Measured performance

Same executable, same synthetic volume and objective, 40 requested iterations and 48 worker candidates. Timings are **CPU solver computation only** (`compute_ms` emitted by the program), median of separate process runs, 5 repetitions for 16/32/64 and 3 for 128. System and input-dependent; no universal speed factor.

| Cube | Iterative median (ms) | Spectral median (ms) | Speedup |
|------|-----------------------|----------------------|---------|
| 16³  | 6.172 | 0.423 | 14.57× |
| 32³  | 48.780 | 4.285 | 11.38× |
| 64³  | 431.728 | 41.773 | 10.34× |
| 128³ | 4484.495 | 400.165 | 11.21× |

For 64³, the final objective difference was roughly `7.1e-15`, and the spectral solver produced a slightly **lower** quadratic objective. This is not a measured 1000× acceleration. The base code had 775 lines and this enhanced variant has 888 lines: lines of code are not a runtime-speed unit.

## Verified tests

- Existing original C++ self-tests: π bounds/CF ledger, tamper rejection, 3D coarse/fine reconstruction, deterministic swarm execution, exact byte round trip and associative memory.
- Independent spectral equation residual with 16³ grid, plus direct objective comparison against the swarm.
- Power-of-two FFT guard and automatic non-power-of-two fallback at `30³`.
- Source 434,313-byte PNG round-tripped byte-for-byte; tampered container rejected.
- Saved/reloaded numerical state preserved the objective.
- Spectral and iterative objectives agreed to < `1e-11` on all benchmark grids.

## Files

- `jarvis_x_dm3d_v2.cpp` — complete core runtime.
- `jarvis_x_dm3d_v2` — compiled Linux x86-64 binary; other platforms compile from source.
- `dm3d_v2_geometry.html` — standalone interactive 3D **harmonic** geometry explanation. It shows an exactly solvable harmonic-only example, not the compiled C++ program embedded in a browser.
- `dm3d_v2_benchmarks.json` — all recorded benchmark sample times and objective values.
- `dm3d_v2_test_results.json` — regression tests.
- `CMakeLists.txt` — portable project build.

## GitHub repository integration

This reference module is staged under `reference/dm3d_spectral_v2/`; it intentionally does **not** change the existing Jarvis-X runtime entrypoints or require a merge into `cpp_runtime`. The GitHub submission contains the C++ source, build configuration, an end-to-end regression test, a scoped CI workflow, a standalone geometry explanation, and recorded local benchmark data. Compiled executables are **not** committed.

In this repository, reproduce the results with:

```sh
cmake -S reference/dm3d_spectral_v2 -B build/dm3d-spectral-v2 -DCMAKE_BUILD_TYPE=Release
cmake --build build/dm3d-spectral-v2 --parallel
ctest --test-dir build/dm3d-spectral-v2 --output-on-failure
```

`benchmarks.json` contains **historical, local CPU-only timings** supplied with the prototype; CI verifies correctness rather than making a speedup claim. The `geometry.html` file is a standalone explanatory visualisation, not a live dashboard connected to the C++ program. The FNV-64 checksum detects ordinary corruption but is not a cryptographic authentication mechanism.
