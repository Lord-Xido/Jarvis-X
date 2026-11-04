# Jarvis-X v11 — Independent local validation report

**Date:** 2026-11-04. **Target:** Linux environment, C++20 native CPU shared library, Python localhost HTTP interface. Results are from this local evaluation environment, not cloud or accelerator measurements.

## Compilation and regression suite

Executed:

```sh
node --check web/app.js
./run_tests.sh
python3 benchmark.py --dimension 10 --iterations 1000
```

Results:

- **C++ native compile:** PASS (`g++ -O2 -std=c++20 -Wall -Wextra -Wpedantic`).
- **JavaScript syntax:** PASS (`node --check`).
- **C++ integration assertions:** PASS, including CRC32-protected spatial JX3D byte roundtrips, tiled adjacency, numerical boundedness, fault recovery, 600 independently checked native A* vs BFS paths, 8 packed bitplanes, byte switching counters, 76-param neural training and exact JXA1 export/import.
- **Python HTTP tests:** 6 of 6 PASS: health/HTML, native state evolution, fault/routing/healing, JX3D corruption rejection and restoration, UTF-8 codec, and JXA1 neural training/roundtrip/corruption rejection.
- **HTTP smoke check:** PASS, `/api/health` returned a healthy C++20 backend; `/api/state` returned 1,000 actual Q8 values and 1,000 native neural reconstructions.
- **Browser screenshot smoke test:** INCONCLUSIVE. The environment's Chromium browser could launch but refused localhost navigation with `ERR_BLOCKED_BY_ADMINISTRATOR`; do not interpret this as a successful visual-rendering test.

## Representative measurements (one run)

| Workload | Observed result | Scope |
|---|---:|---|
| Grid | 10×10×10 = 1,000 voxels | local native CPU |
| Spatial stencil | 1,000 iterations in 0.014675 s | ~68.1 million voxel updates/s, one local run |
| Neural model | 76 float32 trainable parameters | shared 8→4→8 over 125 spatial tiles |
| Training | 100 epochs in 5.8556 ms | full batch, local one-run measurement |
| Training MSE | 0.10135194 → 0.02447394 | on seeded field, continuous latent output |
| Training accepted | 100 of 100 epochs | non-increasing-loss acceptance |
| JXA1 encoded stream | 2,027 B from 1,000 source Q8 bytes | **2.027× expansion**, not compression |
| JX3D snapshot | 1,119 B from 1,000 source Q8 bytes | **1.119× expansion** |
| Native A* route 0→999 | 27 hops | unblocked 10³ lattice |
| Q8 bitplane integrity | all 8 reconstructed exactly | native C++ test |
| Tampered neural stream | rejected, state unchanged | C++ and HTTP tests |

Measurements depend on hardware, compiler, load and field content. The neural loss is for the sampled training field and is not evidence of generalization, model scaling, convergence to a global optimum, or semantic intelligence.

## Code-to-claim traceability

- `src/jx_core.cpp`: spatial field update, bitplane packing, popcount switching activity, A* routing, fault-healing, JX3D/JXA1 streams and native C ABI.
- `src/jx_ann.hpp`: 8→4→8 shared-weight neural forward path; quantized latent encoder/decoder; full-batch gradient and backtracking update.
- `serve.py`: native library compilation, `ctypes` bindings, background simulation worker, controlled JSON and binary HTTP routes.
- `web/index.html`, `web/app.js`, `web/style.css`: user-interface geometry, field/ANN/bitplane selection and visual telemetry (JavaScript syntax checked; graphical runtime not certified).
- `tests/test_core.cpp`, `tests/test_http.py`: independent regression assertions.

## Mathematical guarantees and limitations

On a fixed fault graph, the six-neighbour evolution map is a convex combination with Lipschitz bound 0.96 in sup norm. Training uses gradient backtracking to accept non-increasing full-batch loss on the current frozen training field. The latent path quantizes each tile's four latent outputs to bytes; the decoder cannot reconstruct all original Q8 values without the explicit XOR residual, so the residual is always serialized. CRC32 detects accidental corruption; it is **not cryptographic authentication**. To protect an untrusted network, use authenticated signatures/MACs and transport-layer security beyond this prototype.
