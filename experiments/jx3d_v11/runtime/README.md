# Ξ⃝ Jarvis-X — Native 3D Electronic / Mathematical Engine v11

A **real, locally executable C++20 3D data-processing prototype** built from the supplied Autonomous OS v9 visualization and the tested v10 CPU runtime. This release adds a **trainable 3D tile neural autoencoder**, 8-bit electrical-logic observability, a self-contained neural latent-plus-residual stream, and corresponding verified controls. The electronic interpretation is a **digital CPU/bitwise emulation**; it does not require, control, or claim an actual photonic, analog, ASIC, FPGA, or remote cloud accelerator.

## Immediate launch

Requirements: Python 3.10+, Linux `g++` or macOS `clang++` with C++20 support, and a current browser. No `pip` or `npm` dependencies.

```sh
cd JarvisX_3D_Operational_v11
python3 serve.py
```

Open `http://127.0.0.1:8765`. Native C++ compiles automatically on first launch. `python3 serve.py --dimension 16 --port 8766` uses a 16³ = 4,096 voxel grid. Supported side dimensions are **even integers 2..32**. The server listens on **127.0.0.1** by default. Do not publish the development HTTP server directly to the public Internet.

To test:

```sh
./run_tests.sh
python3 benchmark.py --dimension 10 --iterations 1000
```

The original v9 HTML is included under `source/`. To run the new system use the Python server, not the original standalone HTML.

## Native end-to-end execution — what actually happens

1. **3D addressing:** The 3D grid has `N = D³` real native-memory cells. Address `i = x + D*y + D²*z`. Each cell has a float32 state and quantized unsigned 8-bit view.
2. **Mathematical evolution:** On each native CPU tick, healthy cells evolve using a bounded 6-neighbour spatial stencil. Faulted voxels are temporarily zeroed and restored after 18 native ticks.
3. **Electronic digital path:** Actual `uint8_t` values are converted to eight packed bitplanes with `B_b(i) = (Q8[i] >> b) & 1`. Bit transitions are computed as `Σ popcount(Q8_t[i] XOR Q8_(t+1)[i])`. These count logical switching events **in the simulated Q8 state**, not physical transistor switching, electrical current, energy, or wall-power measurements.
4. **Spatial neural autoencoding:** Raster-order voxels are re-ordered into adjacent **2×2×2 tiles**. Each tile `x∈[0,1]^8` passes through a **shared learned 8→4→8 network** with **76 float32 parameters** (32 encoder weights + 4 encoder biases + 32 decoder weights + 8 decoder biases):

   ```text
   z = tanh(W_e x + b_e)               # 4-dim latent
   z8 = round(127.5 * (z + 1))         # four 8-bit latent symbols
   z' = z8/127.5 - 1                   # decoder latent input
   x_hat = sigmoid(W_d z' + b_d)       # 8 reconstructed voxel intensities
   ```

   Model training is native C++ full-batch analytical backpropagation (SGD with a backtracking, non-increasing-loss acceptance test). Training targets **continuous-latent MSE**. The dashboard displays an 8-bit-quantized-latent reconstruction; rounding means its MSE need not numerically equal the continuous training MSE.
5. **Neural residual encoding:** The neural predictor alone is **lossy**. The `.jxa` file holds the shared model's float32 weights, four Q8 latent codes per tile, and an exact XOR correction for each original Q8 voxel byte. For each tile byte: `R = Q8 XOR Q8_hat`, and reconstruction uses `Q8 = Q8_hat XOR R`. Inner bitstreams are encoded with the existing CRC32-protected JX3D byte codec; the overall neural stream also checks the CRC32 of the reconstructed source bytes.
6. **Self-contained decoding:** `.jxa` import loads the embedded model, decodes the Q8 latents, regenerates predicted Q8 bytes, XORs residuals, checks CRC32 and grid dimensions, then **transactionally** replaces the current state. Bad or tampered streams must not alter the loaded state.
7. **Control plane and 3D projection:** A localhost Python `ctypes` server invokes the C++ library and reports actual state, neural loss, bitplanes, path statistics, measured CPU timings and snapshots. The browser displays a rotatable perspective projection of native voxels. Choose the actual Q8 field, reconstructed ANN field, or any of the eight bitplanes.

## Core equations

For a healthy voxel with at least one healthy face-neighbour:

\[\Psi_{t+1}(i) = 0.78\Psi_t(i) + 0.18\operatorname{mean}_{j\in N_6(i)}\Psi_t(j)+0.04 S(i).\]

If no neighbours are healthy, the neighbour mean is replaced by the same cell state. Each step clamps to `[0,1]`. For fixed source and fault topology, the map is a contraction with sup-norm Lipschitz constant **at most 0.96**. This bound does not apply across discrete fault injection, timed restoration or source changes.

Network training minimizes:

\[L(\theta)=\frac{1}{8T}\sum_{t=1}^{T}\left\|\sigma(W_d\tanh(W_e x_t+b_e)+b_d)-x_t\right\|_2^2,\quad T=D^3/8.\]

A trained latent is not itself invertible: the explicit XOR residual is what guarantees byte-exact reconstruction of **quantized Q8 inputs**, not exact reconstruction of pre-quantization floating-point values. Files may be **larger** than the raw Q8 input; the ANN is a trained processing pipeline, not a demonstrated compression breakthrough.

## API routes

| Method | Endpoint | Action |
|---|---|---|
| GET | `/api/health` | Native core availability |
| GET | `/api/state` | Raw Q8, neural predicted Q8, active faults, routing and measured telemetry |
| GET | `/api/bitplanes` | Eight packed Q8 bitplanes, popcount summaries |
| POST | `/api/auto` | `{"enabled":true/false}` native stepping |
| POST | `/api/step` | `{"steps":1}` bounded native stepping |
| POST | `/api/fault` | `{"index":123}` or `{}` to choose random healthy cell |
| POST | `/api/route` | `{"from":0,"to":999}` shortest path A* |
| POST | `/api/reset` | `{"seed":20261104}` |
| POST | `/api/ann/train` | `{"epochs":100,"learning_rate":1.0}` |
| GET | `/api/ann/preview` | Real 8→4→8 Q8 reconstructed 3D grid |
| GET | `/api/ann/export` | Self-contained `.jxa` neural code + residual |
| POST | `/api/ann/import` | `application/octet-stream` `.jxa` validated import |
| POST | `/api/verify` | Verify original lossless spatial JX3D roundtrip |
| GET | `/api/export` | Original `.jx3` snapshot |
| POST | `/api/import` | Validated `.jx3` import |
| POST | `/api/codec` | UTF-8 lossless text codec demo |

The numerical fields `ann_epochs`, `ann_last_mse`, `switching_bits_total`, `last_switching_bits` and `byte_updates_total` refer to executed CPU operations and model data. `byte_updates_total` counts logical per-tick voxel updates, not cache-line or DRAM bus writes.

## Important boundaries

- **Verified:** C++ compilation; native stencil, true 3D tiling, trainable shared ANN, backprop gradient acceptance, spatial bitplanes, Q8 bit-transition counts, shortest-path A*, fault recovery, CRC checking, two kinds of byte-exact snapshot roundtrip, and HTTP integration.
- **Not claimed:** a trained general-purpose LLM, trillion³ parameters, optical hardware, energy efficiency, live cloud deployment, autonomous code modification, or physical electron-level simulation.
- **Browser validation:** JavaScript passed a Node.js syntax check; HTTP control-plane endpoints were tested. Automated browser screenshot verification was blocked by the evaluation environment's browser administrator policy, so pixel-perfect rendering is **not certified**.
- **Security:** Local development server, not production-authenticated; do not expose `--host 0.0.0.0` without adding authentication, authorization, origin protection, TLS, quotas and deployment hardening.
- **Next scalability layer:** GPU/OpenCL/CUDA kernels, tiled streaming, authenticated distributed jobs, and trained multimodal encoders are further development, not present.

See `VALIDATION_REPORT.md` for the exact tests and measured local benchmark. A sample verified JXA1 file can be found under `samples/`.
