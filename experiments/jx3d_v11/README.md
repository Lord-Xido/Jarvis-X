# Jarvis-X v11 — 3D Neural Encoding Reference + Executable Runtime

This is an **isolated, source-reviewed implementation** of the locally verified Jarvis-X 3D CPU prototype. It does not overwrite existing Jarvis-X cloud, native, dashboard or production pathways.

## Two independently tested entry points

**Compact mathematical reference:** C++20 header and regression tests in `src/jx_ann.hpp` and `tests/reference.cpp`.

```bash
g++ -std=c++20 -O2 -Wall -Wextra -Wpedantic -Werror \
  experiments/jx3d_v11/tests/reference.cpp -o /tmp/jx3d_v11_reference
/tmp/jx3d_v11_reference
```

**Complete local CPU runtime and browser dashboard:** the `runtime/` subtree contains the C++20 C ABI implementation, Python localhost HTTP server, interactive projected 3D voxel view, neural training API, bitplane extraction, A* routing, fault recovery, CRC-checked Q8 snapshot transfer and native + HTTP regressions.

```bash
cd experiments/jx3d_v11/runtime
bash ./run_tests.sh
python3 serve.py
# Open http://127.0.0.1:8765
```

Requirements: Python 3.10+, C++20 compiler and modern browser; Node.js is only needed for checking JS syntax. No third-party Python dependencies. Do not expose the development server publicly.

## Operational mathematical model

For healthy voxels, with face-neighbour average `m_i` and bounded source `S_i`:

```text
i = x + D*y + D*D*z,  0<=x,y,z<D
psi_next[i] = clamp(0.78*psi[i] + 0.18*m_i + 0.04*S_i, 0, 1)
tile = adjacent 2x2x2 Q8 voxels
z = tanh(W_enc*tile + b_enc)        # eight inputs, four latent dimensions
z_q = round(127.5*(z+1))            # Q8 latent
x_hat = sigmoid(W_dec*(z_q/127.5-1)+b_dec)
residual = source_Q8 XOR predicted_Q8
decoded_Q8 = predicted_Q8 XOR residual
```

The 76 trainable float32 parameters are shared across tiles. A valid XOR residual gives *byte-exact Q8* recovery; the pre-quantized float field is not preserved exactly. The lattice map has an L-infinity Lipschitz bound at most 0.96 only under fixed fault topology and source.

## Verification

`.github/workflows/jx3d-v11-native.yml` runs both independent C++20 regression kernels and the full HTTP/JS test suite. `runtime/VALIDATION_REPORT.md` records local benchmark observations and limitations. Original minified v9 HTML and compiled libraries are intentionally not committed; `runtime/web/` is the maintained source-visible UI.

**Not implemented:** trillion-cubed actual ANN parameters, physical transistor/electron or photonic acceleration, cloud deployment, production authentication, certified browser rendering or compression improvement. Full runtime remains a local CPU prototype.
