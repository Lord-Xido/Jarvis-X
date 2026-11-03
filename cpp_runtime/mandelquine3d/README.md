# Dr Moagi Mandelbulb 3D ANN — Inward Verification Suite

An **executable, reproducible C++20 audit** built around the original Dr Moagi C++ multimodal Mandelbulb ANN Quine engine. Includes the original engine, independent gradient and 3D recurrence verifiers, genuine whole-archive inward round trips, generative cascade probes, source Quine verification, real 3D traces, and an offline interactive renderer.

## Contents

| File | Purpose |
|---|---|
| `dm_mandelquine.cpp` | Unmodified original C++20 engine and Quine |
| `dm_inward_verifier.cpp` | Independent C++20 mathematical+pipeline audit, includes original translation unit with renamed `main` |
| `3D_inward_verification_interactive.html` | Offline interactive 3D voxel visualization; rotate, zoom, playback, inspect all four latent dimensions and Mandelbulb field |
| `VERIFICATION_REPORT.md` | Equations, proof, measured results, limitations and security gap |
| `latent_field_t00.png`, `latent_field_t04.png`, `latent_field_t24.png` | Rendered states of C++-measured field |
| `convergence.png` | Measured iteration convergence |
| `demo_outputs/*` | Original text, image, audio and archive sample inputs and outputs |
| `audit_outputs/recurrence.csv` | Measured convergence and perturbation norms, 24 rounds |
| `audit_outputs/spatial_trace.csv` | 729 occupied voxels × 25 time steps with full 4D latent vectors |
| `audit_outputs/loop_summary.csv` | Four modality audit results |
| `audit_outputs/generative_cascade.csv` | 3-generation image output changes |
| `audit_outputs/audit_report.txt` | C++ test log |
| `audit_outputs/cube_729.dmq` | Native 3D archive used for recurrence validation |
| `audit_outputs/tampered_*.dmq` | Adversarial fixtures demonstrating integrity weaknesses |
| `run_all.sh` | Repeat the complete build and audit |

## Usage

```bash
bash run_all.sh
# Alternatively:
g++ -O2 -std=c++20 -Wall -Wextra -pedantic dm_mandelquine.cpp -o dm_mandelquine
g++ -O2 -std=c++20 -Wall -Wextra -pedantic dm_inward_verifier.cpp -o dm_inward_verifier
./dm_inward_verifier audit_outputs dm_mandelquine.cpp
```

Open `3D_inward_verification_interactive.html` directly in a browser. It uses local HTML canvas only; there are no CDN dependencies or external requests.

## What is established

- Correct 3D coordinate indexing on tested shapes, verified Mandelbulb feature boundedness.
- Correct analytic ANN backpropagation checked numerically on **all 269 scalars**.
- Frozen-base recurrence contraction upper bound `0.68 + 0.32×0.18 = 0.7376` in the maximum norm.
- Original recursive field after four steps precisely matches the separately implemented recurrence on the tested 729-cell cube.
- Full modality inputs re-encode and restore identically across **three rounds**; archive bytes are identical under fixed seeds.
- Byte-for-byte self-reproducing Quine reproduces the original C++ source.
- Explicitly demonstrated absence of archive integrity authentication (residual tampering goes undetected).

## What is *not* established

The archive's byte-perfect reconstruction is XOR-residual-dependent; it is **not** evidence of lossy-free neural encoding or compression. The ANN is not trained through the 3D recurrence, and its generated outputs are not shown to reach fixed points. A true semantic LLM, robust multimedia generative model, autonomous code evolution and massive parallel scaling are not implemented.

## GitHub integration

This implementation lives in `cpp_runtime/mandelquine3d/` and is CI-tested using `.github/workflows/mandelquine3d.yml`. Run `bash run_all.sh` here to generate fixtures, compile both C++20 executables, validate the source Quine, and execute the mathematical audit. Generated outputs are excluded from source control.
