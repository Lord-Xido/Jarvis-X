# Jarvis X — Recursive 3D ANN text/autoencoding engine

**Status: executable research prototype.** This is an isolated Jarvis-X application. It embeds trained neural weights and provides both a local browser UI and a CPU PyTorch inference/training path. It does not modify the repository's existing runtimes.

## Run immediately

Open [`index.html`](./index.html) in a modern desktop browser. It has its 27-symbol pretrained **synthetic-vocabulary** ANN and 3D Conv autoencoder weights embedded: no internet connection, server or GPU required. Choose 1 / 100 / 1,000 streams, press **Generate + re-encode**, rotate the 4×4×4 token lattice, inspect the 2³ intermediate and 1³ latent levels, change the feedback stage and verify exact reconstruction with the XOR residual.

## CPU inference and verification

```bash
python -m pip install -r apps/recursive-3d-ann/requirements.txt
python apps/recursive-3d-ann/runtime.py --streams 1000 --seed 20261029 --iterations 8 --output /tmp/recursive-ann-run.json
python -m pytest -q -o addopts='' apps/recursive-3d-ann/test_runtime.py
```

`runtime.py` extracts and shape-checks pretrained weights embedded in `index.html`, recreates the PyTorch model, samples 64 positions per stream, executes the 3D encoder/decoder, iterates an inward latent feedback operator, and measures reconstruction accuracy and generation-only wall time. `test_runtime.py` verifies the PyTorch reconstruction against the reference embedded in the HTML and tests a complete 1,000 × 64 run.

### Retrain the research model

```bash
cd apps/recursive-3d-ann
python engine.py --streams 1000 --steps 64 --corpus 900 --pred-epochs 5 --ae-steps 90 --inward-iterations 8 --output results.json
python build_dashboard.py
```

`engine.py` generates synthetic training sequences from the original uploaded fixed-logit simulation, trains a one-token-context next-token ANN and a Conv3D autoencoder, and writes `results.pt` and `results.json`. `build_dashboard.py` exports the checkpoint weights and renders a fresh standalone `index.html`-equivalent file named `3D_ANN_Live_Engine.html` from `dashboard_template.html`. Do not interpret training as a pretrained general-purpose language model. Use the newly rendered HTML to inspect a retrained checkpoint.

## Core mathematics

\[
X\in\{0,\ldots,V-1\}^{4\times4\times4},\qquad
E: X\mapsto z\in\mathbb R^{24},\qquad
D:z\mapsto\mathbb R^{64\times V}.
\]

\[
z_{k+1}=(1-\alpha)z_k+\alpha E(\operatorname{softmax}(D(z_k))),\quad \alpha=0.25.
\]

The trained decoder is **lossy**. For token IDs that fit in 8 bits, separately retaining the XOR residual `r = X XOR X_hat` allows `X = X_hat XOR r` exactly. This is not an established compression gain. Inward iteration produces finite states under tested settings but **does not have a proved contraction bound**.

## Reference results (seed 20261029)

| Measure | Reference |
| --- | ---: |
| Parallel independent sequences | 1,000 |
| Tokens per sequence | 64 |
| Generated synthetic-vocabulary tokens | 64,000 |
| Trainable predictor weights | 3,171 |
| Trainable 3D autoencoder weights | 22,695 |
| Generated-batch argmax reconstruction accuracy | 76.83% |
| Residual-assisted token recovery | 100% |
| Inward iterations | 8 |
| Short CPU generation-only sample rate | ~2.08 million synthetic tokens/s |

The rate excludes model training, 3D reconstruction, serialization and rendering; it is not a sustained production benchmark or a large-language-model throughput claim. The CPU runtime now also reports measured pipeline wall time (including model loading, generation, autoencoding, feedback and residual verification, excluding JSON serialization), together with latent/residual byte-accounting. For 64 8-bit token IDs, the 24 FP32 latent elements plus 64 residual bytes cost at least 160 bytes before model weights, so lossless compression has not been demonstrated. Inward trajectories report separate one-step and two-step residual norms; neither constitutes a formal convergence proof.

**Dedicated CI:** `recursive-3d-ann.yml` overrides the repository-wide pytest coverage options for the isolated application and runs the complete test suite and 1,000-stream inference receipt on pull requests and pushes to `main`. The check must pass before the ANN is considered verified; it is not an automatic proof of full-system integration. The browser uses precomputed one-token probability distributions to accelerate its tiny vocabulary. The `4³ → 2³ → 1³ → 2³ → 4³` geometry is **data representation**, not physical 3D computing hardware. Refer to [`reference_results.json`](./reference_results.json) for the captured original benchmark, not a fresh measurement on every machine.

## File map

- `index.html` — self-contained 3D browser UI, learned model weights, reference samples, and live generation.
- `runtime.py` — executable PyTorch inference from the embedded weights.
- `engine.py` — original PyTorch training, generation and benchmark implementation.
- `test_runtime.py` — shape, reference-conformance, seeded output and end-to-end tests.
- `reference_results.json` — baseline benchmark from the original prototype.
- `dashboard_template.html` / `build_dashboard.py` — re-export the HTML after retraining.
- `requirements.txt` — local development dependencies.

**Safety boundary:** no integration with real financial systems, vehicles or physical hardware. No privileged external actions; the browser computes locally.
