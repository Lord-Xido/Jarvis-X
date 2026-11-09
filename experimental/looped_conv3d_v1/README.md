# JARVIS-X Ω³ — Looped 3D Conv Autoencoder v1 (experimental)

A **runnable, bounded PyTorch CPU reference implementation** of the looped 3D neural autoencoder diagram. This is a *new independent experimental module*; existing mainline engines and draft PR #485 are unchanged.

## Precisely implemented

- **Input:** one-channel UINT8 64³ or 128³ volume, converted to float32 /255 for inference. Synthetic source used by the CLI is explicitly quantized to UINT8 before conversion to float.
- **Encoder:** five trainable `Conv3d(kernel_size=3, stride=2, padding=1)` stages with channels `[4,8,12,16,16]` and SiLU. Spatial steps: **64→32→16→8→4→2** or **128→64→32→16→8→4**.
- **Latent:** flatten + learned projection to **512 FP32 values** (2,048 bytes), `z∈R^512`. Decoder projects back to the correct 2³ or 4³ bottleneck.
- **Decoder:** five trainable `ConvTranspose3d(kernel_size=4, stride=2, padding=1)` stages with channels `[16,12,8,4,1]`, SiLU between layers and sigmoid output in `[0,1]`; exact 64³ / 128³ dimensional restoration.
- **Loopback:** decode transmitted initial latent **once**; at every later cycle, re-encode the *previous decoder output*, decode it again, and record a hash and reconstruction metrics where the reference exists. The decoder-only path never accesses the original source.
- **Parameter optimization:** full backpropagation through 1–4 unrolled cycles, average original-referenced MSE, and up to 16 candidate updates; explicit line-search with **commit only for lower objective**, otherwise rollback of **every model parameter**. No optimizer momentum/state. Tested with forced highly unstable updates.
- **Original-reference guard:** optionally selects the best prior reconstruction, rejecting later non-improving loopback outcomes. This is for training/evaluation **only**. It is **disallowed** for an archive decoder without the original reference.

### Recurrence

For the original volume `X0`, at cycle `t`:

    z_t = E_theta(X_t)
    Y_t = D_phi(z_t)
    X_(t+1) = Y_t
    L(theta, phi) = mean_t( mean_voxels((Y_t - X0)^2) )

`(theta', phi')` is accepted only if the complete recalculated objective is strictly smaller. This gives **monotone accepted training objective**, *not* monotone fidelity for each individual loopback cycle. Re-encoding an untrained decoder's output can leave reconstruction unchanged or make it worse.

## Checkpoint and latent-stream fidelity

- Model checkpoint: `safetensors` weights with side/format metadata (safe tensor loading rather than Python pickle), SHA-256 digest stored in the latent header. The checkpoint is **required at decode time**.
- Latent archive: **2,128 bytes** = 48-byte header (`JX3DL512`, UINT32 side, UINT32 latent count, 32-byte exact model SHA-256), **2,048 bytes** of little-endian FP32 latents, and 32-byte SHA-256 checksum of preceding bytes.
- Strictly reject wrong archive length/magic/dimensions, model checkpoint mismatch, checksum mismatch, nonfinite latent data, and checkpoint metadata mismatch.
- Raw archive SHA/checkpoint SHA detect accidental corruption; **bare hashes do not authenticate an adversary**. Treat the checkpoint as a trusted dependency.
- Same-machine save/load replay uses exact `torch.equal` between fresh and checkpoint-reconstructed outputs. Cross-platform bitwise identity is **not** promised.

## Actual byte accounting

| Item | 64³ | 128³ |
|---|---:|---:|
| One original UINT8 volume | 262,144 bytes | 2,097,152 bytes |
| 512 FP32 latent-only | 2,048 bytes | 2,048 bytes |
| Full latent archive | 2,128 bytes | 2,128 bytes |
| Trainable FP32 parameters | 184,589 | 1,102,989 |
| Saved model checkpoint (observed locally) | 740,388 bytes | 4,414,004 bytes |
| Raw volume / latent archive, *excluding model* | 123.19× | 985.50× |
| Raw volume / (archive + checkpoint) for a *single* volume | **0.353× (expansion)** | **0.475× (expansion)** |

The much-touted **512×** from `262144 / 512` is only a ratio of **element counts**, not encoded bytes. The model weights are substantial. The demo **trains on the same synthetic volume being encoded**, so it does **not** establish shared-model compression across a corpus. Genuine amortized compression requires a separately trained shared decoder and testing on held-out real datasets.

## Run (Python 3.11+, CPU)

From this directory:

    python -m pip install numpy safetensors pytest
    python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
    python -m pytest -q -W error -o addopts='' test_engine.py
    python engine.py demo --side 64 --updates 2 --cycles 2 --out local/64
    python engine.py demo --side 128 --updates 1 --cycles 2 --out local/128
    python engine.py decode --side 64 --cycles 2 \
        --checkpoint local/64/jx3d-64.safetensors \
        --archive local/64/sample-64.jxz512 \
        --out local/64/reconstruction.raw

The CLI `demo` creates one synthetic source and writes the checkpoint, encoded latent archive and `trace.json`. `decode` reconstructs from **only** archive + matching model. It has no quality guard because the reference volume is absent.

## Auditable evidence (local PyTorch 2.10.0+cpu, 2026-11-09)

| Test | 64³ | 128³ |
|---|---:|---:|
| Initial untrained cycle-0 normalized MSE | 0.02779017 | 0.01752679 |
| Training objective before | 0.02779017 | 0.01752679 |
| Training objective after | 0.02661518 | 0.01719213 |
| Accepted candidate updates | 2/2 | 1/1 |
| Final raw loopback PSNR (against synthetic reference) | 15.749 dB | 17.647 dB |
| After second cycle | No further measured improvement | No further measured improvement |
| Checkpoint + archive replay | Exact on local machine | Exact on local machine |

**6 pytest cases passed** (including both input sizes, latent roundtrip, corruption, 3-cycle loopback, forced parameter rollback and full archive replay). All tests use **synthetic** data; the PSNR is relatively low and does not justify image-quality or speedup claims. No 1,000 GB/s, 1,000×, clinical reliability, GPU throughput, or cloud-service deployment is claimed. CI is a GitHub-hosted CPU check, not a permanent production service.

## Trace discipline

Every cycle records `cycle`, `opcode` (`0x30` decode, `0x10>0x30` loopback), SHA-256 of 512 FP32 latent values, SHA-256 of reconstructed UINT8 bytes, normalized MSE/PSNR where a reference was supplied, and `accepted` / `not_improved` / `rollback` / `unverified`. The opcode labels are **trace tags**, not an executable general-purpose VM. A separate train record lists accepted and rejected update proposals with learning rates.

No existing baseline source was changed; release, merge, model trust, broader dataset benchmarking and infrastructure deployment require separate review.
