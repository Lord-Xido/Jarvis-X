# Moagi Unified 3D Runtime

A single-file Python reference runtime for the Jarvis-X sparse recursive 3D auto-encoding/decoding architecture.

## Canonical architecture

This runtime is an implementation layer beneath the **Dr. Moagi 1K³ Cloud Autoencoding/Decoding Organism**, canonically attributed within Jarvis-X as:

> **Conceived and specified by Dr. Matladi Maxwell Moagi.**

The canonical distributed architecture is defined in:

- [`DR_MOAGI_1K3_CLOUD_ORGANISM.md`](./DR_MOAGI_1K3_CLOUD_ORGANISM.md)

Its invariant is:

```text
1024³ virtual sparse body
  -> 32³ cloud organs
  -> local recurrent latent cores
  -> Omega memory
  -> global latent core
  -> residual reconstruction
  -> Top-K error-guided regeneration
  -> adaptive cloud rescheduling
  -> recur
```

The existing `runtime.py` and `terabyte_3d_engine.py` are concrete software substrates for progressively operationalizing that architecture.

## VOXEL3D ROM permeation layer

The unified runtime now has a reference boundary for the structured `\x7FVOXEL3D` ROM image:

- [`VOXEL3D_ROM_RUNTIME.md`](./VOXEL3D_ROM_RUNTIME.md) — binary identity, ABI rules, state model, CTR boundary, and end-to-end pipeline.
- [`voxel3d_rom_runtime.py`](./voxel3d_rom_runtime.py) — strict ROM parser plus reference orchestration bridge.
- [`test_voxel3d_rom_runtime.py`](./test_voxel3d_rom_runtime.py) — validation, anchor, raw-word, and CTR commit-gate tests.
- [`voxel3d_visualizer.html`](./voxel3d_visualizer.html) — dependency-free live browser visualization of the inward recursive runtime.

The ROM layer follows:

```text
ROM byte image
  -> decode
  -> spatial VM
  -> SVO compute / encoder
  -> latent core Z_t
  -> decoder
  -> reconstruction X_hat_t
  -> residual R_t
  -> CTR verify / correct
  -> scheduler Pi_t
  -> engine state
  -> raymarcher
  -> frame output
  -> recur
```

Unknown VM opcode semantics remain explicitly `UNSPECIFIED` until the ABI defines them. The reference layer therefore does not claim that the observed ROM bytes are native CPU/GPU instructions.

## Operational pipeline

```text
input
  -> sparse virtual 1000^3 field
  -> 1000^3 -> 100^3 -> 10^3 -> 1^3 contraction
  -> residual pyramid
  -> 64-dimensional latent state
  -> 64 relaxed fixed-point folds
  -> Omega temporal memory
  -> sparse multiresolution decode
  -> residual/error field
  -> threshold + Top-K active selection
  -> selective correction
  -> re-encode and recur
```

The `1000^3` domain is a virtual address space. The runtime stores only active points and residual records; it does not allocate a billion-voxel dense tensor.

## Mathematical core

The latent recursion is

```text
z[k+1] = (1-rho) z[k] + rho tanh(Wz z[k] + Wo Omega + b)
```

with the recurrent memory update

```text
Omega[t+1] = beta Omega[t] + (1-beta) z*[t]
```

and selective correction

```text
Hhat[t+1] = Hhat[t] + lambda * 1[J_t] * (H - Hhat[t])
```

where `J_t` is the thresholded Top-K error set.

## Requirements

- Python 3.10+
- NumPy
- Tkinter for GUI mode (normally included with CPython desktop installations)

```bash
pip install numpy
```

## Run

Interactive 3D viewer:

```bash
python apps/moagi-unified-3d/runtime.py
```

Headless verification:

```bash
python apps/moagi-unified-3d/runtime.py --headless --points 10000 --cycles 8
```

Text ingest:

```bash
python apps/moagi-unified-3d/runtime.py --headless --text "recursive sparse volumetric auto encoding decoding"
```

Disable residual quantization for near-exact residual storage:

```bash
python apps/moagi-unified-3d/runtime.py --headless --residual-quantum 0
```

Inspect a VOXEL3D binary or hex image:

```bash
python apps/moagi-unified-3d/voxel3d_rom_runtime.py path/to/image.hex --step --verified
```

Open the live architecture visualization directly in a browser:

```text
apps/moagi-unified-3d/voxel3d_visualizer.html
```

For the hierarchical streaming implementation, see [`TERABYTE_ENGINE.md`](./TERABYTE_ENGINE.md).

## Grounding

This is a software reference engine running on ordinary hardware. Runtime timings are host measurements. Optical/EM and sub-picosecond execution concepts remain hardware research hypotheses rather than properties asserted by this Python implementation.
