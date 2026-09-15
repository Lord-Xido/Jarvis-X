# Moagi Unified 3D Runtime

A single-file Python reference runtime for the Jarvis-X sparse recursive 3D auto-encoding/decoding architecture.

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

## Grounding

This is a software reference engine running on ordinary hardware. Runtime timings are host measurements. Optical/EM and sub-picosecond execution concepts remain hardware research hypotheses rather than properties asserted by this Python implementation.
