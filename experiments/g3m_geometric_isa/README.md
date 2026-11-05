# G3M Geometric ANN + 3D Bytecode Plan (isolated research module)

Status: **experimental, source-only**. Two independent components are included:

1. **`bytecode/`** — Python program emitter and C++17 validator/disassembler for a fixed 15-op, 272-byte `G3MBC001` execution **plan**. The plan checks binary layout, CRC32, typed registers, shapes, and instruction dependencies. It does **not** execute ANN kernels or create an evidence ledger.
2. **`native/`** — bounded C++17 LibTorch geometric point-cloud ANN. Supports XYZ float32 input; support subsampling; kNN, EdgeConv, Euclidean-distance-biased attention, 512D latent, folding decoder, training, checkpoints and empirical candidate-first refinement. It operates **separately** from the bytecode plan.

These are **not yet an integrated bytecode-to-LibTorch dispatcher**. Existing Jarvis-X canonical VM semantics and architecture are unchanged.

## Offline bytecode validation (standard-library only)

```bash
python3 experiments/g3m_geometric_isa/bytecode/tests/test_plan.py
```

## Optional native LibTorch build

```bash
cd experiments/g3m_geometric_isa/native
cmake -S . -B build -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build build -j1
python3 tests/test_engine.py ./build/g3m
```

The native test requires LibTorch. GitHub Actions CI for this module validates the bytecode toolchain only; it does **not** claim native LibTorch ANN execution in hosted CI.

## Accuracy and security boundaries

- The neural autoencoder is **lossy**; 512 latent numbers cannot losslessly represent arbitrary million-point clouds.
- `N=1,000,000` supports million-point input/output, while expensive attention and graph operations use bounded support, typically 128 points. No 1,000,000× speedup or quality-matched advantage is proven.
- The bytecode plan's CRC32 detects accidental errors, not malicious modification. No signatures, authenticated bytecode or secure execution sandbox.
- The fixed plan is not a programmable virtual machine for arbitrary opcodes. The numerical LibTorch implementation is not controlled by this bytecode.
- The latent feedback loop implements a local candidate gate, not a proof of contraction, autonomous learning or correctness.
- CPU-only reference, synthetic datasets; no claims of cloud deployment or production safety.

See [`VALIDATION.md`](VALIDATION.md) and component READMEs for precise bounds. This module is separate from the canonical `cpp_runtime` VM and the merged `moagi3d_operational_v2` codec.
