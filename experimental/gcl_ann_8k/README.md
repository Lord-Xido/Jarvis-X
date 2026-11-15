# JARVIS-X Ω³ — fused 8K Graphics Control Logic ANN (experimental)

Standalone, opt-in **C++17 CPU reference and optional CUDA fused renderer** for the user-proposed 7680 × 4320 GCL-ANN pipeline. Does not change existing repository engines. The architecture uses a **16×16 output tile plus one texel halo (18×18)**; 256 threads cooperatively load 324 `Sample` values (16 bytes each) into ~5.1 KiB shared memory, extract Sobel edges/contrast and a 3×3 Gaussian blur, execute an 8→64→32→16→6 graphics controller, apply **SDR** creative color transformations, and emit one R10G10B10A2 `uint32_t` per output pixel.

## Build / execute

```sh
cmake -S experimental/gcl_ann_8k -B /tmp/jx-gcl8k -DGCL_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/jx-gcl8k -j2
ctest --test-dir /tmp/jx-gcl8k --output-on-failure
/tmp/jx-gcl8k/gcl8k --width 64 --height 36 --output /tmp/demo.rgb10a2
# On a CUDA-equipped machine, omit -DGCL_ENABLE_CUDA=OFF and run:
# /tmp/jx-gcl8k/gcl8k --cuda --width 640 --height 360
# /tmp/jx-gcl8k/gcl8k --cuda --full --output /tmp/frame_8k.rgb10a2
```

The file bytes are **little-endian R10G10B10A2**: bits 0–9 R, 10–19 G, 20–29 B, 30–31 opaque alpha. `--full` requires 132,710,400 output bytes (126.56 MiB) and consumes device/output memory. A 60 Hz output-only lower bound is 7.963 GB/s. No display driver integration, scanout synchronization, video timing or real-time capability is claimed.

## Model & correctness boundary

Weights are reproducible *untrained*, pseudo-random **binary16-representable** values (seed=42). CUDA stores the 3,286 weights/biases as `__half` in constant memory but **accumulates in FP32 scalar arithmetic**. This does **not** invoke Tensor Cores. CPU and GPU share mathematical operations but transcendental functions, rounding and FMA choices may produce small quantization differences. GPU parity and GPU latency must be measured on real supported CUDA hardware; CI does not claim them.

Feature vector: `[u,v,R,G,B,Gaussian-local-contrast,normalized-Sobel-edge,t]`. Output controls map gains 0.4..1.6, sharpening 0..1, gamma 1.6..2.6, dithering 0..1. Sharpening uses a *signed* 3×3 high-pass residual in source sRGB code-value space (illustrative SDR processing). Rec.709→Rec.2020 primary conversion followed by sRGB transfer is **not** HDR10/PQ/HLG. The cube of 10^18 addresses is a **logical visualization coordinate domain** only—no 3D voxel ANN or allocated 10^18-site tensor exists here. No training, backprop, automatic optimization, encryption, Tensor Core speedup, or sustained 8K60 performance is implemented.

## Operational invariants

- Unmodified source is generated in the shared tile **before** inference; `__syncthreads()` occurs before Sobel reads.
- Border pixels use **clamped global source coordinates**, never modulo-buffer aliasing.
- CPU tests check deterministic frame hashes, packed channel ordering, rectangular dimensions and layer layout.
- CUDA is optional; a 0-device GPU environment fails explicitly with a message rather than silently claiming CUDA success.
- Record GPU timing with `cudaEventRecord` and `cudaDeviceSynchronize` in a GPU benchmark environment; compare with a staged cooperative-GEMM/Tensor Core alternative.

## Authorship

Architecture originates with the user's **FUSED 8K GCL-ANN PIPELINE ARCHITECTURE**. This experimental implementation and its engineering caveats are an adaptation; performance/correctness beyond CPU test evidence remains to be verified.
