# ADR-0039: Isolated runtime-dispatched AVX-512 volume pooling codec

Status: proposed integration candidate, 2026-11-13.

## Context

Dr Matladi Maxwell Moagi supplied a C++17 AVX-512 2x2x2 mean-pooling codec.
Its channel planes duplicate one mean, its decoder's vector broadcast is unused,
and global ISA requirements prevent safe execution on unsupported machines.
Existing native volumetric-fold and trainable autoencoder modules implement
different transforms. This addition needs its own mathematical/evidence boundary.

## Alternatives considered

1. Compile the complete program with AVX-512 enabled: simple but unsafe as a
   fallback-capable binary and does not correct redundant per-block/channel work.
2. Keep only a scalar reference: portable but omits the requested SIMD backend.
3. Isolate scalar and targeted AVX-512 kernels behind capability dispatch: selected.

## Decision

Add `experimental/avx512_volume_codec_v1/` with a scalar reference, targeted SIMD
encoder/decoder/MSE, explicit requested/automatic backend selection, and synthetic
voxel receipts. No canonical runtime calls this experiment. Pooling is lossy;
the float-payload ratio is 8/C. Repeated channels are not trained features.
Quantized latents are rejected until a real format is defined.

## Consequences and compatibility

Mean and MSE accumulation use doubles. The draft's public MSE member changes
from float to double; no existing repository API or persistent format is migrated.
Callers compile this isolated API explicitly. Scalar/native numerical equivalence
uses equal coarse/reconstructed floats and a documented MSE tolerance. Caller
storage scales with template dimensions; the example is deliberately bounded.

Malformed data is rejected before codec work, explicit unsupported native requests
throw, automatic requests fall back, and const input state remains unchanged on
failure. There is no on-disk codec or atomic file-state protocol in this API.

## Validation and promotion boundary

Nine runtime groups, four compile-rejection cases, a scalar-only build, independent
Python block/voxel audits, AVX object inspection, ASan/UBSan and focused GCC/Clang CI.
The local host lacks AVX-512: native execution/parity and all performance claims
remain unverified locally. Hardware-dependent checks print their executed/SKIP
status. A future promotion needs suitable hardware evidence and review; there is
no dependency on merging other open research PRs.
