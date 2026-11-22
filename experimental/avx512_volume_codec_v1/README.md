# AVX-512 volume pooling codec v1

**Classification: integration candidate / numerical demonstration.**
Original supplied `AVX512VolumeAutoencoderCodec` concept and prototype:
Dr Matladi Maxwell Moagi. Runtime dispatch, vector lowering, numerical guards,
tests and evidence: OpenAI Codex, 13 November 2026. Repository LICENSE applies.

This isolated C++17 module implements the supplied 2x2x2 mean-pooling and block
replication transform. It has no learned weights, training procedure, entropy
coder, quantizer or serialized compressed bitstream. Existing ANN/codec runtimes
and the canonical VM are not called or changed by this module.

## Transform and storage

Input layout is X-fastest: `i = z*Height*Width + y*Width + x`. For each coarse
coordinate `(z,y,x)`, the encoder averages the eight input voxels beginning at
`(2z,2y,2x)`. Each channel plane stores that same average. The decoder averages
channel values at a coarse coordinate, then replicates the result to its eight
fine voxels. The channel-mean rule also applies to externally supplied latents.

For input size N and C channels:

    latent floats = N*C/8
    input-float-payload / latent-float-payload = 8/C

These counts exclude object padding, the bool flag, buffers and any future file
metadata. C=1 gives 8:1; C=8 gives 1:1; C>8 expands the float payload. Repeated
channels contain no additional learned features. Reconstruction is generally
lossy, with MSE measured against the caller's reference input.

## Changes from the supplied draft

- Require positive even dimensions, positive channels and non-overflowing element/byte counts.
- Reject non-finite input/reference/latent values and reject `is_quantized=true`.
- Compute each coarse mean once, then copy repeated channel planes.
- Vectorize eight independent blocks per encoder accumulator, rather than gathering
  eight samples plus eight zeros into one vector for each channel/block.
- Vectorize decoder block replication in groups of sixteen coarse means.
- Accumulate means and MSE in doubles to handle finite float32 extremes; MSE's
  public member type is deliberately `double` rather than the draft's `float`.
- Isolate AVX-512 code in target-specific functions and dispatch at runtime.
- Keep a scalar backend and a build with all AVX-512 code disabled.

The transform is mathematically the same pooling/replication rule, but double
accumulation changes rounding relative to the draft's float horizontal reduction.
Bit-for-bit equivalence to that draft is not asserted. Scalar and AVX-512 means
use the same ordered double additions. Native parity tests require equal latent
and reconstruction values and MSE agreement within `1e-12*max(1,abs(reference))`.
AVX MSE uses FMA and a different reduction order, so its bits need not match.

## Build and use

```bash
cd experimental/avx512_volume_codec_v1
python3 verify.py --sanitize
# Or directly, with no global -mavx512f / -march=native:
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror \
    -ffp-contract=off scalar.cpp avx512.cpp demo.cpp -o codec_demo
./codec_demo output
```

```cpp
#include "codec.hpp"
using Codec = jx::volume::AVX512VolumeAutoencoderCodec<8,8,8,1>;
Codec codec; // Auto selects a supported backend.
std::array<float, Codec::InputVolumeSize> input{};
const auto latent = codec.encode(input);
const auto reconstruction = codec.decode(latent, input);
```

`Backend::Scalar` always selects the reference path. Explicit `Backend::AVX512`
throws if the compiler/build/CPU is unavailable; automatic selection falls back
to scalar. `-DJX_DISABLE_AVX512` removes native implementation code. GCC/Clang
x86 target attributes and CPU builtins are used; other builds provide scalar
fallback. Never globally enable AVX-512 or fast-math for a fallback-capable build.
See [GCC CPU detection](https://gcc.gnu.org/onlinedocs/gcc/x86-Built-in-Functions.html).

## Validation and actual hardware boundary

`verify.py` builds with warnings as errors, runs nine runtime groups, compiles
four deliberately invalid shapes, tests a scalar-only build, and independently
checks every demo block/voxel using Python. On x86 it inspects the AVX object for
ZMM and FMA instructions. Optional ASan/UBSan checks cover the executable tests.
Leak detection is disabled for the traced hosted runner; no leak check is claimed.

The numerical tests cover analytic ramp/impulse/constant inputs, six shapes,
multiple channel counts, encoder/decoder vector prefixes and tails, long-double
reference MSE, external latent channels, invalid data, float32 extremes, generic
MSE tails, backend rejection and replay. Native AVX-512 parity executes only when
runtime capability detection succeeds; otherwise the output explicitly says SKIP.

The checked-in `evidence/local_validation.json` records a local GCC 14.2 x86_64
run: all scalar checks and sanitizer checks passed; the AVX-512 object compiled,
but native execution was unavailable. No AVX-512 speedup, throughput, energy
measurement or native-hardware parity is claimed from that local run. The CI
workflow runs GCC and Clang, publishing fresh receipts and preserving skip status.

The deterministic synthetic 6x4x34 demo has 816 input floats (3,264 bytes), 102
latent floats (408 bytes), and local scalar MSE `0.2525250002454355`. Evidence
includes all 102 coarse means and all 816 input/reconstruction/residual rows.
Checksums identify the source version that produced the local receipt.

## State, failure and resource contract

The input and latent arguments are const. Encode/decode return newly constructed
values; validation errors and allocation failures leave caller input unchanged.
There is no persistent state, external I/O in codec methods, model loading or
network access. The demo alone writes files. Template dimensions fix work and
memory; arrays returned by value can be large, so callers must provision their
storage/stack appropriately. Decode additionally allocates N/8 float means.
No full-scale volume is silently allocated by the demo, and no process isolation
or hostile-input sandbox is asserted. Internal raw-pointer kernels require the
wrapper's valid dimensions and correctly sized, nonoverlapping buffers.

See [ADR-0039](../../docs/adr/0039-avx512-volume-pooling-codec.md). This is a separate
pooling experiment, not a replacement for the existing trainable autoencoder.
