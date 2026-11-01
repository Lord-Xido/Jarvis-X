# Dr Moagi / Jarvis-X — WebGL 3D Quine Throughput Laboratory

**Review-only browser reference.** Implements the user-supplied 2026-11-01 WebGL source/byte/texture round-trip concept with measured operational throughput. Complements PR #434 (strict JavaScript quine), PR #435 (quine-ROM), and PR #436 (sparse bitwise 3D bridge). Does not replace canonical runtime or modify the Dr Moagi equation.

## Execution chain

1. Paste UTF-8 source or fetch the running page's own HTML. Data is *never executed*.
2. Pack bytes into RGBA8 texture rows with width 256 (4 bytes/pixel), zero-padding the last row.
3. Run an identity GLSL texture sampling pass to an offscreen unsigned-byte RGBA target.
4. Read target pixels synchronously, considering both direct and vertically flipped row ordering.
5. Verify all original bytes. Any mismatch fails closed, and unverified source is not drawn as verified output.
6. Sample **recovered** bytes into a rotating 3D point cloud; the cloud is visualization, not neural computation.
7. Exclude two warm-up rounds, then collect wall-clock samples from packing, texture and target allocation, render, synchronous readback, byte comparison and disposal. Yielding to animation is outside the measured interval.
8. Export a JSON verification and throughput receipt.

## Usage

From repository root: run a local server with Python's http.server on port 8000; open http://localhost:8000/apps/webgl-quine-throughput/ . The browser loads Three.js from unpkg.com (internet required).

Run independent Node.js 18/22 byte-layout and throughput tests:

    node --test apps/webgl-quine-throughput/codec.test.mjs

Run CPU-only reference benchmark (not a GPU measurement):

    node apps/webgl-quine-throughput/benchmark_cpu.mjs 1048576 20

The HTML requires a real browser GPU. Node.js CI checks the shared layout math but cannot establish WebGL driver correctness or GPU throughput.

## Mathematical throughput contract

For useful input bytes B, width W = 256, H = max(1,ceil(B/(4W))), and n verified complete round trips measured over t seconds:

- Useful throughput = B*n/t bytes/s; bit iteration count = 8*B*n/t bits/s.
- Shader fragments = W*H*n. Each fragment samples one RGBA pixel and writes one pixel.
- Minimum *logical* upload-plus-readback byte volume = 2*4*W*H*n, excluding target writes and driver traffic.
- Byte invariant = every recovered[i] equals original[i] for i in [0,B).
- 3D visualization attributes require 24 bytes per drawn byte (XYZ and RGB float32), but the browser caps visible points at 18,000.

Rates use decimal GB (1e9 bytes), not GiB. Browser median timing is measured, not derived from GPU clock assumptions. The CPU benchmark labels its backend distinctly.

## Resource and capability boundaries

- Max source 4 MiB and device MAX_TEXTURE_SIZE must permit the texture dimensions.
- This is self-reading source *transport*, not a strict self-generating quine.
- Identity GLSL byte transport is not compression, a learned autoencoder, 3D neighbourhood computing or proof of the canonical fixed point.
- Verification can fail depending on GPU/browser byte preservation; it must not be bypassed.
- Virtual 3D dimensions do not imply resident bytes, GPU operations or throughput.
- Neither frame rate nor fragment samples are interchangeable with floating-point operations.
- This PR must be reviewed and hardware-benchmarked before integration into any production authority boundary.
