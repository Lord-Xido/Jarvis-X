# Dr Moagi TiB3D 1000 GB/s Target Processor

## Scope

This component operationalises the exact binary 3D mapping

\[
8192 \times 8192 \times 16384 = 2^{40}\ \text{bytes} = 1\ \text{TiB}.
\]

Each \((x,y,z)\) coordinate addresses one byte:

\[
A(x,y,z)=x+8192(y+8192z),
\]

with

\[
0\le x<8192,\quad
0\le y<8192,\quad
0\le z<16384.
\]

One XY plane contains

\[
8192\times8192=2^{26}\ \text{bytes}=64\ \text{MiB}.
\]

Therefore the logical TiB volume is a stream of 16,384 independent 64 MiB plane tasks.

## Execution model

The executable is intentionally streaming and bounded. It does not allocate a 1 TiB resident array.

Each worker owns one aligned 64 MiB plane buffer and repeatedly executes scheduled logical Z-plane tasks. The default worker count is capped at eight, so the default resident working set is at most roughly 512 MiB plus process overhead.

Two reference kernels are provided:

- xor: reversible in-place byte transformation with AVX-512 / AVX2 paths when compiled for those ISA extensions;
- checksum: read-only four-accumulator streaming checksum used as a second bandwidth-oriented path.

The kernel surface is deliberately small so a future 3D encoder/latent/decoder/residual operator can replace the reference kernel without changing the scheduler or throughput accounting.

## 1000 GB/s contract

1000 GB/s is a measured target, not an assumed property of the software.

For one complete TiB payload sweep,

\[
T_{\text{target}}
=
\frac{1,099,511,627,776}{10^{12}}
\approx 1.0995\ \text{s}.
\]

The executable reports measured payload throughput:

\[
B_{\text{payload}}
=
\frac{\text{processed bytes}}{\Delta t}.
\]

For an in-place read/modify/write kernel such as XOR, approximate physical memory-fabric traffic is about twice payload throughput. Consequently, sustaining a measured 1000 GB/s payload rate can require roughly 2000 GB/s aggregate memory traffic before accounting for additional system overheads.

No CI result is interpreted as evidence that arbitrary hardware can sustain 1000 GB/s.

## Bounded versus full traversal

The default run processes only 64 planes:

\[
64\times64\ \text{MiB}=4\ \text{GiB}
\]

of logical payload. This prevents an accidental 1 TiB sweep during development or CI.

Use:

~~~bash
./DrMoagi-TiB3D-1000GBs --threads 64 --full-sweep --kernel xor
~~~

to traverse the complete logical TiB volume.

Machine-readable telemetry:

~~~bash
./DrMoagi-TiB3D-1000GBs \
  --threads 64 \
  --task-limit 256 \
  --kernel checksum \
  --target-gbps 1000 \
  --json
~~~

## Verification

The built-in self-test checks:

1. representative 3D coordinate to linear-address round trips;
2. the final voxel maps exactly to 2^40 - 1;
3. the reversible XOR transform changes a deterministic block after one application;
4. applying the identical XOR transform twice restores the original block exactly.

Run:

~~~bash
./DrMoagi-TiB3D-1000GBs --self-test
~~~

The repository CTest integration runs this self-test in the existing cross-platform C++ workflow.

## Relationship to the 3D autoencoding runtime

The processor is a throughput substrate, not yet a trained compressor.

The intended substitution is:

\[
X_s
\rightarrow
E_\theta(X_s)
\rightarrow
Z_s
\rightarrow
D_\phi(Z_s)
\rightarrow
\hat X_s
\rightarrow
R_s=X_s-\hat X_s
\rightarrow
\text{verify/correct}.
\]

The same plane scheduler can carry that operator, but any compression or reconstruction claim must remain evidence-gated. A geometrically smaller latent tensor is not automatically lossless compression; discarded information must be represented by quantized latent symbols, residuals, side information, or explicitly accepted loss.

## Visual emulator

The end-to-end browser surface lives at `apps/tib3d-processor/index.html`.

It visualizes the same logical TiB geometry as a bounded 3D state machine:

`INPUT -> SHARD -> ENCODE -> LATENT -> DECODE -> RESIDUAL -> VERIFY -> RECUR`.

The browser model preserves the exact address arithmetic and verification semantics while deliberately downsampling the rendered geometry. It is an emulator, not a throughput benchmark or a 1 TiB allocation.

Validate it independently with:

~~~bash
node --test apps/tib3d-processor/test_model.cjs
~~~
## Build

The target requires C++20 for std::barrier and std::jthread; this requirement is target-local and does not move the rest of the C++ runtime off its existing C++17 baseline.

~~~bash
cmake -S cpp_runtime -B build/cpp-runtime -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpp-runtime --target jarvisx-tib3d-stream --parallel
./build/cpp-runtime/DrMoagi-TiB3D-1000GBs --self-test
~~~
