# Dr Moagi 3D — C++17 operational reference

This executable implements the **numerical codec** in the preceding Dr Moagi / Klein-Net construction. It is a 3-channel spatial 3D orthogonal block encoder/decoder with recursive latent updates, an optional Ω feedback state, a CTR accept/reject rule, verification, injected input, and a separate 256-node Klein-inspired visualization mesh export.

It is **not** a trained ANN, an arbitrary-file multimedia codec, a GPU renderer, or a 1 GB virtual-memory allocator. The Klein-inspired 256-node surface is exported as an OBJ mesh for 3D inspection; it is not itself the domain of the codec.

## Build

```bash
g++ -std=c++17 -O3 -Wall -Wextra -Wpedantic dr_moagi_3d_engine.cpp -o dm3d
# Optional parallelization:
g++ -std=c++17 -O3 -fopenmp dr_moagi_3d_engine.cpp -o dm3d_omp
# Or with CMake (if installed):
cmake -S . -B build && cmake --build build --config Release
```

Windows (with MinGW-w64):

```bash
g++ -std=c++17 -O3 dr_moagi_3d_engine.cpp -o dm3d.exe
```

## Run

```bash
./dm3d --self-test
./dm3d --grid 32 --steps 16 --eta .38 --beta .86 --mu .05 \
  --inject-after-eight --csv dr_moagi_cpp_trace.csv \
  --json dr_moagi_cpp_trace.json --obj dr_moagi_klein_3d.obj --phase .65
```

Open `dr_moagi_klein_3d.obj` in any OBJ-capable 3D viewer. Inspect execution metrics in CSV/JSON.

## Field and codec

Let `N` be even, `M=N/2`; `X` has shape `[3,N,N,N]`, and `Z`, `Omega` have shape `[3,M,M,M,4]`.

For each spatial block `B` of 8 scalar values and signs `sx, sy, sz` in `{-1,+1}`:

- `a = sum_B X / 8`
- `bx = sum_B(sx*X) / 8`
- `by = sum_B(sy*X) / 8`
- `bz = sum_B(sz*X) / 8`
- `decode(a,bx,by,bz; sx,sy,sz) = a + sx*bx + sy*by + sz*bz`

These 4 orthogonal basis functions retain four of eight coefficients per block. Coefficient count ratio is exactly 2:1, excluding working storage and metadata. It is lossy for general 3D data.

## Recurrence and CTR

For stationary input, `z_star=encode(X)`.

```text
proposal = z + eta * ((z_star-z) + mu * (Omega-z))
loss(proposal) = mean((X-decode(proposal))^2)
accept iff loss(proposal) <= loss(z) + 1e-10
if rejected: try smaller eta, then fall back to mu=0
Omega <- beta * Omega + (1-beta) * z_accepted
```

Optional `--inject-after-eight` changes one synthetic input channel after steps 8,16,... and rebuilds the target; the MSE is allowed to rise across an injection event. MSE is non-increasing under accepted steps relative to the **same input**. The rule does not guarantee monotonic improvement across input changes.

## Exact-arithmetic identity

Writing `P = decode o encode` (an orthogonal projector),

```text
MSE(z) = floor_MSE + 8 / (3*N^3) * sum((z-encode(X))^2)
floor_MSE = mean((X-PX)^2)
```

For `mu=0`, fixed input and `0 < eta <= 1`:

```text
excess_MSE(t) = (1-eta)^(2t) * excess_MSE(0)
```

The implementation verifies residual orthogonality, `encode(decode(z_star))=z_star`, floor and Pythagorean loss decomposition, with tolerances for float32 rounding.

## Geometry

The `16x16` node map samples a Klein-inspired immersion `(u,v)->(x,y,z)` with logical depth classes `[63,54,45,36,27,18,9,0]`. `--phase` folds exported positions inward. The OBJ includes 256 vertices and 512 triangular faces, with a reversed seam connection in the `u` direction. The depth-dependent scaling makes this a visualization approximation and not a rigorous smooth closed Klein immersion.

## Verification recorded

| Test | Final MSE | Status |
|---|---:|---|
| 16³, 8 steps, mu=0 | 0.0008634864596 | PASS |
| 24³, 8 steps, mu=0 | 0.0004519916642 | PASS |
| 32³, 8 steps, mu=0 | 0.0001730300716 | PASS |
| 24³, 16 steps, mu=.05, injection after step 8 | 0.0004448170452 | PASS |

Both ordinary g++ and OpenMP-enabled g++ self-tests passed on this build environment. These are correctness checks, not a throughput benchmark.
## Jarvis-X integration

This is an isolated C++17 codec under `cpp_runtime/dm3d_klein_codec`.
Both standalone and parent `cpp_runtime/CMakeLists.txt` builds include the
`jarvisx-dm3d-klein-codec` target. The dedicated GitHub Actions workflow
compiles on GCC, Clang, and MSVC and runs the CTest invariants and smoke run.

The fixed 2x2x2 orthogonal projection is lossy, not a replacement for existing
byte-exact DM3D codecs or the separately developed ANN training engines.
Klein-inspired geometry and shell depth labels are visualization exports.
