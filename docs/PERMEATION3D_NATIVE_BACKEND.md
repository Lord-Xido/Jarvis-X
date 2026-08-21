# Jarvis-X Permeation 3D Native Backend

## Purpose

This layer pushes the kinetic 3D runtime one level deeper into the execution stack without changing its observable state law.

```text
Kinetic Spatial IR
  -> backend router
  -> native C++ active-volume kernel
  -> hierarchical residual reconstruction
  -> verification
  -> commit
```

The pure Python implementation remains the correctness oracle. The runtime can now select `native-cpu` when a compiled backend is installed and falls back to `cpu-reference` otherwise.

## Backend contract

Every backend receives:

- current 3D values;
- predicted 3D values;
- `(sx, sy, sz)` shape;
- active threshold;
- coarse block factor;
- fine-refinement threshold.

It must return exactly the same semantic state:

- residual field;
- active indices;
- one coarse residual per active block;
- fine corrections;
- reconstructed world.

The runtime itself still performs verification, integrity hashing, transaction commit, epoch management, telemetry, and API state management. Hardware kernels therefore cannot bypass the verification boundary.

## Native execution

`native/kinetic3d_backend.cpp` exports a C ABI function that performs the active-volume numerical path in compiled C++.

For each cell:

\[
R_i = W_i - \hat W_i
\]

and

\[
i \in A \iff |R_i| > \tau_{active}.
\]

Active cells are grouped into spatial blocks. The kernel computes:

\[
z_b = \frac{1}{|A_b|}\sum_{i\in A_b}R_i.
\]

It reconstructs each active cell as:

\[
\tilde W_i = \hat W_i + z_b + \delta_i,
\]

where the fine correction is emitted only when:

\[
|R_i-z_b| > \tau_{refine}.
\]

Inactive cells remain at their predicted state.

## Native library discovery

The Python runtime loads the shared library only through the explicit environment variable:

```text
JARVISX_KINETIC3D_NATIVE_LIB=/path/to/libjarvisx_kinetic3d.so
```

`backend=auto` uses the native backend if that library exists and loads successfully. Otherwise it falls back to the Python reference implementation.

`backend=native-cpu` fails closed if the library is unavailable.

## Container path

The kinetic Docker image now has a multi-stage build:

1. compile the C++ kernel with `g++ -O3 -fPIC -shared`;
2. copy only the resulting shared library into the Python runtime image;
3. expose its location through `JARVISX_KINETIC3D_NATIVE_LIB`;
4. run the API with backend policy `auto`.

The container smoke test verifies that `native-cpu` is available and that actual state transitions report `telemetry.backend == "native-cpu"`.

## Verification boundary

Backend acceleration does not change transactional semantics:

```text
EXECUTE BACKEND
  -> RECONSTRUCT
  -> COMPUTE MSE / MAX ERROR / SHA-256
  -> VERIFY
  -> COMMIT only if tolerance passes
```

Thus:

\[
\neg VERIFY \Rightarrow \neg COMMIT.
\]

The authoritative 3D world and epoch remain unchanged after failed verification.

## CI evidence

The dedicated workflow validates three levels independently:

- reference-runtime tests with no native library present;
- native parity tests using a compiled shared library;
- Docker end-to-end execution where `auto` resolves to `native-cpu`.

This prevents a native backend from being treated as operational merely because source code exists: it must compile, load, reproduce reference semantics, survive stateful transitions, and pass transaction verification.

## Next permeation boundary

The next backend enters through the same contract:

```text
Kinetic Spatial IR
   |-- cpu-reference
   |-- native-cpu
   |-- triton-gpu        (next)
   |-- cuda-native       (next)
   `-- distributed-path  (later)
```

The target remains measurable rather than declarative:

\[
\eta = \frac{\text{verified useful state transitions}}{\text{time} + \text{bytes moved} + \text{energy} + \text{recovery cost}}.
\]
