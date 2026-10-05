# DM3D Auto-Executing Bytecode Runtime

Native C++ reference backend for the Jarvis-X inward 3D VM.

The control plane is a 17-instruction, fixed-width 64-bit bytecode program. It
models the 1000 KB x 1000 KB proposal as a sparse virtual address fabric:
1,000,000 x 1,000,000 logical positions, while only active media tiles are
physically resident.

## Execution loop

```text
BOOT
  -> CONFIG_FABRIC
  -> OPEN_STREAM
  -> NEXT_WINDOW
  -> DECODE_MEDIA
  -> ENCODE_MODAL
  -> FUSE_3D
  -> FOLD_INWARD
  -> DECODE_MODAL
  -> RESIDUAL
  -> UPDATE_OMEGA
  -> FIXPOINT_CHECK
       | not converged
       +-----------------> FOLD_INWARD
  -> EMIT
  -> CLOSE_WINDOW
  -> NEXT_WINDOW
```

The recurrent state is:

```text
Z0      = Fuse(E_m(X_m), Omega)
Z(k+1)  = Fold3D(Zk, Omega)
Xhat(k) = D_m(Z(k+1))
R(k)    = X - Xhat(k)
Omega   = beta*Omega + (1-beta)*R(k)
```

The loop exits when the residual reaches the fixed-point threshold or the
configured recursion cap is reached.

## 64-bit instruction word

```text
63          56 55      48 47      40 39      32 31                     0
+--------------+----------+----------+----------+------------------------+
| opcode (8)   | dst (8)  | srcA(8) | srcB(8) | immediate / offset (32)|
+--------------+----------+----------+----------+------------------------+
```

## Build

```bash
cmake -S cpp_runtime/dm3d_bytecode -B build/dm3d-bytecode
cmake --build build/dm3d-bytecode --parallel
ctest --test-dir build/dm3d-bytecode --output-on-failure
```

CMake invokes `assemble.py` and creates `dm3d_system.bc` reproducibly in
the build directory.

## Current boundary

The VM is executable end-to-end, but codec and neural kernels are deliberately
reference implementations. `synthetic_media()` stands at the host-adapter
boundary where production FFmpeg/libavcodec, camera, audio, text-tokenizer or
other media ingress adapters should attach. The bytecode control plane can stay
stable while those kernels move to SIMD, CUDA, Metal, Vulkan compute, or NPUs.
