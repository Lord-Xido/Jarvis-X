# Jarvis-X 1000GB × 1000GB × 1000GB 3D Intelligence VM

## Operational contract

The runtime exposes a three-axis virtual byte-coordinate space with a default extent of **1000 decimal GB per axis**:

\[
0 \le x,y,z < 10^{12}.
\]

If interpreted as one addressable byte at every `(x,y,z)` coordinate, the conceptual cube contains

\[
(10^{12})^3 = 10^{36}\ \text{byte coordinates}.
\]

This is an **address space**, not a physical allocation. The engine never attempts to allocate `10^36` bytes. Sparse cubic pages are materialized on demand and held in a bounded resident cache. Dirty pages spill to the state directory and can be loaded again after eviction.

The default resident ceiling is **10 GB**. The executable is intentionally not padded to 10 GB: the 10 GB value is the maximum live page-cache budget available to the virtual engine. Padding a PE file would consume storage without increasing addressability or processing capability.

## Psi execution path

For the current coordinate `(x,y,z)`, the VM evaluates a bounded executable form of the engine operator:

\[
\Psi_{\text{Engine}}(x,y,z,t)
=
\mathcal D_\theta\!\left(
\Omega_{\text{fusion}}\!\left(
M_{\text{octree}}(x,y,z)\,\mathcal E_\phi(V_{x,y,z,t})
\right)
\right).
\]

The reference runtime implements:

- `E_phi`: deterministic 3×3×3 neighborhood projection into a configurable latent vector;
- `M_octree`: the same four-survivor inward octree rule used by the Jarvis-X sparse hierarchy;
- `Omega_fusion`: latent-state fusion with bounded recurrent memory;
- `D_theta`: deterministic latent-to-byte decoding;
- bounded feedback: `PsiLearn` updates the recurrent memory using reconstruction residuals with clipping.

This is a concrete reference engine, not a claim that the deterministic projection is equivalent to a trained frontier model. Learned encoders/decoders can replace these operators behind the same execution contract.

## Bytecode ISA

The binary format begins with `JX3DVM1\0`, followed by the instruction count and fixed-width 16-byte instructions. Sixteen 64-bit registers are available; registers `r0`, `r1`, and `r2` are the active X/Y/Z address.

| Opcode | Meaning |
| --- | --- |
| `NOP` | no operation |
| `MOV_IMM` | load a 64-bit immediate into a register |
| `ADD` | unsigned register addition |
| `XOR` | register XOR |
| `LOAD_VOXEL` | read the sparse byte at `(r0,r1,r2)` |
| `STORE_VOXEL` | write the low byte of a register at `(r0,r1,r2)` |
| `PSI_INFER` | execute encode → octree mask → fusion → decode |
| `PSI_LEARN` | execute the Psi path and one bounded feedback update |
| `JUMP` | absolute instruction jump |
| `JNZ` | jump when the selected register is non-zero |
| `HALT` | stop the current cycle |

Every VM cycle has a configurable maximum-step ceiling. Register indexes, coordinates, jump targets, bytecode size and cache geometry are validated before use.

## Windows executable

CMake target:

```text
jarvisx-intelligence3d
```

Windows output name:

```text
JarvisX-3D-10GB.exe
```

Canonical launch:

```powershell
.\JarvisX-3D-10GB.exe `
  --axis-gb 1000 `
  --resident-gb 10 `
  --page-edge 32 `
  --cycles 100 `
  --state-dir .\jarvisx-3d-state
```

To emit a sample bytecode program:

```powershell
.\JarvisX-3D-10GB.exe --generate-demo .\demo.jxb3 --cycles 1
```

Then execute it:

```powershell
.\JarvisX-3D-10GB.exe --bytecode .\demo.jxb3 --cycles 100
```

## Storage mechanics

A page with edge `e` stores `e^3` bytes. The default `e=32` therefore creates 32,768-byte pages. Page keys retain independent 64-bit X/Y/Z page coordinates, avoiding any need to flatten the complete conceptual cube into a single integer address.

The resident set is least-recently-used bounded. Before materializing a page that would exceed the configured resident ceiling, the oldest resident page is evicted. Dirty pages are written atomically through a temporary file and rename operation.

## Verification

`cpp_runtime/tests/intelligence_vm3d_tests.cpp` verifies:

1. far-edge addresses inside the 1000 GB-per-axis cube without preallocating the cube;
2. resident-cache bounding;
3. eviction → spill → reload data preservation;
4. octree culling behavior;
5. bytecode store/load execution;
6. bytecode serialization round trips;
7. repeated Psi inference and bounded learning steps.

The Windows workflow builds the MSVC Release executable, runs the focused CTest targets, performs a 1000/10 smoke execution, produces a demo `.jxb3`, calculates SHA-256, and uploads the executable as a workflow artifact.
