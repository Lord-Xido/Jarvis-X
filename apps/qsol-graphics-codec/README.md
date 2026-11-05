# QSOL 3D Graphics Auto-Encoder / Decoder

Executable .NET 8 reference implementation of the inward kinetic encoding model for 3D graphics state.

## What is operational

- Procedural animated 3D scene generation (three toroidal meshes).
- Kinetic state update using linear and angular velocity.
- Binary scene encoder and decoder with an explicit versioned container.
- Variable-bit (8/10/12/14/16) XYZ quantization packed at bit level.
- Delta + ZigZag + varint topology/index compaction.
- Brotli payload compaction using only the .NET runtime.
- Automatic codec search: the encoder tries each quantization depth, decodes each candidate, measures geometric reconstruction error, and selects the smallest stream that satisfies the requested error bound.
- PBR material parameters (base color, metallic, roughness) preserved in the stream.
- Transform and kinetic velocity state preserved in the stream.
- OBJ export from the decoded state for inspection in Blender, Unity, Unreal import pipelines, MeshLab, etc.
- Deterministic self-test suitable for CI.
- DM3D evidence/graphics runtime with 20-bit-per-axis Morton addressing over a virtual 1,024,000³ lattice.
- 256-bit XNOR/POPCOUNT evidence similarity and authority/graph gates.
- Inward prefix refinement that reduces local 3D candidate volume by approximately 8× per resolved XYZ bit triplet.
- Deterministic 128 KiB DM3D ROM image generation with immutable bytecode, self-integrity hashing, inverse-render verification opcodes and bounded runtime optimization.
- Transactional runtime tuning: candidate configurations are committed only when verification quality stays above the declared guardrail; otherwise they roll back.

## Operational loop

```text
Scene(t)
  -> kinetic update
  -> encode candidates {8,10,12,14,16 bits/axis}
  -> compact topology + Brotli
  -> decode every candidate
  -> measure max vertex reconstruction error
  -> select smallest valid representation
  -> persist .q3d
  -> decode
  -> verify topology + geometry
  -> export OBJ checkpoint
  -> Scene(t+1)
```

The implemented inward selection criterion is:

```text
Z* = argmin |Z_b|
     over b in {8,10,12,14,16}
     subject to max_vertex_error(decode(Z_b), X) <= epsilon
```

This makes compaction measurable rather than symbolic: a lower-bit representation is promoted only if its decoded geometry remains inside the declared error tolerance.

## DM3D inward evidence / graphics loop

The DM3D control plane extends the same measurable encode/decode invariant to evidence-localized graphics generation:

```text
input/evidence
  -> INT8 encode
  -> 256-bit latent code
  -> project to 3D
  -> Morton3D(20 bits/axis)
  -> octree/prefix-localized retrieval
  -> XNOR + POPCOUNT ranking
  -> authority + graph + contradiction gates
  -> structured claim decode
  -> verification
  -> latent correction
  -> resolve +1 prefix bit on X/Y/Z
  -> contract Top-K by ~8x
  -> repeat
  -> parameter decode
  -> vector render
  -> inverse render
  -> spatial error field
  -> freeze low-error cells / refine high-error cells
  -> profile latency + memory + verification quality
  -> guarded config commit or rollback
```

The virtual address manifold is:

```text
axis = 1,024,000 cells
bits/axis = 20
virtual cells = 1,024,000^3 = 1,073,741,824,000,000,000
```

The lattice is procedural; the implementation does **not** allocate a dense exabyte-scale backing store.

For the initial 16 resolved bits per axis, four bits remain free on each axis:

```text
free bits:        12 -> 9 -> 6 -> 3
candidate cells: 4096 -> 512 -> 64 -> 8
```

Each inward iteration resolves one additional bit on all three axes, giving an approximately 8× reduction in the local 3D candidate volume.

### Run the DM3D self-test

```bash
dotnet run --project apps/qsol-graphics-codec/QSol.GraphicsCodec.csproj -c Release -- --dm3d-self-test
```

It verifies:

1. the 4096 → 512 → 64 → 8 inward volume schedule;
2. Morton XYZ bit interleave invariants;
3. 256-bit XNOR/POPCOUNT identity and complement scores;
4. evidence graph/contradiction gating;
5. the optimizer never commits below the 0.95 verification guardrail;
6. the bounded objective improves over the baseline configuration;
7. ROM generation is deterministic.

The regular `--self-test` also executes the DM3D test.

### Generate the deterministic DM3D ROM

```bash
dotnet run --project apps/qsol-graphics-codec/QSol.GraphicsCodec.csproj -c Release -- \
  --dm3d-rom artifacts/qsol-graphics-codec/dm3d-self-optimizing-v2.rom
```

The generated ROM is 128 KiB and contains fixed-width 16-byte instructions for evidence encoding, Morton localization, XNOR/POPCOUNT retrieval, graph validation, inward contraction, parameter decoding, vector rendering, inverse rendering, spatial-error refinement, profiling and guarded commit/rollback optimization.

The executable bytecode is immutable at runtime. Only bounded configuration, cache and latent state are adaptive.

## Build

```bash
dotnet build apps/qsol-graphics-codec/QSol.GraphicsCodec.csproj -c Release
```

No NuGet packages are required.

## Verify the codec

```bash
dotnet run --project apps/qsol-graphics-codec/QSol.GraphicsCodec.csproj -c Release -- --self-test
```

The self-test verifies:

1. kinetic state advances;
2. encode -> decode preserves entity count;
3. decoded index topology is exact;
4. geometric reconstruction error is within tolerance;
5. persisted binary bytes decode deterministically.

## Generate an encoded 3D animation sequence

```bash
dotnet run --project apps/qsol-graphics-codec/QSol.GraphicsCodec.csproj -c Release -- \
  --frames 120 \
  --target-error 0.0025 \
  --output artifacts/qsol-graphics-codec
```

Outputs:

```text
artifacts/qsol-graphics-codec/
  frame-0000.q3d
  frame-0000.obj
  frame-0001.q3d
  ...
  frame-0119.q3d
  frame-0119.obj
  metrics.csv
```

`metrics.csv` records the selected bit depth, encoded bytes, pre-Brotli bytes, and measured reconstruction error for every frame.

## Decode an existing stream

```bash
dotnet run --project apps/qsol-graphics-codec/QSol.GraphicsCodec.csproj -c Release -- \
  --decode artifacts/qsol-graphics-codec/frame-0000.q3d \
  --obj artifacts/qsol-graphics-codec/reconstructed.obj
```

## Binary pipeline

The `.q3d` stream contains:

- QSC envelope/version/flags/raw length;
- Q3D payload/version/quantization depth;
- global local-geometry bounds;
- per-entity name, transform, linear/angular velocity and PBR material state;
- bit-packed quantized vertices;
- exact delta-coded triangle indices;
- optional Brotli compression.

## Scope

This is a fully executable **graphics-state codec and kinetic animation reference core**. It does not claim that the codec itself performs photorealistic rasterization or path tracing. The decoded scene state is deliberately renderer-agnostic so GPU backends (Direct3D 12, Vulkan, WebGPU, Unity/Unreal adapters, ray/path tracers, neural renderers) can be attached without changing the Q3D encode/decode invariant.

The DM3D evidence layer is likewise a deterministic reference control plane. `HOST_SEARCH` is a host integration boundary: production deployments must supply validated evidence and provenance rather than treating the ROM itself as a web client or source of truth.
