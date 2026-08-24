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
