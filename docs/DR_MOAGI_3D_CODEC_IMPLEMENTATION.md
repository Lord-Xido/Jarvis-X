# Dr Moagi 3D Codec Reference Implementation

**Status:** executable alpha reference  
**Architecture:** ADR-002 and `docs/research/DR_MOAGI_3D_CODEC_RUNTIME.md`  
**Runtime:** `src/jarvisx/dr_moagi_codec_3d.py`

## Purpose

This module operationalizes the minimum bounded transaction defined by ADR-002 while preserving the deterministic Jarvis-X core boundary. It is a correctness-oriented scalar 3D reference, not a trained neural codec and not a production compression benchmark.

## Executed transaction

```text
Volume3D X
  -> validate shape / finite values / resource ceiling
  -> mean-centre reference transform
  -> uniform scalar quantization
  -> signed 64-bit latent Z
  -> zlib entropy coding
  -> versioned JX3D bitstream B
  -> SHA-256 payload verification
  -> bounded entropy decode
  -> dequantize / reconstruct X_hat
  -> measure local MSE, anchor MSE and rate
  -> Pi_Lambda-style admissibility checks
  -> commit Omega_codec statistics or rollback
```

The entropy stage is lossless over the discrete signed-64-bit latent representation. Reconstruction loss is introduced by quantization, not by zlib.

## Bitstream contract

The fixed header binds:

- magic `JX3D`;
- format version;
- codec architecture version;
- entropy version;
- three-dimensional shape;
- quantizer step;
- encoded mean;
- latent count;
- compressed payload length;
- SHA-256 digest of the compressed payload.

Malformed dimensions, incompatible versions, invalid quantizer values, digest mismatch, decompression-size mismatch and resource-limit violations fail closed.

## Transaction boundary

`DrMoagiCodec3D.process()` snapshots the current `Omega_codec` state before evaluation. A candidate is committed only when its telemetry is admissible. Anchor or rate rejection leaves authoritative adaptive memory unchanged.

The immutable first source is retained as the runtime anchor. This allows repeated operations to distinguish local self-consistency from drift relative to the original source.

## Million-step semantics

`virtual_depth` may be configured up to `1_000_000`, matching ADR-002. The reference transaction reports:

```text
virtual_depth
measured_microsteps_executed
wall_clock_seconds
measured_throughput_voxels_per_second
```

The current implementation executes one physical codec transaction per call and reports `measured_microsteps_executed = 1`. It does not infer one million physical transitions from a virtual-depth setting.

## API

The canonical FastAPI service exposes:

```text
GET  /health
POST /run
POST /codec/roundtrip
```

`/run` creates a fresh `CodexVM` per request, avoiding shared mutable VM state across concurrent requests. `/codec/roundtrip` creates a bounded reference codec transaction with an API-level voxel ceiling.

## Validation

Focused invariants cover:

1. deterministic bitstream generation;
2. quantization reconstruction-error bound;
3. payload-corruption detection;
4. anchor-drift rejection with atomic rollback;
5. explicit virtual-depth versus measured-step telemetry;
6. shape and resource-limit rejection.

Run:

```bash
pytest tests/test_dr_moagi_codec_3d.py
```

The dedicated GitHub Actions workflow also builds the package container and probes `/health` after startup.

## Capability boundary

This implementation establishes a reproducible codec transaction and deployable API surface. It does **not** establish:

- trained neural compression quality;
- a literal one-million-pass dense 3D runtime;
- learned entropy modelling;
- autonomous architecture mutation;
- production security certification;
- hardware acceleration;
- superiority over established image/video/volume codecs.

Those remain later Layer 4/5 research stages and require independent benchmarks and evidence artifacts.
