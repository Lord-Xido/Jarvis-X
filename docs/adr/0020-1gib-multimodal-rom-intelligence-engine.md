# ADR-020: 1 GiB Multimodal ROM Intelligence Engine

**Status:** Proposed  
**Date:** 2026-09-17  
**Applies to:** constrained multimodal Jarvis-X runtimes, native container/microVM deployments, WASM clients, WebGPU-capable clients and bounded swarm execution  
**Extends:** ADR-019 and the 1 GiB BitMatrix stream profile

## Context

Jarvis-X already contains two complementary architectural boundaries:

1. the executable 1 GiB BitMatrix stream profile establishes exact 1 GiB geometry, bounded reusable buffers, fixed-point transport verification and measured rather than assumed compression/throughput; and
2. ADR-019 establishes the separation between compact immutable ROM description and potentially much larger sparse dynamic world state.

A deployment-oriented profile is needed between these layers: a multimodal encoder/decoder runtime with a strict 1 GiB host-memory envelope, immutable quantized base parameters, bounded context, vector-quantized latent transport, multiple decoder heads and capability-gated WASM/WebGPU execution.

Several proposed characteristics require explicit engineering boundaries. A four-way 256 MiB partition leaves no unaccounted memory for stacks or allocator/runtime metadata; a 128k FP16 KV cache is architecture-dependent; one million WebGPU invocations are logical work items rather than one million simultaneous hardware lanes; `mmap` reduces copying/eager population but does not guarantee zero startup latency; and high compression ratios are meaningful only with paired distortion/fidelity evidence.

## Decision

Jarvis-X SHALL define a **1 GiB Multimodal ROM Intelligence Engine** as a bounded deployment specialization with canonical state

\[
S_t=[X_t,Q_t,Z_t,\Omega_t,\hat X_t,R_t,\Pi_t,T_t].
\]

The runtime transition is

\[
S_{t+1}=\mathcal M_{ROM}(S_t;\Pi_t),
\]

where the base ROM parameters remain immutable for the invocation and only bounded policy/dynamic state may change.

### 1. Memory accounting

The 1 GiB managed host space is divided into four nominal 256 MiB accounting regions:

```text
ROM     256 MiB  immutable weights/codebooks/tokenizer
LATENT  256 MiB  embeddings, latent ring, context/KV state
EXEC    256 MiB  runtime, stacks, allocator metadata, staging/work buffers
SWARM   256 MiB  virtual-agent state, queues, telemetry and bounded policy
```

Every implementation-controlled host allocation MUST be charged to a region. The EXEC region MUST include implementation reserve for runtime overhead rather than assuming the entire region is available to tensor payloads.

Conforming steady-state native execution MUST satisfy

\[
RSS_{host}\le1\text{ GiB}.
\]

Transient startup peaks and platform-owned browser/runtime memory MUST be reported separately.

Device-local accelerator memory MUST also be reported separately; it MUST NOT be disguised as zero-cost memory merely because it is outside host RSS.

### 2. ROM immutability

The base parameter artifact is read-only during invocation. Native backends MAY use read-only file-backed mapping; browser/WASM backends MAY fetch or embed the artifact into bounded memory.

`zero-disk` means no writable local model state is required. It does not prohibit a read-only file-backed ROM artifact.

Runtime adaptation MAY modify \(\Pi_t\), queue topology, active point counts, quantization modes, context retention and other explicitly mutable controls, but MUST NOT silently rewrite the frozen ROM parameters.

### 3. Context is budget-derived

A conventional KV cache uses

\[
B_{KV}=2LSH_{kv}D_hb.
\]

Therefore resident sequence length MUST be derived from the actual architecture and available LATENT bytes. A 128k logical context MAY be supported through ring eviction, quantized KV, grouped/multi-query attention, recurrent latent state, summaries or retrieval; it is not automatically a 128k resident FP16 KV cache.

### 4. Multimodal pipeline

The canonical lifecycle is

```text
INGEST
 -> TOKENIZE/FRAME/PATCH/PROJECT
 -> FUSE + ENCODE
 -> ACTIVATION QUANTIZE
 -> RVQ + BIT PACK
 -> BOUNDED LATENT/CONTEXT
 -> TEXT/CODE | AUDIO | VISUAL/3D DECODE
 -> LOSS/FIDELITY/BUDGET TELEMETRY
 -> BOUNDED POLICY UPDATE
 -> RECUR
```

The initial modality profile is:

- text/code: compact subword/BPE-like tokenizer, target vocabulary 32k;
- audio: 24 kHz mono baseline with 20 ms framing and manifest-defined STFT/mel parameters;
- visual/spatial: lightweight 2D/3D stems with a target shared width of 1024, reducible under budget pressure.

### 5. Quantization and compression evidence

FP16 -> INT8 activation quantization provides approximately 2:1 scalar-storage reduction before scales/metadata. Other reductions MUST be measured.

For RVQ, the compressed payload includes indices, metadata and residual side information. Compression SHALL be reported as

\[
CR=\frac{B_{source}}{B_{indices}+B_{metadata}+B_{residuals}}.
\]

No fixed compression ratio is canonical. A value such as `1,000,000:1` MAY be reported only from a reproducible workload together with its reconstruction/task fidelity.

### 6. Swarm semantics

The SWARM arena MAY represent up to one million virtual points/agents when the chosen record layout and queues fit the memory budget.

WASM SIMD and WebGPU MAY schedule operations over one million logical items. Documentation and telemetry MUST distinguish logical invocation count from physical simultaneous hardware execution.

### 7. Capability-gated deployment

Three capability profiles are recognized:

- **native container/microVM** — bounded allocator, read-only mapping/equivalent, CPU SIMD, optional GPU;
- **browser/WASM** — bounded linear memory, optional SIMD/threads/WebGPU, AudioWorklet where supported;
- **serverless edge** — provider-specific CPU/WASM execution with optional accelerator support only after capability detection.

No provider name implies universal WebGPU, `mmap`, thread or memory-limit support.

`SharedArrayBuffer` MAY reduce redundant CPU-side copies between cooperating JS/WASM contexts but MUST NOT be described as a universal zero-copy path to device-local GPU memory.

### 8. Benchmark-first performance claims

The architecture fixes the memory envelope but not unmeasured latency, throughput, compression or fidelity numbers.

Benchmark reports MUST include, as applicable:

- peak and steady host RSS;
- per-arena committed bytes;
- accelerator memory;
- cold-start p50/p95;
- encode/decode p50/p95;
- modality throughput;
- complete compressed payload size;
- reconstruction/task fidelity;
- logical swarm work count and dispatch geometry;
- allocation, verifier and queue failures.

Release thresholds beyond the 1 GiB memory constraint SHOULD be added only after reproducible baselines exist.

## Consequences

### Positive

- The 1 GiB concept becomes a falsifiable deployment contract rather than a nominal label.
- Existing 1 GiB stream measurement semantics and ADR-019 ROM/world separation are reused.
- Multimodal and swarm execution can target native, WASM and WebGPU backends without claiming unavailable hardware features.
- Immutable base weights remain compatible with read-only distribution and page sharing.
- Large-context, high-compression and high-parallelism goals remain possible as benchmarked capabilities without being encoded as unsupported constants.

### Costs and constraints

- Strict memory accounting reduces the nominal payload capacity available inside each 256 MiB region because runtime overhead must be budgeted.
- A 256 MiB ROM strongly constrains parameter count, projection width and decoder quality.
- High-fidelity multimodal generation may require specialized codecs or smaller modality-specific models.
- Browser/serverless environments require fallback paths because accelerator and thread capabilities vary.
- Context length and swarm population are runtime-derived quantities rather than unconditional maxima.

## Implementation mapping

The detailed operational specification is:

`docs/DR_MOAGI_1GIB_MULTIMODAL_ROM_INTELLIGENCE_ENGINE.md`

It composes these existing components:

- `cpp_runtime/include/jarvisx/bitmatrix1gib.hpp`;
- `docs/DMVX_BITMATRIX_1GIB_STREAM.md`;
- `cpp_runtime/include/jarvisx/intelligence_media_processor.hpp`;
- `docs/DR_MOAGI_1000GB_3D_CLOUD_ROM_ENGINE.md`;
- `docs/adr/0019-1000gb-3d-cloud-rom-engine.md`.

## Acceptance criteria

ADR-020 may move from **Proposed** to **Accepted** when a reference implementation demonstrates all of the following:

1. deterministic ROM-manifest parsing and integrity validation;
2. four-region memory ledger with allocation accounting;
3. steady-state 1 GiB host-memory conformance on a declared native test profile;
4. at least two modality ingestion paths and one decode path;
5. RVQ/packed latent round-trip at the index/metadata level;
6. compression reporting that includes all required side information;
7. bounded runtime-policy adaptation without base-ROM mutation;
8. CPU reference execution plus at least one capability-gated WASM or WebGPU smoke path;
9. swarm telemetry distinguishing virtual points from hardware concurrency;
10. reproducible benchmark artifacts containing memory, latency, throughput and fidelity evidence.
