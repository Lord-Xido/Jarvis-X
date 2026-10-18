# Multiparallel VM Reference Architecture

## 1. Purpose

The current page establishes the visual language for a future multiparallel Jarvis-X runtime. It intentionally separates **displayed architecture** from **executed mechanism**.

The present execution path is:

```text
requestAnimationFrame
  -> main-thread procedural node update
  -> Three.js instance-matrix staging
  -> WebGL buffer upload
  -> rasterization and presentation
```

The target execution path is:

```text
immutable bytecode epoch
  -> bounded virtual-context scheduler
  -> worker or WASM SIMD execution
  -> candidate state buffer
  -> validation barrier
  -> atomic commit
  -> GPU-resident rendering
```

## 2. Current implementation boundary

The demo has one JavaScript animation loop and 2,400 Three.js instances. The alternating VADD and VMUL display changes DOM text and core colour; it does not mutate executable code.

The page therefore must not be used as evidence of:

- 1,024 active physical cores;
- `SharedArrayBuffer` execution;
- SIMD dispatch;
- exact instruction benchmarking;
- zero-copy rendering;
- or a 1,200,000-node physical simulation.

## 3. Proposed 32-bit instruction formats

A single register format cannot encode both three registers and full addresses. Use typed formats:

```text
R format: [opcode:8][rd:8][ra:8][rb:8]
I format: [opcode:8][rd:8][imm16:16]
B format: [opcode:8][condition:8][relative_offset:16]
D format: [opcode:8][flags:8][descriptor_index:16]
```

Large addresses and render resources belong in validated descriptor tables.

## 4. Runtime state

The minimum authoritative state should be:

```text
VMState = {
  active_code_epoch,
  virtual_contexts,
  ready_queue,
  current_state_buffer,
  candidate_state_buffer,
  validation_state,
  telemetry,
  provenance
}
```

One thousand and twenty-four contexts are virtual scheduling entities. A bounded worker pool maps them onto the processors the browser exposes.

## 5. Safe optimization

Arbitrary XOR mutation of active instructions is unsafe. Runtime optimization should use two code banks:

```text
Code A: active and immutable
Code B: candidate patch
```

The patch cycle is:

```text
build candidate
  -> validate encoding and bounds
  -> verify semantic equivalence
  -> benchmark against identical inputs
  -> commit at an epoch barrier or discard
```

No worker may fetch partially modified instructions.

## 6. Shared-memory execution

A future browser implementation requires:

- an HTTP deployment with cross-origin isolation headers;
- a preallocated Web Worker pool;
- `SharedArrayBuffer` typed-array regions;
- integer atomic control words;
- generation-based barriers;
- nonblocking main-thread coordination;
- and deterministic fault records.

Float vector values may reside in `Float32Array`, but synchronization and ownership must be represented in atomic integer views.

## 7. Rendering integration

The current Three.js path writes instance transforms on the CPU and sets `instanceMatrix.needsUpdate = true`, which schedules GPU upload.

A scalable target keeps swarm state on the GPU:

```text
WebGPU compute shader
  -> GPU storage buffer
  -> render pipeline consumes same buffer
```

This avoids uploading all node transforms each frame. It is GPU-resident reuse, not transparent `SharedArrayBuffer` aliasing.

## 8. Validation and commit

Each VM epoch should calculate a candidate state first:

\[
\widetilde S_{t+1}=F(S_t,I_t).
\]

Validation combines opcode, memory, numerical, budget, ownership, and provenance predicates:

\[
\Lambda_t=\bigwedge_i v_{i,t}.
\]

Commit is transactional:

\[
S_{t+1}=\Pi_{\Lambda_t}[\widetilde S_{t+1};S_t].
\]

Rejected candidates leave the entire committed state unchanged.

## 9. Evolution sequence

1. **Working:** implement decoder, assembler tests, virtual contexts, and deterministic single-thread execution.
2. **Robust:** add bounds checks, double buffering, Atomics coordination, fault injection, and rollback tests.
3. **Portable:** add a pure reference interpreter and conformance vectors before WASM or GPU acceleration.
4. **Elegant:** add immutable code epochs, descriptor tables, canonical receipts, and measured telemetry.
5. **Advanced:** add WebAssembly SIMD, WebGPU compute, safe auto-tuning, sparse tensor paging, and distributed state deltas.

## 10. Acceptance criteria for “operational VM”

The visualization may be described as an operational VM only after tests demonstrate:

- deterministic fetch, decode, execute, and halt;
- valid fixed-width encodings for every instruction class;
- bounded memory and control flow;
- worker-safe synchronization;
- complete candidate rollback;
- deterministic replay;
- semantic verification for code patches;
- and renderer output driven by committed VM state rather than procedural display logic.
