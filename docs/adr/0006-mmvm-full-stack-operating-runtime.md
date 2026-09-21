# ADR-006: MMVM Full-Stack Auto-Encoding/Decoding Operating Runtime

- Status: Accepted
- Date: 2026-08-13

## Context

Jarvis-X already contains deterministic VM, sparse field-runtime, codec and research-layer semantics. The next requirement is an executable system boundary that joins those ideas into a deployable multimodal operating runtime without claiming physical petabyte allocation or silently replacing the canonical deterministic VM.

## Decision

Adopt `jarvisx.mmvm` as a bounded MMVM microkernel/runtime layer with these constitutional properties:

1. **Exact virtual-memory contract.** The logical byte lattice is `100000 x 100000 x 100000 = 10^15` cells. Physical persistence is sparse.
2. **Lossless codec boundary.** Input bytes are compressed into a reversible ZLIB packet and independently projected into a deterministic 128-dimensional latent state.
3. **Xi refinement.** Latent state may be iteratively refined, but the refined state does not replace source truth.
4. **Lambda projection before authority.** Checksum equality, finite/bounded latent state and resource limits must pass before a transaction can commit.
5. **Omega persistence.** Durable event telemetry feeds bounded historical activation back into subsequent cycles.
6. **Collision-resolved object allocation.** Object identity is content-and-modality-derived; virtual addresses are separately allocated with explicit collision resolution.
7. **Task state machine.** Work transitions through queued, running and terminal committed/rejected/failed states.
8. **Symmetric binary ingress.** The API accepts UTF-8 text or arbitrary base64 bytes; internal encoding is modality-agnostic.
9. **Multimedia adapters.** Deterministic local decoders provide text, SVG image, WAV audio, H.264 MP4 and 3D voxel JSON artifacts. Learned model adapters may replace these behind the same boundary later.
10. **Browser is an observability/control client, not a second kernel.** WebGL2 maps real `Xi-dot`, reconstruction error, Omega activation and Lambda state into the framebuffer/bloom pipeline.
11. **Container hardening.** The reference image runs as a non-root user, drops Linux capabilities, persists only `/data`, exposes a health endpoint, and uses explicit resource limits in Compose.

## State transition

For a submitted task, the authoritative transition is:

```text
Observe -> Encode -> Xi refine -> Lambda project -> Decode/verify
        -> sparse allocate -> optional generate -> Omega event -> Commit
```

A generated artifact is never authoritative source state merely because it was decoded or rendered.

## Consequences

The implementation is now a deployable multimodal state-processing runtime with operating-system-like task, memory, policy and device-adapter boundaries. It remains intentionally distinct from a general-purpose host OS, hypervisor or CPU ISA implementation. This preserves precise claims while providing a concrete path toward richer schedulers, model/device adapters, distributed page ownership and capability-based authorization.
