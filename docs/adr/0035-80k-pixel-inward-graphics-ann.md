# ADR-0035: 80K-Pixel Inward Graphics ANN Closure

- **Status:** Proposed with executable implementation
- **Date:** 2026-10-05
- **Scope:** C++ world/graphics runtime

## Context

Jarvis-X already contains volumetric ANN references, verified candidate/rollback mechanics, graphical demonstrations, and a separate `80,000^3` sparse logical MP4 runtime. The 80,000-pixel terminal experiment needs a canonical boundary so that "80K pixels" is not confused with "80K resolution" or the existing `80K^3` field, and so ANN adaptation cannot silently promote a worse candidate.

## Decision

Add a dependency-free C++17 target named `DrMoagi-80K-Pixel-Inward-ANN` with these invariants:

1. internal framebuffer exactly `400 x 200 = 80,000` pixels;
2. rendered observations carry RGB and depth;
3. observations encode into a bounded `16^3 x 8` recurrent neural field;
4. recurrent communication uses six toroidal spatial neighbours;
5. the decoder reconstructs the same framebuffer and produces a measurable residual;
6. observed residuals are back-projected into the latent 3D field;
7. decoder/density adaptation runs in a shadow candidate;
8. a candidate commits only if metrics are finite, same-frame reconstruction MSE is non-regressing within `1e-7`, and feedback energy is at most `0.5`;
9. wall-clock timing is telemetry and never an acceptance criterion;
10. accepted latent state may deform subsequent geometry only through bounded functions.

## Consequences

The graphics path becomes a closed numerical loop rather than a one-way visualizer. Parameter authority remains transactionally bounded and the runtime emits machine-readable receipts suitable for CI.

The implementation is intentionally small and CPU-bound. It is not a production renderer, a general differentiable renderer, an `80,000 x 80,000` display, a physical `80,000^3` resident volume, or evidence of autonomous self-improvement.

## Verification

The target remains covered by CTest and focused GCC/Clang/sanitizer CI. Documentation and the project capability matrix preserve the distinction between 80,000 total pixels and the separate 80K-cubed sparse logical runtime.
