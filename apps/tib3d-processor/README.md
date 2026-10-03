# TiB3D End-to-End Visual Emulator

A dependency-free browser visualization for the Dr Moagi TiB3D streaming processor.

It renders the exact logical geometry

8192 × 8192 × 16384 = 2^40 bytes = 1 TiB

as a bounded 3D scene and walks the runtime through:

X -> shard -> E_theta -> Z -> Phi_inward -> D_phi -> X_hat -> residual -> CTR -> recur.

The browser does not allocate 1 TiB. It visualizes the logical state using a small Canvas framebuffer.

## Run

    python -m http.server 8000 --directory apps/tib3d-processor

Then open http://localhost:8000.

## Controls

- Pause / resume the bounded recurrence.
- Single stage advances exactly one pipeline state.
- Recurrence toggles whether a completed verification cycle feeds back into the next iteration.
- Simulation rate changes visual time only.
- Arrow keys rotate the isometric projection slightly.

## Exact invariants

- NX = 8192
- NY = 8192
- NZ = 16384
- planeBytes = 8192^2 = 67,108,864 = 64 MiB
- totalBytes = 2^40 = 1,099,511,627,776
- A(x,y,z) = x + 8192(y + 8192z)
- final coordinate (8191,8191,16383) maps to 2^40 - 1
- 1000 GB/s target full-sweep time is 1.099511627776 s

## Verification semantics

The emulated codec uses a monotone candidate residual. At VERIFY:

    candidateResidual <= residual  => COMMIT
    candidateResidual > residual   => ROLLBACK

At RECUR, only a committed candidate becomes the next authoritative residual.

This mirrors the repository candidate-first verification boundary without claiming a trained neural codec.

## Validation

    node --test apps/tib3d-processor/test_model.cjs

The test suite validates exact address round trips, TiB arithmetic, stage ordering, bounded residual monotonicity, rollback/commit semantics, and inline JavaScript syntax.

## Capability boundary

This app is a visualization/emulation surface. It does not prove 1000 GB/s hardware throughput, allocate the full TiB volume, perform production compression, or replace the C++ processor. Runtime performance claims remain measurement-gated.
