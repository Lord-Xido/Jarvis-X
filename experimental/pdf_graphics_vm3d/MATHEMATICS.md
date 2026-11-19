# Typed operational interpretation of the proposed M_3D operator

## Original architectural notation (conceptual)

F_out(x,y,z,t) = D_theta( bigoplus_{s in Omega_10^6^3} R_GPU[ W_JIT(E_phi(B_mmap,s)) odot A_cache(s,t) ]) + lambda grad_s P_{t->t+1}.

The proposed expression mixes quantities of incompatible types: executable code, cache residency, RGBA framebuffer, and a prefetch gradient. This reference retains its causal intent but places control feedback into engine state rather than numerically adding it to pixel output.

## Operational type signature

`M_3D : (State, ContentBytes) -> (Frame, State')` where the bounded, explicit state contains the parsed IR, immutable draw plan, metrics, and accept/reject receipt.

1. `parse : Bytes -> IR` supports a strictly limited grammar and produces 128-bit instruction descriptors with out-of-line operands.
2. `lower : IR -> Ordered<Polygon>` interprets `q/Q`, CTM, text matrix, font-size, `Tj`, `re/f`, and color state with documented bounds.
3. `place : Polygon -> (tile_x,tile_y,sequence)` embeds an active polygon in a **virtual** 3D coordinate domain. `|Omega|=10^18` does not imply dense allocation.
4. `render : Ordered<Polygon> -> RGB^(128*128)` rasterizes in order using sample-center coverage. The `z` coordinate records ordering, not volumetric physical geometry.
5. `profile : Implementation -> ObservedTime` measures actual wall-clock time on a fixed example.
6. `select : {reference, specialized} -> accepted` chooses specialized only when exact pixel equality and lower measured time hold.

The code's performance criterion is `accept = same_input && (F_ref == F_plan) && (median(T_plan) < median(T_ref))`. A stronger production gate would add randomized property tests, PDF conformance fixtures, hardware counter profiling, repeated confidence intervals, and sandboxing.

## Preservation predicates

- Original stream retention: `decode_original(parse(B)) == B` (no compression, no re-encoding of source bytes).
- Visual equivalence **within this renderer**: `render(lower(IR)) == render(specialize(IR))` byte-for-byte.
- PDF specification equivalence: **not established**, since font mapping, transparency, curves, JPEG/DCT, clipping, Xref parsing, color management, and many operators are not implemented.
- `FNV64` tracks deterministic debugging provenance only; it is not SHA-256 or authentication.

## Reference geometry

For `0.5 0 0 0.5 20 40 cm` and `1 0 0 1 100 100 Tm`, `C*T` maps the text origin `(0,0)` to `(70,90)`. For the rectangle `(100,40,100,50)` under CTM `C`, the corners become `(70,60)`, `(120,60)`, `(120,85)`, `(70,85)`. Area scales by `abs(det(C_2x2))=0.25`. Paths are transformed at creation (at `re`) and do not retroactively move when a later `cm` occurs.

At 60 Hz, display VSync spacing is ~16.67 ms, not 0.8 ms. Physical performance cannot be inferred from virtual address multiplicity, an instruction count, or a renderer's synthetic screenshot.
