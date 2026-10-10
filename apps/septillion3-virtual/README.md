# DM3D Septillion³ virtual tile emulator

**Status:** Bounded, runnable 3D **procedural codec**. The virtual grid is `(10^24)^3 = 10^72` address positions; none of these sites is allocated in bulk. This is **not a trained neural network**, nor evidence for septillion simultaneous hardware operations.

## Run

- Open [index.html](index.html) locally in a modern browser: 3D field monitor, BigInt XYZ addressing, original/latent/approximate/residual/exact views, finite tile batching, and feedback-gated scalar fold control.
- `node apps/septillion3-virtual/test_cross_language.js` from the repo root runs deterministic Python-generated reference fixtures and negative tests.
- `node -e "const c=require('./apps/septillion3-virtual/virtual_codec'); console.log(c.processTile([0n,0n,0n]).identical)"` prints `true`.

## Processing model

Each selected tile has `8×8×8×3 = 1536` raw RGB8 bytes, with an `4×4×4×3=192` byte pooled latent representation. A 6-neighbour periodic stencil folds the **local latent tile**. Repetition gives an approximate reconstruction. The exact modular residual `R = (X - Xhat) mod 256` restores `X = (Xhat + R) mod 256`.

The GUI demonstrates **sequential finite microbatches**; it does not run distributed workers. The browser keeps residual bytes in memory and does not produce a compressed, portable tile packet. Numerical improvements from adjusting `alpha` are control-parameter selection, **not weight training**.

See [system contract](../../docs/SEPTILLION3_VIRTUAL_TILE_CODEC.md) and pre-existing repository engines `jarvisx.dr_moagi_septillion_swarm` and `jarvisx.dr_moagi_virtual_3d_ae`.
