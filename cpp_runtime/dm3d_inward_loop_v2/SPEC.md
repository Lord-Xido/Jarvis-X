# Mathematics and self-loop verification, DM3D v2

## 1. Spatial virtual memory

Let `A = 10^12`. Coordinates `(x,y,z) ∈ {0,…,A-1}³` represent a purely virtual 3D volume. The reversible index is `i=x+A(y+Az)`; `0≤i<A³=10³⁶`. An unsigned 128-bit integer can represent all values because `10³⁶ < 2^120`.

The byte stream is mapped to consecutive 16×16×16 tiles. `x` is the fastest varying tile-local coordinate; for tile-local `p=x+16y+256z` all predecessors `p-1,p-16,p-256` lie earlier in scan order when available.

## 2. Exactly reversible 3D predictors

For byte array `b[p]`, define `px=b[p-1]` if x>0 else 0; `py=b[p-16]` if y>0 else 0; `pz=b[p-256]` if z>0 else 0. Candidate predictors are:

- `p1 = px if x>0 else (py if y>0 else pz)`;
- `p2 = px XOR py XOR pz`;
- `p3 = px`, `p4 = py`, `p5 = pz`.

For mode m, encode `d[p]=b[p] XOR p_m(b,<p)`; store d with reversible 0-run/literal RLE. Decode in increasing p:

`b_hat[p]=d[p] XOR p_m(b_hat,<p)`.

Induction over p proves exact inversion, because the predictor only refers to previously decoded entries and XOR is involutive. RLE is independently reversible and every candidate is tested on all 4096 original bytes. The raw fallback preserves exactness even when compression fails.

## 3. Self-refining codec operator

Given a tile B and a preferred mode from previous feedback, initialize `best₀=(raw, 4096)`. For each candidate `c_k` in the bounded search order:

`r_k = Decode(Encode_{c_k}(B))`

`verified_k = [r_k = B]` (exact byte equality)

`best_{k+1} = c_k` if verified and its payload is strictly smaller; otherwise retain `best_k`.

The invariant is `Decode(best_k) = B` for all k, and the stored payload size is monotonically nonincreasing. The next tile batch first tests the previous winning compression mode (Ω-controlled preference). This feedback loop **optimizes an operational outcome**; it is not an arbitrary state self-reference or AGI.

## 4. Toroidal latent operator

Mean-pool 4×4×4 voxels per latent cell to form `s∈[0,1]^{4³}`. On the periodic lattice `Z_4³`, let `N₆` mean the 6 periodic axis neighbors:

`z_{k+1}=0.60 z_k+0.40(0.85 s+0.15 N₆(z_k))`, starting at z₀=0.

For fixed input s and the infinity norm, `||N₆(a)-N₆(b)||∞ ≤ ||a-b||∞`, so

`||T(a)-T(b)||∞ ≤ (0.60+0.40×0.15)||a-b||∞=0.66||a-b||∞`.

Thus a unique latent fixed point exists. Importantly, **the codec's exact output is guaranteed by the reversible predictor and raw fallback, not by latent contraction**.

## 5. Persistent memory

For tile-level latent mean `μ_t`, `Ω_{t+1}=0.9 Ω_t+0.1 μ_t`, and an independent preferred-predictor register is updated when compression beats raw. Ω is saved in the archive header, while predictor preference operates within the current file encoding session.

## 6. Verification boundary

Tile-level checks: reconstruct and compare all 4096 bytes during candidate selection; check FNV64 on decoding. Whole-stream checks: FNV64, exact archived byte reconstruction against the independently read original, plus independent restored-output bytewise comparison in `loop`. This is integrity validation for accidental faults, not protection against hostile modification of a mutable archive.

## 7. Scalability

For a physical stream of M bytes and tile size 4096, time is `O(ceil(M/4096) × (P×4096 + K×64))`, where P≤5 candidate passes, K≤1000 fold steps; physically resident working memory scales with the bounded worker count rather than the symbolic 10³⁶ cells. Many parallel threads may be slower for tiny tiles because thread startup dominates.
