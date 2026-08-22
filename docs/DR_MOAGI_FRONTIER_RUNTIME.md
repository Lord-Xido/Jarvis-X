# Dr Moagi Frontier Runtime

## Status

**Empirical frontier candidate. Not a SOTA claim.**

This runtime upgrades the Dr Moagi 3D operating stack along the four gaps identified
against current sparse-3D and equilibrium-system baselines:

1. hierarchical sparse geometry;
2. actual byte-producing entropy coding;
3. implicit fixed-point acceleration;
4. benchmark-gated claims.

The purpose is to make "go beyond SOTA" an executable research program with a
hard evidence gate rather than an architectural label.

## End-to-end recurrence

The frontier cycle is

```text
sparse field X_t
    |
    v
Morton hierarchy H_t
    |
    v
inward fold K(X_t)
    |
    v
sparse encode/decode F(X_t)
    |
    +-------------------------+
    |                         |
    v                         v
plain fixed point         Anderson fixed point
    |                         |
    +------------+------------+
                 |
                 v
      rate-distortion-compute score
                 |
          choose lower objective
                 |
                 v
          transactional gate
                 |
                 v
       entropy packet + journal
                 |
                 v
               X_t+1
```

The fixed-point operator is deliberately pure and bounded:

```text
F(X) = clip(
    X + g * ( D(E(K(X))) - X ),
    -1,
    +1
)
```

where `K` is centroid contraction plus radial attenuation, `E/D` is the existing
sparse block codec, and `g` is `fixed_point_gain`.

## 1. Morton hierarchical sparse geometry

`HierarchicalSparseGrid3D` maps each active coordinate `(x,y,z)` to a Morton
(Z-order) key:

```text
m = interleave_bits(x, y, z)
```

Leaves are stored in increasing Morton order. The runtime can report occupied
blocks at every dyadic scale:

```text
level 0: 1 x 1 x 1 voxels
level 1: 2 x 2 x 2 blocks
level 2: 4 x 4 x 4 blocks
...
```

This is still a portable reference backend, not an fVDB replacement. It creates
an explicit backend boundary for future CUDA/fVDB execution while preserving the
Dr Moagi logical coordinate system.

## 2. Actual entropy packet

The previous bit plane is useful occupancy structure, but it is not by itself a
complete rate-distortion codec.

`SparseEntropyCodec3D` now emits real bytes:

```text
coordinate
  -> Morton order
  -> Morton delta
  -> varint
value
  -> scalar quantization
  -> zig-zag integer
  -> varint
stream
  -> DEFLATE
  -> payload bytes
```

The runtime measures:

```text
encoded_bytes
bits_per_active
compression_ratio
SHA-256 packet checksum
```

The reference raw sparse representation is accounted as 12 coordinate bytes +
4 float32 bytes per active cell.

This is intentionally a deterministic reference entropy layer, not a claim that
DEFLATE beats learned point-cloud entropy models.

## 3. Anderson-accelerated implicit equilibrium

For

```text
X* = F(X*)
```

the plain reference solver executes

```text
X_(k+1) = F(X_k).
```

The frontier solver stores the latest `m` residual fields

```text
G_k = F(X_k) - X_k
```

and computes coefficients `alpha` by the constrained least-squares problem

```text
min || sum_i alpha_i G_i ||^2
subject to sum_i alpha_i = 1.
```

A small regularized KKT system is solved in pure Python. The accelerated proposal is

```text
X_(k+1) = sum_i alpha_i F(X_i)
```

with a configurable damping factor.

The production cycle executes both plain and accelerated solvers. Acceleration is
not trusted automatically.

## 4. Rate-distortion-compute selection

Each candidate receives

```text
J =
    w_D * distortion
  + w_R * encoded_rate
  + w_C * normalized_iterations
```

with lower `J` better.

The accelerated state is authoritative only if

```text
J_anderson <= J_plain.
```

Otherwise the cycle falls back to the plain solver.

This converts the frontier mechanism into a no-regression internal comparison.

## 5. Transactional boundary

A selected candidate is rejected if it:

- exceeds the active-cell budget;
- leaves the logical lattice;
- contains non-finite values;
- deletes all active state from a non-empty authoritative field.

Every attempted cycle is written through the existing SHA-256 hash-chain journal.

## 6. SOTA claim gate

`SOTAClaimGate` is intentionally separate from the internal benchmark.

An external SOTA claim requires explicit `BenchmarkEvidence` containing:

```text
metric
candidate value
reference value
direction
minimum required relative gain
source/provenance
```

A missing source fails the gate.

A claim passes only if every workload-matched metric beats its external reference
by the configured margin.

Therefore the code itself distinguishes:

```text
frontier candidate != verified SOTA
```

until reproduced external comparisons are supplied.

## 7. CLI

Install:

```bash
python -m pip install -e ".[dev]"
```

Run:

```bash
jarvisx-dr-moagi-frontier demo \
  --side 32 \
  --cycles 3 \
  --max-iterations 32 \
  --anderson-depth 4
```

Inspect the initial frontier state:

```bash
jarvisx-dr-moagi-frontier status --side 32
```

Run the deterministic internal benchmark:

```bash
python scripts/benchmark_dr_moagi_frontier.py
```

## 8. CI contract

`.github/workflows/dr-moagi-frontier.yml` checks:

- frontier regression tests;
- quantized entropy packet round-trip;
- hierarchical Morton invariants;
- Anderson acceleration on a known affine contraction;
- no-regression rate-distortion-compute selection;
- CLI smoke execution.

The affine fixed-point benchmark must demonstrate fewer iterations than the plain
reference solver. That test verifies the accelerator works; it does not imply a
world-record result.

## 9. Frontier roadmap

The next backend substitutions are explicit:

```text
portable Morton hierarchy
    -> CUDA/fVDB backend adapter

DEFLATE reference entropy coder
    -> learned entropy model + ANS/range coder

pure-Python Anderson solve
    -> fused GPU equilibrium solver

internal synthetic baseline
    -> standardized external sparse-3D / PCC / DEQ benchmark suite
```

A future "beyond SOTA" declaration should be emitted only after the external
claim gate passes on identical hardware, datasets, precision, preprocessing and
measurement methodology.

## Governing invariant

The frontier runtime uses:

```text
describe -> compress -> equilibrate -> measure -> compare -> verify -> commit
```

or mathematically:

```text
X_(t+1) = Pi_Lambda(
    argmin_{Y in {Y_plain, Y_anderson}}
    [ D(X_t,Y) + lambda_R R(Y) + lambda_C C(Y) ]
)
```

with empirical evidence required before any SOTA status is promoted.
