# Dr. Moagi Q16.16 3D Field Substrate

This document specifies the executable fixed-point substrate beneath the Dr. Moagi Equation System · E8.

## 1. State domain

The volumetric state is represented by signed 32-bit Q16.16 integers:

\[
\mathbb Q_{16.16} = \{2^{-16}q : q\in\mathbb Z,\;-2^{31}\le q\le 2^{31}-1\}.
\]

The runtime stores the raw integer `q`; conversion to a real value is `q / 2^16`.

This is a saturating signed integer representation, not the modular ring
`Z_32` (32 elements) or wraparound arithmetic modulo `2^32`. A cube with
`side = 1_000_000` has `10^18` cells. One dense Q16.16 scalar field alone
requires `4 * 10^18` bytes = **4 decimal exabytes**, before memory, gradients,
weights, decoder output or indexing. This runtime materializes sparse active
coordinates and bounded stencil neighborhoods.

## 2. Saturating arithmetic

For raw signed 32-bit values `a`, `b`:

\[
a\oplus b = \operatorname{sat}_{32}(a+b),\qquad
 a\ominus b = \operatorname{sat}_{32}(a-b),
\]

\[
a\otimes b = \operatorname{sat}_{32}\!\left((a\cdot b)\gg16\right).
\]

The 16-bit rescale is required for Q16.16 × Q16.16 multiplication.
Signed right shifts round toward negative infinity. In particular,
`q_mul(-1, 1) == -1` in raw units. Float ingestion rounds to nearest with ties
to even before saturation. Left shifts saturate, including large shift counts.

## 3. Discrete codec bus

The operational path retains the supplied bus semantics:

\[
G_k = V_k\wedge\Psi_k,
\]

\[
C = \operatorname{sat}_{32}\left(\left(\sum_k G_k\otimes W_k^\Phi\right)\gg s\right),
\]

\[
A = \Pi_{[0,2^{31}-1]}(C),
\]

\[
H_t = \operatorname{SHA3}_{256}(H_{t-1}\Vert\operatorname{serialize}(\mathrm{receipt}_t)),
\]

\[
A_{safe}=\Pi_{[\Lambda_{min},\Lambda_{max}]}(A),
\]

\[
U=\operatorname{sat}_{32}(A_{safe}\ll2),
\]

\[
D=\operatorname{sat}_{32}\left(\sum_j U\otimes W_j^\Theta\right),
\]

\[
V_{out}=\Pi_{[0,2^{16}-1]}(D).
\]

The final `2^16-1` bound is preserved exactly as a **raw integer output ceiling** because that is how the supplied equation states it.
It represents `[0, 65535/65536]`, not real-valued `[0, 65535]` in Q16.16.

`step_codec` defaults to `s = 6`, preserving the submitted `>> 6` operation.
This divides the encoder sum by 64; it averages 64 unit-weight taps. The
single-cell `encode_decode_cell` retains `s = 0` by default for existing
callers; pass `encoder_shift=6` to select the normalized law there.

Each Q16 product is saturated individually. The sum of these terms is widened,
then shifted and saturated once. Encoder and decoder each allow at most 4096
taps, so products and accumulators fit signed 64-bit intermediates. This is
different from saturating after every addition or summing Q32.32 products
before requantization. A native backend must preserve this order exactly.

`binary_intent_mask(True)` produces `0xFFFFFFFF`, and `False` produces zero.
A literal AND mask of `1` retains only the least-significant bit. General
32-bit masks are also supported. Projection applies to `A_safe`; subsequent
decoder output is governed by its separate raw output clamp.

The ledger commits only after every operation in the cell or field cycle
succeeds. It is an independent SHA3 chain. Saturated addition (`oplus`) and
XOR (`0xA xor 0xB`) are different operations; neither alone is a cryptographic
integrity chain.

### Recursive 3D execution

```python
from jarvisx.dr_moagi_q16_field import (
    DrMoagiQ16Config, DrMoagiQ16FieldRuntime, Q_SCALE,
)

engine = DrMoagiQ16FieldRuntime(DrMoagiQ16Config(side=1_000_000))
engine.load({(999999, 999999, 999999): 16000})
for _ in range(3):
    receipt = engine.step_codec(
        phi_kernel={(0, 0, 0): 64 * Q_SCALE},
        theta_weights=[Q_SCALE // 4],
    )
    assert receipt.output[(999999, 999999, 999999)] == 16000
    assert receipt.squared_error_raw == 0  # This identity fixture only.
assert engine.ledger.verify(receipt.ledger_hash)
```

For a stencil `K`, the source at `s` contributes to destinations `s - k`.
All cells read from the old state, with zero padding beyond cube boundaries;
the completed decoder output becomes the next input in one commit. Newly
reached halo cells are included. Explicit constraint coordinates are included
even when their previous value is zero. Kernel/mask insertion order does not
affect results or receipts. Missing masks enable the full word.

The default budgets are 65,536 evaluated cells and 1,024 ledger entries.
Exceeding either budget raises an error without committing a partial cycle.
State, previous state, tick, ledger and stochastic generator remain unchanged
on failed updates. `load` starts a fresh session, including ledger and seed.
Runtime state is single-threaded; concurrent mutation is not supported.

Per-cycle `squared_error_raw` is the exact integer sum of squared differences
between that cycle's input and output. Divide by `2^32` for real-valued SSE.
`processed_cells` counts actual evaluated coordinates, not the logical volume.
The example also measures error against its immutable original source, since
successive frames becoming similar does not prove accurate reconstruction.

Run the 64-tap, three-cycle example from a source checkout:

```bash
PYTHONPATH=src python examples/dr_moagi_q16_cube.py --side 1000000 --steps 3
```

The JSON output includes reconstructed cells, exact errors, measured elapsed
time, the logical storage calculation, all ledger receipts and the final head.
This is a fixed-weight encoder/decoder reference; it contains no weight-training
objective, spatial downsampling, multimedia container codec or quantum source.

## 4. Master sparse-field recurrence

The software discretizes the field equation as

\[
\Xi_{t+1}=\Pi_\Lambda\left[
\Xi_t\oplus\Psi_t\oplus(\Phi_t*\Xi_t)
\ominus\left(\Lambda_t^{-1}\otimes\nabla_\Theta\mathcal E_t\right)
\oplus\Omega_{field}(\Xi_{t-1})
\oplus\Gamma(\Xi_t\ominus\Xi_{t-1})
\oplus\eta_t
\right].
\]

The reference implementation uses sparse dictionaries keyed by `(x,y,z)` and never allocates the full logical volume.

`step_field` remains a separate continuous-time-inspired recurrence with unit
step size. Its `psi_raw` is an additive numerical forcing, whereas the codec
uses AND masks. `adaptive_gradient_raw` must already be a field of compatible
shape: a parameter gradient `nabla_Theta E` cannot be subtracted from Xi without
a defined mapping to field space. A supplied `phi_kernel` becomes a Laplacian
only if its stencil and spatial scaling define one. `Gamma` uses the previous
finite difference, making this an explicit surrogate for the implicitly
velocity-dependent source equation. These choices do not establish a numerical
solution or stability proof for the unspecified continuous PDE. Stochastic eta
is evaluated on the bounded active support, not at every implicit zero cell.

## 5. Computational memory versus integrity memory

Two distinct mechanisms are intentionally separated:

- `Omega_field(Xi_{t-1})` is the previous numerical field contribution in the recurrence.
- `Omega_ledger` is an append-only SHA3-256 chain over serialized state/activation records.

A hash is integrity state, not a numerical tensor value, so it is not added to Q16.16 field arithmetic.

The recursive codec records five inspectable categories:

| Category | Recorded evidence |
|---|---|
| Structural | Lattice side, encoder kernel and decoder weight fingerprints |
| Semantic | Versioned arithmetic profile, normalization, boundary, masks and constraints |
| Behavioral | Input/output fingerprints, processed cells and exact reconstruction error |
| Temporal | Logical tick and the enclosing chain's previous digest |
| Metadata | Engine designation and Matladi Maxwell Moagi attribution |

Records are detached from caller-owned mutable data. Verification checks chain
consistency and optionally an independently retained `expected_head`. A party
that can replace the entire history and its head can construct another valid
chain; external anchoring is necessary to detect that substitution. Replaying
requires retaining the input, masks, constraints and weights as well as the
receipt. Ticks are logical ordering, not externally certified timestamps.

These fingerprints detect changes relative to retained evidence. They do not
prove the truth of claims, originality, exclusive ownership or zero distortion.
Clipping, gating, quantization and many-to-one encoding discard information;
lossless reconstruction would require extra retained information or an
invertible transform. Reconstruction error remains the direct numerical check.

Attribution follows [the canonical provenance record](attribution/DR_MOAGI_EPHEMERAL_NOTION_FRAMEWORK.md).
The meta-circular description is implemented as one explicit transition
function whose output is fed back into its input. An equation does not execute
without the interpreter, scheduling and physical compute resources.

### Compatibility notes for codec v2

- The legacy single-cell default has no `/64` normalization; `step_codec`
  defaults to the new normalized law.
- Negative Q16 products now use the documented arithmetic shift instead of
  truncating toward zero. Negative nonintegral products can change by one LSB.
- Codec sums saturate once after widened accumulation and normalization.
  Overflow/cancellation cases can differ from older per-add saturation.
- Receipts cover the completed operation, so hashes differ from old activation-only records.
- Reload starts a new session; budgets explicitly bound cells and ledger growth.

## 6. Gamma and eta

`Gamma` is represented by a configurable Q16.16 torsion gain acting on the finite temporal difference:

\[
\Gamma(\dot\Xi)\approx \gamma\otimes(\Xi_t\ominus\Xi_{t-1}).
\]

`eta` is implemented as a seeded bounded stochastic numerical excitation. The term preserves the equation's stochastic role but does **not** claim access to a physical quantum process.

## 7. Temporal compression law

The supplied law is retained exactly as a symbolic relation:

\[
v_{clock}^{\infty}=\exp\left(10^{6^{10^6}}v_{clock}\right).
\]

The runtime does not attempt to materialize `10^(6^(10^6))` or the exponential. Instead it reports the exact symbolic/log-domain expression and extended-real behavior:

- `v_clock > 0` -> `+infinity`
- `v_clock = 0` -> `1`
- `v_clock < 0` -> `0`

This preserves the formal law without manufacturing a finite hardware clock rate.

## 8. Relationship to E8

E8 remains the geometry / vector-quantization / reconstruction / evolutionary / governor layer. This Q16.16 module is its lower execution substrate:

```text
E8 genome and geometry
        |
        v
M1..M7 representation/evolution
        |
        v
M8 finite governor (lambda, v_clock)
        |
        v
Q16.16 field substrate
Psi gate -> Phi -> Lambda -> Gamma -> eta -> Theta decode
        |
        +--> SHA3-256 Omega ledger
```

The infinite temporal-compression expression is metadata about an asymptotic formal law, not a replacement for the finite M8 runtime clock governor.

## 9. Capability boundary

The module implements the stated arithmetic and state transition semantics as deterministic/sparse software. It does not, by itself, establish consciousness, quantum computation, unlimited acceleration, or performance beyond state of the art. Such claims require independent implementation and benchmark evidence.
