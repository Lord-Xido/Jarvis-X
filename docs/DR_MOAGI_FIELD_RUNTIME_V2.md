# Dr Moagi Field Runtime v2

## Status

Canonical Layer 4/5 operational research specification adopted by ADR-003.

This document turns the Dr Moagi 3D autoencoding/decoding equation into a bounded sparse state-transition contract. It does **not** replace the deterministic Jarvis-X VM. It defines how a volumetric research runtime proposes a candidate transition that may then be admitted through the existing policy/transaction boundary.

---

## 1. State spaces

Let the logical lattice be

```text
Omega_h = {0, ..., side-1}^3
```

with the default conceptual side length `side = 1000`.

The authoritative working field is

```text
Psi_n : Omega_h -> R
```

but the reference implementation stores only a bounded active support

```text
A_n = {x in Omega_h : Psi_n(x) is materialized}
|A_n| <= max_active_cells.
```

Coordinates outside the materialized support are deterministic zero background. Coordinates outside `Omega_h` obey Dirichlet-zero boundary semantics in the reference implementation.

The initial projected state is frozen as

```text
Psi_anchor = Psi_0.
```

---

## 2. Autoencoder closure

The encoder and decoder are typed separately:

```text
E_theta : sparse field -> latent
D_theta : latent x requested_support -> sparse field
```

Define the same-space reconstruction operator

```text
A_theta = D_theta o E_theta
Psi_hat = A_theta[Psi]
```

and reconstruction residual

```text
R_theta(Psi) = Psi - Psi_hat.
```

The evolution law never subtracts a latent tensor directly from a field tensor.

A scalar latent may be used for deliberately lossy experiments or a restricted decoder family. It is not a general invertible representation of an arbitrary `1000^3` field.

---

## 3. Spatial operators

### Six-neighbour Laplacian

For an interior coordinate `(i,j,k)`:

```text
Delta_6 Psi[i,j,k]
  = Psi[i+1,j,k] + Psi[i-1,j,k]
  + Psi[i,j+1,k] + Psi[i,j-1,k]
  + Psi[i,j,k+1] + Psi[i,j,k-1]
  - 6 Psi[i,j,k].
```

The reference implementation uses the same expression with missing/out-of-domain values equal to zero.

### Moagi glyph kernel

The fixed permeation kernel is

```text
G(0,0,0)      =  1
G(face-neighbour) = -1/6
G(otherwise)  =  0.
```

Therefore

```text
(G * Psi)[i,j,k]
  = Psi[i,j,k] - (1/6) sum(face_neighbours)
  = -(1/6) Delta_6 Psi[i,j,k].
```

The equality is normative for this kernel/sign convention.

---

## 4. Canonical field equation

The runtime master equation is

```text
dPsi/dt
  = -alpha * R_theta(Psi)
    + lambda * Delta_6(R_theta(Psi))
    + eta * (G * Psi).
```

Interpretation:

1. `-alpha R` closes the working field toward its decoded representation.
2. `lambda Delta_6 R` propagates reconstruction mismatch through local spatial structure.
3. `eta G*Psi` applies the fixed permeation drive.

Under the chosen sign convention a positive `eta` is not ordinary diffusion; the fixed glyph is `-Delta_6/6`. Implementations must therefore keep explicit timestep and projection guards.

---

## 5. Discrete transaction

For explicit Euler step size `dt`:

```text
snapshot    = Psi_n
latent      = E_theta(snapshot)
Psi_hat     = D_theta(latent, requested_support)
R_n         = snapshot - Psi_hat
H_n         = Delta_6 R_n
P_n         = G * snapshot
rhs_n       = -alpha R_n + lambda H_n + eta P_n
candidate   = snapshot + dt * rhs_n
projected   = Pi_Lambda(candidate)
```

Commit rule:

```text
if projected is finite
and resource bounds hold
and optional validator accepts:
    Psi_(n+1) = projected
    COMMIT
else:
    Psi_(n+1) = Psi_n
    ROLLBACK
```

One transaction reads one frozen snapshot. No operator observes partial writes from another operator in the same step.

---

## 6. Sparse support closure

When halo expansion is enabled, the requested support is

```text
S_n = A_n union face_neighbours(A_n).
```

This allows one step of local residual/permeation propagation without dense materialization.

The closure is rejected before codec execution when

```text
|S_n| > max_active_cells.
```

A decoder may return values only inside the requested support. Returning coordinates outside that set is a contract violation and fails closed.

---

## 7. Lambda projection

`Pi_Lambda` is the runtime admission operator. The reference projection:

- rejects non-finite coordinates or values;
- rejects coordinates outside the logical lattice;
- clamps values to configured `[value_min, value_max]`;
- optionally prunes values within `prune_epsilon` of zero;
- enforces `max_active_cells`;
- runs an optional caller-supplied candidate validator before commit.

A production implementation may add model-version checks, rate/distortion ceilings, memory reservations, scheduler ownership, checksums, or policy masks.

---

## 8. Stability guard

For a non-expansive codec, the reference configuration uses the conservative operator bounds

```text
||I - A_theta|| <= 2
||Delta_6|| <= 12
||G|| <= 2
```

and rejects a default explicit step when

```text
dt * (2 alpha + 24 lambda + 2 |eta|) > 1.
```

This is an engineering admission guard. It does not prove convergence for an arbitrary learned codec.

Default reference coefficients:

```text
alpha  = 1.0
lambda = 1.0
eta    = 0.1
dt     = 0.025
```

The training coefficients `beta` and `gamma` belong to the model-training objective and are not runtime field coefficients by default.

---

## 9. Training objective

A compatible codec may be trained with

```text
L(Psi, Psi_hat)
  = mean((Psi - Psi_hat)^2)
    + beta * ||z||^2
    + gamma * ||grad(Psi) - grad(Psi_hat)||^2.
```

When decoder weights are tied to encoder weights, the decoder uses the true convolution adjoint: spatial kernel reversal and channel-axis transpose where applicable.

Training updates are not committed into an active runtime model without shadow evaluation and the same candidate-first transaction rule used for other adaptive changes.

---

## 10. Telemetry

Every attempted step reports at least:

```text
cycle
support_cells
active_cells_before
active_cells_after
reconstruction_mse
anchor_mse
max_abs_residual
max_abs_rhs
committed
rejection_reason
```

`anchor_mse` is measured against the immutable `Psi_anchor` so a self-referential codec cannot hide generational drift merely by becoming self-consistent with its latest reconstruction.

---

## 11. System integration

The full Jarvis-X flow is

```text
input / sparse field
    -> Layer 4 support closure
    -> Layer 5 encode
    -> Layer 5 decode
    -> same-space residual
    -> Delta_6 + glyph operators
    -> candidate Euler step
    -> Pi_Lambda
    -> validator / shadow evaluation
    -> canonical transaction boundary
    -> commit + provenance
    -> next inward cycle
```

The canonical 64-bit VM remains the authority substrate. Alternative 256-bit/512-bit tensor ISAs, DMEB-32, CUDA kernels, C++ implementations, FPGA soft cores, and hardware extensions are permitted implementation targets only when they preserve this state-transition and transaction contract.

---

## 12. Bytecode lowering contract

A tensor-oriented backend may lower one field step into conceptual operations such as

```text
BEGIN_TX
LOAD_SPARSE_FIELD
ENCODE3D
DECODE3D
RESIDUAL
LAPLACIAN6
GLYPH6
FUSED_AXPY
CHECK_BOUNDS
CHECK_POLICY
VERIFY
JOURNAL
COMMIT
```

but those names do not redefine the canonical VM instruction format. Backend-specific instruction encodings remain adapters until an ADR promotes them.

Self-optimization follows

```text
active program/model
-> candidate patch
-> deterministic/shadow test
-> reconstruction + anchor + resource checks
-> Pi_Lambda
-> COMMIT or ROLLBACK.
```

Direct unjournaled mutation of authoritative code or model state is outside this contract.

---

## 13. Conformance conditions

A runtime conforms to Field Runtime v2 when it demonstrates:

1. same-space `Psi - D(E(Psi))` residuals;
2. exact six-face-neighbour stencil semantics;
3. bounded sparse support and explicit boundary conditions;
4. deterministic replay for deterministic codecs;
5. immutable run anchors;
6. candidate-first commit/rollback;
7. explicit timestep/resource bounds;
8. honest latent information limits;
9. measured throughput reported separately from virtual/logical depth;
10. tests showing invalid decoder support, non-finite values, and rejected candidates fail closed.
