# Dr Moagi Deep Distiller (DM-DD)

**Status:** Operational reference runtime  
**Product name:** Dr Moagi Deep Distiller  
**Law ID:** `DM-DD`  
**Commit policy:** `Pi_Lambda-atomic`

## 1. Definition

Deep Distiller is the product-facing form of the IP-locked residual auto-iteration network. It repeatedly compresses, reconstructs, measures residual error, updates persistent residual memory, proposes a residual-gradient parameter step, and commits the complete next state only through `Pi_Lambda`.

The canonical loop is

```text
Z_t         = E_Theta(X_t)
X_hat_t     = D_Theta(Z_t)
E_t         = X_t - X_hat_t
Omega_t+1   = rho * Omega_t + (1-rho) * E_t
Theta'_t+1  = Theta_t - eta * grad_Theta ||E_t||^2
X'_t+1      = X_hat_t + omega * Omega_t+1
(X,Omega,Theta)_t+1 = Pi_Lambda(X',Omega',Theta')
```

A tick has exactly two phases:

```text
PROPOSE -> GATE -> ATOMIC COMMIT | REJECT
```

No provisional state, memory, or parameter value is authoritative before the gate passes.

## 2. IP lock

`IP-locked` has a concrete runtime meaning in this implementation:

1. **Commit lock** — `X`, `Omega`, and `Theta` commit together only after `Pi_Lambda` accepts the full candidate.
2. **Parameter lock** — the reference `Theta` changes only through the explicit residual-gradient rule inside `theta_candidate`.
3. **Budget lock** — state and latent active-cell ceilings are hard configuration invariants.
4. **Finite-value lock** — non-finite state, memory, or parameters fail closed.
5. **Value-domain lock** — committed state must remain inside configured numerical bounds.
6. **Audit lock** — every accepted or rejected tick is appended to the hash-chained journal.

Rejected proposals may be recorded for audit but cannot mutate authoritative runtime state.

## 3. Reference codec

The reference implementation uses two bounded learnable scalar parameters:

```text
Theta = {
    encoder_gain,
    decoder_gain,
}
```

Encoding is a deterministic sparse bottleneck:

```text
z_i = encoder_gain * x_i
```

followed by a configured prune threshold and deterministic top-K active-cell budget.

Decoding is

```text
x_hat_i = decoder_gain * z_i
```

on retained latent support.

This deliberately small model makes the gradient and constitutional commit boundary directly testable. A production neural encoder/decoder can replace these scalar maps without changing the transactional law.

## 4. Residual gradient

For mean squared residual loss

```text
L = mean_i (x_i - decoder_gain * encoder_gain * x_i)^2
```

on retained latent support, the reference gradients are

```text
dL/d encoder_gain = -2 * decoder_gain * mean_i(x_i * e_i)
dL/d decoder_gain = -2 * mean_i(z_i * e_i)
```

The discrete top-K support is held fixed during one tick; the implementation therefore uses a straight-through reference approximation across the bottleneck selection boundary.

Each parameter delta is independently capped by `theta_max_delta` and then clipped to the configured `Theta` interval.

## 5. Omega memory

Residual memory is an EMA:

```text
Omega_t+1 = rho * Omega_t + (1-rho) * E_t
```

with optional pruning. The next state proposal is

```text
X'_t+1 = X_hat_t + omega_gain * Omega_t+1
```

so the decoder output is corrected by bounded historical residual information.

## 6. Pi_Lambda constitutional gate

`Pi_Lambda` validates the entire candidate transaction:

```text
Candidate = {
    X_prime,
    Omega_prime,
    Theta_prime,
    latent_cells,
}
```

The built-in gate rejects candidates that violate:

```text
active_cells <= max_active_cells
latent_cells <= max_latent_cells
all values finite
value_min <= X_prime <= value_max
theta_min <= Theta_prime <= theta_max
```

An optional external policy may add further admissibility rules.

If any check fails:

```text
X_t+1     = X_t
Omega_t+1 = Omega_t
Theta_t+1 = Theta_t
```

The rejection is fail-closed and the auto-iteration loop stops.

## 7. Stopping semantics

A run terminates when any of the following occurs:

```text
residual_rms <= residual_tolerance
max_iterations reached
Pi_Lambda rejects a candidate
```

The fixed-point target is operationally

```text
E* ~= 0
X* = Pi_Lambda(X*)
grad_Theta L* ~= 0
```

but convergence is reported from measured residual rather than asserted symbolically.

## 8. Sparse / no-silent-expansion rule

The logical lattice side defaults to `1000`, but the implementation never allocates `1000^3` cells. Only sparse active support is materialized.

The runtime has separate ceilings for:

```text
max_active_cells
max_latent_cells
```

A larger logical lattice is metadata unless an explicit gated expansion policy is implemented.

## 9. Relationship to existing Jarvis-X laws

| Name | Role |
|---|---|
| `DM-vOmegaXi+` | Existing bounded inward fixed-point operator and semantic-gap runtime |
| Latent Moagi flow | Optional continuous latent dynamics |
| DM-Lambda / geometric AE | Research geometry/Ricci branch; not required by DM-DD |
| **Deep Distiller** | Product-facing transactional residual auto-iteration network |

The existing `dm_vomegaxi_fixed_point.py` remains intact. `dm_vo_xi_operational.py` is a compatibility surface that resolves to the Deep Distiller implementation.

## 10. CLI

After installation:

```bash
jarvisx-deep-distiller --side 16 --steps 8
```

The CLI prints JSON containing per-tick residuals, gradient values, committed parameters, hashes, journal status, and final lock state.

## 11. Constitutional invariant

The defining implementation invariant is:

```text
PROVISIONAL != AUTHORITATIVE
```

until

```text
Pi_Lambda(candidate) == ACCEPT
```

This is the operational meaning of:

> Deep Distiller = residual auto-iteration under constitutional commit.
