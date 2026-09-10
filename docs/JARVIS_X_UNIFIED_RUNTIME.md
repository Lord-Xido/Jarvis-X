# JARVIS-X Unified Runtime Permeation

## Status

Bounded coordination layer for the existing Dr Moagi ANN IDE.

This module does **not** replace the canonical Jarvis-X VM, the inward ANN implementations,
the policy/admission layer, or evidence from the external world. It makes the shared
runtime state explicit so those surfaces can exchange measured state rather than decorative
telemetry.

## Unified state

The committed state is

```text
S_t = (
  Psi_t,
  Phi_t,
  Lambda_t,
  Omega_t,
  Theta_t,
  Z_t,
  Xhat_t,
  e_t,
  telemetry_t
)
```

with

```text
S_(t+1) = M(S_t, X_t)
```

and fixed-point intent

```text
S* = M(S*, X).
```

The executable reference is `jarvisx.unified_runtime.UnifiedRuntime`.

## Operators

### Psi

For bounded observation `x_i` and previous residual memory `omega_i`,

```text
Psi_i = tanh(x_i + g_omega * omega_i)
```

where `g_omega = psi_memory_gain`.

`Psi` is an internal coordination state, not a claim that the runtime has measured or
reconstructed the physical world.

### Phi

For at least two state elements,

```text
Phi_i = 0.5 Psi_i + 0.25 Psi_(i-1) + 0.25 Psi_(i+1)
```

using wraparound indexing. A one-element state uses `Phi = Psi`.

This gives the coordination layer a deterministic local manifold operator without a dense
3D allocation.

### Lambda

```text
Lambda_i = tanh(Phi_i + Theta_t Psi_i)
```

`Lambda` here is transform state, not an authorization decision. Canonical policy and
admissibility remain in the existing Jarvis-X transaction/policy machinery.

### Auto-encode / auto-decode

The reference codec uses deterministic block means:

```text
Z_j = mean(Lambda[j*b : (j+1)*b])
Xhat = D(Z)
e = Lambda - Xhat
L_recon = mean(e^2)
```

Re-encoding the decoded state gives

```text
L_cycle = MSE(Z, E(D(Z))).
```

For this exact block codec, `L_cycle = 0` up to floating-point arithmetic. This is an
algebraic cycle-consistency check, not a learned-compression claim.

### Omega

Residual memory closes the inward loop:

```text
Omega_(t+1) = rho Omega_t + (1-rho) e_t
```

The next `Psi` reads this residual through `psi_memory_gain`.

### Theta

The controller state is bounded:

```text
Theta_(t+1) = clip(
    Theta_t + eta (L_recon - epsilon),
    -Theta_max,
    +Theta_max
)
```

It does not rewrite source code, alter policy, or grant execution authority.

## Measured H_MMM

Earlier browser prototypes used decorative sinusoidal HUD values. The unified runtime
instead reports

```text
H_MMM =
    L_recon
    + 0.25 L_cycle
    + 0.10 Delta_state
    + L_resource
```

with

```text
Delta_state = RMS(Psi_t - Psi_(t-1))
L_resource  = resource_weight * dimensions / max_dimensions.
```

Lower values indicate a smaller internal reconstruction/state-change objective. `H_MMM`
is a runtime diagnostic, not a physical Hamiltonian or universal-intelligence score.

## Verification and stability

A state is `verified` only when all committed numeric values are finite.

A state is `stable` only when

```text
verified
and L_recon <= stability_epsilon
and Delta_state <= stability_epsilon.
```

Every committed state receives a deterministic SHA-256 digest over numeric state, metrics,
cycle count, and verification flags. The hash supports replay/audit identity; it does not
establish semantic correctness or external-world truth.

## Relationship to the merged contour operator

`DMvOmegaXiContourOperator` remains a measurable inner kernel that may produce a bounded
`Delta Psi` before a unified runtime tick:

```text
(Phi, grad Lambda, Omega, Xi+, DM)
        -> contour operator
        -> Delta Psi
        -> bounded candidate/observation
        -> UnifiedRuntime.step(...)
```

This preserves the contour law as an explicit mathematical kernel without letting it bypass
validation or become an implicit source of truth.

## IDE integration

The repository already exposes:

```text
Dr Moagi ANN IDE
  |- transactional CodexVM
  |- deterministic bounded refactorer
  |- Inward4D ANN
  |- SQLite project persistence
  |- telemetry journal / WebSocket
  `- Dr Moagi 3D OS mount
```

The unified runtime is mounted as an additional coordination plane at `/runtime` through
`jarvisx.unified_runtime_api`.

Typical flow:

```text
POST   /runtime/v1/sessions
POST   /runtime/v1/sessions/{id}/tick
GET    /runtime/v1/sessions/{id}
POST   /runtime/v1/sessions/{id}/reset
DELETE /runtime/v1/sessions/{id}
```

Long-term, Chat, Editor, VM, ANN, and 3D visualization should read the same committed state
envelope rather than maintain unrelated synthetic metrics.

## Security and authority boundaries

The coordination layer:

- exposes no `eval`, `exec`, shell, GPIO, or host-process execution;
- contains no provider API secret and makes no remote LLM call;
- does not allocate symbolic virtual 3D spaces densely;
- does not treat internal reconstruction error as evidence about the external world;
- does not replace transaction, policy, rollback, or verification boundaries;
- uses bounded dimensions, bounded value magnitude, finite checks, deterministic hashing,
  and a bounded session count.

## Reference use

```python
from jarvisx.unified_runtime import UnifiedRuntime

runtime = UnifiedRuntime()
state = runtime.step([0.2, -0.4, 0.8, 0.1])

print(state.h_mmm)
print(state.state_hash)
```

The next call re-enters through `Omega` and `Theta`:

```text
X_t
 -> Psi
 -> Phi
 -> Lambda
 -> E
 -> Z
 -> D
 -> Xhat
 -> e
 -> Omega_(t+1)
 -> Theta_(t+1)
 -> next Psi
```
