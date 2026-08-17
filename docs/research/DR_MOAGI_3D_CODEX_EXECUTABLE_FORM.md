# Dr Moagi 3D Codex — Hardened Canonical Executable Form

**Designation:** `Xi^recur_Phi_3D`  
**Layer:** bounded Jarvis-X Layer-5 research operator  
**Status:** canonical hardened executable research semantics  
**Safety boundary:** hypothesis generation is separated from authoritative observation, learning, permeation, and output release

## 1. Canonical composition

The Dr Moagi 3D system is represented as four coupled spaces rather than one undifferentiated expression:

- observation / scene space `X`;
- latent state space `Z` / `Xi`;
- parameter space `Theta`;
- spatial permeation / rendered-output space `(Phi, vpx)`.

The system may generate and recursively refine hypotheses, but a generated state becomes authoritative only after epistemic admission.

### 1.1 Encode

```text
Z_t = E_3D(O_t)
```

where `O_t` is an admissible externally sourced observation and `E_3D` is the bounded 3D encoder.

### 1.2 Recursive inward refinement

```text
Z*_t = FixedPoint(R_inward, Z_t; epsilon_fp, I_max)
```

Operationally:

```text
Z_(m+1) = R_inward(Z_m)
```

until

```text
||Z_(m+1) - Z_m||_2 <= epsilon_fp
```

or the configured actual-iteration ceiling is reached.

The design may retain a virtual depth label such as

```text
N_virtual = 10^(6^10^6)
```

but physical work is always reported separately as the measured fixed-point iteration count. No claim is made that an arbitrary `N`-fold nonlinear recurrence can be exactly executed in logarithmic time.

### 1.3 Latent candidate update

The same-space latent update is

```text
Z_raw = Z*_t + P_t - K_epsilon * epsilon_t - eta_Z * grad_Z L_t
Z_smooth = S_dt(Z_previous, Z_raw)
Xi_(t+1)^cand = Pi_Lambda(Z_smooth)
```

Equivalently,

```text
Xi_(t+1)^cand = Pi_Lambda[
    S_dt(
        R_inward^*(E_3D(O_t))
        + P_t
        - K_epsilon * epsilon_t
        - eta_Z * grad_Z L_t
    )
]
```

`Pi_Lambda` is a real admissibility projection. The reference implementation uses projection onto the Euclidean latent ball

```text
||Xi||_2 <= Lambda_max
```

rather than componentwise clipping.

## 2. Parameter learning remains in parameter space

Parameter learning is not added directly to the latent state. It is a separate candidate update:

```text
Theta_(t+1)^cand = Theta_t - eta_Theta * grad_Theta L_t
```

This separation is mandatory because `grad_Theta L` inhabits parameter space while `Xi`, `Z`, and `P` inhabit latent space. Direct addition is undefined unless an explicit transport or Jacobian map is supplied.

The candidate parameter update is committed only after epistemic admission.

## 3. Decode to a hypothesis, not automatically to truth

The projected latent is decoded as a generated hypothesis:

```text
H_(t+1) = D_3D(Xi_(t+1)^cand)
```

`H_(t+1)` is inspectable and may participate in bounded internal reasoning, but it is not automatically promoted to an external observation or authoritative state.

The core epistemic invariant is

```text
H -> O    prohibited by default
```

A hypothesis may recurse, but self-consistency or fixed-point convergence is not evidence of factual correctness.

## 4. Epistemic verification and commit

Verification compares the decoded hypothesis against:

- the current admissible external observation `O_t`;
- the immutable run anchor `A_0`;
- independently identified external evidence `E_1 ... E_n`;
- provenance, source confidence, support, and configured error thresholds.

```text
V_t = Verify(H_(t+1), O_t, A_0, E_1...E_n)
```

The authoritative state transition is

```text
Xi_(t+1) = Xi_(t+1)^cand    if V_t == ADMIT
Xi_(t+1) = Xi_t             if V_t == REJECT
```

and parameter learning is

```text
Theta_(t+1) = Theta_(t+1)^cand    if V_t == ADMIT
Theta_(t+1) = Theta_t             if V_t == REJECT
```

Rejected candidates are quarantined. They do not alter authoritative parameters and do not release authoritative permeation.

## 5. Canonical permeation source

The bounded latent is mapped to a scalar spatial source field before Green propagation:

```text
q_(t+1)(r') = Q[Xi_(t+1); r']
```

A reference executable source form is

```text
q(r') = gamma * |M(Xi)(r') - q_eq(r')| + beta * g(r')
```

where `M` is an explicit latent-to-source mapper and `g(r')` is a compatible scalar source-gradient field.

This avoids adding a scalar norm directly to a latent-space vector gradient.

## 6. 3D permeation field

For an admitted state,

```text
Phi_(t+1)(r) = integral_VLambda G_k(r,r') * q_(t+1)(r') dV'
```

with

```text
G_k(r,r') = exp(i*k*R) / (4*pi*R)
R = max(||r-r'||_2, epsilon_G)
```

The bounded discrete implementation is

```text
Phi(r) = sum_r' [exp(i*k*R)/(4*pi*R)] * q(r') * DeltaV
```

where `epsilon_G > 0` regularizes coincident source and target points.

Epistemic admission gates release:

```text
Phi_(t+1)(r) = GreenPermeate(q_(t+1))(r)    if V_t == ADMIT
Phi_(t+1)(r) = 0 / unreleased               if V_t == REJECT
```

The reference kernel is a computational Helmholtz/Green operator. It is not represented as physical electromagnetic radiation unless a separate validated physical model supplies units, constitutive relations, source physics, boundary conditions, and empirical evidence.

## 7. Canonical 3D output

A camera- or ray-conditioned decoder may define

```text
vpx_(t+1)(r) = D_3D(Xi_(t+1), gamma(r))
```

where `gamma(r)` denotes the camera/ray conditioning used by the rendering backend.

For a ray parameterization,

```text
gamma_p(s) = o + s * d_p
```

and a volumetric renderer may decode density and appearance

```text
D_3D(Xi) -> (sigma(r), c(r,d))
```

before integrating along the ray.

The authoritative-output rule is

```text
vpx_authoritative = Render(Xi_(t+1))    if V_t == ADMIT
vpx_authoritative = unreleased          if V_t == REJECT
```

The decoded rejected hypothesis may remain available for diagnostics, but it is labelled as a hypothesis and must not be presented as an authoritative observation.

## 8. Hardened master equation

The complete executable research law is therefore

```text
Z_t = E_3D(O_t)
Z*_t = FixedPoint(R_inward, Z_t; epsilon_fp, I_max)
Z_smooth = S_dt[Z*_t + P_t - K_epsilon*epsilon_t - eta_Z*grad_Z L_t]
Xi_(t+1)^cand = Pi_Lambda(Z_smooth)
H_(t+1) = D_3D(Xi_(t+1)^cand)
V_t = Verify(H_(t+1), O_t, A_0, E_1...E_n)

if V_t == ADMIT:
    Xi_(t+1) = Xi_(t+1)^cand
    Theta_(t+1) = Theta_t - eta_Theta*grad_Theta L_t
    q_(t+1) = Q[Xi_(t+1)]
    Phi_(t+1) = GreenPermeate(q_(t+1))
    vpx_(t+1) = Render(Xi_(t+1))
else:
    Xi_(t+1) = Xi_t
    Theta_(t+1) = Theta_t
    Phi_(t+1) = unreleased
    vpx_(t+1) = unreleased
    quarantine(H_(t+1))
```

In mathematical indicator notation,

```text
Phi_(t+1) = 1_[V_t=ADMIT] * GreenPermeate(Q[Xi_(t+1)])
```

with the implementation representing rejection by withholding the authoritative field rather than manufacturing a physical zero field.

## 9. Hallucination-control invariants

1. **External-observation boundary:** model-generated or hypothesis-derived packets are not admissible observations by default.
2. **No self-certification:** convergence, reconstruction consistency, or repeated agreement with the model's own outputs cannot satisfy independent evidence requirements.
3. **Immutable anchor:** the run retains an immutable anchor for long-horizon drift measurement.
4. **Independent evidence:** duplicate or correlated evidence cannot inflate corroboration count.
5. **Verified-only learning:** rejected hypotheses cannot update authoritative `Theta`.
6. **Verified-only permeation:** rejected hypotheses cannot release authoritative `Q` or `Phi`.
7. **Verified-only output:** rejected hypotheses cannot become authoritative `vpx` or observation state.
8. **Fail closed:** missing, conflicting, low-confidence, non-finite, unsupported, or provenance-invalid evidence causes rejection.
9. **Virtual depth is not throughput:** `N_virtual` is metadata; actual work is measured.
10. **Authenticated provenance is a deployment requirement:** software labels alone do not prove that an untrusted adapter is genuinely a sensor, instrument, user, or retrieval source.

## 10. Conservation boundary

The expression

```text
Energy_in = E_encode + E_recurse + E_decode + E_radiate
```

is not assumed to be a conservation law. A conserved quantity requires a defined energy functional and either proof or empirical validation that the participating operators preserve it.

## 11. Reference implementation

Core bounded Codex:

```text
src/jarvisx/dr_moagi_codex.py
```

Epistemic admission wrapper:

```text
src/jarvisx/dr_moagi_epistemic.py
```

Conformance tests:

```text
tests/test_dr_moagi_codex.py
tests/test_dr_moagi_epistemic.py
```

Architecture decisions:

```text
docs/adr/0010-dr-moagi-geometric-state-equation.md
docs/adr/0011-dr-moagi-epistemic-admission-gate.md
```

## 12. Operational loop

```text
EXTERNAL SENSE
  -> ENCODE
  -> BOUNDED FIXED-POINT INWARD RECURSE
  -> LATENT PREDICTION / CORRECTION
  -> SMOOTH
  -> Pi_Lambda PROJECT
  -> DECODE HYPOTHESIS
  -> VERIFY
       -> REJECT: QUARANTINE / NO LEARNING / NO PERMEATION / NO AUTHORITATIVE OUTPUT
       -> ADMIT:  COMMIT / LEARN / MAP SOURCE / PERMEATE / RENDER
  -> NEXT EXTERNALLY OBSERVED CYCLE
```

This is the hardened canonical executable interpretation of `Xi^recur_Phi_3D` within Jarvis-X's bounded research architecture.
