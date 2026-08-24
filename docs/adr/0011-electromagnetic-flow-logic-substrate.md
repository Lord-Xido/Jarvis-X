# ADR-011: Electromagnetic Flow Logic as the Physical Substrate Boundary

- Status: Proposed integration
- Date: 2026-08-19
- Scope: Dr Moagi state-space runtime, fixed-point switching telemetry, hardware-facing research layers

## Context

Jarvis-X increasingly represents computation as bounded kinetic state evolution:

```text
Observe -> Encode -> Evolve -> Decode -> Compare -> Correct
       -> Lambda -> Commit/Rollback -> Omega -> Re-enter
```

At the physical implementation layer, any electronic realization of those state transitions is ultimately carried by charge, current and electromagnetic fields. However, the repository must preserve a strict distinction between:

1. abstract computational state,
2. its canonical digital bit representation,
3. an electrical switching model, and
4. a physically validated electromagnetic field model.

Collapsing those layers would create false claims. A software state equation is not itself a Maxwell field, and a logical toggle count does not determine radiated or conducted emissions without hardware geometry and electrical parameters.

## Decision

Jarvis-X will treat **electromagnetic flow logic** as a layered substrate model:

```text
kinetic state transition
        |
        v
canonical fixed-point logic image
        |
        v
bit-transition / Hamming activity
        |
        v
explicit electrical switching model
        |
        v
hardware current-density / geometry model
        |
        v
Maxwell / EM field solution
```

The canonical logical-to-electrical bridge is observational and fail-closed. It does not acquire authority over the computational state merely because it can estimate switching cost.

## Canonical mapping

For a state transition

```text
Xi_t -> Xi_{t+1}
```

an observer first projects selected scalar state into an explicit Q16.16 logic image:

```text
B_t = Q16.16(Xi_t)
B_{t+1} = Q16.16(Xi_{t+1})
```

The raw switching count is

```text
H_t = sum popcount(B_t XOR B_{t+1})
```

and the measured activity factor is

```text
alpha_t = H_t / N_bits.
```

The reference Data Bus Inversion (DBI) payload model reports

```text
H_DBI = sum min(H_word, 32 - H_word)
```

for each 32-bit Q16.16 word. This is an ideal payload transition bound only. It does not assert that the executing CPU, GPU, memory controller or physical bus implements DBI, and it excludes the separate inversion-control line.

When and only when explicit hardware constants are supplied, the first-order electrical model may estimate

```text
P_dyn = alpha * C_eff * V_dd^2 * f
I_avg = P_dyn / V_dd.
```

`C_eff`, `V_dd` and `f` must come from an identified hardware model, design specification or measurement. Jarvis-X must not infer them from logical state.

A current-slew proxy may be reported as

```text
|dI/dt| ~= |I_t - I_{t-1}| / Delta_t
```

when the current estimates and timing interval are explicit.

## Physical boundary

The repository will not derive electric or magnetic field fidelity directly from logical telemetry.

A physical EM claim requires, at minimum, an identified mapping for current density and geometry:

```text
J(r,t) = G_layout[I(t), placement, routing, package, return paths, materials]
```

followed by an appropriate electromagnetic solution such as Maxwell's equations:

```text
div E = rho / epsilon
div B = 0
curl E = -dB/dt
curl B = mu J + mu epsilon dE/dt.
```

Without that layer, Jarvis-X may report **logical switching telemetry** or an **electrical first-order estimate**, but not electromagnetic field strength, radiation, emissions compliance, signal integrity or hardware power fidelity.

## Architectural interpretation

The physical-to-logical hierarchy is therefore:

```text
(E, B, rho, J)
      -> electrical circuit state
      -> transistor / interconnect switching
      -> digital words
      -> Q16.16 latent state
      -> kinetic logic
      -> 3D adaptive computation.
```

Conversely, an implemented kinetic transition can be traced downward only as far as evidence permits:

```text
Delta Xi
  -> Delta Q16.16 words
  -> measured bit toggles
  -> alpha
  -> P/I estimate when hardware constants exist
  -> physical EM only when layout/current-density model exists.
```

## Relationship to the Psi-Phi-Lambda-Omega-Theta stack

The symbols remain computational abstractions. A hardware implementation may map them onto physical structures, but the mapping must be explicit:

- `Psi`: observation / ingress coupling
- `Phi`: state propagation
- `Lambda`: admissibility and authority constraints
- `Omega`: persistent correction and evidence state
- `Theta`: configurable transfer parameters

No symbol is automatically identified with a Maxwell field quantity.

## Implementation

The C++ reference layer provides `jarvisx/electromagnetic_flow.hpp` with:

- deterministic Q16.16 switching projection,
- raw Hamming transition counts,
- ideal DBI payload transition counts,
- raw and DBI activity factors,
- optional first-order electrical power/current estimates requiring explicit hardware parameters,
- a current-slew proxy requiring an explicit time interval.

The Dr Moagi state-space reference executable observes every state transition through this layer and reports raw and DBI activity without altering authoritative state.

## Invariants

1. **Observation does not grant authority.** EM/switching telemetry cannot mutate the state transition.
2. **No implicit hardware model.** Physical constants must be supplied explicitly.
3. **No field claim from toggle count.** Bit activity is not Maxwell-field fidelity.
4. **Canonical representation before comparison.** State is projected to Q16.16 rather than hashing or comparing implementation-defined floating-point object bytes.
5. **DBI claims are scoped.** Reported DBI values describe ideal payload transition reduction, not actual interconnect behavior unless hardware support is demonstrated.
6. **Finite bounded inputs.** Non-finite state or invalid physical parameters fail closed.
7. **Measured-vs-asserted separation.** Logical activity, electrical estimates and physical EM measurements remain distinct telemetry classes.

## Consequences

### Positive

- Provides a concrete bridge from kinetic logic to switching activity.
- Makes the earlier Q16.16 EM-leanness concept measurable.
- Enables future optimization objectives that include both reconstruction loss and switching cost.
- Preserves honest separation between software simulation and physical hardware evidence.
- Supports future GALS, phase-staggering and placement studies without changing authority semantics.

### Trade-offs

- Q16.16 telemetry is a canonical observation image, not necessarily the representation used by the physical compiler or processor.
- First-order dynamic power estimates omit leakage, short-circuit current, clock-tree effects, memory subsystem behavior, parasitics, voltage droop and thermal coupling.
- Accurate EM prediction requires a separate hardware/layout solver and empirical calibration.

## Future work

1. Bind switching telemetry into unified transaction receipts as non-authoritative metrics.
2. Add per-region / per-agent activity maps for the 3D swarm.
3. Measure deterministic phase-staggering and GALS scheduling effects on peak activity.
4. Add an explicit hardware-description contract for `C_eff`, voltage domains, frequency domains and interconnect topology.
5. Integrate package/layout current-density data before introducing Maxwell-field simulation.
6. Validate estimates against measured power/current traces on identified hardware.

## Claim boundary

This ADR establishes **electromagnetic flow logic as the physical substrate interpretation of electronic computation** while keeping the executable repository claim narrower:

> Jarvis-X can measure canonical logical switching activity and, with explicit hardware parameters, compute a first-order electrical switching estimate. It does not yet simulate or validate the physical electromagnetic fields of a real implementation.
