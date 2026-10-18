# ADR-006: Adopt the Dr Moagi multimodal 3D I/O operator as a bounded research runtime

**Status:** Accepted  
**Date:** 2026-08-15

## Context

ADR-002 established the Dr Moagi 3D adaptive codec-runtime as a bounded research architecture outside the authoritative deterministic VM core. The next system requirement is to apply the same encode/decode/verify discipline to all input and output media rather than treating display, audio, touch, RF, networking, storage and actuator paths as unrelated peripherals.

A useful common abstraction is:

```text
physical phenomenon -> transducer/sensor -> encoder -> common 3D state
common 3D state -> decoder -> actuator/transducer -> physical phenomenon
```

This abstraction is operational only when channel boundaries, feedback provenance, resource ceilings and transaction semantics are explicit. Capability metadata is not physical feedback: for example, display EDID or link status cannot establish what photons were actually emitted without an optical sensor or equivalent loopback.

## Decision

Jarvis-X adopts the multimodal macro-operator

```text
G_{Omega Xi,IO}^3D :
    (Xi_t^3D, X_t, Omega_t^3D, Theta_t)
    ->
    (Xi_(t+1)^3D, X_hat_t, Omega_(t+1)^3D, Theta_(t+1))
```

with the canonical bounded recurrence

```text
Xi_(t+1)^3D =
Pi_Lambda_t [
    Xi_t^3D
    + P_(1:M)^inward(
        Xi_t^3D,
        F^3D E_(1:M)(X_t)
      )
    - E_t^3D
    + Omega_t^3D
    + U_t^3D
    - eta_t grad_Theta L_t
]
```

and the world loop

```text
X_t
  -> E_(1:M)
  -> Z_t^3D
  -> Xi_t^3D
  -> D_(1:N)
  -> X_hat_t
  -> physical/digital environment
  -> X_(t+1)
```

The initial reference implementation is `src/jarvisx/dr_moagi_multimodal_io.py`.

### Runtime interpretation

One macro-instruction performs:

```text
SENSE
-> TRANSDUCE
-> ENCODE
-> FUSE
-> PREDICT
-> DECODE CURRENT STATE
-> OBSERVE/LOOPBACK WHEN AVAILABLE
-> COMPUTE RESIDUAL
-> UPDATE OMEGA CANDIDATE
-> COMPUTE NEXT 3D STATE
-> PI_LAMBDA PROJECT
-> VALIDATE
-> ATOMIC COMMIT OR ROLLBACK
-> DECODE COMMITTED OUTPUTS
```

The reference runtime uses a sparse 3D lattice of bounded vectors. It is deliberately not a dense physical simulation of every medium.

### Channel contract

A medium adapter provides three explicit operations:

```text
encode_input(observation) -> sparse 3D field
decode_output(committed_field) -> medium-specific output
observe_output(output) -> sparse 3D loopback field
```

`observe_output` is invoked only when a target is supplied. If no real or digital loopback exists, an adapter must not fabricate one.

### Shared state and memory

All media fuse into one common state `Xi_t^3D`. Persistent residual memory is:

```text
Omega_(t+1)^3D =
    rho * Omega_t^3D
    + eta_Omega * E_t^3D
```

and is committed atomically with the candidate state.

### Loss

The reference objective exposes the same resource dimensions as the mathematical specification:

```text
L_t =
    lambda_D * distortion
    + lambda_tau * latency
    + lambda_B * bandwidth
    + lambda_P * energy
    + lambda_C * compute
```

The runtime accepts externally measured resource costs. It does not invent physical power, bandwidth or latency telemetry.

## Required invariants

1. **Deterministic fusion:** fixed channel order, inputs, weights and configuration produce the same fused field.
2. **Explicit physical feedback:** physical output correctness is claimed only when a corresponding observation/loopback exists.
3. **Sparse bounded state:** active cells and vector width are explicit and bounded.
4. **Finite numeric state:** non-finite values are rejected.
5. **Pi-Lambda projection:** all committed values are projected into configured bounds.
6. **Atomic state/memory commit:** rejected candidates mutate neither `Xi` nor `Omega`.
7. **Known channels only:** undeclared input or target channels fail closed.
8. **No authority escalation:** the multimodal research runtime cannot directly mutate the canonical VM implementation or bypass its policy/ledger boundaries.
9. **Measured resource terms:** latency, bandwidth and energy penalties come from supplied measurements.
10. **No dense-world implication:** a logical `side^3` lattice does not imply resident allocation of every cell.

## Initial media mapping

The contract is intended to host, without privileging any one medium:

- vision/display;
- microphone/speaker audio;
- text/API/keyboard/printer channels;
- touch and haptics;
- motion/IMU/robotic actuation;
- RF and antenna paths;
- network packet paths;
- storage read/write paths;
- thermal, pressure, chemical and electrical sensors/actuators.

Individual adapters remain responsible for real hardware protocol, calibration, authorization and safety.

## Validation

The initial reference implementation is accepted when tests demonstrate:

- multimodal weighted fusion into a shared 3D state;
- explicit target/output residual feedback;
- persistent bounded `Omega` update;
- atomic rollback on validator rejection;
- value clamping under `Pi_Lambda`;
- unknown-channel rejection;
- active-cell budget enforcement;
- no implicit claim of physical observation without loopback.

## Consequences

The Dr Moagi research architecture now has one uniform I/O law instead of separate metaphors for display, sound, touch, RF, storage and actuation. The cost is that physical correctness remains adapter-specific: a mathematically shared operator cannot replace hardware drivers, calibration, real sensors, real actuators, safety interlocks or protocol conformance.

This ADR extends ADR-002 and does not supersede the deterministic-core separation established by ADR-001.
