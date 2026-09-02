# ADR-012: Bound DM-vOmegaXi+ as a mechatronic control reference

**Status:** Proposed  
**Date:** 2026-09-02  
**Extends:** ADR-001, ADR-002, ADR-007

## Context

The DM-vOmegaXi+ stack maps Psi, Phi, Lambda, Omega, and Theta onto sensing,
one-bit delta-sigma description, XNOR/popcount arithmetic, 16-bit recurrent
memory, and safety-constrained actuation. That mapping is implementable, but it
crosses three domains with materially different semantics: computation, power
electronics, and a mechanical plant.

Without a boundary, a software state transition can be mistaken for electrical
switching speed, an electrical edge can be mistaken for closed-loop mechanical
response, and an XOR prediction error can be mistaken for an ethical system.

## Decision

Jarvis-X adopts a deterministic, hardware-neutral reference with these rules:

1. Psi is a normalized sensor vector supplied by a caller or test fixture.
2. Phi is a first-order one-bit delta-sigma bank with explicit integrator state.
3. Lambda uses the exact identity `dot = 2 * popcount(XNOR(x,w)) - N`.
4. Omega is a bounded 16-bit rotate/XOR recurrent register.
5. Theta is an admissibility governor: clamp, slew limit, emergency stop,
   direction-reversal dead time, and per-leg shoot-through exclusion.
6. The software endpoint is a hardware-neutral gate command. A real driver,
   MOSFET stage, motor, sensor, and watchdog are outside the reference boundary.
7. XOR is a prediction-error bit only. Ethical or policy claims require an
   independently specified policy model, authority source, and validation set.
8. No latency is called zero. Computational, electrical, and plant timings are
   measured and reported separately when physical hardware exists.

## Consequences

- Every conceptual operator has a testable transition and bounded state.
- Gate commands fail closed on invalid inputs and emergency stop.
- Reversal cannot immediately energize the opposite bridge direction.
- The reference is suitable for simulation, trace generation, and HDL/MCU
  conformance fixtures, but not direct deployment to power hardware.
- Physical deployment additionally requires isolated gate drivers, hardware
  dead time, over-current and thermal cut-offs, watchdogs, safe-state analysis,
  and plant-specific control validation.

## Acceptance evidence

- exact XNOR/popcount equivalence tests;
- delta-sigma density test over a bounded sample window;
- deterministic 16-bit recurrence test;
- clamp, slew, emergency-stop, reversal, and shoot-through invariants;
- complete trace coverage from sensor sample through gate command;
- hardware-in-the-loop evidence before any physical-performance claim.

The reference implementation is
`src/jarvisx/dm_vomegaxi_mechatronic.py`; its focused tests are
`tests/test_dm_vomegaxi_mechatronic.py`.
