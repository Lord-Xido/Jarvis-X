# ADR-013: Bit-serial mixed-signal control reference

- **Status:** Accepted
- **Date:** 2026-09-02
- **Decision owners:** Jarvis-X maintainers

## Context

The canonical DM-vOmegaXi+ fixed-point runtime models sparse internal state,
description, bounded latent projection, recurrent memory, and transactional Theta
policy. The newer sensor-to-actuator formulation adds delta-sigma acquisition,
XNOR/popcount inference, 16-bit register adaptation, binary error, pulse-density
output, and an H-bridge/MOSFET boundary.

Those mechanisms need executable semantics without implying that a Python model
is analog hardware, a trained controller, or a production-safe gate driver.

## Decision

Add `dm_vomegaxi_mixed_signal.py` as a dependency-free digital reference with the
following ordered boundary:

```text
bounded samples
  -> stateful delta-sigma
  -> binary matrix score
  -> finite Lambda projection
  -> 16-bit Omega recurrence
  -> Theta mask and Hamming metric
  -> independent fail-closed interlock
  -> mutually exclusive PDM gate intent
```

The runtime must:

1. reject shape, range, and capacity violations;
2. retain explicit bit ordering and 16-bit word semantics;
3. require the integrator to supply plant-specific hardware limits;
4. keep Theta policy separate from electrical/thermal/timing interlocks;
5. emit data only, with no hardware or deployment side effects;
6. distinguish internal controller fixed points from physical equilibrium;
7. report deterministic state hashes and measurable transition gaps.

The term “zero latency” is not accepted as a physical claim. Implementations may
report measured or synthesis-derived latency under a declared platform and clock.

## Authority boundary

Theta cannot authorize physical output when the independent interlock is tripped.
Conversely, passing this software interlock is not sufficient authority to drive
power electronics. A real system requires plant-specific control analysis and
independent hardware protection.

The emitted positive/negative vectors are logic intent. They are not pin timing,
do not synthesize dead time, and never bypass a gate driver.

## Consequences

Positive:

- the complete symbolic stack now has deterministic, testable bit-level semantics;
- Boolean arithmetic, memory, policy, safety, and output evidence are separated;
- exact dimensions and finite resource limits replace unbounded language;
- interlock trips are visible without erasing the cognitive-state trace;
- future FPGA/ASIC implementations have a conformance reference.

Costs and limitations:

- Python is not a timing model for hardware;
- first-order delta-sigma and PDM are reference algorithms, not optimized filters;
- fixed binary weights are supplied by the caller and are not trained here;
- discrete internal convergence says nothing about plant stability;
- numeric thresholds require calibration outside this repository.

## Alternatives rejected

### Put electrical protection inside Theta

Rejected because application policy must not be the sole barrier between a model
output and hazardous power switching.

### Emit GPIO or PWM directly from the package

Rejected because the repository has no plant, driver, timing, isolation, or
certification context that could make direct actuation safe.

### Preserve “zero-latency” as an implementation requirement

Rejected because propagation, synchronization, memory, and switching delays are
non-zero in every physical realization.

### Document equations without an executable reference

Rejected because ordering, bit packing, recurrence, trip behavior, and state
mutation would remain ambiguous and untested.

## Validation

The acceptance suite covers arithmetic identities, state ordering, malformed
inputs, capacity bounds, every interlock, fail-closed emission, deterministic
replay, mutual exclusion, and internal fixed-point measurement.
