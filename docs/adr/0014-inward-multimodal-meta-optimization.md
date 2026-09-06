# ADR-014: Bounded inward multimodal meta-optimization

**Status:** Proposed  
**Date:** 2026-09-05

## Context

The inward multimodal 3D research runtime closes media generation through a decode/re-encode fixed-point
operator. A further requirement is to turn the runtime inward onto its own operating configuration so it
can refine task, swarm, inward, memory and geometry controls from measured performance.

Unconstrained self-modifying code would conflict with the repository's candidate-first, auditable research
boundary. A scalar fitness alone is also insufficient: an optimizer can improve global coherence while
silently degrading fixed-point integrity, stability or resource use.

## Decision

Jarvis-X will represent self-optimization of the inward multimodal runtime as a **bounded outer-loop
configuration search**.

The optimizer may mutate only an explicit numerical genotype derived from `Swarm3DConfig`. Every candidate
must execute in a caller-controlled shadow evaluation and return normalized quality metrics plus measured
fixed-point and resource penalties.

Promotion requires all of the following:

1. the candidate is stable;
2. fixed-point error does not exceed the incumbent by more than the configured allowance;
3. resource cost remains within the configured regression allowance;
4. composite fitness clears the minimum improvement threshold.

The default fixed-point regression allowance is zero.

No candidate may mutate executable source, canonical VM state, repository policy, permissions or deployment
controls through this subsystem.

## Consequences

### Positive

- adaptation is reproducible under an explicit seed;
- the incumbent remains unchanged during candidate evaluation;
- improvements are evidence-driven rather than narrative claims;
- fixed-point integrity is a hard constraint instead of a weak weighted objective;
- workload-specific evaluators can incorporate task quality, cross-modal coherence, latency, memory and
  hardware telemetry without coupling those measurements into the search engine;
- the design composes with existing Jarvis-X candidate-first transaction patterns.

### Negative

- search explores only the declared parameter surface;
- local configuration improvement does not imply model-weight improvement or general intelligence gain;
- workload-specific fitness design remains the caller's responsibility;
- deterministic evolutionary search can be computationally expensive as branch width and generations grow;
- hard non-regression gates can reject useful Pareto trade-offs unless explicitly relaxed by policy.

## Mathematical contract

For incumbent configuration `theta_g`, define a bounded neighborhood `N(theta_g)` and measured fitness
vector `F(theta)`.

The promotion law is

```text
theta_(g+1) = argmax score(theta')
              theta' in N(theta_g)
```

subject to

```text
stable(theta') = true
R_fp(theta') <= R_fp(theta_g) + epsilon_fp
C_resource(theta') <= (1 + epsilon_resource) C_resource(theta_g)
score(theta') > (1 + epsilon_gain) score(theta_g).
```

If the admissible set is empty, `theta_(g+1) = theta_g`.

The inner representation recurrence remains

```text
Phi(Z) = E(D(Z)),
```

while the outer runtime recurrence is

```text
theta -> run -> measure -> candidate search -> shadow verify -> theta_next.
```

These recurrences are intentionally distinct.

## Validation

This decision is successful when:

- candidate generation is deterministic for a fixed seed and generation;
- unstable incumbents cannot enter optimization;
- higher scalar fitness cannot bypass fixed-point or resource gates;
- a genuinely superior admissible candidate can be promoted;
- the report records every evaluated candidate and promotion reason;
- ordinary canonical VM tests remain independent of this research layer.
