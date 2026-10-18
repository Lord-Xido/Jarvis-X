# Inward Multimodal Meta-Optimization

**Status:** Research-layer companion specification  
**Runtime:** `src/jarvisx/inward_multimodal_swarm3d.py`  
**Meta-optimizer:** `src/jarvisx/inward_multimodal_meta_optimizer.py`

## 1. Purpose

This layer turns the multimodal runtime inward onto its own **bounded runtime configuration**.
It does not rewrite executable source code and it does not autonomously alter the canonical VM.
Instead, it treats a declared subset of `Swarm3DConfig` as a genotype, evaluates candidate
configurations in shadow execution, and promotes only candidates that satisfy hard verification gates.

The outer control loop is:

```text
RUN
  -> MEASURE
  -> MUTATE BOUNDED CONFIG
  -> SHADOW EVALUATE
  -> VERIFY
  -> PROMOTE OR REJECT
  -> RUN AGAIN
```

The inner multimodal loop remains:

```text
bytes/media -> encode -> 3D coordination -> inward E(D(z))
            -> decode -> re-encode -> verify
```

## 2. Runtime genotype

The current optimizer mutates only bounded numerical controls from `Swarm3DConfig`, including:

- integration step `dt`;
- Riemannian metric gain `alpha_metric`;
- task, inward, swarm and memory gains;
- feature-mixing and feature-similarity gains;
- geometry-distance gain;
- maximum inward iteration count.

Resource bounds, particle limits, position bounds and the executable implementation remain outside the
mutation surface unless a future ADR explicitly promotes them.

Let the tunable configuration be

```text
theta = (
    dt,
    alpha_metric,
    k_task,
    k_inward,
    k_swarm,
    k_memory,
    k_feature,
    k_similarity,
    k_geometry,
    N_steps,
)
```

and let one shadow run return

```text
F(theta) = (
    task_score,
    semantic_coherence,
    feature_coherence,
    fixed_point_error,
    resource_cost,
    stable,
).
```

The scalar ranking score is intentionally subordinate to hard constraints:

```text
score(theta) =
    0.30 task_score
  + 0.25 semantic_coherence
  + 0.20 feature_coherence
  + 0.15 / (1 + fixed_point_error)
  + 0.10 / (1 + resource_cost)
```

## 3. Candidate-first promotion law

For incumbent `theta_g`, generate a bounded candidate neighborhood

```text
N(theta_g) = {theta'_1, ..., theta'_B}.
```

Each candidate is evaluated without mutating the incumbent runtime.
A candidate is eligible only if all hard gates pass:

```text
stable(theta') = true
fixed_point_error(theta') <= fixed_point_error(theta_g) + epsilon_fp
resource_cost(theta') <= (1 + epsilon_resource) resource_cost(theta_g)
score(theta') > (1 + epsilon_gain) score(theta_g)
```

The default fixed-point regression allowance is zero:

```text
epsilon_fp = 0.
```

This encodes an important systems constraint: a candidate may not purchase better task or geometric
coherence by silently degrading inward fixed-point integrity.

If no candidate passes every gate, the incumbent is retained unchanged.

## 4. Mathematical outer loop

The bounded optimization law is

```text
theta_(g+1) =
    argmax score(theta')
    over theta' in N(theta_g)
    subject to verification gates,
```

with rollback-to-incumbent when the admissible candidate set is empty.

This is distinct from gradient training of neural weights. It is closer to deterministic evolutionary
configuration search under transactional promotion constraints.

## 5. Why this is an inward loop

The first inward loop checks representation self-consistency:

```text
Phi(z) = E(D(z))
```

and drives

```text
Phi(z*) ~= z*.
```

The second inward loop checks runtime self-consistency:

```text
runtime(theta)
  -> telemetry
  -> candidate theta'
  -> shadow runtime(theta')
  -> verification
  -> theta_(next).
```

Therefore the architecture has two nested recurrences:

```text
DATA LOOP:     z -> D(z) -> E(D(z)) -> z'
RUNTIME LOOP:  theta -> run -> measure -> propose(theta') -> verify -> theta_next
```

## 6. Verification contract

The meta-optimizer itself does not decide what `task_score`, semantic coherence or resource cost mean.
The caller must supply an evaluator grounded in the workload being optimized.

A production evaluator should include, where applicable:

- fixed-point residual `||Phi(Z)-Z||`;
- task loss / answer quality / reconstruction quality;
- cross-modal semantic disagreement;
- 3D coordination spread or graph-consensus error;
- latency and throughput;
- RAM/VRAM use;
- thermal or power telemetry when running on real hardware;
- constraint violations and numerical instability.

Repository CI and workload-specific benchmarks remain authoritative. A locally improved scalar score is
not by itself evidence of general capability improvement.

## 7. Safety and architectural boundary

This layer is deliberately bounded:

- no arbitrary source-code rewriting;
- no mutation of repository permissions or deployment policy;
- no bypass of candidate verification;
- no direct modification of the canonical VM state during shadow evaluation;
- no claim that 3D semantic coordinates are literal physical space;
- no claim that the optimizer provides global or unbounded self-improvement.

Its operational identity is:

```text
SELF-OBSERVING CONFIGURATION SEARCH
+ SHADOW EXECUTION
+ HARD INTEGRITY GATES
+ TRANSACTIONAL PROMOTION
```

That is the defensible form of "turn the runtime inward onto itself."
