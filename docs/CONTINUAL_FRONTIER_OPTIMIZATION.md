# Continual Frontier Optimization

## Status

Normative research/runtime contract for evidence-gated recursive improvement of the Adaptive Corporation orchestration layer.

This document does **not** claim literal infinite performance, perpetual state-of-the-art status, or permission for unrestricted self-modification. "Indefinite" means the runtime may execute repeated finite improvement cycles for as long as new candidates, resources, valid benchmarks, and authorized operating conditions exist.

The core rule is:

```text
candidate proposes
benchmark measures
policy constrains
shadow observes
canary tests
production promotes
rollback restores
memory records
```

No candidate may self-certify or self-promote.

## 1. Objective

The optimizer turns the Adaptive Corporation from a fixed orchestration runtime into an evidence-driven champion/challenger system:

```text
active champion C_g
        |
        v
candidate generation
        |
        +--> c_1
        +--> c_2
        +--> ...
        +--> c_M
                |
                v
       repeatable benchmark harness
                |
                v
          Pi_Lambda gates
                |
                v
        SHADOW -> CANARY
                |
          verified gain?
          /           \
        yes            no
         |              |
         v              v
    PRODUCTION       ROLLBACK
         |
         v
      C_(g+1)
         |
         +---------------------> next finite generation
```

## 2. State

For generation `g`, define:

```text
C_g       active production champion
B(C_g)    repeated benchmark evidence for champion
F_t       versioned external frontier snapshot
Q_g       set of candidate changes derived from C_g
Sigma_t   security/integrity state
Lambda    immutable and mutable policy constraints
Omega_g   accumulated optimization evidence
```

A candidate is:

```text
q = (
  candidate_id,
  parent_id,
  description,
  risk_score,
  change_scope,
  evidence
)
```

The parent identity is mandatory. A stale candidate derived from an old champion cannot silently become the new champion.

## 3. Metrics

Each metric declares:

```text
name
direction = maximize | minimize
weight >= 0
optional hard minimum
optional hard maximum
maximum tolerated regression
```

The system does not collapse safety, integrity, or other non-tradeable properties into a single weighted score when they are configured as hard bounds.

For metric `k`:

```text
Delta_k = challenger_k - champion_k         when maximizing
Delta_k = champion_k - challenger_k         when minimizing
```

A hard bound fails closed before utility aggregation.

## 4. Repeated evaluation

One score is not evidence of superiority. The benchmark harness runs each subject repeatedly:

```text
x_(k,1), x_(k,2), ..., x_(k,n)
```

and records:

```text
mean_k
stdev_k
sample vector
benchmark digest
```

The digest binds the evaluated subject identity to its observed samples and summary statistics.

Production backends should additionally bind:

```text
source commit
artifact digest
benchmark version
dataset version
runtime version
hardware class
configuration
seed set
```

before making strong comparative claims.

## 5. Conservative improvement gate

For champion and challenger standard deviations `s_c` and `s_q`, the reference runtime computes:

```text
SE_k = sqrt(s_c^2 / n_c + s_q^2 / n_q)
```

and a conservative directional improvement:

```text
Delta_k^- = Delta_k - z * SE_k
```

where `z` is a configurable confidence multiplier.

The normalized aggregate utility is:

```text
Delta J^- =
  sum_k w_k * (Delta_k^- / max(|champion_k|, 1))
  ------------------------------------------------
                     sum_k w_k
```

A candidate may enter shadow mode only when:

```text
Delta J^- >= minimum_utility_gain
```

and every hard gate passes.

This is a promotion heuristic, not a universal statistical proof. Production benchmark programs should choose inferential methods appropriate to their data-generating process.

## 6. Immutable invariants

Optimization authority is subordinate to invariants.

Examples include:

```text
no bypass of authorization
no removal of auditability
no unrestricted credentials inside model context
no direct probabilistic control of safety-critical actuators
no weakening of security below configured floors
no hidden benchmark substitution
no mutation of protected constitutional policy
```

Formally:

```text
q admissible only if I_j(q) = true for every immutable invariant I_j
```

The optimizer may optimize *inside* the admissible region. It does not redefine the region by itself.

## 7. Security-state coupling

A candidate that would mutate production state is additionally gated by the current security state:

```text
Sigma_t.confidence >= threshold
AND intrusion_detected = false
```

Therefore optimization authority contracts during suspected intrusion or integrity degradation.

```text
security confidence down
        -> mutation authority down
        -> promotion authority down
        -> observation/diagnosis remains available
```

Recursive improvement must never become a privilege-escalation path.

## 8. Frontier tracking

State-of-the-art is external, moving, domain-specific, and benchmark-dependent. The runtime therefore represents it explicitly as a versioned `FrontierSnapshot`:

```text
F_t = (
  benchmark_id,
  benchmark_version,
  metrics
)
```

For each shared metric, the runtime reports a frontier delta:

```text
frontier_delta_k = challenger_k - frontier_k    maximize
frontier_delta_k = frontier_k - challenger_k    minimize
```

A negative frontier delta means the candidate remains below the declared frontier on that metric. A positive delta means it exceeds that particular declared benchmark value; it does **not** establish universal SOTA.

Frontier snapshots must be refreshed externally from authoritative benchmark evidence. The optimizer does not invent its own frontier and call it SOTA.

## 9. Promotion ladder

The promotion ladder is deliberately asymmetric:

```text
REJECTED
   ^
   |
SHADOW -> CANARY -> PRODUCTION
             |          |
             +----------+
               rollback
```

### Shadow

The candidate operates without authoritative production mutation. It gathers comparative evidence against real or replayed workloads.

### Canary

A bounded fraction of authorized work is exposed to the candidate. Canary observations carry:

```text
success
value_delta
integrity_ok
```

### Production

Only a candidate with successful benchmark evidence and canary evidence may become the next champion.

## 10. Canary policy

The reference controller requires:

```text
minimum observation count
minimum success rate
minimum mean value delta
all integrity observations true
```

Any failed condition produces a rejected promotion decision with rollback semantics.

Production systems should additionally support:

```text
error-budget thresholds
latency-budget thresholds
security incident tripwires
customer-impact bounds
financial exposure limits
blast-radius limits
automatic traffic restoration
```

## 11. Champion immutability during evaluation

A challenger is evaluated against a fixed champion snapshot.

```text
candidate.parent_id == champion.subject_id
```

This prevents moving-baseline ambiguity and stale candidate promotion.

Once a candidate reaches `PRODUCTION`, `commit_production` re-evaluates the artifact before changing the active champion identity.

## 12. Separation of proposal and authority

The design separates:

```text
proposal generation
benchmarking
governance
release control
production mutation
```

An AI system may generate candidate descriptions or code, but it does not thereby receive authority to:

```text
change its own policy
change benchmark definitions
modify production credentials
promote itself
suppress failed evaluations
rewrite the evidence ledger
```

Those functions remain separate control-plane responsibilities.

## 13. Continual optimization recurrence

For generation `g`:

```text
Q_g = Generate(C_g, Omega_g, F_t)
```

For each candidate `q_i`:

```text
B_i = Evaluate(q_i)
D_i = Gate(C_g, B(C_g), B_i, Sigma_t, Lambda, F_t)
```

A passing candidate traverses:

```text
SHADOW -> CANARY -> PRODUCTION
```

Only then:

```text
C_(g+1) <- q_i
Omega_(g+1) <- Omega_g + evaluation evidence + release evidence
```

and the next finite cycle may begin.

## 14. Relationship to the Adaptive Corporation recurrence

The corporate recurrence remains:

```text
Xi_(t+1) = Pi_Lambda[
    Xi_t
  + P_(1:M)(Xi_t)
  - E_t
  + Omega_t
  + U_t
]
```

The optimizer adds a slower meta-loop over the mechanisms that implement `P`, `U`, verification, and orchestration:

```text
Theta_(g+1) = Promote_
  Lambda,Sigma,Evidence(
    Candidate(Theta_g)
  )
```

where `Theta` is the mutable implementation/configuration state and the promotion operator is evidence-gated.

The meta-loop may not directly mutate `Lambda`'s protected invariants.

## 15. Beyond-frontier interpretation

The engineering objective is:

```text
maximize durable risk-adjusted utility
while continuously reducing the measured frontier gap
subject to immutable invariants
```

or:

```text
minimize Gap(B(C_g), F_t)
while Delta J^- > 0
and Lambda = PASS
and Sigma = PASS
```

If a validated candidate exceeds the current declared frontier on one or more metrics, the evidence receipt records that fact. It does not convert a local benchmark result into a universal claim.

## 16. Why literal infinity is rejected

Physical systems have finite resources, irreducible uncertainty, measurement error, changing environments, and multi-objective trade-offs. Some improvements approach asymptotes. Some metrics conflict. Some frontiers move.

Therefore the runtime implements **open-ended improvement capacity**, not a mathematical promise of unbounded improvement.

The safe interpretation of "indefinite" is:

```text
repeat finite evidence-gated improvement cycles
for as long as useful admissible improvements continue to exist
```

## 17. Reference implementation

The dependency-light reference implementation is:

```text
src/jarvisx/continual_optimizer.py
```

with tests in:

```text
tests/test_continual_optimizer.py
```

Core types:

```text
MetricSpec
OptimizationCandidate
BenchmarkHarness
EvidencePromotionGate
FrontierSnapshot
ReleaseController
ContinualOptimizer
```

## 18. Production evolution path

The reference runtime should evolve toward:

```text
versioned benchmark registry
held-out evaluation sets
adversarial/red-team suites
trace grading
causal and counterfactual evaluation
multi-objective Pareto selection
resource/cost-aware optimization
formal invariant checking where feasible
signed artifact provenance
reproducible build attestations
durable workflow execution
human approval queues
sandboxed candidate generation
staged deployment infrastructure
continuous frontier ingestion
independent rollback authority
```

## 19. External engineering alignment

This architecture is intentionally compatible with established production principles:

- durable workflow engines should persist execution state across failures rather than relying on model context;
- agent systems should use tools, guardrails, tracing/evaluation, and human intervention for high-risk actions;
- zero-trust systems should avoid implicit trust based only on network location;
- AI risk controls should be integrated throughout design, deployment, evaluation, and operation.

These references inform the architecture but do not replace Jarvis-X's own empirical validation requirements.

## 20. Canonical rule

The continual optimizer is governed by one non-negotiable statement:

```text
NO EVIDENCE -> NO PROMOTION
NO INTEGRITY -> NO MUTATION
NO CANARY SUCCESS -> NO PRODUCTION
NO VERIFIED ADVANTAGE -> NO SOTA CLAIM
```

The purpose of recursive optimization is not to make the system believe it is superior. The purpose is to make every claimed improvement increasingly difficult to fake.
