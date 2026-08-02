# Hyperion Verifiable Audit Engine

## Status

Hyperion is an **experimental deterministic audit kernel** for semantically aligned, multi-source event arithmetic. It makes every fusion, derivative, anomaly score, model version and proof commitment reproducible.

It does not claim that a camera, microphone, database or CPU trace is truthful merely because its arithmetic verifies. Input authenticity remains a chain-of-custody and attestation problem.

## Design goals

1. Never average incompatible quantities or units.
2. Preserve event identity instead of expanding sparse evidence onto a millisecond matrix.
3. Use elapsed time in all derivatives.
4. Predict from prior state before calculating a continuity residual.
5. Use robust, two-sided statistics for contaminated audit windows.
6. Normalize the composite anomaly score to `[0, 1]`.
7. Require explicit labels before reporting supervised precision or adapting score weights.
8. Version and hash every model and configuration.
9. Bound the health score to `[0, 100]`.
10. Export deterministic fixed-point witnesses suitable for an external SNARK/STARK circuit.

## Observation contract

Each source produces an `Observation`:

```python
Observation(
    source="csv",
    timestamp_ms=1_700_000_000_000,
    value=149_000.00,
    quantity="amount",
    unit="ZAR",
    correlation_id="transaction-42",
    confidence=1.0,
    label="beneficiary-name",
)
```

The pair `(quantity, unit)` is a dimensional contract. An identity match encoded as `(identity_match, boolean)` is not fused into `(amount, ZAR)`.

Unknown sources are excluded by default. Duplicate observations from one source do not gain extra voting weight; Hyperion selects one deterministic, highest-confidence observation per source and event.

## Event-time alignment

Correlation identifiers are authoritative when supplied. Otherwise, observations are grouped into deterministic time buckets of `event_tolerance_ms`.

For event `k`, compatible source observations are:

\[
\mathcal O_k = \{x_{s,k}: q_{s,k}=q^*,\;u_{s,k}=u^*\}.
\]

No cross-modal Kalman imputation is performed without an explicit state-space model. Missing streams are excluded and reflected in the fusion confidence.

## Fixed-point fusion

For each compatible source:

\[
X_{s,k}=\operatorname{round}(S_x x_{s,k}),
\]

\[
W_{s,k}=\operatorname{round}(S_w w_s c_{s,k}).
\]

The circuit-ready arithmetic is:

\[
N_k = \sum_s W_{s,k}X_{s,k},
\qquad
D_k = \sum_s W_{s,k},
\]

\[
M_k = \operatorname{round}\!\left(\frac{N_k}{D_k}\right),
\]

\[
N_k = D_kM_k + r_k,
\qquad
2|r_k|\le D_k.
\]

The decoded fused value is:

\[
\mu_k=M_k/S_x.
\]

`ArithmeticWitness.verify()` checks the integer relation and commitment. `circuit_inputs()` exports only integer arrays and public relation values.

## Time-correct mechanics

For elapsed time:

\[
\Delta t_k=\max(t_k-t_{k-1},\Delta t_{\min})/1000,
\]

Hyperion calculates:

\[
\Delta\mu_k=\mu_k-\mu_{k-1},
\]

\[
v_k=\frac{\Delta\mu_k}{\Delta t_k},
\qquad
a_k=\frac{v_k-v_{k-1}}{\Delta t_k},
\qquad
j_k=\frac{a_k-a_{k-1}}{\Delta t_k}.
\]

The minimum time resolution is explicit. Simultaneous events therefore do not cause division by zero, and the chosen resolution remains part of the configuration hash.

## Independent continuity prediction

The current observation is never used to manufacture its own prediction. Hyperion uses prior committed state:

\[
\widehat\mu_{k|k-1}
=
\mu_{k-1}+v_{k-1}\Delta t_k+\tfrac12a_{k-1}\Delta t_k^2.
\]

The residual is:

\[
R_k=\mu_k-\widehat\mu_{k|k-1}.
\]

The scale-aware tolerance is:

\[
\tau_k=\tau_{\mathrm{abs}}+\tau_{\mathrm{rel}}
\max(|\widehat\mu_{k|k-1}|,|\mu_{k-1}|).
\]

The continuity flag activates when `|R_k| / tau_k > 1`.

## Robust filters

All filters produce both a boolean flag and a continuous severity in `[0, 1]`.

### Acceleration spike

The engine uses a causal window and median absolute deviation:

\[
z_k^{\mathrm{robust}}
=
\frac{0.67448975(a_k-\operatorname{median}(a))}
{\operatorname{MAD}(a)+\epsilon}.
\]

The test is two-sided.

### Precision strike

Given lower balance bound `L`:

\[
B_k^{\mathrm{projected}}=B_{k-1}+\Delta\mu_k,
\qquad
\operatorname{buffer}_k=B_k^{\mathrm{projected}}-L.
\]

A candidate requires a negative delta, debit magnitude above the configured historical quantile, and a non-negative buffer inside the limit-proximity band.

### Ghost entity

An event must have no non-empty label and exhibit a robust magnitude anomaly. Missing descriptive metadata alone is never treated as proof of wrongdoing.

### Bytecode divergence

CSV and CPU values are compared only inside the same aligned event. The test uses absolute plus relative tolerance:

\[
|x_{csv,k}-x_{cpu,k}|
>
\tau_{abs}+\tau_{rel}|x_{csv,k}|.
\]

A divergence proves disagreement between committed observations. It does not by itself identify the cause.

## Composite anomaly score

Hyperion uses a bounded logistic model:

\[
\operatorname{CAS}_k
=
\sigma\!\left(b+\sum_iw_i s_{i,k}\right),
\qquad
s_{i,k}\in[0,1].
\]

The score is always in `[0, 1]`. Model parameters are non-negative, bounded, versioned and hashed.

Supervised fitting requires `TrainingExample` labels. Hyperion does not derive precision from unlabelled data and does not silently reinterpret reconstruction loss as classification precision.

## Geometric health score

Critical exposure ratio:

\[
A=
\frac{\sum_k|\mu_k|\mathbf 1[CAS_k\ge\tau_c]}
{\sum_k|\mu_k|+\epsilon}.
\]

Critical event-frequency ratio:

\[
F=\frac{N_{critical}}{N_{events}}.
\]

The bounded score is:

\[
GHS=100\operatorname{clip}(1-\lambda_AA-\lambda_FF,0,1).
\]

## Tamper-evident report

Hyperion builds separate Merkle roots for:

- canonical input observations;
- arithmetic witnesses;
- output audit points.

The final report digest commits to:

```text
model hash
configuration hash
input Merkle root
witness Merkle root
output Merkle root
GHS
```

This is deterministic computation attestation, not a zero-knowledge proof by itself. A proving backend can consume `ArithmeticWitness.circuit_inputs()` and prove the same fixed-point relation without disclosing private source values.

## Reproducibility boundary

A defensible audit must retain:

- raw evidence commitments;
- source acquisition and clock metadata;
- correlation identifiers;
- Hyperion configuration hash;
- score-model hash and version;
- source code or release hash;
- report digest;
- reviewer labels used for any supervised update.

Adaptive fitting should occur on a training partition. The evidentiary audit partition should use a frozen model.

## Example

```python
from jarvisx.hyperion import HyperionEngine, Observation

observations = [
    Observation("csv", 1_000, 149_000.00, "amount", "ZAR", "tx-1"),
    Observation("audio", 1_003, 149_000.00, "amount", "ZAR", "tx-1", 0.92),
    Observation("cpu", 1_001, 149_000.00, "amount", "ZAR", "tx-1"),
]

report = HyperionEngine().audit(observations)
assert report.verify()
print(report.report_digest)
```

## Benchmark policy

The repository includes `scripts/benchmark_hyperion.py` for deterministic synthetic evaluation. Throughput and anomaly-quality numbers must be reported with hardware, Python version, workload size, seed, configuration hash and model hash. No state-of-the-art claim should be made without comparison against named baselines on public or independently reviewable data.
