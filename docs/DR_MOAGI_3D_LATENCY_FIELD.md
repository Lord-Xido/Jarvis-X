# Dr Moagi 3D Latency-Field Optimizer

## Status

This document specifies the executable latency-routing controller in
`src/jarvisx/dr_moagi_latency3d.py`.

The controller does **not** claim a universal 50--250 ms LLM response bound.
It treats propagation as a lower bound and all other latency as measured or
estimated engineering overhead.

## 1. Operational state

Each inference endpoint is embedded into a three-axis latency field:

\[
\mathbf p_i=(x_i,y_i,z_i)
\]

with

\[
x_i=T_{\mathrm{network},i},
\qquad
y_i=T_{\mathrm{compute},i},
\qquad
z_i=T_{\mathrm{queue},i}.
\]

The axes are operational rather than literal Cartesian geography:

- **X -- network geometry:** propagation floor plus measured RTT and network overhead;
- **Y -- compute geometry:** prompt prefill plus first-token decode and, when requested,
  decode tail;
- **Z -- service-load geometry:** queueing delay.

The scalar routing potential is

\[
U_i=w_xx_i+w_yy_i+w_zz_i+b_i
\]

where \(b_i\) is a bounded feedback residual learned from observed minus predicted latency.

The selected endpoint is

\[
i^*=\arg\min_{i\in\mathcal H}U_i
\]

over the healthy endpoint set \(\mathcal H\).

## 2. Causality floor

For one-way path distance \(d_i\) and configured signal speed \(v\),

\[
T_{\mathrm{causal},i}=\frac{2d_i}{v}.
\]

When RTT telemetry is present, the controller uses

\[
T_{\mathrm{transport},i}=\max(T_{\mathrm{causal},i},T_{\mathrm{RTT},i})
\]

so bad or synthetic telemetry cannot imply propagation below the configured physical floor.

The network axis is therefore

\[
x_i=T_{\mathrm{transport},i}+T_{\mathrm{network-overhead},i}.
\]

The default propagation speed is \(2\times10^8\) m/s, a useful engineering approximation
for optical-fiber propagation. It is configurable.

## 3. Compute axis

For input length \(L\), measured prefill rate \(R_p\), and measured decode rate \(R_d\),

\[
T_{\mathrm{prefill}}=1000\frac{L}{R_p}\;\mathrm{ms}
\]

and

\[
T_{\mathrm{decode},1}=1000\frac{1}{R_d}\;\mathrm{ms}.
\]

For a time-to-first-token objective,

\[
y_i=T_{\mathrm{prefill}}+T_{\mathrm{decode},1}.
\]

For an \(n\)-token completion objective,

\[
y_i=T_{\mathrm{prefill}}+1000\frac{n}{R_d}.
\]

This deliberately uses effective measured rates rather than peak FLOP/s.

## 4. Total predicted latency

The user-visible prediction is

\[
T_i=x_i+y_i+z_i+T_{\mathrm{render}}.
\]

`render_ms` is request-side overhead and is not part of the 3D routing coordinate.

## 5. Kinetic damping by hysteresis

Repeated routing decisions can oscillate when two endpoints have nearly equal potential.
If the incumbent endpoint is \(c\) and the best challenger is \(j\), the controller switches
only when

\[
U_c-U_j>h
\]

where \(h\ge0\) is `hysteresis_ms`.

This is the discrete routing analogue of damping: small field fluctuations do not cause
route churn.

## 6. Closed feedback loop

After execution, the caller may submit observed latency \(T_i^{obs}\). The prediction
residual is

\[
e_i=T_i^{obs}-T_i^{pred}.
\]

The endpoint bias is updated with an exponentially weighted moving average:

\[
b_i^{t+1}=(1-\alpha)b_i^t+\alpha e_i
\]

and clipped to

\[
|b_i|\le b_{\max}.
\]

The next routing decision therefore incorporates recent prediction error without rewriting
endpoint telemetry.

The end-to-end loop is

\[
\boxed{
\text{measure}
\rightarrow
\text{embed in 3D}
\rightarrow
\text{predict}
\rightarrow
\arg\min U
\rightarrow
\text{execute}
\rightarrow
\text{observe error}
\rightarrow
\text{update bias}
\rightarrow
\text{repeat}
}
\]

## 7. Relationship to the existing inward optimizer

This controller is complementary to `src/jarvisx/dr_moagi_meta_optimizer.py`.

The existing meta-optimizer searches bounded internal runtime configuration axes. The
latency-field optimizer searches bounded **deployment choices supplied by the caller**.
Neither component rewrites source code or autonomously provisions remote infrastructure.

A higher-level orchestration layer can therefore compose the two:

\[
\text{request}
\rightarrow
\text{3D endpoint selection}
\rightarrow
\text{bounded runtime configuration}
\rightarrow
\text{verified execution}.
\]

## 8. Example

```python
from jarvisx.dr_moagi_latency3d import (
    EndpointState3D,
    LatencyField3DOptimizer,
    RequestProfile,
)

optimizer = LatencyField3DOptimizer(
    [
        EndpointState3D(
            name="edge",
            one_way_distance_km=100,
            network_overhead_ms=2,
            queue_ms=1,
            prefill_tokens_per_s=10_000,
            decode_tokens_per_s=20,
        ),
        EndpointState3D(
            name="regional-gpu",
            one_way_distance_km=5_000,
            network_overhead_ms=2,
            queue_ms=1,
            prefill_tokens_per_s=50_000,
            decode_tokens_per_s=200,
        ),
    ]
)

decision = optimizer.select(
    RequestProfile(input_tokens=20, output_tokens=1, objective="ttft")
)

print(decision.endpoint.name)
print(decision.estimate.point3d)
```

For long generated outputs, set `objective="completion"` and provide the expected
`output_tokens`; the optimizer can then prefer a more distant endpoint when its decode
throughput compensates for the additional network distance.
