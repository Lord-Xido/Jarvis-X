# DMSO 3D Operational Runtime

This document defines the executable reference implemented in `jarvisx.dmso_runtime`.
It operationalizes the inward-facing equations as a bounded sparse 3D state machine; it does
not claim consciousness, unrestricted self-modification, continuous `SO(3)` invariance, or
native-code acceleration from symbolic compression alone.

## State and dynamics

The authoritative state is a sparse field

\[
\mathcal S_t : \mathcal G \to \mathbb R^C,
\qquad
\mathcal G=\{0,\ldots,N-1\}^3.
\]

Each active coordinate reads all 26 Chebyshev neighbours. The front decoder chooses the
smallest occupied `z` for each `(x,y)` ray. The inward map is

\[
F_\theta(\mathcal S,\mathcal U)
=\mathcal E_\theta(\mathcal D(\mathcal S),\mathcal U,\mathcal S),
\]

and the fast state clock uses relaxed iteration

\[
\mathcal S_{t+1}
=(1-\alpha)\mathcal S_t+\alpha F_\theta(\mathcal S_t,\mathcal U_t).
\]

The fixed-point residual is the RMS norm of `F_theta(S,U)-S`. `settle()` stops when that residual
is below the configured tolerance or the bounded iteration budget is exhausted.

## Parameter clock

Five scalar mechanics parameters are explicit: self, neighbour, projection and external-input
gains plus bias. When targets are supplied, `step(..., learn=True)` performs one analytic MSE
gradient update after computing the state candidate. Parameters are clamped to a configured
finite range. State and parameter updates therefore remain distinct clocks.

## Exact operator promotion

Every active cell executes the deterministic primitive trace

```text
LOAD_SELF -> AGGREGATE_26 -> DECODE_FRONT -> LOAD_INPUT
-> AFFINE -> TANH -> RELAX
```

Repeated traces can be promoted to exact macro definitions. The macro keeps its complete child
expansion and is marked verified because promotion aliases the same primitive semantics.
Reported compression is **description compression**: a promoted trace can be represented by one
four-byte operator reference instead of seven four-byte primitive references. No claim of runtime
speedup is made until a compiled or fused backend is measured.

Higher-order abstractions have explicit depth. Depths above `auto_approve_depth` require the
`human_approved=True` gate and all definitions are bounded by `max_operator_depth`.

## Transaction and verification boundary

`step()` normalizes inputs and computes candidates before authoritative mutation. Invalid or
non-finite inputs fail before commit. `verify()` checks state finiteness and bounds, parameter
bounds, operator provenance, abstraction depth and the human gate. `state_digest()` provides a
canonical SHA-256 digest for deterministic replay comparison.

## Minimal use

```python
from jarvisx.dmso_runtime import DMSOConfig, DMSORuntime

runtime = DMSORuntime(DMSOConfig(side=8, promotion_repeats=4))
runtime.seed({(3, 3, 3): 0.25, (4, 3, 3): -0.10})

metrics = runtime.step(
    external={(3, 3, 3): 0.5},
    targets={(3, 3, 3): 0.4},
    learn=True,
)

assert runtime.verify()
print(metrics)
```

## Capability boundary

This is an executable mathematical reference. It demonstrates sparse 3D coordination,
projection-with-prior recurrence, fixed-point telemetry, bounded parameter adaptation, exact
macro abstraction and deterministic verification. It does not prove general intelligence or
state-of-the-art performance. Those claims require workload benchmarks against fixed baselines,
including the discovery, verification, lookup and memory costs of the abstraction layer.
