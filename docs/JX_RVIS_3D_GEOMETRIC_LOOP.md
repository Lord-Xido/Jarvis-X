# JX-RVIS 3D Geometric Auto-Encoding/Decoding Multiparallel Loop

## Operational identity

The shell is turned inward by feeding every committed decoded output into the next geometric cycle:

\[
X_{t+1}=Y_t,
\qquad
Z_t=E_G(X_t),
\qquad
Y_t=D_G\!\left(\operatorname{Select}_{\Lambda}\{\mathcal L_k(Z_t,\Omega_t)\}_{k=1}^{K}\right).
\]

Each cycle is transactional:

```text
Observe public shell values
  -> Q3 geometric encoding
  -> 3D coordinate mapping
  -> multiparallel candidate lanes
  -> multiresolution 2x2x2 condensation
  -> top-down decoding
  -> reconstruction scoring
  -> Lambda projection
  -> commit or rollback
  -> feed committed output inward
```

The implementation visualises explicit public state and shell events. It does not expose private hidden chain-of-thought.

## Geometric latent state

\[
Z_G=(V,E,C,A,M,\Omega,\Lambda)
\]

- `V`: voxel samples;
- `E`: implicit six-neighbour relations;
- `C`: multiresolution 2x2x2 cells;
- `A`: signed-three-bit activations;
- `M`: lattice metric and dimensions;
- `Ω`: cumulative reconstruction correction;
- `Λ`: admissibility constraints.

Linear and geometric addresses are bijective:

\[
\gamma^{-1}(x,y,z)=x+W(y+Hz).
\]

## Multiparallel lanes

Every lane reads the same committed checkpoint and produces an isolated candidate:

| Lane | Evolution |
|---|---|
| `identity` | preserves the encoded voxel value |
| `diffusion` | blends each voxel with its six-neighbour mean |
| `memory` | applies cumulative Ω correction |
| `hybrid` | combines local state, neighbourhood and Ω |

The lanes are executed concurrently as pure functions. Results are gathered in a fixed order, so selection and replay remain deterministic.

The selected branch is:

\[
k^*=\arg\min_{k\in\mathcal K_{\mathrm{admissible}}}
\|X_t-D_G(Z_t^{(k)})\|_1.
\]

## Lambda projection

A lane is admissible only when:

\[
E^{(k)}_{\mathrm{reconstruction}}\leq E_{\max}
\]

and:

\[
N^{(k)}_{\mathrm{active}}\leq N_{\max}.
\]

When no lane passes, the complete committed geometric state remains unchanged.

## Cumulative memory

\[
\Omega_{t+1}
=
\operatorname{clip}
\left(
\rho\Omega_t+
\eta(X_t-Y_t),
-\Omega_{\max},
\Omega_{\max}
\right).
\]

Ω improves candidate generation but never bypasses Λ verification.

## Python API

```python
from jarvisx.geometric_rvis import GeometricFeedbackRuntime

runtime = GeometricFeedbackRuntime()
cycles = runtime.run_feedback([3, 1, -1, -3], cycles=4)

for cycle in cycles:
    print(cycle.selected_lane)
    print(cycle.metrics)
    print(cycle.events)
```

## CLI

```bash
jarvisx geometry3d --cycles 4 3 1 -1 -3
```

The command returns JSON containing every cycle, parallel lane result, 3D hierarchy, Λ decision, public visualization event, and final committed state.

## Browser shell

Open `geometric-rvis-shell.html`, then either run its built-in demonstration or paste/load JSON produced by `jarvisx geometry3d`.

The browser shell provides a rotatable 3D point-lattice projection, encoded/evolved/decoded stage selection, cycle replay, lane telemetry, commit status, and event timeline controls.

## Governing invariant

\[
\boxed{
\text{Parallel lanes may propose geometric evolution; only Λ may commit it.}
}
\]
