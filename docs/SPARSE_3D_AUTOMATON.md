# Sparse 3-D Auto-Encoding/Decoding Processing Automaton

## Operational status

This module converts the Jarvis-X geometric automaton specification into a
runnable, deterministic Python reference implementation.

The nominal coordinate universe is

\[
\mathcal U = \{0,\ldots,1000^{1000}-1\}^3
\]

and therefore contains

\[
|\mathcal U|=(1000^{1000})^3=10^{9000}
\]

virtual cells. The implementation does **not** allocate a dense tensor of this
size. It uses exact arbitrary-precision coordinates, a deterministic procedural
background field, and a bounded sparse active frontier.

## State

For each materialised coordinate \(\mathbf r=(x,y,z)\), the committed cell state
is

\[
C_{\mathbf r,t}=(B_{\mathbf r,t},\Omega_{\mathbf r,t},a_{\mathbf r,t},v_{\mathbf r,t})
\]

where:

- \(B\) is the scalar field value;
- \(\Omega\) is persistent residual memory;
- \(a\) counts inactive cycles for pruning;
- \(v\) is the cell revision.

The global state is

\[
\Sigma_t=(\mathcal A_t,C_t,\mathcal M_t,J_t)
\]

with active frontier \(\mathcal A_t\), mechanics \(\mathcal M_t\), and
transaction journal hash \(J_t\).

## Local 3-D autoencoder

Each cell reads the seven-value axis-aligned neighbourhood

\[
V_{\mathbf r,t}=
(B_{\mathbf r,t},B_{\mathbf r+e_x,t},B_{\mathbf r-e_x,t},
 B_{\mathbf r+e_y,t},B_{\mathbf r-e_y,t},
 B_{\mathbf r+e_z,t},B_{\mathbf r-e_z,t}).
\]

The deterministic ANN executes

\[
Z_{\mathbf r,t}=E_\theta(V_{\mathbf r,t}),
\qquad
\widehat Z_{\mathbf r,t+1}=P_\theta(Z_{\mathbf r,t},\Omega_{\mathbf r,t}),
\]

\[
\widehat V_{\mathbf r,t+1}=D_\theta(\widehat Z_{\mathbf r,t+1}).
\]

The centre reconstruction residual is

\[
E_{\mathbf r,t}=\widehat V_{\mathbf r,t+1}[0]-B_{\mathbf r,t}.
\]

## Automaton update

Persistent correction memory evolves as

\[
\Omega_{\mathbf r,t+1}
=\rho_\Omega\Omega_{\mathbf r,t}-\eta_\Omega E_{\mathbf r,t}.
\]

The six-neighbour discrete Laplacian is

\[
\nabla_h^2 B_{\mathbf r,t}
=\sum_{\mathbf q\in\mathcal N(\mathbf r)}
(B_{\mathbf q,t}-B_{\mathbf r,t}).
\]

The transaction proposal is

\[
\widetilde B_{\mathbf r,t+1}=B_{\mathbf r,t}+\Delta t
\left[D\nabla_h^2B_{\mathbf r,t}-K E_{\mathbf r,t}
+\Omega_{\mathbf r,t+1}\right].
\]

The candidate state is committed only after the \(\Lambda\) verification gate
checks finiteness, coordinate validity, magnitude bounds, energy budget, and
active-cell budget:

\[
\Sigma_{t+1}=\operatorname{COMMIT}
\left(\Pi_\Lambda[\widetilde\Sigma_{t+1}]\right).
\]

Failure produces an atomic rollback to \(\Sigma_t\).

## Procedural storage

An untouched cell is reconstructed as

\[
B_{\mathbf r,0}=H(s,\mathbf r)
\]

using keyed BLAKE2b. Sampling a virtual coordinate therefore does not allocate
it. A cell is materialised only when it is injected, becomes active, or lies on
the immediate causal boundary of an active cell.

For \(A_t\) active cells, latent dimension \(d\), and constant neighbourhood
size \(k=6\), one cycle is approximately

\[
T_t=O(A_t(d+k)),\qquad M_t=O(A_t).
\]

The nominal \(10^{9000}\) universe affects address range, not dense runtime
cost.

## Bounded inward optimisation

`BoundedMechanicsOptimizer` tests a finite declared set of mechanics candidates
on forked shadow states. A candidate is adopted only when every transaction
commits and its score is lower than the baseline:

\[
C=\sum_t \operatorname{MSE}_t+\lambda_A|\mathcal A_t|.
\]

No source code is rewritten and no candidate can bypass the same verification
gate used by the baseline engine.

## Run

```bash
pip install -e .
jarvisx universe
jarvisx automaton --steps 20 --side 3
jarvisx automaton --steps 20 --side 3 --auto-optimize --json
```

Equivalent module execution:

```bash
python -m jarvisx automaton --steps 20
```

## API

```bash
jarvisx api
```

Endpoints:

- `GET /health`
- `POST /run`
- `GET /automaton`
- `POST /automaton/step`

Example step body:

```json
{
  "injections": [
    {"x": 0, "y": 0, "z": 0, "value": 1.0}
  ]
}
```

## Determinism

Given the same seed, mechanics, initial state, and ordered input transactions,
the engine produces the same state and SHA-256 journal chain. Tests cover exact
replay, sparse budgeting, coordinate wrapping, procedural reconstruction,
transaction commit, and rollback.
