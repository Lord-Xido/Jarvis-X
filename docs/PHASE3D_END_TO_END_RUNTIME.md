# Dr Moagi Phase3D End-to-End Runtime

## Status

This specification operationalizes the current DM-vOmegaXi+ iteration as one measurable 3D execution loop:

```text
3D phase-space state
-> relativistic momentum update
-> implicit shell representation
-> differentiable geometry learning
-> measured runtime benchmark
-> bounded device/runtime search
-> canonical kinetic transaction
-> semantic/resource/performance validation
-> commit or rollback
-> hash-chained receipt
-> next cycle
```

The executable reference is:

```text
examples/dr_moagi_phase3d_runtime.py
```

It extends, rather than replaces, the self-referential PyTorch/Inductor runtime defined in `SELF_REFERENTIAL_TRITON_RUNTIME.md` and reuses the backend-neutral transaction protocol in `src/jarvisx/kinetic_runtime.py`.

## 1. Separation of authorities

The runtime deliberately keeps three optimization domains separate.

### Phase dynamics

The authoritative physical state is

\[
K_t=(X_t,P_t,M),
\]

where `X` is an `N x 3` position tensor, `P` is relativistic momentum, and `M` is the per-node mass tensor.

The neural model does not directly write to `X` or `P`.

### Neural representation

The implicit field parameters are

\[
\Theta_t.
\]

They are updated only through a differentiable signed-distance objective.

### Runtime mechanics

The execution configuration is

\[
C_t=(\text{chunk size},\text{compile mode},\ldots).
\]

It is updated only from measured benchmark evidence and only after a canonical Jarvis-X transaction commits the candidate.

Therefore

\[
\boxed{
\text{phase evolution}\neq\text{gradient learning}\neq\text{runtime tuning}
}
\]

while all three participate in the same outer recurrence.

## 2. True 3D toroidal initialization

The prior single-index construction coupled the two toroidal angles and sampled a one-dimensional closed curve. Phase3D instead creates independent angular grids

\[
\theta_i\in[0,2\pi),
\qquad
\phi_j\in[0,2\pi)
\]

and samples

\[
\rho=R+a\cos\phi,
\]

\[
x=\rho\cos\theta,
\qquad y=\rho\sin\theta,
\qquad z=a\sin\phi.
\]

The resulting state samples a genuine two-parameter torus surface before being flattened into an `N x 3` tensor for parallel execution.

## 3. Dimensionally explicit radial field

The conservative force is

\[
F_r(r)=\frac{A}{r^2}-kr,
\]

where

\[
[A]=\mathrm{N\,m^2},
\qquad
[k]=\mathrm{N/m}.
\]

The potential is

\[
U(r)=\frac{A}{r}+\frac12kr^2.
\]

The equilibrium shell satisfies

\[
F_r(r_*)=0,
\]

therefore

\[
\boxed{r_*=(A/k)^{1/3}}.
\]

With the reference values

\[
A=15\times10^6\ \mathrm{N\,m^2},
\qquad
k=5\times10^6\ \mathrm{N/m},
\]

we obtain

\[
r_*=\sqrt[3]{3}\approx1.44225\ \mathrm{m}.
\]

Optional damping is represented as

\[
\frac{dP}{dt}=F(X)-\zeta P.
\]

When `zeta = 0`, the shell is a stable conservative equilibrium and phase points oscillate around it. When `zeta > 0`, the shell becomes dissipatively attracting.

## 4. Relativistic momentum integration

The runtime does not use Newtonian `F = ma` followed by an artificial light-speed clamp.

Instead it evolves momentum:

\[
P_{t+1}=P_t+F(X_t)\Delta t.
\]

The Lorentz factor is recovered from momentum:

\[
\gamma_{t+1}
=
\sqrt{1+\frac{\|P_{t+1}\|^2}{m^2c^2}}.
\]

Velocity is then

\[
V_{t+1}
=
\frac{P_{t+1}}{\gamma_{t+1}m},
\]

and position advances semi-implicitly:

\[
X_{t+1}=X_t+V_{t+1}\Delta t.
\]

This construction enforces

\[
\|V\|<c
\]

without manual clipping.

## 5. Energy telemetry

The stable relativistic kinetic-energy expression is

\[
K
=
\frac{p^2}{m(\gamma+1)},
\]

summed over nodes.

The mechanical energy is

\[
E_{mech}=K+U.
\]

The runtime records

\[
\delta_E
=
\frac{E_t-E_0}{|E_0|}.
\]

For `damping = 0`, this is an integrator-conservation diagnostic. For non-zero damping, a declining mechanical energy is expected and must not be misclassified as numerical failure.

## 6. Neural shell representation

The current phase coordinates are passed to the implicit field as

\[
X_t\in\mathbb R^{1\times N\times3}.
\]

The supervision target is the analytic equilibrium-shell signed distance

\[
d^*(x)=\|x\|_2-r_*.
\]

The field receives normalized execution telemetry as context, but its weights are updated only by the differentiable geometry loss:

\[
\Theta_{t+1}
=
\Theta_t-\eta\nabla_\Theta L_{geometry}.
\]

Measured wall-clock latency and throughput remain outside autograd.

## 7. Measured runtime candidate search

For the current phase coordinates and one frozen telemetry snapshot, the runtime evaluates a bounded neighborhood of execution configurations.

Each candidate is:

1. selected or compiled;
2. warmed up outside the timed interval;
3. measured using synchronized execution;
4. compared against the eager semantic reference;
5. returned as benchmark evidence without changing the authoritative runtime configuration.

Search and authority are separate. A fast candidate is not yet a committed candidate.

## 8. Canonical kinetic transaction gate

Every scheduled runtime tuning attempt is passed to `KineticTransactionEngine` using the repository's canonical sequence:

```text
snapshot
-> observe
-> encode
-> propose
-> shadow
-> verify
-> commit | rollback
-> journal
-> reenter
```

The authoritative state supplied to the transaction is the current `RuntimeConfig`.

The proposal contains the measured incumbent, the best bounded candidate, and the number of evaluated configurations.

Three validators are mandatory:

### Semantic validator

\[
V_{semantic}
=
(E_{semantic}\le\epsilon_s).
\]

### Memory validator

\[
V_{memory}
=
(M_{peak}\le M_{budget}).
\]

### Performance validator

For a valid incumbent,

\[
V_{performance}
=
q(C^{best})
\ge
(1+\rho)q(C_t),
\]

and the best configuration must differ from the incumbent.

If the incumbent itself is unavailable, a valid replacement may be committed as a recovery transition.

The transaction commits only when

\[
\boxed{
V_{semantic}\land V_{memory}\land V_{performance}
}
\]

is true.

Otherwise the original runtime snapshot is restored.

## 9. Hash-chained runtime receipts

Every tuning attempt emits a `KineticReceipt` containing:

```text
transaction_id
parent_state_hash
candidate_hash
resulting_state_hash
decision
stages
validator results
shadow telemetry
previous_receipt_hash
receipt_hash
```

Thus runtime self-optimization is auditable as a state-transition chain:

\[
R_{t-1}
\rightarrow
R_t
\rightarrow
R_{t+1}.
\]

A rejected candidate still receives a receipt; it simply records a rollback rather than a commit.

## 10. Measurement contract

The runtime reports two independent measured throughput classes.

### Phase throughput

\[
q_{phase}
=
\frac{N}{\Delta t_{wall}}
\]

in node updates per second.

### Neural/runtime throughput

\[
q_{field}
=
\frac{N_{queries}}{\Delta t_{wall}}
\]

in implicit-field queries per second.

No artificial `10^12` or other symbolic multiplier is applied to either measurement.

The console reports:

```text
phase node updates/s
mean radius
shell RMSE
maximum velocity
relative mechanical-energy drift
geometry loss
geometry RMSE
implicit-field queries/s
implicit-field latency
peak allocated memory
effective compiler/runtime mode
chunk size
transaction decision
receipt hash prefix
```

## 11. Full recurrence

One complete cycle is

\[
\boxed{
\begin{aligned}
F_t &= -\nabla_X U(X_t)-\zeta P_t,\\
P_{t+1} &= P_t+F_t\Delta t,\\
X_{t+1} &= X_t+\frac{P_{t+1}}{\gamma_{t+1}M}\Delta t,\\
Z_t &= E_{\Theta_t}(X_{t+1},\mathcal T_t),\\
\widehat d_t &= D_{\Theta_t}(Z_t),\\
\Theta_{t+1} &= \Theta_t-\eta\nabla_\Theta L_t,\\
\mathcal T_{t+1} &= \operatorname{PROFILE}(\Theta_{t+1},C_t),\\
C^{trial}_{t+1} &= \operatorname{SEARCH}(C_t,\mathcal T_{t+1}),\\
(C_{t+1},R_{t+1}) &= \operatorname{KINETIC\_TRANSACTION}(C_t,C^{trial}_{t+1}).
\end{aligned}
}
\]

The combined runtime state is

\[
\mathcal S_t=(K_t,\Theta_t,C_t,\mathcal T_t,R_t)
\]

with recurrence

\[
\boxed{\mathcal S_{t+1}=\mathcal M(\mathcal S_t)}.
\]

## 12. Validation workflow

`.github/workflows/phase3d-runtime.yml` installs the optional PyTorch dependency and validates:

1. source compilation;
2. equilibrium-radius arithmetic;
3. `N x 3` phase tensor shapes;
4. finite position and momentum evolution;
5. subluminal velocity;
6. finite energy telemetry;
7. canonical rollback receipt when no improvement exists;
8. canonical commit receipt for deterministic valid speedup evidence;
9. conservative end-to-end CPU execution;
10. damped end-to-end CPU execution.

This is a runtime smoke/contract suite. It is not a GPU performance benchmark.

## 13. Execution

Install optional PyTorch support:

```bash
pip install -e '.[torch]'
```

CPU reference:

```bash
python examples/dr_moagi_phase3d_runtime.py \
  --device cpu \
  --cycles 5 \
  --nodes 16384 \
  --compile-mode eager
```

CUDA/Inductor reference:

```bash
python examples/dr_moagi_phase3d_runtime.py \
  --device cuda \
  --cycles 20 \
  --nodes 65536 \
  --compile-mode default \
  --tune-every 2 \
  --peak-memory-mb 8192
```

Dissipative shell experiment:

```bash
python examples/dr_moagi_phase3d_runtime.py \
  --device cuda \
  --cycles 200 \
  --damping 50
```

## 14. Claim boundary

This runtime is an executable research architecture, not evidence of new physical law, quantum mechanics, or state-of-the-art hardware performance.

`torch.compile`/Inductor can generate Triton-backed CUDA kernels where supported; the reference does not contain a hand-written Triton kernel.

External performance claims require matched hardware, workloads, baselines, statistical treatment, and reproducible benchmark provenance.
