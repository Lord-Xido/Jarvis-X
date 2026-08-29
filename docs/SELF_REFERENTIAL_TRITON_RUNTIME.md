# Self-Referential 3D Implicit Field Runtime

## Status

This document defines the PyTorch/Inductor integration path for a self-referential 3D implicit field inside Jarvis-X.

It is intentionally compatible with the existing bounded meta-optimization architecture in `src/jarvisx/dr_moagi_meta_optimizer.py` and the kinetic commit/rollback adapter in `src/jarvisx/dr_moagi_kinetic_system.py`.

The key design rule is that **neural optimization and runtime optimization are different optimization spaces**.

Measured wall-clock throughput is not differentiable with respect to model weights. It must therefore be optimized by an outer measurement-driven controller rather than inserted into a PyTorch loss as if autograd could differentiate through `time.perf_counter()`.

---

## 1. Canonical state

At cycle `t`, define

\[
S_t = (X_t, Z_t, \theta_t, C_t, M_t),
\]

where:

- `X_t` is the 3D query field;
- `Z_t` is the learned meta-latent representation;
- `theta_t` is the neural parameter state;
- `C_t` is the runtime configuration;
- `M_t` is measured execution telemetry.

The operational recurrence is

\[
S_{t+1}=\mathcal M(S_t).
\]

The fixed-point target is

\[
S^*=\mathcal M(S^*)
\]

subject to explicit quality and resource constraints.

---

## 2. Inner differentiable loop

The implicit field receives 3D coordinates and normalized execution telemetry.

Spatial projection:

\[
h_s = \operatorname{SiLU}(W_s x+b_s).
\]

Execution-state conditioning:

\[
\widetilde M_t =
\begin{bmatrix}
\log(1+q_t/q_{target})\\
\log(1+\tau_t)
\end{bmatrix}.
\]

Meta encoding:

\[
Z_t=E_\phi([h_s,\widetilde M_t]).
\]

Implicit field evaluation:

\[
\widehat d_t=F_\theta([Z_t,h_s]).
\]

The reference example supervises a non-trivial sphere signed-distance field:

\[
d^*(x)=\|x\|_2-r.
\]

Neural parameters are updated only through a differentiable geometry objective:

\[
\theta_{t+1}=\theta_t-\eta\nabla_\theta L_{geometry}.
\]

This avoids the collapse induced by training every point toward zero SDF.

---

## 3. Outer runtime loop

Let the bounded runtime configuration be

\[
C_t=(\text{chunk size},\text{compile mode},\ldots).
\]

A candidate runtime is measured using the same coordinates and model state as the incumbent.

For each candidate:

1. compile or select the runner;
2. execute warm-up passes outside the benchmark;
3. synchronize CUDA before timing;
4. measure one or more steady-state passes;
5. synchronize CUDA after timing;
6. compute query throughput, latency, and peak allocated memory;
7. compare candidate output against the eager semantic reference;
8. reject candidates outside the declared numerical tolerance.

The outer objective is empirical:

\[
C_{t+1}
=
\arg\max_{C\in\mathcal N(C_t)}q(C)
\]

subject to

\[
\|F_{\theta,C}(X)-F_{\theta,ref}(X)\|_\infty\le\epsilon.
\]

The reference implementation uses bounded neighboring chunk sizes and PyTorch compile modes.

No wall-clock value is added to the differentiable loss.

---

## 4. Why the original throughput penalty was not operational

A construction such as

```python
current_lps = processed_operations / elapsed_time
throughput_penalty = torch.exp(
    torch.tensor((target_lps - current_lps) / target_lps)
)
total_loss = sdf_loss + 0.01 * throughput_penalty
```

does not optimize throughput by gradient descent.

`current_lps` was produced by a host timer outside the autograd graph. Therefore

\[
\frac{\partial L_{throughput}}{\partial\theta}=0.
\]

The scalar can change the printed loss value, but it cannot tell AdamW how to modify model parameters to make the kernel faster.

The corrected architecture makes throughput a **measured control signal and runtime-selection criterion**, not a fake differentiable term.

---

## 5. Measurement contract

Every reported performance number must carry its provenance.

For CUDA benchmarking:

```text
warmup
cuda synchronize
start timer
execute fixed workload
cuda synchronize
stop timer
```

The first compiled execution is not a steady-state benchmark because graph capture, code generation, compilation, and autotuning may occur.

The performance unit in this runtime is **queries per second** because each `(x,y,z)` point is one implicit-field query. It must not be labeled lines per second unless a higher-level line-processing contract explicitly maps lines to queries.

---

## 6. PyTorch, Inductor, and Triton boundary

`torch.compile` can route eligible PyTorch graphs through TorchDynamo and Inductor. On supported CUDA configurations, Inductor can generate Triton kernels.

Accordingly:

- the reference runtime is accurately described as a **PyTorch/Inductor implicit-field accelerator**;
- Triton may be present in the generated backend path;
- it is **not** a hand-written Triton kernel unless explicit `triton.jit` kernels are added and versioned in the repository.

This distinction must remain explicit in benchmark and architecture claims.

---

## 7. Integration with the existing Jarvis-X inward optimizer

Jarvis-X already has a bounded three-axis runtime meta-optimizer:

```text
X: compression geometry
Y: adaptive dynamics
Z: spatial dynamics
```

The PyTorch implicit-field runtime adds a hardware execution subspace rather than replacing that optimizer.

The combined system can be represented as

\[
C_t=(C_t^{system},C_t^{device}),
\]

where:

- `C_system` contains the existing Dr Moagi OS mechanics;
- `C_device` contains execution configuration such as chunking and compile strategy.

A future production integration should pass device candidates through the same kinetic transaction law already used by `CanonicalKineticDrMoagiSystem`:

```text
observe
-> encode
-> propose candidate
-> shadow benchmark
-> semantic/resource validators
-> commit or rollback
-> append receipt
```

The model weights and production runtime configuration therefore remain independently recoverable.

---

## 8. Reference implementation

The executable reference is:

```text
examples/self_referential_triton_engine.py
```

Install the optional dependency:

```bash
pip install -e '.[torch]'
```

Run on CPU:

```bash
python examples/self_referential_triton_engine.py --device cpu --compile-mode eager
```

Run on a CUDA host:

```bash
python examples/self_referential_triton_engine.py --device cuda --compile-mode default
```

The example deliberately uses a bounded autotuning search. It does not rewrite source code, mutate arbitrary host configuration, or claim external state-of-the-art performance without matched benchmark evidence.

---

## 9. Fixed-point acceptance criterion

A runtime can be treated as operationally converged only when both neural and runtime constraints are simultaneously satisfied:

\[
L_{geometry}<\epsilon_g,
\qquad
q\ge q_{target},
\qquad
\tau\le\tau_{target},
\qquad
M_{peak}\le M_{budget},
\qquad
E_{semantic}\le\epsilon_s.
\]

This produces the complete inward loop:

\[
(X_t,M_t,C_t,\theta_t)
\rightarrow
\widehat d_t
\rightarrow
L_{geometry}
\rightarrow
\theta_{t+1}
\rightarrow
\operatorname{BENCHMARK}
\rightarrow
M_{t+1}
\rightarrow
C_{t+1}
\rightarrow
(X_{t+1},M_{t+1},C_{t+1},\theta_{t+1}).
\]

That is the operational boundary between **self-referential conditioning** and **actual self-optimization**.
