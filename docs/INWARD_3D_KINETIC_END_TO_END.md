# Inward 3D: Operational and Kinetic End-to-End Specification

## Status

This document is the operational companion to [Dr Moagi Inward-Turned Self-Optimizing Runtime](DR_MOAGI_INWARD_SELF_OPTIMIZING_RUNTIME.md). It connects the live Inward Optimizer 3D Lab visualization to the executable Jarvis-X components without confusing displayed telemetry with measured runtime evidence.

Its typed 4D theoretical extension is defined in [Dr. Moagi 4D Quantum-Inspired Autoencoding Equation](DR_MOAGI_4D_QUANTUM_INSPIRED_AUTOENCODING.md).

The system is an original composite architecture built from established primitives: autoencoding, gradient descent, finite-difference directional derivatives, vector-Jacobian pullbacks, damped dynamics, spherical inversion, bounded search, deterministic replay, validation gates, atomic promotion, journaling, and rollback.

## 1. Capability boundary

| Layer | Role | Repository status |
|---|---|---|
| Inward 3D browser lab | Visualizes input, latent state, parameter tokens, attractor, telemetry, fixed-point progress, and a 3-bit directive | Deterministic demonstration; hosted separately |
| Hugging Face inward optimizer | Applies a bounded geometric control displacement to real PyTorch parameters | Implemented in hf_model/inward_self_optimizer.py |
| Runtime meta-optimizer | Searches a bounded 3D configuration lattice and promotes only verified candidates | Implemented in src/jarvisx/dr_moagi_meta_optimizer.py |
| Canonical inward runtime | Specifies shadow execution, semantic equivalence, canary deployment, commit, rollback, and meta-memory | Specification in docs/DR_MOAGI_INWARD_SELF_OPTIMIZING_RUNTIME.md |
| Three-bit directive display | Maps a displayed 3-bit state to one of eight labels | Visualization-only until connected to a versioned VM opcode table |
| Nanosecond horizon and browser validation latency | Communicates the intended execution scale | Simulated telemetry unless produced by a benchmark clock and recorded with provenance |

A visual transition is not automatically an optimizer update. An optimizer update is not automatically a committed runtime change. The complete system requires all three transitions:

\[
\text{visual state}
\longleftarrow
\text{measured execution state}
\longleftarrow
\text{verified committed state}.
\]

## 2. Complete state

At cycle \(t\), define:

\[
\Xi_t =
(X_t,\widehat X_t,Z_t,W_t,G_t,P_t,V_t,
\Omega_t,\Theta_t,\mathcal M_t,\mathcal T_t,\mathcal J_t,
E_t^{sys},P_t^{net}).
\]

| Symbol | Operational meaning |
|---|---|
| \(X_t\) | Current six-channel input state; the reference demo uses position and velocity |
| \(\widehat X_t\) | Decoder reconstruction |
| \(Z_t\) | Latent autoencoder representation |
| \(W_t\) | Flattened trainable base-model parameters |
| \(G_t=\nabla_W L_t\) | Gradient of the current objective |
| \(P_t\in[0,L]^3\) | Three-dimensional parameter-control tokens |
| \(V_t\) | Token velocity induced by the descent direction |
| \(\Omega_t\) | Residual or optimization memory |
| \(\Theta_t\) | Constraints, numerical bounds, policy, and objective weights |
| \(\mathcal M_t\) | Runtime mechanics configuration |
| \(\mathcal T_t\) | Measured telemetry |
| \(\mathcal J_t\) | Append-only optimization journal |
| \(E_t^{sys}\) | Measured or estimated stored system energy, in joules |
| \(P_t^{net}\) | Net physical power crossing the declared system boundary, in watts |

The browser may render a projection of this state. The authoritative state belongs to the executable optimizer or runtime, not to the canvas.

## 3. End-to-end operational clock

One accepted outer cycle is:

\[
\boxed{
X_t
\rightarrow
(Z_t,\widehat X_t)
\rightarrow
L_t
\rightarrow
G_t
\rightarrow
P_t
\rightarrow
(V_t,F_t)
\rightarrow
P_{t+1}^{trial}
\rightarrow
\Delta W_t^{trial}
\rightarrow
W_{t+1}^{trial}
\rightarrow
\operatorname{VERIFY}
\rightarrow
\operatorname{COMMIT/ROLLBACK}
\rightarrow
X_{t+1}
}
\]

### Phase 0 — initialize

1. Validate dimensions, finite values, bounds, seed, device, determinism class, and resource budget.
2. Load \(X_0\), model parameters \(W_0\), optimizer bounds, and any recoverable journal.
3. Separate base-model parameters from control parameters. The projector and hyperparameter anchors are not flattened into their own optimization target.
4. Record the initial state hash and configuration version.

### Phase 1 — sense and encode

The model observes the current state:

\[
Z_t=E_{\theta_t}(X_t),
\qquad
\widehat X_t=D_{\phi_t}(Z_t).
\]

For the reference demonstration:

\[
X_t\in\mathbb R^{B\times 6}
=
[x,y,z,v_x,v_y,v_z].
\]

The reconstruction objective is:

\[
L_t=\operatorname{MSE}(\widehat X_t,X_t).
\]

This loss supplies the ordinary optimization direction. The geometry does not replace the objective.

### Phase 2 — differentiate

Compute gradients only for the trainable base-model parameters:

\[
G_t=\nabla_{W_t}L_t.
\]

Unused parameters receive zero gradients. Parameter shapes are flattened deterministically, and their original ordering is retained for later scattering.

### Phase 3 — chunk and project into 3D

Partition \(W_t\) into chunks of \(C\) scalars. The implementation defaults to \(C=256\). For chunk \(q_i\):

\[
\widetilde q_i=
\frac{q_i}
{\sqrt{\operatorname{mean}(q_i^2)}+\epsilon}.
\]

A learned projector maps each normalized chunk into a three-dimensional control token:

\[
P_{t,i}
=
L\,
\sigma\left(
A_2\,\operatorname{GELU}(A_1\widetilde q_i+b_1)+b_2
\right),
\]

where \(P_{t,i}\in[0,L]^3\) and the default logical cube extent is \(L=1000\).

This is a control projection, not a lossless encoding of all chunk values into three scalars. Parameter information remains in \(W_t\).

### Phase 4 — measure kinetic velocity

Probe a small ordinary descent displacement:

\[
W_t^{probe}=W_t-\varepsilon G_t.
\]

Project the probe and estimate the directional derivative:

\[
V_t=
\frac{P(W_t-\varepsilon G_t)-P(W_t)}
{\varepsilon}.
\]

Therefore the displayed expression \(v=J_P(W)G\) requires a sign convention: the implementation estimates motion along \(-G\).

### Phase 5 — fold inward

Let the cube center be:

\[
c=(L/2,L/2,L/2).
\]

For token \(p\), define a bounded spherical inversion:

\[
R(p)=
\operatorname{clip}_{[0,L]^3}
\left[
c+
\frac{r^2(p-c)}
{\|p-c\|_2^2+\epsilon}
\right].
\]

The shell \(\|p-c\|_2=r\) is fixed by the inversion. With the reference defaults, \(r=500\).

Important: spherical inversion alone is not a proof of an attracting dynamical system. Attraction is created by the controlled force and damped integrator:

\[
F_t=\alpha(R(P_t)-P_t)+\eta V_t,
\]

\[
V_{t+1}
=
V_t+\Delta t(-\beta V_t+F_t),
\]

\[
P_{t+1}^{trial}
=
\operatorname{clip}_{[0,L]^3}
(P_t+\Delta t V_{t+1}).
\]

Here:

- \(\alpha\) controls inward geometric attraction;
- \(\beta\) damps kinetic motion;
- \(\eta\) couples the gradient-induced velocity into the force;
- \(\Delta t\) is the integration step.

### Phase 6 — pull the 3D displacement back to parameter space

Compute:

\[
\Delta P_t=P_{t+1}^{trial}-P_t.
\]

The parameter-space geometric direction is the vector-Jacobian product:

\[
H_t=J_P(W_t)^T\Delta P_t.
\]

Normalize it against the gradient norm:

\[
\widetilde H_t
=
H_t
\frac{\|G_t\|_2}
{\max(\|H_t\|_2,\epsilon)}.
\]

Blend ordinary descent with the geometric control term:

\[
\Delta W_t^{raw}
=
-\lambda G_t
+
\lambda\mu\widetilde H_t,
\]

where \(\lambda\) is the learning rate and \(\mu\) is the geometry mix.

### Phase 7 — bound the trial update

Let \(\rho\) be the maximum update-to-parameter ratio:

\[
\|\Delta W_t\|_2
\le
\rho\|W_t\|_2.
\]

The executable reference scales the complete update when this bound is exceeded:

\[
s_t=
\min\left(
1,
\frac{\rho\|W_t\|_2}
{\max(\|\Delta W_t^{raw}\|_2,\epsilon)}
\right),
\]

\[
\Delta W_t=s_t\Delta W_t^{raw},
\qquad
W_{t+1}^{trial}=W_t+\Delta W_t.
\]

The flattened result is scattered back using the retained parameter ordering.

### Phase 8 — reconstruct and evaluate

Run the updated model:

\[
Z_{t+1}^{trial}=E_{\theta_{t+1}^{trial}}(X_t),
\]

\[
\widehat X_{t+1}^{trial}
=
D_{\phi_{t+1}^{trial}}(Z_{t+1}^{trial}),
\]

\[
L_{t+1}^{trial}
=
\operatorname{MSE}(\widehat X_{t+1}^{trial},X_t).
\]

A production gate must compare the trial against the previous committed checkpoint. A lower training loss alone is insufficient.

### Phase 9 — verify

A candidate is admissible only if:

\[
V_{opt}=
V_{finite}
\land
V_{shape}
\land
V_{bounds}
\land
V_{loss}
\land
V_{semantics}
\land
V_{determinism}
\land
V_{resources}
\land
V_{recovery}
\land
V_{policy}.
\]

Recommended minimum gate:

1. all tensors and metrics are finite;
2. tensor shapes and parameter count are unchanged;
3. the update norm obeys its declared bound;
4. reconstruction regression remains within tolerance;
5. anchor fidelity remains within tolerance;
6. deterministic replay matches the declared equivalence class;
7. memory and latency remain within budget;
8. the prior checkpoint is restorable;
9. the journal record can be committed atomically.

### Phase 10 — commit or rollback

If every gate passes:

\[
W_{t+1}=\operatorname{COMMIT}(W_{t+1}^{trial}).
\]

Otherwise:

\[
W_{t+1}=W_t,
\qquad
\operatorname{ROLLBACK}(W_{t+1}^{trial}).
\]

Record the decision, metrics, hashes, configuration, seed, update scale, and rollback reference in \(\mathcal J_{t+1}\).

The current Hugging Face optimizer norm-bounds and scatters its update directly. Connecting it to the canonical shadow/canary gate is a remaining integration step.

### Phase 11 — turn the reconstructed state inward

The optional self-feedback operator updates the next input every \(k\) cycles:

\[
X_{t+1}
=
(1-\chi)X_t+\chi\widehat X_{t+1}.
\]

The reference defaults correspond to \(\chi=0.5\) every two cycles. When feedback is disabled:

\[
X_{t+1}=X_t.
\]

This changes the optimization target over time and must be reported separately from parameter improvement on a fixed dataset.

### Phase 12 — adapt control anchors

The executable reference conservatively changes \(\alpha,\beta,\eta,\lambda\) according to the observed loss trend and clamps every value to a declared interval.

If loss improves, attraction, forcing, and learning rate increase slightly while damping decreases slightly. If loss worsens, the direction reverses. This is bounded heuristic adaptation, not gradient-based hyperparameter optimization.

### Phase 13 — render telemetry

The visual instrument should consume an immutable cycle report:

~~~text
cycle
accepted
loss_before
loss_after
gradient_norm
parameter_flux
voxel_velocity
voxel_flux
update_norm
update_scale
alpha
beta
eta
learning_rate
self_consistency
directive_bits
directive_id
measured_duration_ns
telemetry_source
state_hash
energy_system_j
power_input_w
power_compute_w
power_memory_w
power_network_w
power_cooling_w
power_net_w
energy_source
~~~

Every timing value requires a source label:

- measured_cpu;
- measured_gpu;
- measured_browser;
- simulated;
- unavailable.

A simulated value must never be displayed as measured.

### Phase 14 — close the power–energy–time loop

Power and energy are the same physical account viewed through different time operators:

\[
\boxed{
P^{net}(t)=\frac{dE^{sys}(t)}{dt},
\qquad
E^{sys}(t)=E^{sys}(t_0)+\int_{t_0}^{t}P^{net}(\tau)\,d\tau.
}
\]

For a sampled Jarvis-X cycle of measured duration \(\Delta t_t\):

\[
E_{t+1}^{sys}
=
E_t^{sys}
+
\Delta t_t P_t^{net}
+
\varepsilon_t^{meter},
\]

where the balance residual \(\varepsilon_t^{meter}\) must remain within the declared metering tolerance. The net boundary flow is:

\[
P_t^{net}
=
P_t^{in}
-
P_t^{compute}
-
P_t^{memory}
-
P_t^{network}
-
P_t^{cooling}
-
P_t^{other}.
\]

The dimensional contract is strict:

\[
[\,P\,]=\mathrm{W}=\mathrm{J\,s^{-1}},
\qquad
[\,E\,]=\mathrm{J},
\qquad
[\,\Delta t\,]=\mathrm{s}.
\]

Loss, entropy, latent magnitude, gradient norm, and \(\Omega_t\) are algorithmic quantities. They are not physical joules or watts unless a calibrated measurement model explicitly maps them to those units.

The trial gate therefore adds:

\[
V_{energy}
=
(E_{t+1}^{trial}\le E_{budget})
\land
(P_{peak}^{trial}\le P_{max})
\land
(|\varepsilon_t^{meter}|\le\epsilon_{meter}).
\]

At energetic equilibrium:

\[
\frac{dE^{sys}}{dt}=0
\Longleftrightarrow
P^{net}=0.
\]

This means the inflows and outflows balance; it does not mean that every physical power flow has stopped. Accepted cycles integrate measured net power into the authoritative energy ledger. Rolled-back trials retain their metering record but cannot rewrite committed algorithmic state.

## 4. Three-bit directive projection

Three boolean fields produce:

\[
b_2,b_1,b_0\in\{0,1\},
\qquad
d=4b_2+2b_1+b_0\in\{0,\ldots,7\}.
\]

Operationally:

\[
(Z_1,Z_2,Z_3)
\rightarrow
(b_2,b_1,b_0)
\rightarrow
d
\rightarrow
\operatorname{DIRECTIVE}[d].
\]

The hosted demonstration displays changing directive names. Those labels are not yet canonical Jarvis-X opcodes. To become executable, the system needs:

1. a versioned eight-entry directive table;
2. threshold definitions for each bit;
3. an admissibility policy for every directive;
4. deterministic unit tests for all eight combinations;
5. an adapter from directive ID to existing VM or runtime actions;
6. a journal record linking the directive to its measured input state.

Until then, the bit engine is a visual projection of state.

## 5. Runtime-level inward turn

The PyTorch optimizer changes model parameters. The runtime meta-optimizer changes bounded mechanics:

\[
m_t=
(\text{compression},\text{adaptation},\text{dynamics}).
\]

It searches signed neighbouring displacements:

\[
\delta m=(d_x,d_y,d_z),
\qquad
d_x,d_y,d_z\in\{-1,0,1\},
\]

excluding \((0,0,0)\). The axes are:

| Axis | Mechanics |
|---|---|
| X | Block size, quantization, and pruning |
| Y | Distiller learning rate, residual memory, gain, pass depth, and latent budget |
| Z | Contraction, attenuation, and fixed-point depth |

The implemented end-to-end runtime sequence is:

1. snapshot the authoritative sparse state;
2. hash the authoritative state;
3. evaluate the baseline under deterministic workloads;
4. generate bounded neighbouring configurations;
5. probe candidates;
6. rank candidates by the multi-metric score;
7. confirm the best survivors over more cycles;
8. reject any candidate with a failed cycle;
9. require the minimum relative improvement;
10. reject excessive reconstruction or anchor-drift regression;
11. confirm the authoritative state hash has not changed;
12. create a new kernel with the promoted configuration;
13. restore adaptive residual state and logical cycle;
14. atomically replace the active kernel;
15. append the result to a hash-chain meta-journal.

The score implemented by the meta-optimizer is:

\[
S=
6L_{rec}
+4E_{distill}^2
+2E_{fixed}^2
+4E_{anchor}
+0.010C_{transport}
+0.20C_{compute}
+0.25V_{phase}
+P_{reject}.
\]

Promotion requires:

\[
\frac{S_{base}-S_{candidate}}{S_{base}}
\ge 0.01,
\]

no rejected workload, and no reconstruction or anchor-drift regression beyond the declared tolerance, which defaults to five percent.

## 6. Fixed point and lock

The visual lock condition is not merely a high displayed consistency percentage. A practical fixed point requires:

\[
\|X_{t+1}-X_t\|_2\le\epsilon_X,
\]

\[
\|W_{t+1}-W_t\|_2\le\epsilon_W,
\]

\[
|L_{t+1}-L_t|\le\epsilon_L,
\]

\[
\|\mathcal T_{t+1}-\mathcal T_t\|_W\le\epsilon_T,
\]

for \(N\) consecutive cycles, with no violated gate.

Then:

\[
\operatorname{LOCK}_t=
C_X\land C_W\land C_L\land C_T\land V_{opt}.
\]

The lock is provisional, workload-specific, and reversible. A distribution shift, new input, policy change, or resource change unlocks the system.

## 7. Browser event sequence

~~~text
User presses STEP or RUN
        ↓
scheduler requests one bounded cycle
        ↓
cycle reads the last committed snapshot
        ↓
encoder produces latent state and reconstruction
        ↓
loss and gradient are calculated
        ↓
parameters are projected into 3D tokens
        ↓
velocity, inward force, and damped motion are integrated
        ↓
3D displacement is pulled back into parameter space
        ↓
trial update is bounded
        ↓
trial is verified
        ↓
commit or rollback is recorded
        ↓
optional self-feedback creates the next input
        ↓
bit/directive projection is derived
        ↓
immutable telemetry report is emitted
        ↓
canvas and readouts render that report
~~~

The renderer must not manufacture authoritative state. Pause stops scheduling new cycles; it does not interrupt an atomic cycle. Reset restores a versioned initial snapshot, clears transient trails, and retains or explicitly rotates the audit journal.

## 8. Repository implementation map

| Operational phase | Executable location |
|---|---|
| Parameter target selection | InwardSelfOptimizer._target_named_parameters |
| Flatten and zero unused gradients | InwardSelfOptimizer._flatten |
| Chunking | InwardSelfOptimizer._chunk |
| Three-dimensional projection | InwardSelfOptimizer.param_to_voxel |
| Spherical inward fold | inward_fold |
| Velocity, force, integration, and pullback | InwardSelfOptimizer.kinetic_param_step |
| Norm-bounded parameter update | InwardSelfOptimizer.kinetic_param_step |
| Scatter to model | InwardSelfOptimizer._scatter |
| Control-anchor adaptation | InwardSelfOptimizer._adapt_hyperparameters |
| Reconstruction loop and self-feedback | InwardSelfOptimizer.forward_self_optimising |
| Bounded mechanics lattice | DrMoagi3DMetaOptimizer.candidate_config |
| Deterministic candidate replay | DrMoagi3DMetaOptimizer._evaluate |
| Promotion gate | DrMoagi3DMetaOptimizer._promotion_gate |
| Authoritative kernel swap | SelfOptimizing3DSystem.turn_inward |
| Hash-chain meta-journal | SelfOptimizing3DSystem.meta_journal |
| Shadow, canary, commit, and rollback contract | DR_MOAGI_INWARD_SELF_OPTIMIZING_RUNTIME.md |

## 9. Executable pseudocode

~~~text
function inward_cycle(committed_state, committed_model, bounds):
    checkpoint = snapshot(committed_state, committed_model)
    x = checkpoint.input

    z, x_hat = model.forward(x)
    loss_before = mse(x_hat, x)
    gradients = grad(loss_before, base_model_parameters)

    weights = flatten(base_model_parameters)
    voxels = project_chunks_to_3d(weights)
    probe = project_chunks_to_3d(weights - epsilon * gradients)
    velocity = (probe - voxels) / epsilon

    folded = spherical_inversion(voxels)
    force = alpha * (folded - voxels) + eta * velocity
    velocity_trial = velocity + dt * (-beta * velocity + force)
    voxels_trial = clamp(voxels + dt * velocity_trial)
    voxel_delta = voxels_trial - voxels

    pullback = transpose_jacobian(projector, weights) * voxel_delta
    pullback = normalize_to_gradient_norm(pullback, gradients)
    update = -lr * gradients + lr * geometry_mix * pullback
    update = norm_bound(update, max_ratio * norm(weights))
    model_trial = scatter(weights + update)

    x_hat_trial = model_trial.reconstruct(x)
    loss_after = mse(x_hat_trial, x)
    verdict = verify(checkpoint, model_trial, loss_before, loss_after, bounds)

    if verdict.accepted:
        committed_model = atomic_commit(model_trial)
    else:
        committed_model = rollback(checkpoint.model)

    x_next = feedback(x, x_hat_trial) if feedback_enabled else x
    report = journal_and_emit(checkpoint, verdict, measured_metrics)
    return x_next, committed_model, report
~~~

## 10. Required integration tests

1. The same seed and input produce identical cycle reports within the declared determinism class.
2. The projector always returns finite coordinates in \([0,L]^3\).
3. Points on the fold shell remain fixed within tolerance.
4. The center singularity remains finite.
5. Parameter order and count are invariant through flatten and scatter.
6. The update norm never exceeds the declared ratio.
7. A NaN, infinite loss, shape change, or failed reconstruction gate causes rollback.
8. Self-feedback disabled leaves the input target unchanged.
9. Self-feedback enabled applies exactly the declared blend and cadence.
10. Pause prevents new cycles after the current atomic cycle.
11. Reset reproduces the versioned initial snapshot.
12. All eight bit combinations map deterministically to a versioned directive table.
13. Displayed timing equals recorded benchmark timing or is labelled simulated.
14. Meta-optimization cannot mutate the authoritative state during candidate evaluation.
15. A promoted configuration is journaled and recoverable.
16. A failed candidate leaves both world state and mechanics state unchanged.
17. The discrete energy update satisfies the metered balance within tolerance and rejects unit or budget violations.

## 11. Operational identity

The end-to-end mechanism is:

\[
\boxed{
\text{Observe}
\rightarrow
\text{Encode}
\rightarrow
\text{Differentiate}
\rightarrow
\text{Project into 3D}
\rightarrow
\text{Apply inward kinetics}
\rightarrow
\text{Pull back}
\rightarrow
\text{Bound}
\rightarrow
\text{Verify}
\rightarrow
\text{Commit or rollback}
\rightarrow
\text{Account for power and energy}
\rightarrow
\text{Describe}
\rightarrow
\text{Repeat}
}
\]

In Dr Moagi notation:

\[
\boxed{
\Xi_{t+1}
=
\operatorname{COMMIT}_{\Theta_t}
\left[
\operatorname{VERIFY}_{\Theta_t}
\left(
\Pi_{\Lambda}
\left[
\Xi_t
-\lambda\nabla L_t
+\lambda\mu J_P(W_t)^T\Delta P_t
+\Omega_t
\right]
\right)
\right].
}
\]

This is the defensible operational claim: a bounded residual-driven optimizer uses a three-dimensional geometric control field to influence ordinary parameter updates, while a separate verified meta-loop searches and promotes runtime configurations. The visual lab makes the mechanism observable; it does not replace measurement, validation, or commit authority.
