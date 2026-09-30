# Inward Multiparallel Multimodal 3D Swarm Runtime

**Status:** Experimental research/runtime layer  
**Implementation:** `src/jarvisx/inward_multimodal_swarm3d.py`  
**Tests:** `tests/test_inward_multimodal_swarm3d.py`  
**Related:** `docs/DR_MOAGI_3D_GEOMETRIC_DIFFUSION_RUNTIME.md`, `docs/INWARD_3D_KINETIC_END_TO_END.md`, `docs/research/DR_MOAGI_3D_ELECTROMAGNETIC_OPERATION.md`

## 1. Purpose

This runtime closes the 3D auto-encoding/decoding loop inward across multiple media modalities while keeping the existing Jarvis-X boundary between the canonical VM and research layers.

The system is intentionally split into three contracts:

1. **Operational:** modality-specific codecs map real media representations into and out of a shared bounded 3D control chart.
2. **Kinetic:** many modality-tagged particles evolve in parallel under a Riemannian task force, inward decode/re-encode contraction, graph coupling, and bounded memory forcing.
3. **Electromagnetic analogue:** the inward and graph-relaxation terms are mapped to a bounded RC-network equation. This is an algorithm-to-circuit correspondence, not a Maxwell field solver.

The 3D coordinates are virtual semantic/control coordinates. They are not asserted to be physical spacetime coordinates, and three scalars are not treated as a lossless representation of arbitrary text, image, audio, video, geometry or code. Production codecs may retain high-dimensional state outside the 3D control chart.

---

## 2. Operational state

For modalities

```text
M = {text, image, audio, video, geometry, code, data}
```

each registered codec satisfies the narrow interface

```text
encode_m(X_m) -> z_m in R^3
decode_m(z_m) -> X_hat_m
```

All encoded items become one population

```text
Z = [z_1, z_2, ..., z_N] in R^(N x 3).
```

Each `Particle3D` additionally carries

```text
(position, shared_feature, modality, confidence, time_coordinate, source_id).
```

The shared feature vector is used to compute dynamic cross-modal interaction weights. A caller may supply a `feature_encoder` that projects heterogeneous modalities into one common feature width.

The complete operational loop is

```text
multimodal inputs
    -> parallel modality codecs
    -> N x 3 particle population
    -> dynamic cross-modal graph
    -> inward/kinetic relaxation
    -> consensus control state
    -> one or more modality decoders
    -> generated/projected surfaces
    -> re-encode through the same codecs
    -> inward correction
```

The decode/re-encode operator is

```text
Phi_m(z) = E_m(D_m(z)).
```

A self-consistent local state satisfies

```text
Phi_m(z*) ~= z*.
```

---

## 3. Riemannian task geometry

Let `phi(z)` be a caller-defined local semantic/task potential. The runtime uses the rank-one metric

```text
G(z) = I + alpha grad(phi) grad(phi)^T.
```

Its inverse is exact by Sherman-Morrison:

```text
G^-1 = I - alpha grad(phi) grad(phi)^T
             / (1 + alpha ||grad(phi)||^2).
```

A Euclidean task gradient is therefore transformed into the local Riemannian gradient

```text
grad_g J = G^-1 grad J.
```

The task force is

```text
F_task = -k_task grad_g J.
```

This makes motion along the local potential gradient more resistant as `alpha` and `||grad(phi)||` increase.

The implementation exposes:

- `riemannian_metric_inverse(...)`
- `local_riemannian_gradient(...)`

No global geodesic solver is claimed. Consensus and inter-particle displacement use the local/tangent-chart approximation.

---

## 4. Dynamic cross-modal swarm

For particles `i` and `j`, the runtime computes

```text
score_ij = k_f cosine(h_i, h_j) - k_d ||z_i - z_j||^2
```

and row-normalizes the scores with softmax:

```text
A_ij = softmax_j(score_ij).
```

`A` is therefore a dynamic directed adjacency matrix that changes as the particles and shared features evolve.

The local chart form of the graph force is

```text
F_swarm_i = k_swarm sum_j A_ij (z_j - z_i).
```

For a symmetric/static graph this has the familiar graph-Laplacian form

```text
dZ/dt = -gamma L Z,
L = D - A.
```

The implementation does not claim that attention weights are a physical force. They are computational interaction coefficients.

---

## 5. Inward contraction

For particle `i`, the runtime computes

```text
z_phi_i = Phi_mi(z_i) = E_mi(D_mi(z_i))
```

and applies

```text
F_inward_i = lambda (z_phi_i - z_i).
```

The cycle energy is

```text
E_cycle = mean_i ||Phi_mi(z_i) - z_i||^2.
```

If the codec cycle is contractive in the region of interest, repeated application drives the state toward a local fixed point. The runtime measures this through `fixed_point_error`; it does not assume Banach contraction unless a codec actually supplies that property.

---

## 6. Memory forcing

A bounded caller-provided memory target may be keyed by `source_id`:

```text
F_memory_i = k_memory (z_memory_i - z_i).
```

Memory is optional. The reference runtime neither performs RAG retrieval nor owns a vector database; retrieval systems can inject their selected evidence through encoded particles and/or memory targets.

---

## 7. Complete kinetic equation

The implemented first-order local-chart equation is

```text
dz_i/dt =
    - k_task G_i^-1 grad J_i
    + lambda (Phi_i(z_i) - z_i)
    + gamma sum_j A_ij (z_j - z_i)
    + rho (z_memory_i - z_i).
```

The explicit-Euler implementation is

```text
z_i[t+1] = clamp(
    z_i[t] + dt * F_i,
    -position_bound,
    +position_bound
).
```

Every positional update is additionally capped by `max_position_step`.

Shared features undergo a bounded-rate neighbor mixing step:

```text
h_i[t+1] = h_i[t]
           + dt * k_h * (sum_j A_ij h_j[t] - h_i[t]).
```

The runtime terminates when both movement and enabled inward fixed-point error fall below tolerance, or when `max_steps` is reached.

---

## 8. Measured state

`Swarm3DMetrics` reports:

```text
step
particle_count
energy
task_energy
cycle_energy
coupling_energy
fixed_point_error
consensus_error
max_displacement
```

The diagnostic energy is

```text
E = k_task E_task
    + lambda E_cycle
    + 1/2 gamma E_coupling.
```

It is a runtime diagnostic, not a universal physical Hamiltonian.

---

## 9. Multiparallel generation surface

After relaxation, `decode_consensus(...)` computes a confidence-weighted local-chart consensus

```text
z_bar = sum_i c_i z_i / sum_i c_i
```

and sends the same control state through any requested modality heads:

```text
text_hat     = D_text(z_bar)
image_hat    = D_image(z_bar)
audio_hat    = D_audio(z_bar)
video_hat    = D_video(z_bar)
geometry_hat = D_geometry(z_bar)
code_hat     = D_code(z_bar).
```

This is the orchestration contract. The concrete quality and dimensionality of generated media remain the responsibility of the registered codecs/generator heads.

---

## 10. Electromagnetic/circuit analogue

The runtime exposes a separate RC-network analogue with node voltage vector `V_i`, decode/re-encode target voltage `V_phi_i`, adjacency `A_ij`, capacitance `C`, feedback conductance `g_phi`, and coupling conductance `g_c`:

```text
C dV_i/dt =
    g_phi (V_phi_i - V_i)
    + g_c sum_j A_ij (V_j - V_i)
    + I_ext_i.
```

This corresponds structurally to

```text
dz_i/dt =
    lambda (Phi_i(z_i) - z_i)
    + gamma sum_j A_ij (z_j - z_i)
    + F_ext_i.
```

The mapping is

```text
algorithmic state z_i        <-> encoded node voltage V_i
inward gain lambda           <-> g_phi / C
graph coupling gamma A_ij    <-> g_c A_ij / C
external/memory force        <-> injected current I_ext / C
```

`electrical_rhs(...)` evaluates the continuous RC right-hand side and `electrical_step(...)` advances it by one bounded Euler step.

This layer deliberately stops at circuit dynamics. The repository's electromagnetic research specification remains the source for the lower physical boundary: voltages, charges, currents and conductances must ultimately be realized by electromagnetic fields satisfying Maxwell's equations. The software helper itself does not solve those equations.

---

## 11. Safety and numerical bounds

`Swarm3DConfig` makes the research runtime finite by construction:

- finite input validation;
- bounded position domain;
- capped per-step displacement;
- bounded particle count;
- bounded iteration count;
- common shared feature width;
- non-negative gains;
- explicit convergence tolerance.

`ElectricalAnalogueConfig` similarly bounds capacitance, conductances, integration step and voltage interval.

These constraints are part of the executable contract rather than visualization metadata.

---

## 12. Example adapter

```python
from jarvisx.inward_multimodal_swarm3d import (
    InwardMultimodalSwarm3D,
    Modality,
)


class TextControlCodec:
    def encode(self, value):
        # Replace with a trained/text embedding -> 3D control projection.
        return (0.2, -0.1, 0.4)

    def decode(self, position):
        # Replace with a conditioned text generator/projection head.
        return {"latent_control": position}


runtime = InwardMultimodalSwarm3D({Modality.TEXT: TextControlCodec()})
particles = runtime.encode_modalities({Modality.TEXT: ("query",)})
result = runtime.relax(particles)
output = runtime.decode_consensus(result.state.particles)
```

The adapter intentionally makes the system extensible without pulling optional model frameworks into the canonical package dependency set.

---

## 13. Promotion boundary

This feature remains a research layer under ADR-001. Promotion into a canonical execution path requires, at minimum:

1. concrete modality codecs with reproducible fixtures;
2. benchmarked convergence and stability envelopes;
3. demonstrated task-quality benefit over simpler baselines;
4. explicit RAG integration contract if retrieval is enabled;
5. hardware measurements before electromagnetic acceleration claims;
6. canonical transaction/provenance integration if the state is allowed to affect authoritative VM behavior.

Until then, the runtime is an executable mathematical reference for the inward multiparallel multimodal swarm architecture.
