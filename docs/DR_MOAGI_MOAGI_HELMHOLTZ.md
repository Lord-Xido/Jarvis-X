# Dr Moagi Moagi-Helmholtz Unified Generative Functional

## Status

Canonical Layer 4/5 orchestration specification for multimodal-conditioned 3D generation, geometric refinement, rendering, archival coding, inverse inference, verification and transactional re-entry.

This specification extends `DR_MOAGI_FIELD_RUNTIME_V2.md` and ADR-004. It does not replace the deterministic Jarvis-X VM or claim that the dependency-free reference implementation is a production neural or video-codec stack.

## 1. State spaces

Let

```text
M  : multimodal observation
c  : conditioning state
z  : latent state
G  : mesh / geometric state
F  : rendered frame sequence
B  : coded multimedia bitstream
A  : archive/container bytes
Omega : persistent metrics/adaptation state
```

The authoritative orchestration state is

```text
Xi = (M, c, z, G, F, A, Theta, Omega, telemetry).
```

Every boundary is explicit. A renderer does not emit a latent; a container is not a codec; an inverse model does not become an exact inverse merely by naming.

## 2. Forward functional

```text
c      = Phi(M)
z      = E_thetae(G_source)
V0     = D_thetad(z, c)
Vstar  = Refine_E_MH(V0)
F      = R(Vstar; camera, material, light)
B      = C_q(F)
A      = Mux(B, side_info)
```

The conditional geometric fixed-point objective is

```text
Vstar = argmin_{V in G_adm} [
    E_MH(V)
    + (xi/2) ||V - D(E(V), c)||_F^2
].
```

## 3. Moagi-Helmholtz energy

```text
E_MH(V) =
    (rho/2) tr(V^T L(V) V)
  + (lambda/2) ||V - C(V)||_F^2
  + (gamma/2) ||L(V)V||_F^2
  + (mu/2) ((Area(V)-A_target)/A_target)^2
  + nu B_geom(V,F).
```

`L(V)` is a cotangent Laplacian under an explicitly selected sign convention. `B_geom` is a bounded barrier/penalty used to reject or discourage degenerate or inadmissible geometry.

Refinement is dissipative gradient flow:

```text
dV/dtau = - grad E_MH(V)
```

or a named numerical approximation such as a lagged-Laplacian update.

A discrete candidate step is

```text
V_candidate = V_k - eta_k * g_E(V_k)
V_(k+1)     = Pi_G(V_candidate).
```

## 4. Geometry projection

`Pi_G` enforces at least:

- finite coordinates;
- vertex-count/resource limits;
- valid triangle indices;
- declared topology constraints;
- bounded displacement where configured;
- non-degenerate geometry where required.

Invalid candidates fail closed.

## 5. Rendering

Rendering is a calibrated map

```text
F_t = R(Vstar, Faces, Camera_t, Materials_t, Lights_t).
```

Perspective or orthographic projection is backend-specific but camera calibration and depth conventions must be explicit whenever reverse geometric inference depends on them.

Optional geometry-preserving channels include depth, normals, segmentation, camera transforms and topology descriptors.

## 6. Multimedia archive

The archival contract is

```text
B = VideoEncode_q(F)
A = ContainerMux(B, side_info).
```

For an MP4 backend, MP4 is the container contract. The actual video codec, transform, prediction, quantization and entropy-coding tools are separate backend details.

The rate-distortion objective is

```text
J_RD = D(F, F_hat) + lambda_R * Rate(B).
```

Both distortion and bit count must be reported for codec claims.

## 7. Reverse inference

```text
(B_hat, side_info) = ContainerDemux(A)
F_hat              = VideoDecode(B_hat)
(z_hat, c_hat)     = I_phi(F_hat, side_info)
V_hat0             = D_thetad(z_hat, c_hat)
V_hatstar          = Refine_E_MH(V_hat0).
```

This is an inference/reconstruction path. A global inverse renderer is not assumed.

Exact deterministic reconstruction may be claimed only when the archive contains sufficient information to establish it. Otherwise the result is an inferred compatible geometry.

## 8. Side-information contract

A geometry-aware archive may carry any explicitly versioned subset of:

```text
camera parameters
frame timestamps
latent code
conditioning code
mesh/topology descriptor
depth
normals
segmentation/material IDs
model versions
codec parameters
integrity digest
transaction identifier
```

Side information is part of the information budget and may not be ignored when making compression claims.

## 9. Cycle verification

Let

```text
Vstar      = forward refined geometry
V_hatstar  = reconstructed refined geometry.
```

The reference cycle metric is a topology-preserving RMS displacement when topology matches:

```text
E_cycle = rms(Vstar, V_hatstar).
```

Production implementations may additionally use Chamfer distance, normal consistency, topology metrics, perceptual render metrics and task-specific scores.

The immutable source anchor provides drift telemetry:

```text
E_anchor = d_G(V_anchor, Vstar).
```

## 10. Grand training objective

A compatible training objective is

```text
L_MH =
    w_CD      L_chamfer
  + w_normal  L_normal
  + w_edge    L_edge
  + w_KL      L_KL
  + w_render  L_render
  + w_rate    Rate
  + w_station ||grad E_MH(Vstar)||^2
  + w_cycle   L_cycle
  + w_cond    L_condition.
```

Training and runtime coefficients are distinct configuration domains.

## 11. Transactional permeation

The full system cycle is

```text
OBSERVE
-> CONDITION
-> ENCODE
-> GENERATE
-> REFINE
-> RENDER
-> CODE
-> ARCHIVE
-> DECODE
-> INFER
-> REGENERATE
-> MEASURE
-> PROPOSE ADAPTATION
-> VERIFY
-> COMMIT / ROLLBACK
-> JOURNAL
-> RE-ENTER
```

No candidate model, tile schedule, geometry, codec policy or bytecode mutation becomes authoritative before validation.

## 12. Master operator

Let `M_MH` denote the composed generative/archive/reconstruction candidate operator. The bounded Jarvis-X transition is

```text
Xi_(t+1) = Pi_Lambda(M_MH(Xi_t)).
```

The broader inward recurrence remains

```text
Xi_(t+1) = Pi_Lambda_t[
    Xi_t
  + P_(1:M)^inward(Xi_t)
  - E_t
  + Omega_t
  + kappa_t R_t^inward
  - eta_t grad_Theta L_MH
  - zeta_t grad_H C_t
].
```

The Moagi-Helmholtz functional therefore supplies the concrete generative, geometric and archival semantics inside the broader Jarvis-X transactional loop.

## 13. Fixed points

A bounded orchestration fixed point satisfies

```text
Xi_star = Pi_Lambda(M_MH(Xi_star)).
```

A geometric stationary point satisfies

```text
grad_V [
    E_MH(Vstar)
    + (xi/2)||Vstar - D(E(Vstar),c)||_F^2
] = 0.
```

Unique convergence is claimed only when a sufficient condition such as contractivity is demonstrated on the admissible state space.

## 14. Reference implementation

`src/jarvisx/moagi_helmholtz.py` provides protocol boundaries for:

```text
Conditioner
GeometryEncoder
GeometryDecoder
GeometryRefiner
Renderer
ArchiveCodec
InverseModel
StateValidator
```

and `MoagiHelmholtzEngine.step()` implements candidate-first publication.

The bundled conformance components intentionally use:

- a deterministic identity-like mesh descriptor instead of a trained neural codec;
- an identity geometric refiner instead of a cotangent-energy solver;
- a logical frame serializer instead of a rasterizer;
- deterministic JSON bytes instead of MP4;
- archived latent/condition side information instead of learned inverse inference.

This makes orchestration invariants executable without overstating backend capability.

## 15. Backend permeation map

The contract is intended to lower into the wider Jarvis-X ecosystem as follows:

```text
Python reference     -> orchestration and conformance
C++ runtime          -> geometry/refinement kernels
CUDA/GPU             -> tensor and rendering acceleration
Neural backends      -> Phi, E, D, I_phi
Video codec backend  -> C_q and container adapter
DMEB / tensor ISA    -> bounded accelerator lowering
FPGA/native ISA      -> verified hardware acceleration
Browser/3D UI        -> visualization and telemetry only unless explicitly promoted
```

All backends inherit the same transaction, evidence and authority boundaries.

## 16. Required telemetry

A production run should expose at least:

```text
cycle
source geometry size
latent size
conditioning version
refinement iterations
geometric energy
stationarity norm
render dimensions / frame count
encoded bytes
bitrate or rate estimate
render distortion
cycle reconstruction error
anchor drift
model/codec versions
validator decision
commit/rollback result
measured latency and resident memory
```

Virtual/logical scale is always reported separately from measured hardware performance.

## 17. Canonical interpretation

The Moagi-Helmholtz system is a conditional 3D generative, refinement and multimedia archival architecture. Its canonical strength is not a claim of magical inversion; it is the explicit coupling of generation, geometry, codec economics, inverse inference, evidence and reversible adaptation inside one auditable Jarvis-X state machine.

## 18. Orthogonal transform precision gate

Transform-based latent or archive adapters that claim an orthonormal reconstruction boundary additionally inherit ADR-005.

For

```text
X       = D x
A_k     = round_nearest(X_k / delta_k)
Xhat_k  = delta_k A_k
xhat    = D^T Xhat
D^T D   = I
```

the deterministic precision envelope is

```text
B_Q = 0.5 * sqrt(sum_k delta_k^2)
||x - xhat||_2 <= B_Q.
```

For a uniform step `Delta`,

```text
B_Q = Delta * sqrt(M) / 2.
```

The transform gate is

```text
Lambda_Q = ||x - xhat||_2 / B_Q <= 1.
```

A precision-gate failure is diagnosed before the enclosing codec or geometry tolerance is changed.  In particular, the runtime must distinguish transform normalization/inverse defects from genuine quantization distortion.

The canonical verification order is

```text
render / latent state
-> declared transform
-> verify D^T D ~= I
-> quantize / dequantize
-> reconstruct with D^T
-> Lambda_Q
-> rate/distortion accounting
-> archive reconstruction / cycle metrics
-> Pi_Lambda
-> COMMIT or ROLLBACK.
```

The precision receipt reports its own transform error, quantization bound and gate ratio separately from render distortion and `E_cycle`.  This prevents one numerical layer from hiding defects inside a broader multimedia error budget.

The reference implementation is `src/jarvisx/orthogonal_quantization.py`; the normative numerical contract is `DR_MOAGI_ORTHOGONAL_QUANTIZATION.md`.
