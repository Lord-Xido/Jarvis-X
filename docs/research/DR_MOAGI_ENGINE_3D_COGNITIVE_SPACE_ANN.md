# Dr Moagi Engine — 3D Cognitive-Space ANN

**Canonical neural-core research specification**  
**Designation:** `DM-3D-Cognitive-Space-ANN`  
**Status:** Locked research architecture  
**Architecture record:** `docs/adr/0013-dr-moagi-engine-3d-cognitive-space-network-of-networks.md`

## 1. Definition

The **Dr Moagi Engine** is the neural core of the wider Jarvis-X research stack:

> A recursive, multiscale 3D auto-encoding/decoding Cognitive Space implemented as a dynamically routed Network of Networks.

The core operator is

```text
DM_3D = D_3D o R_C^* o E_3D
```

where:

```text
E_3D  volumetric encoder
R_C   recurrent Cognitive-Space Network-of-Networks transform
D_3D  volumetric decoder
```

The neural core is intentionally separated from bytecode authority, system audit/commit, provenance authentication, Helmholtz permeation, rendering/container formats, and native accelerator implementations.

## 2. State spaces

### Input volume

A bounded reference input is

```text
X_t in R^(B x C x D x H x W).
```

`C` may encode density, occupancy, color, signed distance, learned channels, or another explicitly declared volumetric representation.

### Volumetric latent field

The encoder produces a spatial latent rather than immediately flattening to one vector:

```text
Z_t in R^(B x C_z x D_z x H_z x W_z).
```

### Cognitive Space

The shared recurrent neural state is

```text
C_t^3D = (Z_t, G_t, Omega_t, A_t, H_t)
```

with:

```text
Z_t      volumetric latent features
G_t      explicit spatial/relational topology
Omega_t  recurrent ANN memory
A_t      attention and routing state
H_t      bounded active latent hypotheses/candidates
```

The Cognitive Space is computational state. The term does not claim consciousness or subjective cognition.

## 3. Encoder

The canonical encoder map is

```text
Z_t = E_ThetaE^3D(X_t).
```

A practical reference family is a hierarchical 3D residual encoder:

```text
X
 -> Conv3D / residual block
 -> downsample
 -> Conv3D / residual block
 -> downsample
 -> ...
 -> Z.
```

For one layer,

```text
F_(l+1) = sigma(N_l(W_l *_3D F_l + b_l)).
```

The encoder must expose tensor shapes and compression ratio so spatial collapse is measurable.

## 4. Network of Networks

The Cognitive Space is updated by specialist neural modules. The initial canonical registry is:

```text
N_local       local 3D feature refinement
N_regional    region-to-region integration
N_global      whole-volume/global context
N_attention   spatial/semantic attention and routing
N_memory      recurrent latent memory update
N_prediction  bounded candidate prediction
N_correction  residual/error-driven correction
N_decoder     one or more volumetric reconstruction heads
```

These are roles, not mandatory class names. Implementations may merge or split roles if the resulting data contracts remain explicit.

## 5. Typed transports

Specialist network outputs are not assumed to share shapes or units. Before combination, each contribution is mapped through an explicit transport:

```text
T_k : Output(N_k) -> Component(C).
```

The generic recurrence is

```text
C_(m+1) = Pi_Lambda_C[
    C_m
    + T_L N_local(C_m)
    + T_R N_regional(C_m)
    + T_G N_global(C_m)
    + T_A N_attention(C_m)
    + T_M N_memory(Omega_t, C_m)
    + T_P N_prediction(C_m)
    - T_E N_correction(E_t, C_m)
].
```

This is a compositional notation. An executable implementation should usually update individual Cognitive-Space components through typed staged operations rather than literally adding a heterogeneous tuple.

## 6. 3D relational computation

A local latent site can be represented as

```text
z_i = (h_i, p_i, omega_i)
```

where `p_i` is a computational 3D coordinate.

A message-passing implementation may define

```text
m_ij = phi(h_i, h_j, p_i - p_j)
h_i' = psi(h_i, aggregate_j(m_ij), omega_i).
```

The topology may be a regular voxel neighborhood, sparse graph, octree neighborhood, windowed attention lattice, or another bounded validated representation.

## 7. Attention and dynamic routing

Full global attention over a dense `DHW` volume scales quadratically in the number of sites and is not the default reference target.

Preferred mechanisms include:

```text
windowed 3D attention
axial attention
sparse graph attention
hierarchical global tokens
mixture-of-experts routing
```

For routed experts,

```text
w_k >= 0
sum_k w_k = 1
Y = sum_k w_k N_k(X)
```

or an explicitly documented top-k sparse variant.

Routing telemetry should include active experts, utilization, capacity, overflow/rejection behavior, and routing entropy where useful.

## 8. Multiscale Cognitive Space

The preferred architecture is pyramidal:

```text
C^(0) <-> C^(1) <-> ... <-> C^(L)
```

with `C^(0)` carrying finer spatial detail and higher levels carrying progressively coarser/global representations.

A generic level update is

```text
C_l^(m+1) = R_l(
    C_l^m,
    Up(C_(l+1)^m),
    Down(C_(l-1)^m),
    Omega_l,
    A_l
).
```

Boundary levels omit nonexistent neighbors.

## 9. Recursive inward refinement

After encoding:

```text
C_t^(0) = Lift(E_3D(X_t)).
```

The Network of Networks iterates

```text
C_t^(m+1) = Pi_Lambda_C[R_C(C_t^m)].
```

Stopping is based on measured execution:

```text
||C_t^(m+1) - C_t^m|| <= epsilon_fp
```

or

```text
m == I_max.
```

The fixed-point notation

```text
C_t^* = FixedPoint(R_C, C_t^(0))
```

does not imply that every learned network is mathematically contractive. Contraction must be proven or measured on the declared operating domain if claimed.

## 10. Decoder

The volumetric decoder reconstructs

```text
X_hat_t = D_ThetaD^3D(C_t^*).
```

A reference implementation mirrors the encoder hierarchy through residual 3D blocks and upsampling. Decoder heads may separately predict quantities such as occupancy, density, signed distance, color, normals, or learned features, provided output semantics are explicit.

## 11. Autoencoding closure and correction

The reconstruction residual is

```text
E_t = X_t - X_hat_t
```

for representations where direct subtraction is defined.

For other domains, define an explicit residual/metric operator.

A corrective loop may re-encode residual information:

```text
Z_E,t = E_error^3D(E_t)
C'_t = Correct(C_t^*, Z_E,t)
```

before another bounded refinement/decode cycle.

The operational identity associated with `I AM = I DESCRIBE` is therefore limited to

```text
state -> latent description -> reconstruction -> residual correction.
```

## 12. ANN memory

Within the neural core, `Omega_t` is recurrent ANN memory. A simple reference rule may be

```text
Omega_(t+1) = beta * Omega_t + (1-beta) * M(C_t^*, E_t).
```

The exact memory network is trainable and versioned.

ANN memory must remain distinct from:

```text
VM journal
system audit ledger
external evidence provenance
security logs
persistent authoritative task state.
```

## 13. Training objective

A reference objective may be

```text
L_total =
    lambda_rec     * L_rec
  + lambda_grad    * L_grad
  + lambda_cycle   * L_cycle
  + lambda_fp      * L_fixed_point
  + lambda_sparse  * L_sparse
  + lambda_route   * L_route_balance
  + lambda_memory  * L_memory.
```

Possible components include

```text
L_rec         reconstruction L1/MSE or domain loss
L_grad        volumetric gradient/edge preservation
L_cycle       latent or reconstruction cycle consistency
L_fixed_point ||R_C(C*) - C*||^2
L_sparse      latent activation/resource sparsity
L_route       expert utilization/capacity regularization
L_memory      bounded temporal consistency objective
```

No loss term should be presented as a physical energy unless a separate physical model defines units and conservation/dissipation laws.

## 14. Parameter update

Parameters are updated in parameter space:

```text
Theta_(t+1) = Theta_t - eta_Theta * grad_Theta L_total.
```

A parameter gradient is not directly additive with Cognitive-Space state. If a learned optimizer transports parameter information into latent state, the transport must be explicit.

## 15. Preferred first trainable reference

A practical first experiment is intentionally small enough to benchmark honestly:

```text
Input
  B x C x 64 x 64 x 64

Encoder
  32 channels @ 32^3
  64 channels @ 16^3
  128 channels @ 8^3

Cognitive Space
  B x 128 x 8 x 8 x 8 base latent
  4-8 bounded recurrent refinement steps
  local 3D residual expert
  windowed-attention expert
  memory expert
  prediction/correction expert
  sparse router

Decoder
  64 @ 16^3
  32 @ 32^3
  output @ 64^3
```

This shape is a reference starting point, not a canonical scale claim.

## 16. Baselines and ablations

The architecture earns value only through controlled comparison.

Minimum baseline family:

```text
A. conventional 3D autoencoder
B. 3D VAE where relevant
C. 3D autoencoder + attention
D. Dr Moagi Network-of-Networks without recurrence
E. Dr Moagi recurrent Network-of-Networks
```

Ablations should isolate at least:

```text
multiscale hierarchy
routing
memory
recurrent depth
attention
residual correction
fixed-point regularization
```

## 17. Metrics

Report metrics appropriate to the representation, including where applicable:

```text
reconstruction MSE / MAE
PSNR / volumetric structural quality
Chamfer / IoU / SDF error for geometry
latent size / bitrate
fixed-point residual
actual refinement iterations
VRAM / resident memory
training/inference latency
samples or volumes per second
expert utilization
parameter count / FLOPs or measured accelerator work
```

Virtual spatial extents and symbolic recursion depth must be reported separately from measured execution.

## 18. Boundary with the wider Jarvis-X system

The Dr Moagi Engine produces neural candidates and reconstructions. Wider Jarvis-X components may then provide:

```text
epistemic verification
capability projection
bytecode lowering
canonical VM execution
transaction audit/commit
permeation
rendering/UI
hardware acceleration
```

These enclosing layers do not become part of the ANN merely because they interact with its outputs.

## 19. Safety and epistemic boundary

The ANN may converge to a stable representation that is wrong about external reality.

Therefore:

```text
fixed point != truth
low reconstruction error != factual correctness
self-consistency != independent evidence
```

The wider epistemic and authority layers remain responsible for external claims and side effects.

## 20. Canonical closed form

The neural core is summarized as

```text
Z_t = E_ThetaE^3D(X_t)
C_t^(0) = Lift(Z_t, G_t, Omega_t, A_t, H_t)
C_t^* = FixedPoint(Pi_Lambda_C o R_ThetaR^3D, C_t^(0); epsilon_fp, I_max)
X_hat_t = D_ThetaD^3D(C_t^*)
E_t = Residual(X_t, X_hat_t)
Omega_(t+1) = M_ThetaM(Omega_t, C_t^*, E_t)
Theta_(t+1) = Theta_t - eta_Theta grad_Theta L_total
```

or compactly

```text
Dr Moagi Engine = D_3D o R_C^* o E_3D
```

with `R_C` realized as a bounded dynamically routed Network of Networks over a multiscale 3D Cognitive Space.

## 21. Locked designation

```text
DR MOAGI ENGINE

Recursive Multiscale 3D Auto-Encoding/Decoding Cognitive Space
implemented as a Dynamically Routed Network of Networks.
```

This is the canonical neural-core definition for subsequent implementation, benchmarking, accelerator lowering, and system integration work.
