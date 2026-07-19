# Dr Moagi 3D Visual Image Memory Runtime

## Status

Executable deterministic reference semantics for Jarvis-X. The implementation
lives in `src/jarvisx/geometric_memory.py` and is bounded, auditable, and
integrated into the authoritative `CodexVM` control plane.

The module is ANN-compatible but is not yet a trained neural network. Its
encoder, decoder, memory addressing, and residual update are explicit
mathematical operators. Future tensor, GPU, and learned implementations must be
verified against these reference semantics.

---

## 1. Unified state

Let the observed 3D image be:

\[
X_t\in[0,1]^{D\times H\times W}.
\]

The compact latent state is:

\[
Z_t\in\mathbb R^{d\times h\times w\times c}.
\]

The finite associative memory is:

\[
\Omega_t=\{(k_i,v_i,u_i)\}_{i=1}^{S},
\]

and the validated mechanics state is:

\[
\mathcal M_t=(d,h,w,c,S,K,\eta,\beta,\gamma,L_{max},\lambda_C).
\]

The VM-authoritative state includes bytecode execution and geometry:

\[
\Sigma_t=(R_t,M_t,PC_t,Z_t,\Omega_t,\mathcal M_t,J_t).
\]

A geometric transaction is identified as `V3D.PERMEATE`; it can be denied by
the Lambda policy gate and is written to the VM ledger and tracer after a
successful commit.

---

## 2. Geometric encoding

The source volume is partitioned into a compact 3D lattice. For every latent
coordinate `p`, the associated source region `R_p` contributes local mean,
variance, directional differences, and radial position:

\[
Z_0=E_G(X_t).
\]

\[
\mu_p=\frac{1}{|R_p|}\sum_{x\in R_p}X_t(x),
\qquad
\sigma_p^2=\frac{1}{|R_p|}\sum_{x\in R_p}(X_t(x)-\mu_p)^2.
\]

The first channel preserves coarse intensity as `2 mu - 1`; remaining channels
preserve local variation and deterministic higher-channel mixtures.

---

## 3. Associative memory

A global scene query is currently formed by averaging latent vectors:

\[
q_k=\frac{1}{dhw}\sum_p Z_k(p).
\]

Retrieval uses cosine similarity with an anti-monopoly usage penalty:

\[
a_i=\operatorname{softmax}_i\left(4\cos(q_k,k_i)-0.05u_i\right).
\]

The recalled correction is:

\[
r_k=\sum_i a_i v_i.
\]

The negative usage term prevents repeatedly selected slots from acquiring an
unbounded positive-selection advantage. Keys and values are updated through
bounded interpolation rather than unconstrained replacement.

---

## 4. Spatial residual permeation

The decoder produces:

\[
\hat X_k=D_G(Z_k).
\]

The full residual field is retained:

\[
E_k(z,y,x)=X^\star(z,y,x)-\hat X_k(z,y,x).
\]

The residual encoder partitions this field using the same geometric map as the
source encoder:

\[
\Delta Z_k(p)=E_G^{res}\left(E_k|_{R_p}\right).
\]

Each latent cell is updated from its own spatial residual rather than one global
mean:

\[
Z_{k+1}(p)=\Pi_{[-L_{max},L_{max}]}
\left[
Z_k(p)+\eta\left(
\beta\Delta Z_k(p)+\gamma(r_k-Z_k(p))
\right)
\right].
\]

This prevents positive and negative errors in unrelated regions from cancelling
before correction reaches the latent geometry.

Every refinement step records numerical telemetry:

- reconstruction loss;
- latent norm;
- residual norm;
- recalled-memory norm;
- selected memory slot.

These values are auditable state telemetry, not textual private reasoning.

---

## 5. Bounded inward optimisation

The runtime does not permit arbitrary source-code mutation. It evaluates a
finite declared candidate set that may adjust:

- learning rate down or up;
- residual gain up;
- memory gain down or up;
- refinement depth by one step within the configured maximum.

Every candidate starts from an identical memory snapshot. Candidate cost is:

\[
J(\mathcal M)=L_{recon}(\mathcal M)+\lambda_C\widehat C(\mathcal M),
\]

with:

\[
L_{recon}=\frac{1}{DHW}\|X^\star-\hat X\|_2^2.
\]

A mechanics change is committed only when:

\[
J(\mathcal M')<J(\mathcal M_t).
\]

Ties preserve the baseline. The journal stores all candidate measurements for
the experiment, not merely the winner.

---

## 6. Transactional cycle

```text
OBSERVE_VOLUME
ENCODE_GEOMETRY
SNAPSHOT_MEMORY
GENERATE_BOUNDED_CANDIDATES
FOR EACH CANDIDATE:
    RESTORE_IDENTICAL_MEMORY_SNAPSHOT
    RECALL_ASSOCIATIVE_MEMORY
    DECODE_VOLUME
    COMPUTE_SPATIAL_RESIDUAL_FIELD
    ENCODE_RESIDUAL_PER_LATENT_REGION
    UPDATE_MEMORY_SLOT
    PROJECT_EACH_LATENT_CELL
    MEASURE_RECONSTRUCTION_AND_COST
SELECT_MINIMUM_VALID_OBJECTIVE
POLICY_CHECK_V3D_PERMEATE
COMMIT_CONFIG_AND_MEMORY
JOURNAL_ALL_CANDIDATES
WRITE_VM_LEDGER_AND_TRACE
RETURN_RECONSTRUCTION_AND_TELEMETRY
```

This instantiates:

\[
\boxed{
\Xi_{t+1}=\Pi_{\Lambda_t}
[\Xi_t+P(\Xi_t)-E_t+\Omega_t+U_t]
}
\]

through validation, projection bounds, identical shadow snapshots, objective
comparison, policy control, and conservative commit semantics.

---

## 7. Interfaces and isolation

The CLI routes 3D execution through `CodexVM.run_visual_memory`. The FastAPI
surface exposes:

```text
GET  /health
POST /run
POST /visual-memory
```

Each HTTP or TCP request uses an isolated VM and in-memory ledger. Development
servers bind to `127.0.0.1` by default. Production exposure still requires an
external authentication, authorisation, TLS, rate-limiting, and process
isolation boundary.

---

## 8. Verification

The test suite verifies:

1. volume indexing and exact self-MSE;
2. deterministic equality from identical initial state;
3. shape, trace, finiteness, and voxel bounds;
4. bounded candidate count and non-regression against baseline;
5. full-candidate experiment journaling;
6. spatial residual locality;
7. repeated VM program execution after HALT;
8. VM ledger and trace integration for `V3D.PERMEATE`;
9. policy denial of geometric execution;
10. FastAPI route integrity.

---

## 9. Cloud and learned progression

The current kernel is a portable semantic oracle, not a high-throughput trainer.
Its contracts map to future infrastructure as follows:

| Reference object | Accelerated implementation |
|---|---|
| `Volume3D` | dense tensor, Zarr/N5 chunk, sparse voxel brick |
| `LatentField` | sharded GPU tensor or 3D texture |
| `SpatialMemory` | batched key-value tensor or memory service |
| candidate snapshot | immutable checkpoint version |
| candidate run | isolated GPU worker |
| candidate journal | metrics and provenance stream |
| commit | atomic authoritative configuration pointer |

The progression order is:

1. NumPy vectorisation;
2. PyTorch or JAX differentiable spatial residual projection;
3. learned encoder, decoder, and local memory attention;
4. sparse voxel and Gaussian-splat adapters;
5. 3D Fourier, gradient, and topology losses;
6. GPU candidate batching;
7. object-store checkpoints and cloud worker orchestration;
8. WebGPU visual telemetry.

The invariant is:

\[
\boxed{
\text{Acceleration may change mechanics; it may not silently change meaning.}
}
\]
