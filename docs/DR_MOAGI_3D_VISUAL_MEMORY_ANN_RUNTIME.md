# Dr Moagi 3D Visual Image Memory ANN Runtime

## Status

Executable reference permeation for Jarvis-X. The implementation lives in
`src/jarvisx/geometric_memory.py` and is deliberately dependency-free,
deterministic, bounded, and auditable.

It converts the earlier conceptual Encoder -> Memory -> Recursive Refinement ->
Decoder architecture into a runnable geometric state-transition system.

---

## 1. State

Let the observed 3D image be a scalar voxel field:

\[
X_t \in [0,1]^{D\times H\times W}.
\]

The compact geometric latent state is:

\[
Z_t \in \mathbb{R}^{d\times h\times w\times c}.
\]

The associative memory is a finite set of key-value slots:

\[
\Omega_t=\{(k_i,v_i,u_i)\}_{i=1}^{S},
\]

where `u_i` is a deterministic usage counter.

The mechanics state is the validated configuration:

\[
\mathcal M_t=(d,h,w,c,S,K,\eta,\beta,\gamma,L_{max},\lambda_C).
\]

The fields respectively specify latent geometry, channel count, memory slots,
refinement steps, learning rate, residual gain, memory gain, projection bound,
and compute-cost weight.

---

## 2. Geometric encoding

The source volume is partitioned into a compact 3D lattice. Each latent cell
encodes local mean intensity, variance, three directional gradients, radial
position, and deterministic higher-channel mixtures:

\[
Z_0=E_G(X_t).
\]

For a source region `R_p` associated with latent coordinate `p`:

\[
\mu_p=\frac{1}{|R_p|}\sum_{x\in R_p}X_t(x),
\qquad
\sigma_p^2=\frac{1}{|R_p|}\sum_{x\in R_p}(X_t(x)-\mu_p)^2.
\]

The first latent channel preserves coarse image intensity as `2 mu - 1`; the
remaining channels preserve local geometric variation.

---

## 3. Associative visual memory

A global query is formed by averaging the latent vectors:

\[
q_k=\frac{1}{dhw}\sum_p Z_k(p).
\]

Memory attention is cosine similarity with deterministic softmax weighting:

\[
a_i=\operatorname{softmax}_i\left(4\cos(q_k,k_i)+0.05u_i\right).
\]

The recalled vector is:

\[
r_k=\sum_i a_i v_i.
\]

The selected memory slot is updated by a bounded interpolation rather than an
unconstrained overwrite.

---

## 4. Recursive latent refinement

The decoder produces a reconstructed voxel volume:

\[
\hat X_k=D_G(Z_k).
\]

The residual is:

\[
E_k=X^\star-\hat X_k.
\]

The reference implementation uses a deterministic residual summary and memory
correction to update every latent vector:

\[
Z_{k+1}(p)=\Pi_{[-L_{max},L_{max}]}
\left[
Z_k(p)+\eta\left(\beta\bar E_k+\gamma(r_k-Z_k(p))\right)
\right].
\]

This is the executable inward turn: the system reconstructs its input, measures
its own error, projects correction back into its geometric latent state, consults
memory, and repeats.

The implementation records numerical telemetry for each refinement step:

- reconstruction loss;
- latent norm;
- residual norm;
- recalled-memory norm;
- selected memory slot.

These records are auditable latent-state telemetry. They are not textual private
chain-of-thought traces.

---

## 5. Bounded self-optimisation

Jarvis-X does not permit arbitrary source-code mutation through this runtime.
The admissible candidate set is finite and declared. Current candidates vary:

- learning rate down;
- learning rate up;
- residual gain up;
- one additional refinement step, provided the configured maximum is not crossed.

Every candidate starts from an identical memory snapshot. Candidate cost is:

\[
J(\mathcal M)=L_{recon}(\mathcal M)+\lambda_C\widehat C(\mathcal M),
\]

where:

\[
L_{recon}=\frac{1}{DHW}\|X^\star-\hat X\|_2^2.
\]

The deterministic operation estimate is:

\[
\widehat C= DHW(2c+3)+K(dhw)c(S+8).
\]

A mechanics change is committed only when:

\[
J(\mathcal M')<J(\mathcal M_t).
\]

Ties preserve the baseline. The selected measurement is appended to the local
optimisation journal.

---

## 6. Transactional permeation cycle

```text
OBSERVE_VOLUME
ENCODE_GEOMETRY
SNAPSHOT_MEMORY
GENERATE_BOUNDED_CANDIDATES
FOR EACH CANDIDATE:
    RESTORE_IDENTICAL_MEMORY_SNAPSHOT
    RECALL_ASSOCIATIVE_MEMORY
    DECODE_VOLUME
    COMPUTE_RESIDUAL
    PROJECT_RESIDUAL_INTO_LATENT
    UPDATE_MEMORY_SLOT
    PROJECT_LATENT_BOUNDS
    MEASURE_RECONSTRUCTION_AND_COST
SELECT_MINIMUM_VALID_OBJECTIVE
COMMIT_CONFIG_AND_MEMORY
JOURNAL_MEASUREMENT
RETURN_RECONSTRUCTION_AND_TELEMETRY
```

This operationally instantiates:

\[
\boxed{
\Xi_{t+1}=\Pi_{\Lambda_t}
[\Xi_t+P(\Xi_t)-E_t+\Omega_t+U_t]
}
\]

with `Pi_Lambda` implemented through configuration validation, latent bounds,
finite candidate enumeration, identical shadow snapshots, objective comparison,
and conservative commit semantics.

---

## 7. Cloud execution mapping

The current implementation is a portable reference kernel, not a high-throughput
GPU trainer. Its contracts map directly onto a distributed cloud runtime:

| Reference object | Cloud/GPU implementation |
|---|---|
| `Volume3D` | object-store chunk, Zarr/N5 volume, sparse voxel brick |
| `LatentField` | sharded GPU tensor or 3D texture |
| `SpatialMemory` | replicated key-value tensor or vector-memory service |
| candidate snapshot | immutable checkpoint/object-store version |
| candidate run | isolated GPU job, pod, or serverless accelerator task |
| measurement | metrics stream and optimisation journal |
| commit | atomic configuration pointer update |

Candidate runs are embarrassingly parallel because they begin from the same
snapshot. The commit gate remains serial and authoritative.

---

## 8. Verification

`tests/test_geometric_memory.py` verifies:

1. volume indexing and exact self-MSE;
2. deterministic equality from identical initial state;
3. output shape, latent shape, trace length, numerical finiteness, and voxel bounds;
4. bounded candidate count, refinement-step ceiling, journal creation, and
   non-regression of the selected objective relative to baseline.

The CLI demonstration is:

```bash
jarvisx visual-memory 12
```

It returns the selected mechanics and the complete numerical refinement trace as
JSON.

---

## 9. Next executable progression

The reference runtime establishes semantics before acceleration. The next
implementation layers should preserve this API while replacing kernels in order:

1. NumPy vectorisation;
2. PyTorch/JAX differentiable residual projection;
3. sparse voxel or Gaussian-splat input adapters;
4. 3D Fourier-domain loss;
5. GPU candidate batching;
6. object-store checkpoints and cloud worker orchestration;
7. visual WebGPU telemetry for the latent lattice and memory attention field.

The invariant is that acceleration may change mechanics, but it may not silently
change the declared semantic, numerical, determinism, resource, or commit
contract.
