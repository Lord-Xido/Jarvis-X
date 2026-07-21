# Sparse Tetration 3-D Auto-Encoding/Decoding Field Automaton

## 1. Virtual address manifold

Define the base-1000 tetration tower

\[
T_1=1000,
\qquad
T_{k+1}=1000^{T_k}.
\]

The virtual cubic universe is

\[
\mathcal U_k=\{0,\ldots,T_k-1\}^3,
\qquad
|\mathcal U_k|=T_k^3.
\]

`TetrationUniverse` stores only this symbolic definition. It never allocates
\(T_k^3\) cells.

For an explicit raw coordinate, the theoretical naming length is

\[
b_k=3\left\lceil\log_2T_k\right\rceil
=3\left\lceil T_{k-1}\log_2 1000\right\rceil.
\]

At \(k=2\), this is 29,898 bits. For \(k\ge3\), even the raw coordinate string is
impractical. The executable runtime therefore uses a finite symbolic chart ID and
exact signed local offsets:

\[
\mathbf r=(k,\text{chart},x,y,z).
\]

This is a compressed coordinate description and a local chart, not a claim that
all arbitrary raw tetration coordinates fit in memory.

## 2. Physical state

Only materialised bricks exist in RAM:

\[
\Sigma_t=(\mathcal A_t,B_t,Z_t,\Omega_t,\mathcal D_t,J_t).
\]

- \(\mathcal A_t\): bounded active brick set;
- \(B_t(\mathbf r)\in\mathbb R^{3\times4\times4\times4}\): 192-value brick;
- \(Z_t\): transient latent state;
- \(\Omega_t(\mathbf r)\in\mathbb R^{192}\): correction memory;
- \(\mathcal D_t\): collision-chained sparse directory;
- \(J_t\): deterministic journal hash.

The invariant is

\[
\operatorname{cost}(t)=O(M_t),
\qquad
M_t=|\mathcal D_t|,
\qquad
O(M_t)\ne O(|\mathcal U_k|).
\]

## 3. Sparse allocation and active mask

The finite directory uses

\[
\operatorname{idx}(\mathbf r)=
\bigl(
5147x\oplus9293y\oplus11257z\oplus H(\text{chart},k)
\bigr)\bmod P.
\]

Each slot contains a collision chain. Equality is checked against the full
canonical address, so collisions do not alias state.

The active mask is a predicate:

\[
W_{\mathcal M}(\mathbf r)=
\begin{cases}
1,&\mathbf r\in\mathcal A_t,\\
0,&\text{otherwise}.
\end{cases}
\]

In code this is directory membership and threshold testing, not a dense tensor
multiplication.

## 4. Full brick encoder

Flatten the whole brick:

\[
b_{\mathbf r}=\operatorname{flat}(B_t(\mathbf r))\in\mathbb R^{192}.
\]

After box normalisation, encoding is

\[
z_{\mathbf r}=\tanh(W_{enc}b_{\mathbf r}+c_{enc}),
\qquad
W_{enc}\in\mathbb R^{d\times192}.
\]

All 192 values participate. No prefix such as `flat[:16]` is used.

## 5. Omega conditioning and top-1 MoE

Correction memory conditions the latent state:

\[
\widetilde z_{\mathbf r}=
\tanh\left(z_{\mathbf r}+W_\Omega\Omega_t(\mathbf r)\right),
\qquad
W_\Omega\in\mathbb R^{d\times192}.
\]

For experts \(i=1,\ldots,N_E\),

\[
\ell_i=R_i^T\widetilde z,
\qquad
g_i=\frac{e^{\ell_i}}{\sum_j e^{\ell_j}},
\qquad
k^*=\arg\max_i g_i.
\]

Routing is explicitly **top-1**:

\[
z'_{\mathbf r}=E_{k^*}(\widetilde z_{\mathbf r}).
\]

The committed brick records `expert_index = k*` for auditing.

## 6. Full decoder

The decoder is a full matrix projection:

\[
\widehat b_{\mathbf r}=W_{dec}z'_{\mathbf r}+c_{dec},
\qquad
W_{dec}\in\mathbb R^{192\times d},
\]

\[
\widehat V(\mathbf r)=
\operatorname{reshape}_{3\times4\times4\times4}(\widehat b_{\mathbf r}).
\]

It does not replace the brick with `mean(z) * ones`.

## 7. Residual and Omega recurrence

\[
E_{\mathbf r,t}=\widehat V(\mathbf r)-B_t(\mathbf r).
\]

Omega is a leaky integral:

\[
\Omega_{t+1}(\mathbf r)=
\rho\Omega_t(\mathbf r)-\eta E_{\mathbf r,t},
\qquad
0<\rho<1.
\]

The characteristic decay time is approximately

\[
\tau_\Omega\approx\frac{1}{1-\rho}
\]

cycles when \(\rho\) is close to one.

## 8. True cross-brick six-face diffusion

For every channel and voxel, the discrete Laplacian is

\[
\Delta B_t(\mathbf r,u)=
\sum_{q\in\mathcal N_6(\mathbf r,u)}B_t(q)-6B_t(\mathbf r,u).
\]

At an internal voxel, neighbours are read from the same brick. At a face voxel,
the neighbour is resolved through the sparse directory into the adjacent brick,
using the opposite face coordinate. This is not `np.roll` with periodic wrapping
inside one brick.

## 9. Projection

For the box constraint

\[
\Lambda=[16,235]^{192},
\]

the orthogonal projection is element-wise clipping:

\[
\Pi_\Lambda(v)_j=\min(235,\max(16,v_j)).
\]

For a general admissible set,

\[
\Pi_\Lambda(v)=\arg\min_{y\in\Lambda}\|y-v\|_2.
\]

## 10. Operational transaction

For each active brick and its causal face frontier:

\[
\widetilde B_{t+1}=B_t+\Delta t\left[
D\Delta B_t
-K\left(\mathcal D_\theta(\mathcal G_{MoE}(\mathcal E_\theta(B_t)\mid\Omega_t))-B_t\right)
+\Omega_{t+1}+U_t
\right].
\]

Then

\[
B_{t+1}=\Pi_\Lambda(\widetilde B_{t+1}).
\]

The unified sparse-field form is

\[
\boxed{
\Sigma_{t+1}=\Pi_\Lambda\left[
\Sigma_t+W_{\mathcal M}\odot
\left(D\Delta\Sigma_t
-K(\mathcal D_\theta(\mathcal G_{MoE}(\mathcal E_\theta(\Sigma_t)\mid\Omega_t))-\Sigma_t)
+\Omega_{t+1}+U_t\right)
\right]
}
\]

where \(W_{\mathcal M}\) denotes sparse scheduling semantics rather than a
materialised dense mask.

All updates are calculated against an immutable transaction snapshot. The
candidate is checked for finite values, projection bounds, Omega bounds, expert
index validity, relative energy, and materialisation budget before the directory
and journal are replaced atomically.

## 11. Frontier control

The next frontier is

\[
\mathcal F_t=\mathcal A_t\cup\partial_6\mathcal A_t.
\]

It is ranked and capped by `max_active_bricks`. A brick is pruned after
`prune_after` low-activity cycles when both deviation from its procedural
background and Omega magnitude fall below `activation_threshold`.

Without thresholding and pruning, six-face expansion may grow as much as

\[
M_{t+1}\lesssim7M_t
\]

before overlap, eventually exhausting the finite cache.

## 12. Stability contract

The reference explicit scheme enforces

\[
\Delta tD\le\frac16,
\qquad
\Delta tK\le1,
\qquad
0<\rho<1.
\]

It also clips the projected field, bounds Omega, caps active bricks, and rolls
back non-finite or over-energy candidates.

## 13. Complexity

For latent dimension \(d\) and \(M_t\) materialised bricks:

\[
T_t=O\left(M_t(192d+d^2+192\cdot6)\right),
\qquad
S_t=O(M_t\cdot192).
\]

The tetration height changes the symbolic address description. It does not change
the number of bricks physically evaluated in one cycle.

## 14. Run

```bash
jarvisx universe --tower-height 2
jarvisx universe --tower-height 4
jarvisx automaton --steps 20 --tower-height 4 --max-active 128
pytest tests/test_tetration_field.py
```
