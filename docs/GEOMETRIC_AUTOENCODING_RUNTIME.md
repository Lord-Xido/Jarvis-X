# Geometric Auto-Encoding and Decoding Runtime

## Status

Canonical Jarvis-X implementation of the geometric state cycle:

\[
X \xrightarrow{E_G} Z_G \xrightarrow{\mathcal T} Z'_G \xrightarrow{D_G} \hat X
\]

The runtime treats geometry as a computational representation rather than a
post-processing visualization. Arithmetic state is assigned exact lattice
coordinates, transformed through a bijective coordination map, validated, and
committed transactionally.

## 1. Address–coordinate bijection

For a lattice with depth `D`, height `H`, and width `W`, arithmetic address
`a` is decoded into

\[
\gamma(a)=
\left(
\left\lfloor\frac{a}{HW}\right\rfloor,
\left\lfloor\frac{a\bmod HW}{W}\right\rfloor,
 a\bmod W
\right).
\]

The exact inverse is

\[
\gamma^{-1}(z,y,x)=x+W(y+Hz).
\]

The implementation validates the identity

\[
\gamma^{-1}(\gamma(a))=a
\]

for every active address before accepting the geometry.

## 2. Canonical latent state

The implemented finite-grid latent state is

\[
Z_G=(C,A,E,\Omega,\Lambda),
\]

where:

- `C` is the coordinate domain;
- `A` is the value or attribute attached to each coordinate;
- `E` is six-connected lattice topology;
- `Omega` is spatial residual memory;
- `Lambda` is the constraint dictionary governing valid execution.

The Python representation is `GeometricLatent` in
`src/jarvisx/geometric_codec.py`.

## 3. Geometric transformation

A discrete geometric coordination transform is represented by a permutation
`pi` of all `N = D*H*W` arithmetic addresses. Its geometric action is the
conjugated map

\[
T_{\pi}=\gamma\circ\pi\circ\gamma^{-1}.
\]

Because `gamma` and `pi` are bijective,

\[
T_{\pi}^{-1}\circ T_{\pi}=I.
\]

`PermutationTransform` rejects duplicate, missing, negative, and out-of-range
destinations at construction time.

## 4. Validation gate

Every candidate state is checked for:

1. arithmetic-coordinate round-trip identity;
2. exact lattice cardinality;
3. complete coordinate-domain coverage;
4. symmetric and in-bounds topology;
5. finite numeric values.

The transaction law is

\[
Z_{t+1}=
\begin{cases}
\Pi_{\Lambda}(\hat Z_{t+1}), & V(\hat Z_{t+1})=1,\\
Z_t, & V(\hat Z_{t+1})=0.
\end{cases}
\]

In the current exact permutation kernel, projection is validation-preserving:
a valid candidate is committed unchanged. Later continuous, mesh, field, or
accelerated kernels may implement nontrivial projection behind this same gate.

## 5. Transactional execution

`GeometricRuntime` exposes the state machine:

```text
LOAD -> PROPOSE -> VALIDATE -> COMMIT
                         \-> ROLLBACK
```

Every transition records:

- monotonic journal sequence;
- operation type;
- latent version;
- lattice shape;
- SHA-256 state digest;
- validation outcome;
- commit status.

A pending candidate never replaces the committed state until `commit()` is
called after successful validation.

## 6. Spatial Omega memory

Observed and predicted values generate a residual at each coordinate:

\[
E_t(\mathbf r)=X_t^{observed}(\mathbf r)-X_t^{predicted}(\mathbf r).
\]

Memory updates locally:

\[
\Omega_{t+1}(\mathbf r)
=\rho\Omega_t(\mathbf r)+\eta E_t(\mathbf r).
\]

This preserves where a correction occurred instead of collapsing all residual
information into one global scalar.

## 7. CodexVM integration

`CodexVM` owns one `GeometricRuntime` as `vm.geometry`.

```python
from jarvisx.core import CodexVM

vm = CodexVM()
vm.load_geometry(list(range(8)), (2, 2, 2))
vm.execute_geometry((1, 2, 3, 4, 5, 6, 7, 0))

values = vm.geometry.codec.decode(vm.geometry.committed)
```

This establishes geometry as a first-class VM subsystem alongside registers,
memory, decoding, execution, ethics, reflex stabilization, sandbox enforcement,
tracing, and the persistent ledger.

## 8. Current boundary

The current kernel is deliberately exact, finite, and deterministic. It does
not claim compression, continuous-manifold inference, rendering, or GPU
acceleration. Those capabilities should be added as interchangeable kernels
behind the same contracts:

```text
encode -> stage -> validate -> project -> commit/rollback -> decode -> journal
```

The control plane remains stable while the execution plane can evolve from
pure Python to NumPy, native SIMD, GPU, sparse octree, mesh, graph, or neural
field implementations.

## Governing invariant

\[
\boxed{
\text{Arithmetic identifies state; geometry coordinates state; topology relates
state; decoding manifests state.}
}
\]
