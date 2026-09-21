# Dr Moagi 3D Volumetric Bytecode — Bounded Reference

## Status

Software reference implementation. This layer operationalizes the supplied
volumetric pseudocode without replacing the canonical `JX3DVM1` authority
boundary.

## 1. Logical versus physical execution

The supplied target

```text
1,000,000 ^ 1,000,000 = 10 ^ 6,000,000
```

is stored as an **unevaluated symbolic iteration target**. The runtime never
constructs that integer and never attempts to enumerate that many cycles.

The default logical spatial side is `10^24`, so one modality has a logical
address domain of `(10^24)^3 = 10^72` voxel coordinates. The reference keeps
only non-zero active voxels in a sparse map.

The physical execution budget is separately bounded by:

- `max_active_voxels`;
- `max_iterations`;
- `octree_depth`;
- the fixed 8-bit reference lane width.

The invariant is:

```text
astronomical logical extent != resident memory != physical instruction count
```

## 2. Reference cycle

The executable cycle is:

```text
SPARSE_MASK
 -> TRAVERSE_OCT
 -> SIMD_ENCODE
 -> FOLD_XYZ_MIRROR
 -> INJECT_MIRROR_UNION
 -> SIMD_DECODE
 -> DELTA_XOR
 -> HAMMING_CHECK
 -> VERIFY_CTR
 -> PERMEATE
 -> recur | halt
```

`PERMEATE` publishes a verified snapshot/receipt. It does not grant authority to
mutate external systems.

## 3. Typed lane semantics

The reference implementation uses 8-bit voxels. A 512-bit SIMD register would
therefore contain **64 byte lanes**, not 512 voxels. A future bit-plane backend
may process 512 one-bit voxels per AVX-512 vector, but the two representations
must remain distinct.

The reference encoder is an invertible one-bit rotate within each byte and the
decoder is the inverse rotate. This intentionally simple codec makes exact
round-trip verification testable:

```text
decode(encode(x)) == x
```

for every 8-bit lane.

## 4. Exact 3D mirror geometry

For logical side `N`, the mirror operator is:

\[
\mu(x,y,z)=(N-1-x,\;N-1-y,\;N-1-z).
\]

This differs from the AVX-512 blockwise reverse kernel. The blockwise kernel
operates on flattened 512-bit blocks; this runtime computes the actual XYZ
coordinate mirror.

The supplied `INJECT` step is specialized as mirror union:

\[
Z'(q)=Z(q)\;\lor\;Z(\mu(q)).
\]

This is a monotone binary closure. After decoding, repeated application reaches
a mirror-symmetric fixed point for the reference operator.

## 5. Sparse octree traversal

`TRAVERSE_OCT DEPTH=d` partitions each logical axis into `2^d` bins but only
materializes IDs for bins containing active voxels. It does not allocate the
full tree.

Thus work scales with the active support rather than the virtual volume:

\[
W_t = O(|\mathcal A_t|)
\]

for the sparse reference bookkeeping.

## 6. Delta and convergence

The binary delta plane is:

\[
\Delta_t=S_{t+1}\oplus S_t.
\]

The changed-bit count is:

\[
C_t=\operatorname{popcount}(\Delta_t).
\]

Two fractions are reported:

\[
f_{\rm logical}=\frac{C_t}{N_{\rm logical\ bits}},
\qquad
f_{\rm active}=\frac{C_t}{N_{\rm active\ union\ bits}}.
\]

The huge logical denominator can make `f_logical` tiny even when every active
voxel changed. Therefore **convergence is gated by `f_active`, not by the global
logical fraction**.

This fixes an important failure mode in literal interpretations of the supplied
pseudocode.

## 7. Verification and commit

Before a candidate state is promoted, the reference verifies the local codec
invariant:

```text
codec_roundtrip_error_bits == 0
```

and all configured bounds must already have passed. This is a local CTR-style
receipt. It does not claim that a mirror-symmetric state is semantically correct
for an external task.

## 8. Operator exponentiation boundary

The equation

\[
F^{[2^{k+1}]}=F^{[2^k]}\circ F^{[2^k]}
\]

is valid as a description of repeated operator composition. It does **not** mean
an arbitrary stateful autoencoder can automatically jump to iteration `T` in
`O(log T)` time. Such acceleration requires a closed/composable representation
of the operator and a cheaper way to compose powers than to replay their work.

The reference therefore preserves `10^6,000,000` as symbolic metadata while
executing a finite fixed-point budget.

## 9. Example

```python
from jarvisx.volumetric_bytecode import VolumetricBytecodeVM, VolumetricConfig

vm = VolumetricBytecodeVM(
    VolumetricConfig(axis_extent=16, max_iterations=8)
)
vm.load_modalities(
    {
        "gui": {(0, 1, 2): 0x03},
        "audio": {(3, 4, 5): 0x55},
        "user": {(7, 1, 0): 0x80},
    }
)
result = vm.run()

assert result.converged
assert result.symbolic_target == "10^6000000"
```

## 10. Final execution law

\[
X_t
\rightarrow \mathcal A_t
\rightarrow E(X_t)
\rightarrow \Phi_{\rm mirror}(E(X_t))
\rightarrow D
\rightarrow \Delta_t
\rightarrow \mathrm{CTR}
\rightarrow \mathrm{commit|rollback}
\rightarrow \mathrm{snapshot}
\rightarrow \mathrm{recur}.
\]

The logical depth may be enormous. The actual executor remains finite,
measurable, sparse, deterministic, and bounded.
