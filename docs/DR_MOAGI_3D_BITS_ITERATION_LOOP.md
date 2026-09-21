# Dr Moagi 3D Bits Iteration Loop — Bounded Reference

## Status

Integration candidate. This document describes the executable adapter in
`src/jarvisx/recursive_bits3d.py`.

It specializes the existing volumetric bytecode runtime into an explicit
bit/byte-to-3D mapping and adds a symbolic septillion power-tower horizon. It
does not replace JX3DVM1, ADR-016 transaction authority, ADR-017 operational
autoencoding semantics, or the existing `VolumetricBytecodeVM`.

## 1. Exact logical state

The default reference geometry is

[
N=64,qquad C=4,qquad B=8	ext{ bits/lane}.
]

Therefore

[
V=N^3C=64^3cdot4=1,048,576
]

byte lanes and

[
N_{m bits}=VB=8,388,608
]

logical raw bits are present in one complete dense state.

The physical implementation is sparse: zero lanes are implicit and only active
lanes are resident in the wrapped volumetric VM.

## 2. Bit-address mapping

A scalar byte-lane index is

[
i=(((zN+y)N+x)C+c).
]

The inverse is

[
c=imod C,
]

[
x=leftlfloor i/Cightfloormod N,
]

[
y=leftlfloor i/(CN)ightfloormod N,
]

[
z=leftlfloor i/(CN^2)ightfloor.
]

A bit address adds the lane bit

[
bin{0,ldots,7},
qquad
j=8i+b.
]

Thus the VM never requires physically cubic RAM. Ordinary linear memory is
given 3D semantics by deterministic indexing.

## 3. End-to-end bounded cycle

The executable cycle is

```text
finite input bytes
  -> omit zero lanes
  -> linear byte index
  -> (x,y,z,channel)
  -> sparse active state
  -> octree accounting
  -> exact 8-bit encode
  -> XYZ mirror fold
  -> mirror union
  -> exact decode
  -> XOR residual
  -> Hamming evidence
  -> CTR verification
  -> commit OR rollback
  -> permeation receipt
  -> recur within max_iterations
```

The current adapter deliberately delegates the core state transition to
`VolumetricBytecodeVM`. There is one state authority, one verification path
and one bounded physical iteration budget.

## 4. Inward recursion

The reference operator can be written

[
S_{t+1}=mathcal M(S_t).
]

The loop is

[
S_0	omathcal M(S_0)	omathcal M^2(S_0)	ocdots
]

until one of the following occurs:

1. active-change convergence reaches the configured threshold;
2. verification fails and promotion is rejected;
3. the finite physical iteration budget is exhausted.

The current reference mirror-union operator is monotone and reaches a
mirror-symmetric fixed point for bounded sparse states.

## 5. Septillion power-tower horizon

Define

[
a=10^{24}.
]

The requested logical horizon is

[
T=a^{(a^a)}.
]

The adapter stores this as the syntax tree

```text
a^(a^(a))
```

with

```text
a = 1000000000000000000000000
```

and an equivalent symbolic base-10 form

```text
10^(24*10^(24*10^(24)))
```

No integer equal to `T` is created. No loop counter is allocated to `T`.
No array has `T` elements.

The contract is

[
oxed{	ext{symbolic logical horizon}
eq	ext{physical iterations}
eq	ext{measured throughput}}.
]

## 6. Why the symbolic horizon is still useful

An enormous logical horizon can remain meaningful as metadata for future
closed-form, fixed-point, memoized, sparse or jump-ahead operators.

For a composable operator,

[
J_0=mathcal M,
qquad
J_{k+1}=J_kcirc J_k,
qquad
J_k=mathcal M^{2^k}.
]

But this identity alone does not establish an acceleration for an arbitrary
adaptive nonlinear runtime. A jump implementation must demonstrate that
composition is cheaper than replay and must verify the resulting state against
the canonical evidence boundary.

The current adapter therefore does not claim to execute the power tower in
physical time.

## 7. Reference usage

```python
from jarvisx.recursive_bits3d import Bits3DConfig, RecursiveBits3DVM

vm = RecursiveBits3DVM(
    Bits3DConfig(
        side=64,
        channels=4,
        max_iterations=64,
    )
)

vm.load_bytes(b"Jarvis-X")
result = vm.run()

print(result.symbolic_iteration_target)
print(result.physical_iterations)
print(result.final_state_hash)
```

Run the bounded demonstration with

```bash
python -m jarvisx.recursive_bits3d --demo
```

## 8. Permeation law

The repository-level specialization is

[
oxed{
	ext{bits}
ightarrow
	ext{3D address}
ightarrow
E
ightarrow
Phi_{m 3D}
ightarrow
Omega
ightarrow
D
ightarrow
Delta
ightarrow
mathrm{CTR}
ightarrow
mathrm{commit|rollback}
ightarrow
	ext{receipt}
ightarrow
mathrm{recur}
}
]

with the symbolic target carried beside, rather than substituted for, the
finite executor.

## 9. Evidence boundary

This reference establishes deterministic addressing, sparse residency,
bounded recursion, codec round-trip verification, residual accounting and
symbolic power-tower representation.

It does not establish:

- septillion-tower physical iterations per second;
- a speedup of that magnitude;
- lossless compression without retained residual information;
- convergence of arbitrary learned nonlinear operators;
- hardware-level electromagnetic wave computation;
- AGI, consciousness or semantic correctness.

Performance claims require reproducible benchmark data on named hardware.
