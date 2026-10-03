# ADR-034: Exact Bitwise Fixed-Point Verification

- **Status:** Accepted for reference implementation
- **Date:** 2026-10-03
- **Applies to:** Dr Moagi 3D bit self-loop, recursive verification, sparse pathway gating, candidate-state promotion
- **Extends:** ADR-016, ADR-017, ADR-031, and `docs/research/DR_MOAGI_3D_BIT_SELF_LOOP.md`

## Context

Jarvis-X already defines a bounded binary specialization of the inward
encode/refine/decode/verify loop. The remaining ambiguity was at the lowest
machine level: what exactly counts as convergence, how a million pathway
verification claim is represented, how memory writes remain bounded, and how
the terminal `32^3 -> ... -> 1^3` fold behaves under literal binary shifts.

A hardware-level interpretation must distinguish:

- mathematical fixed-point equality from approximate floating-point convergence;
- bit integrity checks from numerical invariants;
- spatial address contraction from information-theoretic uncertainty reduction;
- virtual pathway count from physically executed work.

## Decision

### 1. Exact binary fixed-point residual

For a deterministic finite binary state

[
B_tin{0,1}^N
]

and transition operator (mathcal F), define

[
D_t=mathcal F(B_t)oplus B_t,
]

[
E_{m bit}=operatorname{popcount}(D_t),
]

and

[
e_{m bit}=rac{E_{m bit}}{N}.
]

Exact machine-state closure is

[
oxed{E_{m bit}=0}
]

which is equivalent to

[
oxed{mathcal F(B^*)=B^*}
]

and

[
oxed{mathcal F(B^*)oplus B^*=0^N}.
]

This is the canonical exact binary fixed-point test. Floating-point and learned
kernels may still use numerical tolerances, but those are distinct acceptance
contracts.

### 2. Million-pathway verification bitmap

For the one-million-pathway profile, define

[
Vin{0,1}^{1,000,000}.
]

Each bit represents one declared pathway verification receipt:

```text
V[i] = 1  pathway i passed
V[i] = 0  pathway i failed
```

The bitmap therefore occupies exactly

[
1,000,000 {m bits}=125,000 {m bytes}
]

and exactly

[
15,625
]

64-bit words.

Global pass requires

[
oxed{operatorname{popcount}(V)=1,000,000}.
]

A report may not state that all pathways were verified unless the corresponding
evidence mask or equivalent auditable receipt supports that claim.

### 3. Masked memory mutation

For committed memory (Omega), candidate information (I), and write mask
(M), the reference bounded write is

[
oxed{
Omega'=(Omegaland
eg M)lor(Iland M).
}
]

A zero mask bit preserves the committed bit. A one mask bit permits the
candidate bit to replace it.

### 4. Binary transaction boundary

For verification bit (V_tin{0,1}), baseline state (B_t), and candidate
state (B^{cand}_{t+1}), authoritative selection is a digital multiplexer:

[
select=0-V_t,
]

[
oxed{
B_{t+1}
=
(B_tland
eg select)
lor
(B^{cand}_{t+1}land select).
}
]

This is the machine-level realization of the ADR-016 candidate-first promotion
rule.

### 5. Saturating spatial contraction

Plain unsigned shifting gives

```text
1 >> 1 == 0
```

so it does not make `1` a fixed point.

The canonical spatial side-length contraction is therefore

[
oxed{
N_{k+1}=max(1,N_kgg1).
}
]

The dyadic cascade

[
32	o16	o8	o4	o2	o1
]

uses axis-address widths

[
5	o4	o3	o2	o1	o0.
]

In 3D this removes one address bit per coordinate axis at each fold. It is a
coarse-graining of spatial address precision; it is not automatically a
one-bit reduction in Shannon entropy.

### 6. Hysteretic activity pruning

One threshold can cause repeated activation/deactivation near the boundary.
The bit substrate therefore uses two thresholds:

[
epsilon_{m on}>epsilon_{m off}.
]

```text
inactive -> active  when error > epsilon_on
active   -> inactive when error < epsilon_off
otherwise retain previous activity state
```

This keeps sparse scheduling stable while permitting reactivation when a
previously converged region becomes inconsistent.

## Numerical invariants versus integrity checks

A mathematical norm invariant such as

[
|v|_2^2=c
]

is not equivalent to parity or checksum equality. Jarvis-X therefore keeps two
verification classes separate:

[
V_{m numerical}
]

for mathematical residuals and tolerances, and

[
V_{m integrity}
]

for CRC/hash/parity or representation-integrity checks.

A composite gate may require both, but one must not be reported as the other.

## Reference implementation

The accepted C++ reference lives in:

```text
cpp_runtime/include/jarvisx/bit_self_loop3d.hpp
cpp_runtime/tests/bit_self_loop3d_tests.cpp
```

It exposes:

- `fixed_point_residual`;
- `is_exact_fixed_point`;
- `masked_overwrite`;
- `verified_mux`;
- `saturating_half_extent`;
- `dyadic_axis_bits`;
- `hysteretic_activity_gate`;
- `VerificationBitmap1M`.

The regression suite verifies exact XOR/popcount residuals, one-million-bit
pass accounting, mask-preserving writes, saturating spatial closure,
hysteresis, and candidate commit/rollback behavior.

## Consequences

The binary interpretation is now executable rather than metaphorical. Jarvis-X
can report exact Hamming distance to the next deterministic state and exact
per-pathway pass/fail evidence.

This does **not** establish external correctness, intelligence, zero latency,
zero floating-point drift, or state-of-the-art performance. Exact internal
binary closure means only that the declared finite transition would reproduce
the declared finite state.

## Canonical compact law

[
oxed{
egin{aligned}
B^{cand}_{t+1} &= mathcal F(B_t,Omega_t,Pi_t),\
D_t &= B^{cand}_{t+1}oplus B_t,\
E_t &= operatorname{popcount}(D_t),\
V_t &= V_{m invariant}land V_{m integrity}land V_{m CTR}land V_{m resource},\
B_{t+1} &= operatorname{MUX}(V_t,B_t,B^{cand}_{t+1}).
end{aligned}
}
]

The exact binary fixed point is

[
oxed{
operatorname{popcount}(mathcal F(B^*)oplus B^*)=0.
}
]
